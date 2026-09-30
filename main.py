"""PocketSmart: AI Budget Planner – FastAPI application."""
import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, Optional

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

import auth
import config
import database
import gemini_utils
from models import HomeBudgetInput, JewelryBudgetInput, PartyBudgetInput, RegisterUser

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pocketsmart")

# username -> {"login_time", "last_activity", "user_data"}
active_sessions: Dict[str, Dict[str, Any]] = {}
SESSION_TIMEOUT_SECONDS = 1800


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _cleanup_expired_sessions() -> None:
    while True:
        await asyncio.sleep(300)
        for name, s in list(active_sessions.items()):
            if (_now() - s["last_activity"]).total_seconds() > SESSION_TIMEOUT_SECONDS:
                active_sessions.pop(name, None)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    database.init_db()
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    mode = f"Gemini ({config.GEMINI_MODEL})" if gemini_utils.ai_enabled() else "OFFLINE fallback (no API key)"
    log.info("Starting PocketSmart: AI Budget Planner - AI mode: %s", mode)
    task = asyncio.create_task(_cleanup_expired_sessions())
    yield
    task.cancel()


app = FastAPI(title="PocketSmart: AI Budget Planner", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(config.UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(config.BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(config.BASE_DIR, "templates"))


def render(request: Request, name: str, **ctx: Any):
    return templates.TemplateResponse(request, name, ctx)


# --------------------------------------------------------------------------- #
# Authentication helpers
# --------------------------------------------------------------------------- #
class NotAuthenticated(Exception):
    """Raised by page routes so the visitor is redirected to /login."""


@app.exception_handler(NotAuthenticated)
async def _redirect_to_login(_request: Request, _exc: NotAuthenticated):
    return RedirectResponse("/login", status_code=302)


def _user_from_request(request: Request) -> Optional[Dict[str, Any]]:
    token = request.cookies.get("access_token")
    if not token:
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            token = header[7:]
    if not token:
        return None
    username = auth.decode_token(token)
    return database.get_user(username) if username else None


def _touch_session(username: str) -> Dict[str, Any]:
    session = active_sessions.get(username)
    if session is None:
        session = {"login_time": _now(), "last_activity": _now(), "user_data": {}}
        active_sessions[username] = session
    session["last_activity"] = _now()
    return session


def page_user(request: Request) -> Dict[str, Any]:
    user = _user_from_request(request)
    if not user:
        raise NotAuthenticated()
    _touch_session(user["username"])
    return user


def api_user(request: Request) -> Dict[str, Any]:
    user = _user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated. Please sign in.")
    _touch_session(user["username"])
    return user


# --------------------------------------------------------------------------- #
# Public pages
# --------------------------------------------------------------------------- #
@app.get("/")
def index(request: Request):
    return render(request, "index.html", user=_user_from_request(request))


@app.get("/health")
def health():
    return {"status": "ok", "ai_mode": "gemini" if gemini_utils.ai_enabled() else "fallback",
            "model": config.GEMINI_MODEL}


@app.get("/login")
def login_page(request: Request):
    if _user_from_request(request):
        return RedirectResponse("/dashboard", status_code=302)
    return render(request, "login.html", user=None)


@app.get("/register")
def register_page(request: Request):
    if _user_from_request(request):
        return RedirectResponse("/dashboard", status_code=302)
    return render(request, "register.html", user=None)


# --------------------------------------------------------------------------- #
# Auth API
# --------------------------------------------------------------------------- #
@app.post("/register", status_code=201)
def register(payload: RegisterUser):
    if database.get_user(payload.username):
        raise HTTPException(status_code=409, detail="That username is already taken.")
    if database.get_user_by_email(payload.email):
        raise HTTPException(status_code=409, detail="An account with that email already exists.")
    database.create_user(payload.username, payload.email.lower(), payload.full_name,
                         auth.hash_password(payload.password))
    return {"message": "Account created. You can now sign in."}


@app.post("/token")
def login_for_access_token(form: OAuth2PasswordRequestForm = Depends()):
    user = database.get_user(form.username)
    if not user or not auth.verify_password(form.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password.",
                            headers={"WWW-Authenticate": "Bearer"})
    token = auth.create_access_token(user["username"])
    active_sessions[user["username"]] = {"login_time": _now(), "last_activity": _now(), "user_data": {}}
    response = JSONResponse({"access_token": token, "token_type": "bearer"})
    response.set_cookie("access_token", token, httponly=True, samesite="lax",
                        max_age=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    return response


def _end_session(request: Request) -> None:
    user = _user_from_request(request)
    if user:
        active_sessions.pop(user["username"], None)


@app.get("/logout")
def logout_get(request: Request):
    _end_session(request)
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("access_token")
    return response


@app.post("/logout")
def logout_post(request: Request):
    _end_session(request)
    response = JSONResponse({"message": "Logged out"})
    response.delete_cookie("access_token")
    return response


@app.get("/session-info")
def session_info(user: dict = Depends(api_user)):
    s = _touch_session(user["username"])
    return {
        "username": user["username"],
        "login_time": s["login_time"].isoformat(),
        "last_activity": s["last_activity"].isoformat(),
        "session_duration_minutes": int((_now() - s["login_time"]).total_seconds() // 60),
        "user_data": s["user_data"],
    }


@app.post("/session-data")
def update_session_data(data: Dict[str, Any] = Body(...), user: dict = Depends(api_user)):
    s = _touch_session(user["username"])
    s["user_data"].update(data)
    return {"message": "Session data updated", "data": s["user_data"]}


# --------------------------------------------------------------------------- #
# Protected pages
# --------------------------------------------------------------------------- #
@app.get("/dashboard")
def dashboard(request: Request, user: dict = Depends(page_user)):
    recent = database.list_history(user["username"], limit=5)
    return render(request, "dashboard.html", user=user, recent=recent)


@app.get("/home-planner")
def home_planner(request: Request, user: dict = Depends(page_user)):
    return render(request, "home_planner.html", user=user)


@app.get("/party-planner")
def party_planner(request: Request, user: dict = Depends(page_user)):
    return render(request, "party_planner.html", user=user)


@app.get("/jewelry-planner")
def jewelry_planner(request: Request, user: dict = Depends(page_user)):
    return render(request, "jewelry_planner.html", user=user)


@app.get("/history")
def history_page(request: Request, user: dict = Depends(page_user)):
    return render(request, "history.html", user=user)


# --------------------------------------------------------------------------- #
# Recommendation API
# --------------------------------------------------------------------------- #
def _finish(user: dict, rec_type: str, input_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    session = _touch_session(user["username"])
    session["user_data"][f"last_{rec_type}_budget"] = {
        "timestamp": _now().isoformat(),
        "budget": result["total_budget"],
        "type": rec_type,
    }
    result["history_id"] = database.save_history(
        user["username"], rec_type, result["total_budget"], result["remaining_budget"], input_data, result)
    return result


@app.post("/home-budget")
@app.post("/generate-home")
def plan_home_budget(budget_input: HomeBudgetInput, user: dict = Depends(api_user)):
    result = gemini_utils.get_home_recommendations(budget_input)
    return _finish(user, "home", budget_input.model_dump(), result)


@app.post("/party-budget")
@app.post("/generate-party")
def plan_party_budget(budget_input: PartyBudgetInput, user: dict = Depends(api_user)):
    result = gemini_utils.get_party_recommendations(budget_input)
    return _finish(user, "party", budget_input.model_dump(), result)


async def _read_image(upload: UploadFile):
    """Validate and store an uploaded outfit image. Returns (bytes, mime, saved_filename)."""
    data = await upload.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large (max 5 MB).")
    try:
        img = Image.open(BytesIO(data))
        fmt = (img.format or "").upper()
        img.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(fmt)
    if not mime:
        raise HTTPException(status_code=400, detail="Only JPG, PNG or WEBP images are supported.")
    ext = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}[fmt]
    filename = f"{_now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}.{ext}"
    with open(os.path.join(config.UPLOAD_DIR, filename), "wb") as f:
        f.write(data)
    return data, mime, filename


@app.post("/jewelry-budget")
@app.post("/generate-jewelry")
async def plan_jewelry_budget(
    total_budget: float = Form(...),
    occasion: str = Form(...),
    preferences: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    user: dict = Depends(api_user),
):
    try:
        budget_input = JewelryBudgetInput(
            total_budget=total_budget, occasion=occasion.strip(), preferences=(preferences or "").strip() or None)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="; ".join(e["msg"] for e in exc.errors()))

    image_bytes = mime = filename = None
    if image is not None and image.filename:
        image_bytes, mime, filename = await _read_image(image)

    result = await run_in_threadpool(gemini_utils.get_jewelry_recommendations, budget_input, image_bytes, mime)
    input_data = budget_input.model_dump()
    input_data["has_image"] = image_bytes is not None
    if filename:
        input_data["image"] = filename
    return _finish(user, "jewelry", input_data, result)


@app.get("/recommendation-history")
def recommendation_history(user: dict = Depends(api_user)):
    return {"history": database.list_history(user["username"], limit=100)}


@app.get("/recommendation-details/{recommendation_id}")
def recommendation_details(recommendation_id: str, user: dict = Depends(api_user)):
    item = database.get_history_item(user["username"], recommendation_id)
    if not item:
        raise HTTPException(status_code=404, detail="Recommendation not found.")
    return item


if __name__ == "__main__":
    import uvicorn

    print("Starting PocketSmart: AI Budget Planner on http://127.0.0.1:8000 ...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)