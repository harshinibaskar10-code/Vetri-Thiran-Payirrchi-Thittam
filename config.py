"""Central configuration – loads values from the .env file."""
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FORCE_MOCK = os.getenv("FORCE_MOCK", "").lower() in ("1", "true", "yes")

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "pocketsmart.db"))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
    if o.strip()
]