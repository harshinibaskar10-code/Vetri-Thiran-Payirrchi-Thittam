"""Password hashing and JWT helpers."""
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt

import config

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], hashed.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(username: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": username, "exp": exp}, config.SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[str]:
    try:
        return jwt.decode(token, config.SECRET_KEY, algorithms=[ALGORITHM]).get("sub")
    except jwt.PyJWTError:
        return None