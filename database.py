"""SQLite persistence for users and recommendation history (standard library only)."""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import config


@contextmanager
def db():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db() -> None:
    with db() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS users(
                username TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL)"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS history(
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                rec_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                budget REAL,
                remaining REAL,
                input_json TEXT,
                result_json TEXT)"""
        )


def create_user(username: str, email: str, full_name: Optional[str], password_hash: str) -> None:
    with db() as c:
        c.execute(
            "INSERT INTO users(username,email,full_name,password_hash,created_at) VALUES (?,?,?,?,?)",
            (username, email, full_name, password_hash, now_iso()),
        )


def get_user(username: str) -> Optional[Dict[str, Any]]:
    with db() as c:
        row = c.execute("SELECT * FROM users WHERE lower(username)=lower(?)", (username,)).fetchone()
    return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with db() as c:
        row = c.execute("SELECT * FROM users WHERE lower(email)=lower(?)", (email,)).fetchone()
    return dict(row) if row else None


def save_history(username: str, rec_type: str, budget: float, remaining: float,
                 input_data: Dict[str, Any], result: Dict[str, Any]) -> str:
    hid = uuid.uuid4().hex[:12]
    with db() as c:
        c.execute(
            "INSERT INTO history(id,username,rec_type,created_at,budget,remaining,input_json,result_json)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (hid, username, rec_type, now_iso(), budget, remaining,
             json.dumps(input_data, default=str), json.dumps(result, default=str)),
        )
    return hid


def list_history(username: str, limit: int = 50) -> List[Dict[str, Any]]:
    with db() as c:
        rows = c.execute(
            "SELECT id,rec_type,created_at,budget,remaining,input_json FROM history"
            " WHERE username=? ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (username, limit),
        ).fetchall()
    return [
        {"id": r["id"], "type": r["rec_type"], "timestamp": r["created_at"],
         "budget": r["budget"], "remaining": r["remaining"],
         "input": json.loads(r["input_json"] or "{}")}
        for r in rows
    ]


def get_history_item(username: str, hid: str) -> Optional[Dict[str, Any]]:
    with db() as c:
        r = c.execute("SELECT * FROM history WHERE id=? AND username=?", (hid, username)).fetchone()
    if not r:
        return None
    return {"id": r["id"], "type": r["rec_type"], "timestamp": r["created_at"],
            "input": json.loads(r["input_json"] or "{}"),
            "full_result": json.loads(r["result_json"] or "{}")}