"""SQLite-backed persistence for authentication, conversations, and chat history."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
import secrets

from utils.config import settings
from utils.security import hash_password, hash_token, verify_password


def _utcnow() -> datetime:
    """Return the current UTC time."""

    return datetime.now(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    """Format a datetime for SQLite storage."""

    return (value or _utcnow()).isoformat()


@contextmanager
def get_connection(db_path: Path | None = None):
    """Open a SQLite connection with row access by column name."""

    connection = sqlite3.connect(str(db_path or settings.database_path), check_same_thread=False)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_database(db_path: Path | None = None) -> None:
    """Create all required SQLite tables and bootstrap the admin user."""

    database_path = db_path or settings.database_path
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            );
            """
        )

    bootstrap_admin_user(database_path)


def bootstrap_admin_user(db_path: Path | None = None) -> None:
    """Create the initial admin user when it does not already exist."""

    admin_username = settings.admin_username.strip()
    admin_password = settings.admin_password.strip()
    if not admin_username or not admin_password:
        return

    if get_user_by_username(admin_username, db_path=db_path) is None:
        create_user(admin_username, admin_password, role="admin", db_path=db_path)


def create_user(username: str, password: str, role: str = "user", db_path: Path | None = None) -> dict:
    """Create a new user record."""

    normalized_username = username.strip()
    normalized_password = password.strip()
    if not normalized_username:
        raise ValueError("username must not be empty")
    if not normalized_password:
        raise ValueError("password must not be empty")

    try:
        with get_connection(db_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (normalized_username, hash_password(normalized_password), role, _timestamp()),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        raise ValueError("username already exists") from exc

    return {"id": user_id, "username": normalized_username, "role": role}


def get_user_by_username(username: str, db_path: Path | None = None) -> dict | None:
    """Look up a user by username."""

    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT id, username, password_hash, role, created_at FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def create_access_token(user_id: int, db_path: Path | None = None) -> str:
    """Create and persist an opaque bearer token for a user."""

    token = secrets.token_urlsafe(32)
    expires_at = _utcnow() + timedelta(minutes=settings.token_ttl_minutes)
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO tokens (user_id, token_hash, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, hash_token(token), _timestamp(), _timestamp(expires_at)),
        )
    return token


def authenticate_user(username: str, password: str, db_path: Path | None = None) -> dict:
    """Authenticate credentials and issue a bearer token."""

    user = get_user_by_username(username, db_path=db_path)
    if user is None or not verify_password(password, user["password_hash"]):
        raise ValueError("invalid username or password")

    token = create_access_token(int(user["id"]), db_path=db_path)
    return {"access_token": token, "token_type": "bearer", "user": {"id": user["id"], "username": user["username"], "role": user["role"]}}


def get_user_by_token(token: str, db_path: Path | None = None) -> dict | None:
    """Resolve an authenticated user from a bearer token."""

    token_hash = hash_token(token)
    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT users.id, users.username, users.role, tokens.expires_at
            FROM tokens
            JOIN users ON users.id = tokens.user_id
            WHERE tokens.token_hash = ?
            """,
            (token_hash,),
        ).fetchone()
    if row is None:
        return None

    expires_at = datetime.fromisoformat(row["expires_at"])
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at <= _utcnow():
        return None
    return {"id": row["id"], "username": row["username"], "role": row["role"]}


def create_conversation(user_id: int, title: str | None = None, db_path: Path | None = None) -> dict:
    """Create a new conversation for a user."""

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO conversations (user_id, title, created_at)
            VALUES (?, ?, ?)
            """,
            (user_id, title, _timestamp()),
        )
    return {"id": cursor.lastrowid, "user_id": user_id, "title": title}


def list_conversations(user_id: int, db_path: Path | None = None) -> list[dict]:
    """Return all conversations owned by a user."""

    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, user_id, title, created_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(conversation_id: int, db_path: Path | None = None) -> dict | None:
    """Fetch a conversation by id."""

    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT id, user_id, title, created_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    return dict(row) if row else None


def add_message(conversation_id: int, role: str, content: str, db_path: Path | None = None) -> dict:
    """Store a message in a conversation."""

    normalized_content = content.strip()
    if not normalized_content:
        raise ValueError("message content must not be empty")

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, role, normalized_content, _timestamp()),
        )
    return {"id": cursor.lastrowid, "conversation_id": conversation_id, "role": role, "content": normalized_content}


def get_recent_messages(conversation_id: int, limit: int = 6, db_path: Path | None = None) -> list[dict]:
    """Return the most recent messages in a conversation."""

    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, conversation_id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def build_user_prompt_context(messages: Iterable[dict], max_messages: int | None = None) -> str:
    """Format recent messages into a prompt-friendly conversation memory block."""

    recent_messages = list(messages)
    if max_messages is not None:
        recent_messages = recent_messages[-max_messages:]
    if not recent_messages:
        return ""

    lines = ["Conversation memory:"]
    for message in recent_messages:
        speaker = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {message['content']}")
    return "\n".join(lines)
