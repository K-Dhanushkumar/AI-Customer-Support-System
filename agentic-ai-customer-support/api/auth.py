"""Authentication helpers for FastAPI dependencies."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from utils.storage import get_user_by_token


def _extract_bearer_token(authorization: str | None) -> str:
    """Extract a bearer token from the Authorization header."""

    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing authorization header")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid authorization header")
    return token.strip()


def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    """Resolve the authenticated user from the bearer token."""

    token = _extract_bearer_token(authorization)
    user = get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or expired token")
    return user


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """Ensure the authenticated user has admin privileges."""

    if current_user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="admin access required")
    return current_user
