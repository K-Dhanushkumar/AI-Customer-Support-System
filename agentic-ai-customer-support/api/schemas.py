"""Pydantic schemas for the API layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    """Create-user request payload."""

    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginRequest(BaseModel):
    """Login request payload."""

    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class UserResponse(BaseModel):
    """Public user representation."""

    id: int
    username: str
    role: str


class TokenResponse(BaseModel):
    """Bearer token response payload."""

    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class ConversationResponse(BaseModel):
    """Conversation metadata."""

    id: int
    user_id: int
    title: str | None = None


class ConversationCreateRequest(BaseModel):
    """Conversation creation payload."""

    title: str | None = None


class MessageResponse(BaseModel):
    """Stored chat message payload."""

    id: int
    conversation_id: int
    role: str
    content: str


class AskRequest(BaseModel):
    """Request payload for answer generation."""

    query: str = Field(..., min_length=1)
    conversation_id: int | None = None


class AskResponse(BaseModel):
    """Response payload for answer generation."""

    conversation_id: int
    query: str
    answer: str


class UploadResponse(BaseModel):
    """Admin upload response payload."""

    uploaded_files: list[str]
    vectorstore_rebuilt: bool
