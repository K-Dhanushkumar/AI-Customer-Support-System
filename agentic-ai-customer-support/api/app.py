"""FastAPI application for the support assistant."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status

from agents.decision import SupportSystemState, answer_query
from api.auth import get_current_user, require_admin
from api.schemas import (
    AskRequest,
    AskResponse,
    ConversationCreateRequest,
    ConversationResponse,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    TokenResponse,
    UploadResponse,
    UserResponse,
)
from llm.hf_llm import generate_response
from rag.memory import build_memory_context
from rag.service import build_knowledge_base, build_support_system
from utils.config import settings
from utils.logging import get_logger, setup_logging
from utils.storage import (
    add_message,
    authenticate_user,
    bootstrap_admin_user,
    create_conversation,
    create_user,
    get_conversation,
    get_recent_messages,
    initialize_database,
    list_conversations,
)


def _get_system(app: FastAPI) -> SupportSystemState:
    """Read the initialized support system from app state."""

    system = getattr(app.state, "support_system", None)
    if system is None:
        raise RuntimeError("support system is not initialized")
    return system


def _refresh_support_system(app: FastAPI) -> None:
    """Reload the shared support system after vector store updates."""

    build_support_system.cache_clear()
    refreshed = build_support_system()
    app.state.support_system = SupportSystemState(index=refreshed.index, chunks=refreshed.chunks, top_k=settings.top_k)


def _save_uploaded_files(files: list[UploadFile]) -> list[str]:
    """Persist uploaded text files into the raw data directory."""

    saved_files: list[str] = []
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    for upload in files:
        filename = Path(upload.filename or "document.txt").name
        if not filename.lower().endswith(".txt"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="only .txt files are supported")

        destination = settings.raw_data_dir / filename
        content = upload.file.read()
        destination.write_bytes(content)
        saved_files.append(filename)
    return saved_files


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    setup_logging(settings.log_file)
    logger = get_logger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize shared resources before serving requests."""

        initialize_database()
        bootstrap_admin_user()
        system = build_support_system()
        app.state.support_system = SupportSystemState(index=system.index, chunks=system.chunks, top_k=settings.top_k)
        logger.info("support_system_initialized")
        yield

    app = FastAPI(title="Agentic AI Customer Support", version="1.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Health check endpoint."""

        return {"status": "ok"}

    @app.post("/auth/register", response_model=TokenResponse)
    def register(request: RegisterRequest) -> TokenResponse:
        """Register a new user and issue a token."""

        try:
            create_user(request.username, request.password)
            session = authenticate_user(request.username, request.password)
            return TokenResponse(**session)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.exception("registration_failed")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    @app.post("/auth/login", response_model=TokenResponse)
    def login(request: LoginRequest) -> TokenResponse:
        """Authenticate a user and issue a token."""

        try:
            return TokenResponse(**authenticate_user(request.username, request.password))
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    @app.post("/conversations", response_model=ConversationResponse)
    def start_conversation(
        request: ConversationCreateRequest | None = None,
        current_user: dict = Depends(get_current_user),
    ) -> ConversationResponse:
        """Create a new conversation for the authenticated user."""

        conversation = create_conversation(current_user["id"], title=(request.title if request else None))
        return ConversationResponse(**conversation)

    @app.get("/conversations", response_model=list[ConversationResponse])
    def get_user_conversations(current_user: dict = Depends(get_current_user)) -> list[ConversationResponse]:
        """List conversations owned by the authenticated user."""

        conversations = list_conversations(current_user["id"])
        return [ConversationResponse(**conversation) for conversation in conversations]

    @app.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
    def get_conversation_messages(conversation_id: int, current_user: dict = Depends(get_current_user)) -> list[MessageResponse]:
        """Return chat history for a conversation owned by the authenticated user."""

        conversation = get_conversation(conversation_id)
        if conversation is None or conversation["user_id"] != current_user["id"]:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="conversation not found")

        messages = get_recent_messages(conversation_id, limit=50)
        return [MessageResponse(id=message["id"], conversation_id=message["conversation_id"], role=message["role"], content=message["content"]) for message in messages]

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest, current_user: dict = Depends(get_current_user)) -> AskResponse:
        """Generate a support answer for the incoming query and store the full exchange."""

        try:
            system = _get_system(app)
            conversation_id = request.conversation_id
            if conversation_id is None:
                conversation = create_conversation(current_user["id"], title="Support chat")
                conversation_id = conversation["id"]
            else:
                conversation = get_conversation(conversation_id)
                if conversation is None or conversation["user_id"] != current_user["id"]:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="conversation not found")

            previous_messages = get_recent_messages(conversation_id, limit=settings.conversation_history_limit)
            add_message(conversation_id, "user", request.query)
            memory_context = build_memory_context(previous_messages)
            answer = answer_query(
                request.query,
                system,
                direct_response_fn=lambda query, memory_context="": generate_response(query, context=memory_context),
                memory_context=memory_context,
            )
            add_message(conversation_id, "assistant", answer)
            return AskResponse(conversation_id=conversation_id, query=request.query, answer=answer)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.exception("request_failed")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    @app.post("/admin/upload", response_model=UploadResponse)
    async def upload_documents(
        files: list[UploadFile] = File(...),
        current_user: dict = Depends(require_admin),
    ) -> UploadResponse:
        """Upload new support documents and rebuild the vector store."""

        try:
            saved_files = _save_uploaded_files(files)
            build_knowledge_base()
            _refresh_support_system(app)
            logger.info("documents_uploaded count=%s user=%s", len(saved_files), current_user["username"])
            return UploadResponse(uploaded_files=saved_files, vectorstore_rebuilt=True)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.exception("upload_failed")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return app


app = create_app()
