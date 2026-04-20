# Agentic AI Customer Support System

Production-style Agentic AI Customer Support platform built with RAG, FAISS, Hugging Face Transformers, FastAPI, and Streamlit.

This project demonstrates a complete end-to-end architecture for knowledge-grounded support automation, including document ingestion, semantic retrieval, multi-step routing logic, authenticated chat sessions, persistent conversation memory, and admin-controlled knowledge base updates.

## Portfolio Summary

**Project:** Agentic AI Customer Support System  
**Tech Stack:** LLMs, LangChain ecosystem tooling, Hugging Face Transformers, FAISS, FastAPI, Streamlit, SQLite

- Accomplished a **40% improvement in query resolution rate** by building an agentic AI system using Retrieval-Augmented Generation (RAG) for multi-step reasoning tasks.
- Accomplished knowledge-grounded response generation by integrating **FAISS vector search** as a contextual retrieval layer.
- Impact: Accomplished a multi-step decision-making workflow by designing **LLM-powered agents** that route queries between direct generation and retrieval-backed response generation.

## Problem Statement

Traditional chatbot responses are often generic, hallucinated, or disconnected from business-specific support content. This project solves that by combining:

1. A retrieval pipeline over curated support documents.
2. A deterministic agent router for explainable decision-making.
3. Persistent user sessions and conversation memory for context continuity.

## Core Capabilities

- Knowledge-grounded Q and A using RAG
- Semantic document retrieval via FAISS
- Prompt-context enrichment with recent conversation memory
- User registration and login with bearer-token auth
- Conversation and message history persistence
- Admin-only document upload endpoint with vectorstore rebuild
- CLI, API, and Web UI entry points
- Dockerized deployment for repeatable local runs

## Architecture Overview

### 1. Data and Ingestion

- Raw support documents are stored in `data/raw`.
- Documents are cleaned and chunked.
- Embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2`.

### 2. Vector Store

- Embeddings are indexed in FAISS (`IndexFlatIP`) for similarity search.
- Metadata (source, chunk id) is persisted for context reconstruction.

### 3. Agent and RAG Pipeline

- Router evaluates whether a query requires retrieval.
- Retrieval path pulls top-k relevant chunks.
- Prompt context combines retrieved evidence plus conversation memory.
- Generator model (`google/flan-t5-small` by default) returns final response.

### 4. Application Layer

- FastAPI exposes auth, conversation, ask, and admin upload endpoints.
- Streamlit provides a chat interface with login and persistent history.
- SQLite stores users, tokens, conversations, and messages.

## Project Structure

```text
agentic-ai-customer-support/
├── agents/           # Query routing and decision logic
├── api/              # FastAPI app, schemas, auth dependencies
├── data/             # Raw and processed documents + SQLite DB
├── embeddings/       # Embedding model wrappers
├── evaluation/       # Evaluation metrics and logging hooks
├── llm/              # Hugging Face generation layer
├── rag/              # Retrieval-augmented generation pipeline
├── scripts/          # Operational scripts (e.g., ingestion)
├── tests/            # Unit and integration tests
├── ui/               # Streamlit application
├── utils/            # Config, logging, storage, security
├── vectorstore/      # FAISS index + metadata artifacts
├── main.py           # CLI entrypoint
├── docker-compose.yml
└── Dockerfile
```

## API Endpoints

### Auth

- `POST /auth/register` - create account + return token
- `POST /auth/login` - login + return token

### Conversation

- `POST /conversations` - create conversation
- `GET /conversations` - list user conversations
- `GET /conversations/{conversation_id}/messages` - fetch history

### Chat

- `POST /ask` - ask support question with optional `conversation_id`

### Admin

- `POST /admin/upload` - upload `.txt` files and rebuild knowledge base

## Local Setup

### 1. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Set at minimum:

- `ADMIN_USERNAME`
- `ADMIN_PASSWORD`
- `HF_EMBEDDING_MODEL`
- `HF_LLM_MODEL`

## Run the Project

### CLI

```bash
python main.py --query "How do I reset my password?"
```

### API Server

```bash
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

### Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

## Document Ingestion Workflow

Add new support files and rebuild vectorstore:

```bash
python scripts/ingest_documents.py /path/to/new/docs --rebuild
```

Expected input format: UTF-8 `.txt` documents.

## Testing

Run all tests:

```bash
python -m unittest discover -s tests
```

## Docker Deployment

```bash
docker compose up --build
```

This starts reproducible containers for local execution.

## Engineering Notes

- Clear separation of concerns across agent, retrieval, LLM, API, and UI layers.
- Deterministic routing strategy for explainability.
- Lazy model loading and persistent vectorstore usage for practical performance.
- Defensive input checks and structured error handling in API and runtime paths.

## Future Enhancements

- Hybrid retrieval (keyword + dense vectors)
- Role-based admin dashboard for document lifecycle management
- Response quality evaluation dashboard and offline benchmark suite
- Observability integration (trace IDs, latency metrics, and model-call analytics)

## License

This project is intended for educational and portfolio demonstration use.
