"""Integration tests for the API and RAG flow."""

from __future__ import annotations

import sys
import uuid
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - optional dependency boundary
    TestClient = None

from agents.decision import SupportSystemState, answer_query
from evaluation.metrics import evaluate_exact_match
from utils.schema import DocumentChunk

try:
    from api.app import create_app
except ImportError:  # pragma: no cover - optional dependency boundary
    create_app = None


@unittest.skipIf(create_app is None or TestClient is None, "fastapi is not installed")
class IntegrationTests(unittest.TestCase):
    """End-to-end checks for routing and the API layer."""

    def setUp(self) -> None:
        """Prepare a deterministic support system for the tests."""

        self.system = SupportSystemState(
            index=object(),
            chunks=[DocumentChunk(text="Reset instructions", source="faq.txt", chunk_id=0)],
            top_k=1,
        )

    def test_answer_query_uses_retrieval_branch(self) -> None:
        """A knowledge question should use the RAG path."""

        with patch("agents.decision.requires_retrieval", return_value=True), patch(
            "agents.decision.retrieve_relevant_chunks", return_value=self.system.chunks
        ), patch("agents.decision.answer_with_rag", return_value="Use the reset link"):
            answer = answer_query("How do I reset my password?", self.system, direct_response_fn=lambda query: "direct")

        self.assertTrue(evaluate_exact_match("Use the reset link", answer))

    def test_answer_query_uses_direct_branch_for_non_knowledge_query(self) -> None:
        """A direct question should bypass retrieval."""

        answer = answer_query("Hello", self.system, direct_response_fn=lambda query: "direct response")
        self.assertTrue(evaluate_exact_match("direct response", answer))

    def test_api_health_and_ask_endpoints(self) -> None:
        """The FastAPI app should expose health and ask endpoints."""

        app = create_app()
        app.state.support_system = self.system

        with TestClient(app) as client:
            health_response = client.get("/health")
            username = f"bob_{uuid.uuid4().hex[:8]}"
            register_response = client.post("/auth/register", json={"username": username, "password": "secret123"})
            token = register_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            conversation_response = client.post("/conversations", headers=headers, json={"title": "Support"})
            conversation_id = conversation_response.json()["id"]
            ask_response = client.post(
                "/ask",
                headers=headers,
                json={"query": "Hello", "conversation_id": conversation_id},
            )

        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(register_response.status_code, 200)
        self.assertEqual(conversation_response.status_code, 200)
        self.assertEqual(ask_response.status_code, 200)
        self.assertEqual(ask_response.json()["conversation_id"], conversation_id)
        self.assertIn("answer", ask_response.json())

    def test_admin_upload_endpoint(self) -> None:
        """Admins should be able to upload documents without breaking the API contract."""

        app = create_app()
        app.state.support_system = self.system

        with TestClient(app) as client:
            login_response = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
            token = login_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            with patch("api.app.build_knowledge_base", return_value=app.state.support_system), patch(
                "api.app._refresh_support_system"
            ):
                upload_response = client.post(
                    "/admin/upload",
                    headers=headers,
                    files={"files": ("new_faq.txt", b"How do I update my plan?\nUse settings.", "text/plain")},
                )

        self.assertEqual(login_response.status_code, 200)
        self.assertEqual(upload_response.status_code, 200)
        self.assertTrue(upload_response.json()["vectorstore_rebuilt"])


if __name__ == "__main__":
    unittest.main()
