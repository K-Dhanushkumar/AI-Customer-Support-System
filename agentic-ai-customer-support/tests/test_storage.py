"""Tests for the SQLite-backed persistence helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.storage import (
    add_message,
    authenticate_user,
    create_conversation,
    create_user,
    get_recent_messages,
    initialize_database,
)


class StorageTests(unittest.TestCase):
    """Persistence and authentication behavior."""

    def test_user_auth_and_chat_history(self) -> None:
        """Users should authenticate and store/retrieve conversation messages."""

        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "app.db"
            initialize_database(database_path)
            create_user("alice", "secret123", db_path=database_path)
            session = authenticate_user("alice", "secret123", db_path=database_path)
            self.assertIn("access_token", session)

            conversation = create_conversation(session["user"]["id"], title="Support", db_path=database_path)
            add_message(conversation["id"], "user", "Hello", db_path=database_path)
            add_message(conversation["id"], "assistant", "Hi there", db_path=database_path)

            messages = get_recent_messages(conversation["id"], limit=10, db_path=database_path)
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]["role"], "user")
            self.assertEqual(messages[1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
