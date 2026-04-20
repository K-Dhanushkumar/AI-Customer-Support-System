"""Conversation memory formatting for prompt injection."""

from __future__ import annotations


def build_memory_context(messages: list[dict], max_messages: int = 6) -> str:
    """Format recent conversation messages for use in the LLM prompt."""

    if not messages:
        return ""

    recent_messages = messages[-max_messages:]
    lines = ["Conversation memory:"]
    for message in recent_messages:
        speaker = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {message['content']}")
    return "\n".join(lines)
