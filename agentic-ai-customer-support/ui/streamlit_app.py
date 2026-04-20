"""Streamlit interface for the support assistant."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from agents.decision import SupportSystemState, answer_query
from llm.hf_llm import generate_response
from rag.memory import build_memory_context
from rag.service import build_support_system
from utils.config import settings
from utils.logging import setup_logging
from utils.storage import (
    add_message,
    authenticate_user,
    create_conversation,
    create_user,
    get_recent_messages,
    initialize_database,
    list_conversations,
)


@st.cache_resource(show_spinner="Loading support system...")
def get_system() -> SupportSystemState:
    """Build the support system for the UI session."""

    system = build_support_system()
    return SupportSystemState(index=system.index, chunks=system.chunks, top_k=settings.top_k)


def _ensure_authenticated() -> dict | None:
    """Render login controls and return the current user if signed in."""

    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    with st.sidebar:
        st.header("Account")

        if st.session_state.current_user:
            st.success(f"Signed in as {st.session_state.current_user['username']}")
            if st.button("Log out"):
                st.session_state.current_user = None
                st.session_state.conversation_id = None
                st.rerun()
        else:
            login_tab, register_tab = st.tabs(["Login", "Register"])

            with login_tab:
                with st.form("login_form"):
                    username = st.text_input("Username", key="login_username")
                    password = st.text_input("Password", type="password", key="login_password")
                    submit_login = st.form_submit_button("Login")
                if submit_login:
                    try:
                        session = authenticate_user(username, password)
                        st.session_state.current_user = session["user"]
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            with register_tab:
                with st.form("register_form"):
                    username = st.text_input("Username", key="register_username")
                    password = st.text_input("Password", type="password", key="register_password")
                    submit_register = st.form_submit_button("Create account")
                if submit_register:
                    try:
                        create_user(username, password)
                        session = authenticate_user(username, password)
                        st.session_state.current_user = session["user"]
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

    return st.session_state.current_user


def run_ui() -> None:
    """Render the Streamlit chat interface."""

    setup_logging(settings.log_file)
    initialize_database()
    st.set_page_config(page_title="Agentic AI Customer Support", page_icon="💬", layout="centered")
    st.title("Agentic AI Customer Support")
    st.caption("RAG + FAISS + Hugging Face Transformers")

    current_user = _ensure_authenticated()
    if current_user is None:
        st.info("Sign in to start a conversation.")
        st.stop()

    with st.sidebar:
        st.header("Conversations")
        conversations = list_conversations(current_user["id"])
        conversation_options = {f"{conversation['id']}: {conversation['title'] or 'Untitled'}": conversation['id'] for conversation in conversations}
        conversation_labels = ["New conversation"] + list(conversation_options.keys())
        selected_label = st.selectbox("Select conversation", conversation_labels)
        if selected_label == "New conversation":
            if st.button("Start conversation"):
                conversation = create_conversation(current_user["id"], title="Support chat")
                st.session_state.conversation_id = conversation["id"]
                st.rerun()
        else:
            st.session_state.conversation_id = conversation_options[selected_label]

    if st.session_state.conversation_id:
        history = get_recent_messages(st.session_state.conversation_id, limit=50)
        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_query = st.chat_input("Ask a customer support question")
    if user_query:
        system = get_system()

        if not st.session_state.conversation_id:
            conversation = create_conversation(current_user["id"], title="Support chat")
            st.session_state.conversation_id = conversation["id"]

        conversation_id = st.session_state.conversation_id
        previous_messages = get_recent_messages(conversation_id, limit=settings.conversation_history_limit)
        add_message(conversation_id, "user", user_query)
        memory_context = build_memory_context(previous_messages)

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            try:
                answer = answer_query(
                    user_query,
                    system,
                    direct_response_fn=lambda query, memory_context="": generate_response(query, context=memory_context),
                    memory_context=memory_context,
                )
                st.markdown(answer)
                add_message(conversation_id, "assistant", answer)
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Failed to generate answer: {exc}")


if __name__ == "__main__":
    run_ui()
