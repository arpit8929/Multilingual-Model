import json
import tempfile
from pathlib import Path

import streamlit as st

from src.ingest import ingest_file
from src.qa import build_chain, clean_answer
from src.vector_store import VectorStore


st.set_page_config(page_title="Hindi/English PDF Q&A", page_icon="📄")
st.title("📄 Hindi/English/Hinglish PDF QA")
st.caption("PyMuPDF + Tesseract OCR + Chroma + LLaMA-3.2-3B-Instruct Q4_K_S")

# Chat history file
CHAT_HISTORY_FILE = Path("chat_history.json")

def load_chat_history():
    """Load chat history from file."""
    if CHAT_HISTORY_FILE.exists():
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat_history(messages):
    """Save chat history to file."""
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # Silently fail if can't save

# Initialize components
if "store" not in st.session_state:
    st.session_state.store = VectorStore()
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = build_chain(st.session_state.store)

# Initialize chat history - always load from file on first run
# This ensures persistence across page refreshes and app restarts
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()
    st.session_state.last_chat_update = CHAT_HISTORY_FILE.stat().st_mtime if CHAT_HISTORY_FILE.exists() else 0
else:
    # Reload from file if it was modified externally (e.g., another session)
    if CHAT_HISTORY_FILE.exists():
        current_mtime = CHAT_HISTORY_FILE.stat().st_mtime
        if current_mtime > st.session_state.get("last_chat_update", 0):
            st.session_state.messages = load_chat_history()
            st.session_state.last_chat_update = current_mtime

with st.sidebar:
    st.header("Upload PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    ingest_status = st.empty()
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)
        try:
            count, db_path = ingest_file(tmp_path, st.session_state.store)
            ingest_status.success(f"Ingested {count} chunks → {db_path}")
            # Optional: clear previous chat when a new document is loaded
            st.session_state.messages = []
            save_chat_history([])
            if CHAT_HISTORY_FILE.exists():
                st.session_state.last_chat_update = CHAT_HISTORY_FILE.stat().st_mtime
        except Exception as exc:
            ingest_status.error(f"Ingest failed: {exc}")

# Sidebar: Clear chat history button
with st.sidebar:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        save_chat_history([])
        if CHAT_HISTORY_FILE.exists():
            st.session_state.last_chat_update = CHAT_HISTORY_FILE.stat().st_mtime
        st.rerun()

# Render existing chat history - display all messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat-style input
prompt = st.chat_input("Type your question in English/Hindi/Hinglish")

if prompt:
    # Add user message to session state
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    
    # Save immediately after adding user message
    save_chat_history(st.session_state.messages)
    # Update last modification time
    if CHAT_HISTORY_FILE.exists():
        st.session_state.last_chat_update = CHAT_HISTORY_FILE.stat().st_mtime
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.qa_chain.invoke({"query": prompt})
                raw_answer = response.get("result", "")
                # Clean and improve the answer quality
                answer = clean_answer(raw_answer)
            except Exception as e:
                answer = f"Error generating response: {str(e)}"
        
        st.markdown(answer)
    
    # Add assistant message to session state
    assistant_message = {"role": "assistant", "content": answer}
    st.session_state.messages.append(assistant_message)
    
    # Save complete conversation (user + assistant)
    save_chat_history(st.session_state.messages)
    # Update last modification time
    if CHAT_HISTORY_FILE.exists():
        st.session_state.last_chat_update = CHAT_HISTORY_FILE.stat().st_mtime

