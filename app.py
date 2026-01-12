import json
import tempfile
from pathlib import Path

import streamlit as st

from src.ingest import ingest_file
from src.qa import build_chain
from src.vector_store import VectorStore


st.set_page_config(page_title="QnA Assistant")
st.title("Multilang QnA Assistant")
#st.caption("PyMuPDF + Tesseract OCR + Chroma + LLaMA-3.2-3B-Instruct Q4_K_S")

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

if "store" not in st.session_state:
    st.session_state.store = VectorStore()
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = build_chain(st.session_state.store)
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

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
        except Exception as exc:
            ingest_status.error(f"Ingest failed: {exc}")

# Sidebar: Clear chat history button
with st.sidebar:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        save_chat_history([])
        st.rerun()

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat-style input
prompt = st.chat_input("Type your question in English/Hindi/Hinglish")

if prompt:
    # Show user message and store it
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_history(st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke({"query": prompt})
        answer = response.get("result", "")
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_chat_history(st.session_state.messages)

