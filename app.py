import json
import os
import tempfile
import warnings
from pathlib import Path
import re
import streamlit as st
import time

# Suppress warnings and telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore")

from src.ingest import ingest_file
from src.qa import build_chain, clean_answer
from src.vector_store import VectorStore

# -------------------------
# Utility Functions
# -------------------------

def detect_language(text: str) -> str:
    if re.search(r'[\u0900-\u097F]', text):
        return "hindi"
    if any(w in text.lower() for w in ["kya", "ka", "ki", "hai", "ko", "me", "se"]):
        return "hinglish"
    return "english"


def answer_supported_by_sources(answer: str, source_docs: list) -> bool:
    """
    Validate that the answer is grounded in retrieved source documents.
    Prevents hallucination in a generalized way.
    """
    if not answer or not source_docs:
        return False

    answer_tokens = set(answer.lower().split())
    context_tokens = set()

    for doc in source_docs:
        context_tokens.update(doc.page_content.lower().split())

    # Require meaningful lexical overlap
    return len(answer_tokens.intersection(context_tokens)) >= 2


# -------------------------
# Streamlit Setup
# -------------------------

st.set_page_config(
    page_title="Multilang QnA Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💬 Multilang QnA Assistant")
st.caption("Ask questions about your PDF documents in English, Hindi, or Hinglish")

CHAT_HISTORY_FILE = Path("chat_history.json")

def load_chat_history():
    if CHAT_HISTORY_FILE.exists():
        try:
            return json.loads(CHAT_HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_chat_history(messages):
    CHAT_HISTORY_FILE.write_text(
        json.dumps(messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

# -------------------------
# Initialize State
# -------------------------

if "store" not in st.session_state:
    st.session_state.store = VectorStore()

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = build_chain(st.session_state.store)

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# -------------------------
# Sidebar – Upload
# -------------------------

with st.sidebar:
    st.header("📄 Upload PDF")
    clear_before_upload = st.checkbox("Clear existing documents before upload", value=False)
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        if clear_before_upload:
            st.session_state.store.clear()

        count, _ = ingest_file(tmp_path, st.session_state.store)
        st.session_state.qa_chain = build_chain(st.session_state.store)
        st.session_state.messages = []
        save_chat_history([])

        st.success(f"Ingested {count} chunks")

    st.divider()
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        save_chat_history([])
        st.rerun()

# -------------------------
# Chat History Display
# -------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("source_docs"):
            with st.expander("🔍 View Source Documents"):
                for i, d in enumerate(msg["source_docs"], 1):
                    st.write(f"**Source {i}:** {Path(d['source']).name} | Page {d['page']}")
                    st.text_area(
                        f"Content {i}",
                        d["content"][:400],
                        height=120,
                        label_visibility="collapsed",
                        key=f"src_{hash(str(d))}_{i}",
                    )

# -------------------------
# Chat Input
# -------------------------

prompt = st.chat_input("Ask a question…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_history(st.session_state.messages)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke({"query": prompt})

            raw_answer = response.get("result", "").strip()
            source_docs = response.get("source_documents", [])
            lang = detect_language(prompt)

            # 🔒 HARD ANTI-HALLUCINATION CHECK
            if raw_answer != "NOT_FOUND":
                if not answer_supported_by_sources(raw_answer, source_docs):
                    raw_answer = "NOT_FOUND"

            if raw_answer == "NOT_FOUND":
                answer = "उत्तर संदर्भ में नहीं मिला" if lang == "hindi" else "Answer not found in context"
            else:
                answer = clean_answer(raw_answer)

            st.markdown(answer)

    # Save assistant message
    serialized_sources = [
        {
            "source": d.metadata.get("source", "Unknown"),
            "page": d.metadata.get("page", "?"),
            "content": d.page_content,
        }
        for d in source_docs
    ]

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "source_docs": serialized_sources}
    )
    save_chat_history(st.session_state.messages)
