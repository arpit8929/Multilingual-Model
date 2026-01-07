import tempfile
from pathlib import Path

import streamlit as st

from src.ingest import ingest_file
from src.qa import build_chain
from src.vector_store import VectorStore


st.set_page_config(page_title="Hindi/English PDF Q&A", page_icon="📄")
st.title("📄 Hindi/English/Hinglish PDF QA")
st.caption("PyMuPDF + Tesseract OCR + Chroma + LLaMA-3.2-3B-Instruct Q4_K_S")

if "store" not in st.session_state:
    st.session_state.store = VectorStore()
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = build_chain(st.session_state.store)

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
        except Exception as exc:
            ingest_status.error(f"Ingest failed: {exc}")

st.subheader("Ask a question")
question = st.text_input("Type in English/Hindi/Hinglish")

if st.button("Submit") and question:
    with st.spinner("Thinking..."):
        response = st.session_state.qa_chain(question)
    answer = response.get("result", "")
    st.write(answer)

