import json
import os
import tempfile
import warnings
from pathlib import Path
import re
import streamlit as st

# Suppress warnings and telemetry errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"  # Disable ChromaDB telemetry
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*telemetry.*")

from src.ingest import ingest_file
from src.qa import build_chain, clean_answer
from src.vector_store import VectorStore

def detect_language(text: str) -> str:
    if re.search(r'[\u0900-\u097F]', text):
        return "hindi"
    if any(word in text.lower() for word in ["kya", "ka", "ki", "hai", "ko", "me", "se"]):
        return "hinglish"
    return "english"

st.set_page_config(
    page_title="QnA Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like interface
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Chat container styling */
    .stApp {
        background-color: #f7f7f8;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Scrollable chat area */
    .chat-container {
        height: calc(100vh - 200px);
        overflow-y: auto;
        padding-bottom: 1rem;
    }
    
    /* Input area styling */
    .stChatInput {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: white;
        padding: 1rem;
        border-top: 1px solid #e5e5e5;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("💬 Multilang QnA Assistant")
st.caption("Ask questions about your PDF documents in English, Hindi, or Hinglish")

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
    try:
        st.session_state.store = VectorStore()
    except Exception as e:
        st.error(f"Failed to initialize vector store: {e}")
        st.stop()

if "qa_chain" not in st.session_state:
    try:
        st.session_state.qa_chain = build_chain(st.session_state.store)
    except Exception as e:
        st.error(f"Failed to initialize QA chain: {e}")
        st.error("This might be due to:")
        st.error("- Model file not found")
        st.error("- Model file corrupted")
        st.error("- Insufficient memory")
        st.stop()

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
    
    # Option to clear vector store before uploading new PDF
    clear_before_upload = st.checkbox("Clear existing documents before upload", value=False,
                                       help="If checked, removes all previously ingested documents before adding the new PDF")
    
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    ingest_status = st.empty()
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)
        try:
            # Clear vector store if requested
            if clear_before_upload:
                st.session_state.store.clear()
                st.session_state.qa_chain = build_chain(st.session_state.store)
                ingest_status.info("Cleared existing documents")
            
            count, db_path = ingest_file(tmp_path, st.session_state.store)
            # Rebuild chain after adding documents
            st.session_state.qa_chain = build_chain(st.session_state.store)
            ingest_status.success(f"Ingested {count} chunks → {db_path}")
            # Optional: clear previous chat when a new document is loaded
            st.session_state.messages = []
            save_chat_history([])
            if CHAT_HISTORY_FILE.exists():
                st.session_state.last_chat_update = CHAT_HISTORY_FILE.stat().st_mtime
        except Exception as exc:
            ingest_status.error(f"Ingest failed: {exc}")

# Sidebar: Status and controls
with st.sidebar:
    st.divider()
    st.header("📊 Status")
    doc_count = st.session_state.store.get_document_count()
    if doc_count > 0:
        st.success(f"✅ {doc_count} documents loaded")
    else:
        st.warning("⚠️ No documents loaded")
        st.caption("Upload a PDF to get started")
    
    st.divider()
    st.header("🗑️ Clear")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        save_chat_history([])
        if CHAT_HISTORY_FILE.exists():
            st.session_state.last_chat_update = CHAT_HISTORY_FILE.stat().st_mtime
        st.rerun()

# Create scrollable chat container
chat_container = st.container()

with chat_container:
    # Render existing chat history - display all messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show source documents if available (for assistant messages)
            if msg["role"] == "assistant" and "source_docs" in msg:
                source_docs_data = msg.get("source_docs", [])
                if source_docs_data:
                    with st.expander("🔍 View Source Documents", expanded=False):
                        for i, doc_data in enumerate(source_docs_data, 1):
                            source_info = doc_data.get("source", "Unknown")
                            page_num = doc_data.get("page", "?")
                            doc_type = doc_data.get("type", "text")
                            content_preview = doc_data.get("content", "")[:300] + "..." if len(doc_data.get("content", "")) > 300 else doc_data.get("content", "")
                            
                            st.write(f"**Source {i}:** {Path(source_info).name} | Page {page_num} | Type: {doc_type}")
                            st.text_area(f"Content {i}", content_preview, height=100, key=f"source_{hash(str(msg))}_{i}", label_visibility="collapsed")

# Chat-style input at the bottom
prompt = st.chat_input("Type your question in English/Hindi/Hinglish")

if prompt:
    # Add user message to session state
    import time
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    
    # Save immediately after adding user message
    save_chat_history(st.session_state.messages)
    # Update last modification time
    if CHAT_HISTORY_FILE.exists():
        st.session_state.last_chat_update = CHAT_HISTORY_FILE.stat().st_mtime
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                # Check if vector store has documents
                doc_count = st.session_state.store.get_document_count()
                if doc_count == 0:
                    answer = "⚠️ **No documents found in the database.**\n\nPlease upload a PDF first using the sidebar."
                    source_docs = []
                else:
                    # Proceed with query
                    status_text = st.empty()
                    status_text.info(f"🔍 Searching through {doc_count} documents...")
                    
                    response = st.session_state.qa_chain.invoke({"query": prompt})
                    status_text.empty()  # Clear status
                    
                    raw_answer = response.get("result", "").strip()
                    source_docs = response.get("source_documents", [])

                    lang = detect_language(question)

                    if raw_answer == "NOT_FOUND":
                        if lang == "hindi":
                            answer = "उत्तर संदर्भ में नहीं मिला"
                        else:
                            answer = "Answer not found in context"
                    else:
                        answer = clean_answer(raw_answer)
                    
                    # Check if answer is empty
                    if not raw_answer or not raw_answer.strip():
                        answer = "⚠️ **No response generated.**\n\nThe model did not produce an answer. This might be due to:\n- No relevant documents found\n- Model loading issue\n- Context window exceeded\n\nPlease check the source documents below or try re-uploading the PDF."
                    else:
                        # Clean and improve the answer quality
                        answer = clean_answer(raw_answer)
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                answer = f"❌ **Error generating response:**\n\n```\n{str(e)}\n```\n\n**Full error details:**\n```\n{error_details}\n```"
                source_docs = []
                st.error("An error occurred. Check the error message below.")
        
        # Display answer
        if answer:
            message_placeholder.markdown(answer)
        else:
            message_placeholder.warning("No answer was generated. Please check the error messages above.")
        
        # Show source documents for debugging (collapsible)
        if source_docs:
            with st.expander("🔍 View Source Documents", expanded=False):
                for i, doc in enumerate(source_docs, 1):
                    source_info = doc.metadata.get("source", "Unknown")
                    page_num = doc.metadata.get("page", "?")
                    doc_type = doc.metadata.get("type", "text")
                    content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    
                    st.write(f"**Source {i}:** {Path(source_info).name} | Page {page_num} | Type: {doc_type}")
                    st.text_area(f"Content {i}", content_preview, height=100, key=f"source_{int(time.time())}_{i}", label_visibility="collapsed")
    
    # Add assistant message to session state with source docs (serialize for JSON)
    source_docs_serialized = []
    if 'source_docs' in locals() and source_docs:
        for doc in source_docs:
            source_docs_serialized.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "type": doc.metadata.get("type", "text"),
                "content": doc.page_content
            })
    
    assistant_message = {
        "role": "assistant", 
        "content": answer,
        "source_docs": source_docs_serialized
    }
    st.session_state.messages.append(assistant_message)
    
    # Save complete conversation (user + assistant)
    save_chat_history(st.session_state.messages)
    # Update last modification time
    if CHAT_HISTORY_FILE.exists():
        st.session_state.last_chat_update = CHAT_HISTORY_FILE.stat().st_mtime

