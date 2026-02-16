"""
FastAPI backend for Multilingual QnA Assistant
"""
import json
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional


# Add parent directory to Python path to import src modules
backend_dir = Path(__file__).parent
project_root = backend_dir.parent
sys.path.insert(0, str(project_root))

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Suppress warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*telemetry.*")

from src.ingest import ingest_file
from src.qa import build_chain, clean_answer
from src.vector_store import VectorStore

app = FastAPI(title="Multilingual QnA Assistant API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
vector_store = None
qa_chain = None
CHAT_HISTORY_FILE = project_root / "chat_history.json"

uploaded_documents = []  # Track uploaded document names

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
        pass


def extract_answer_from_sources(question: str, source_docs: List) -> str:
    """Fallback: Extract answer from source documents when model fails."""
    if not source_docs:
        return ""
    
    import re
    
    # Look for common patterns in the question
    question_lower = question.lower()
    
    # Helper to get content from doc (handles both Document objects and dicts)
    def get_doc_content(doc):
        if hasattr(doc, 'page_content'):
            return doc.page_content
        elif isinstance(doc, dict):
            return doc.get('content', str(doc))
        else:
            return str(doc)
    
    

class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    source_documents: List[dict] = []
    question: Optional[str] = None  # Include question for history


class StatusResponse(BaseModel):
    document_count: int
    status: str
    document_name: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize vector store and QA chain on startup."""
    global vector_store, qa_chain
    try:
        # Change to project root directory for relative paths
        os.chdir(project_root)
        vector_store = VectorStore()
        qa_chain = build_chain(vector_store)
        print("✅ Backend initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize backend: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Multilingual QnA Assistant API", "status": "running"}


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get current status of the system."""
    global vector_store, uploaded_documents
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    doc_count = vector_store.get_document_count()
    document_name = uploaded_documents[-1] if uploaded_documents else None
    document_name = uploaded_documents[-1] if uploaded_documents else None
    return StatusResponse(
        document_count=doc_count,
        status="ready" if doc_count > 0 else "no_documents",
        document_name=document_name
    )


@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    clear_existing: bool = False
):
    """Upload and ingest a PDF file."""
    global vector_store, qa_chain
    
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        # Clear vector store if requested
        global uploaded_documents
        if clear_existing:
            vector_store.clear()
            qa_chain = build_chain(vector_store)
            uploaded_documents = []
            uploaded_documents = []
        
        # Ensure we're in project root for relative paths
        original_cwd = os.getcwd()
        try:
            os.chdir(project_root)
            # Ingest the PDF
            count, db_path = ingest_file(tmp_path, vector_store)
            
            # Rebuild chain after adding documents
            qa_chain = build_chain(vector_store)
        finally:
            os.chdir(original_cwd)
        
        # Clean up temp file
        filename = file.filename
        filename = file.filename
        
        # Store document name
        if filename:
            uploaded_documents.append(filename)
        
        tmp_path.unlink()
        
        return {
            "success": True,
            "message": f"Successfully ingested {count} chunks",
            "chunk_count": count,
            "document_count": vector_store.get_document_count(),
            "document_count": vector_store.get_document_count(),
            "document_name": filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest PDF: {str(e)}")


@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an answer."""
    global qa_chain, vector_store
    
    if qa_chain is None:
        raise HTTPException(status_code=500, detail="QA chain not initialized")
    
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    # Check if documents exist
    doc_count = vector_store.get_document_count()
    if doc_count == 0:
        return QuestionResponse(
            answer="⚠️ **No documents found in the database.**\n\nPlease upload a PDF first.",
            source_documents=[]
        )
    
    try:
        # Get response from QA chain
        response = qa_chain.invoke({"query": request.question})
        raw_answer = response.get("result", "")
        source_docs = response.get("source_documents", [])
        
        # Debug logging
        # Debug logging
        print(f"DEBUG: Raw answer length: {len(raw_answer) if raw_answer else 0}")
        print(f"DEBUG: Raw answer preview: {raw_answer[:200] if raw_answer else 'EMPTY'}")
        
        # Clean the answer
        if raw_answer and raw_answer.strip():
            answer = clean_answer(raw_answer)
            # Check if cleaning removed everything
            if not answer or not answer.strip():
                print("WARNING: clean_answer removed all content, using raw answer")
                answer = raw_answer.strip()
            
            # Post-process: Check if answer seems hallucinated for limitations questions
            question_lower = request.question.lower()
            if any(kw in question_lower for kw in ["limitation", "limitations", "कमी", "सीमाएं"]):
                # Get all source content to check if specific limitations are mentioned
                all_source_content = " ".join([
                    doc.page_content if hasattr(doc, 'page_content') else str(doc) 
                    for doc in source_docs
                ]).lower()
                
                # Check if sources only vaguely mention limitations without details
                vague_mentions = [
                    "limitations were identified",
                    "limitations and shortcomings were identified",
                    "limitations were",
                    "shortcomings were",
                ]
                has_vague_mention = any(phrase in all_source_content for phrase in vague_mentions)
                
                # Check if answer contains generic patterns not in source
                generic_patterns_in_answer = [
                    r"real-time.*network.*traffic",
                    r"handle.*network.*traffic",
                    r"analyze.*network.*traffic",
                ]
                has_generic = any(re.search(pattern, answer, re.IGNORECASE) for pattern in generic_patterns_in_answer)
                
                # If sources only vaguely mention limitations and answer is generic, likely hallucinated
                if has_vague_mention and has_generic:
                    # Look for specific limitations in sources
                    if not any(specific in all_source_content for specific in [
                        "limitation is", "limitation of", "first limitation", "second limitation",
                        "main limitation", "key limitation", "primary limitation"
                    ]):
                        # No specific limitations found - answer is likely hallucinated
                        is_hindi = any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in request.question)
                        answer = "उत्तर संदर्भ में नहीं मिला" if is_hindi else "Answer not found in context"
        else:
            # Fallback: Try to extract answer from source documents
            print("WARNING: No raw answer, attempting to extract from source documents")
            answer = extract_answer_from_sources(request.question, source_docs)
            if not answer:
                answer = "⚠️ **No response generated.**\n\nThe model did not produce an answer. Please check the source documents below."
        
        # Serialize source documents
        source_docs_serialized = []
        for doc in source_docs:
            source_docs_serialized.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "type": doc.metadata.get("type", "text"),
                "content": doc.page_content[:500]  # Limit content length
            })
        
        # Save to chat history
        try:
            messages = load_chat_history()
            messages.append({
                "role": "user",
                "content": request.question
            })
            messages.append({
                "role": "assistant",
                "content": answer,
                "source_documents": source_docs_serialized
            })
            save_chat_history(messages)
        except Exception as e:
            print(f"Warning: Failed to save chat history: {e}")
        
        return QuestionResponse(
            answer=answer,
            source_documents=source_docs_serialized,
            question=request.question
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")


@app.post("/api/clear")
async def clear_documents():
    """Clear all documents from the vector store."""
    global vector_store, qa_chain
    
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        vector_store.clear()
        qa_chain = build_chain(vector_store)
        # Also clear chat history
        save_chat_history([])
        return {"success": True, "message": "All documents cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")


@app.get("/api/chat-history")
async def get_chat_history_endpoint():
    """Get chat history."""
    try:
        messages = load_chat_history()
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load chat history: {str(e)}")


@app.post("/api/chat-history/clear")
async def clear_chat_history_endpoint():
    """Clear chat history."""
    try:
        save_chat_history([])
        return {"success": True, "message": "Chat history cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear chat history: {str(e)}")


if __name__ == "__main__":
    # Ensure we're in project root
    os.chdir(project_root)
    uvicorn.run(app, host="0.0.0.0", port=8000)
