## Hindi/English PDF Q&A LLM

End-to-end retrieval-augmented QA for Hindi, English, or Hinglish PDFs, including table extraction. Stack: PyMuPDF, Tesseract OCR (hin+eng), ChromaDB + multilingual MiniLM embeddings, LangChain, LLaMA-3.2-3B-Instruct Q4_K_S (llama.cpp), Streamlit UI.

### Project Flow
- **Ingest**: Load PDF with PyMuPDF → extract text and tables (`find_tables`) → render page images and OCR with Tesseract (`hin+eng`) for scanned content → normalize and merge sources.
- **Chunk & Embed**: Split text (overlapping chunks) → embed via `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- **Store**: Persist embeddings in ChromaDB.
- **Serve**: Build LangChain RetrievalQA using `LlamaCpp` with the local Q4_K_S model.
- **UI**: Streamlit app to upload a PDF, ingest it, and query in Hindi/English/Hinglish.

### Prerequisites
- Python 3.10+
- Tesseract OCR installed and on PATH; Hindi data (`hin.traineddata`) present in `tessdata`. On Windows, install from UB Mannheim build; set `TESSDATA_PREFIX` if custom dir.
- VC++ Build Tools (for `llama-cpp-python` on Windows).
- Download LLaMA-3.2-3B-Instruct Q4_K_S GGUF and place in `models/llama-3.2-3b-instruct-q4_K_S.gguf`.

### Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # on PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

### Run Ingestion + UI
```bash
# 1) Put PDFs in data/ or upload via UI
streamlit run app.py
# or pre-ingest a PDF from CLI
python -m src.ingest --pdf data/sample.pdf
```

### Environment knobs
- `MODEL_PATH` (default `models/llama-3.2-3b-instruct-q4_K_S.gguf`)
- `CHROMA_DIR` (default `chroma_db`)
- `TESSDATA_PREFIX` for custom Tesseract data location
- `N_THREADS` / `N_CTX` for llama.cpp

### Notes
- Tables: uses PyMuPDF `find_tables`; falls back to OCR text if not structurally found.
- OCR language pack uses `hin+eng`, enabling Hinglish questions and answers based on retrieved context.
- No fastText dependency; embeddings use multilingual MiniLM via sentence-transformers.

