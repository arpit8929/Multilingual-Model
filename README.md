# Multilingual PDF Q&A Assistant

This project is a local PDF question-answering assistant for English, Hindi, and Hinglish questions. It uses a React frontend, a FastAPI backend, ChromaDB for vector search, OCR for scanned/image PDFs, and a local GGUF Llama model for answer generation.

The current backend is designed to treat English and Hindi uploads differently:

- English documents use the legacy-style English path: no query translation, simpler retrieval, English-only OCR when the page is clearly English, and an English-focused QA prompt.
- Hindi documents use the Hindi-aware path: Hindi/legacy-Hindi detection, PaddleOCR Hindi, Tesseract fallback, query translation to Hindi for English/Hinglish questions, neighboring chunk expansion, and answer-language cleanup.
- Hinglish questions on Hindi documents use the original Hinglish query plus a Hindi retrieval query. The English query-translation branch is intentionally disabled because it can produce answer-like text and hurt retrieval.

## Project Structure

```text
BISAG-212/
  app.py                       # Legacy Streamlit UI
  backend/
    main.py                    # FastAPI API server
    run.py                     # Alternative backend runner
  frontend/
    src/App.jsx                # React app state and API flow
    src/services/api.js        # Frontend API client
    src/components/            # Chat and sidebar components
  src/
    config.py                  # Central settings and env overrides
    ingest.py                  # PDF extraction, OCR, chunking, metadata
    vector_store.py            # ChromaDB embeddings, retrieval, reranking
    qa.py                      # LLM loading, prompts, query translation, cleanup
  models/
    llama-3.2-3b-instruct-q4_K_S.gguf
  chroma_db/                   # Local persisted vector database
  chat_history.json            # Saved chat messages
  requirements.txt             # Python dependencies
```

## Current Flow

1. The user uploads a PDF from the React UI.
2. The frontend sends it to `POST /api/upload`.
3. FastAPI stores the PDF temporarily and calls `src.ingest.ingest_file`.
4. `src/ingest.py` extracts text, tables, and OCR text page by page.
5. The ingestion layer detects whether the whole document is English or Hindi.
6. Text is split into chunks with page/chunk metadata.
7. Chunks are embedded with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
8. Chunks are stored in ChromaDB under `chroma_db/`.
9. Questions are sent to `POST /api/ask`.
10. `src/qa.py` formats the query, optionally translates the query for Hindi documents, retrieves context, calls the local GGUF model, cleans the output, and returns source snippets.

## English Document Behavior

For English PDFs and clearly English scanned/image PDFs:

- Query translation is skipped.
- Answer rewrite translation is skipped.
- Retrieval uses a simpler similarity path.
- The English QA prompt is used.
- OCR can use an English-only PaddleOCR path when text is clearly English.

This is intentional because English documents had better accuracy before the Hindi translator path was added.

Expected log:

```text
[QueryTranslation] Skipping query translation for English document
```

## Hindi Document Behavior

For Hindi PDFs:

- The document is marked with `document_lang=hi`.
- Legacy Hindi font patterns are treated as Hindi signals.
- PaddleOCR English, PaddleOCR Hindi, and Tesseract candidates may be compared.
- English and Hinglish questions can be translated into Hindi for retrieval.
- The final answer is rewritten into the question language when needed.

Expected logs:

```text
[Ingest] Active document language set to hi
[QueryTranslation] Using Hindi retrieval query: ...
```

Chunk translation is currently disabled by default:

```text
ENABLE_HINDI_TRANSLATION=false
```

That keeps Hindi PDF upload faster. Query-time translation is the main Hindi retrieval helper.

## Requirements

Recommended environment:

- Windows PowerShell
- Python 3.10 or newer
- Node.js 18 or newer
- Tesseract OCR installed and available in `PATH`
- Tesseract Hindi and English language data installed
- Local GGUF model file in `models/`

Default model path:

```text
models/llama-3.2-3b-instruct-q4_K_S.gguf
```

You can override it with `MODEL_PATH`.

## Python Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PaddleOCR or Torch gives installation trouble, follow the pinned versions in `requirements.txt`. PaddleOCR downloads OCR model files on first use.

## Frontend Setup

```powershell
cd frontend
npm install
cd ..
```

## Run the App

Start the backend:

```powershell
.\.venv\Scripts\Activate.ps1
python -m backend.main
```

The backend runs at:

```text
http://localhost:8000
```

Start the frontend in a second terminal:

```powershell
cd frontend
npm run dev
```

The frontend usually runs at:

```text
http://localhost:5173
```

## Optional Streamlit App

The main UI is React + FastAPI, but the legacy Streamlit app still exists:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## API Endpoints

```text
GET  /api/status
POST /api/upload
POST /api/ask
POST /api/clear
GET  /api/chat-history
POST /api/chat-history/clear
```

Upload accepts only PDF files.

## Important Environment Variables

Create a `.env` file in the project root if you want to override defaults.

```env
MODEL_PATH=models/llama-3.2-3b-instruct-q4_K_S.gguf
CHROMA_DIR=chroma_db
COLLECTION_NAME=pdf_qa

CHUNK_SIZE=800
CHUNK_OVERLAP=150

N_CTX=4096
N_THREADS=8
TEMPERATURE=0.1

TOP_K=10
FINAL_TOP_K=5
MAX_CONTEXT_DOCS=6
MAX_CONTEXT_CHARS=9000
PAGE_NEIGHBOR_WINDOW=1

USE_RERANKER=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

ENABLE_QUERY_TRANSLATION=true
ENABLE_HINDI_TRANSLATION=false

TESSERACT_LANG=hin+eng
```

## Testing and Verification

Check Python syntax without writing bytecode:

```powershell
@'
import ast, pathlib
for path in [
    pathlib.Path("src/qa.py"),
    pathlib.Path("src/ingest.py"),
    pathlib.Path("src/vector_store.py"),
    pathlib.Path("backend/main.py"),
    pathlib.Path("app.py"),
]:
    ast.parse(path.read_text(encoding="utf-8"))
    print("OK", path)
'@ | .\.venv\Scripts\python -
```

Build the frontend:

```powershell
cd frontend
npm run build
```

Run the setup verifier if needed:

```powershell
.\.venv\Scripts\python verify_setup.py
```

## Operational Notes

- Clear and re-upload PDFs after changing ingestion, OCR, chunking, or metadata logic.
- Existing ChromaDB chunks are not automatically updated after code changes.
- English documents should show query translation skipped.
- Hindi documents should show active language set to `hi`.
- If a Hindi PDF is legacy-font encoded, the OCR path should avoid the English-only shortcut.
- Uploading scanned Hindi PDFs can be slow because OCR is expensive.
- PostHog telemetry errors from ChromaDB are harmless and do not affect answers.

## Troubleshooting

### Model file not found

Make sure the GGUF file exists:

```text
models/llama-3.2-3b-instruct-q4_K_S.gguf
```

Or set:

```powershell
$env:MODEL_PATH="D:\path\to\model.gguf"
```

### Hindi document is detected as English

Clear and re-upload the PDF after the latest ingestion changes. Look for:

```text
[Ingest] Active document language set to hi
```

If it says `en`, the text/OCR is not providing enough Hindi or legacy-Hindi signal.

### English document accuracy drops

Make sure the terminal shows:

```text
[QueryTranslation] Skipping query translation for English document
```

If it does not, clear and re-upload the English PDF.

### Context window error

Lower these values in `.env`:

```env
MAX_CONTEXT_DOCS=4
MAX_CONTEXT_CHARS=7000
```

### Tesseract error

Install Tesseract OCR and ensure it is available in `PATH`. For Hindi OCR, install Hindi trained data as well.

## Current Limitations

- Hindi accuracy depends heavily on OCR quality.
- Legacy Hindi font PDFs may still require cleanup if OCR output is noisy.
- The local 3B quantized model can struggle with long, noisy Hindi context.
- Chunk translation is disabled by default to keep uploads faster.
- The React app is the preferred UI; Streamlit is kept as a legacy option.
