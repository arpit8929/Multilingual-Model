# Multilingual QnA Assistant - FastAPI Backend

FastAPI backend for the Multilingual QnA Assistant.

## Setup

1. Install dependencies:
```bash
pip install -r ../requirements.txt
```

2. Run the server (from project root):
```bash
# Option 1: From project root
cd ..
python -m backend.main

# Option 2: From backend directory
cd backend
python main.py

# Option 3: Using uvicorn from project root
cd ..
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /api/status` - Get system status
- `POST /api/upload` - Upload and ingest PDF
- `POST /api/ask` - Ask a question
- `POST /api/clear` - Clear all documents

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
