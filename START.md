# Getting Started with React Frontend

## Quick Start Guide

### Prerequisites
- Python 3.10+ with virtual environment activated
- Node.js 18+ and npm installed

### 1. Install Dependencies

**Backend:**
```bash
# Navigate to project root
cd Multilingual-Model

# Activate virtual environment (if using one)
.venv\Scripts\Activate.ps1

# Install Python dependencies
pip install -r requirements.txt
```

**Frontend:**
```bash
# Navigate to frontend directory
cd Multilingual-Model\frontend

# Install Node.js dependencies
npm install
```

### 2. Start Backend Server

**Open Terminal 1:**

```bash
# From project root
cd Multilingual-Model

# Activate virtual environment (if using one)
.venv\Scripts\Activate.ps1

# Start backend (Recommended - from project root)
python -m backend.main

# OR from backend directory:
cd backend
python main.py
```

✅ Backend will run on `http://localhost:8000`

### 3. Start Frontend Server

**Open Terminal 2 (new terminal):**

```bash
# Navigate to frontend directory
cd Multilingual-Model\frontend

# Start React development server
npm run dev
```

✅ Frontend will run on `http://localhost:3000`

### 3. Access the Application

Open your browser and go to: `http://localhost:3000`

## Features

✅ Modern React UI with ChatGPT-like interface
✅ FastAPI backend with RESTful API
✅ PDF upload and ingestion
✅ Real-time Q&A with source document viewing
✅ Responsive design
✅ Multilingual support (English, Hindi, Hinglish)

## Project Structure

```
Multilingual-Model/
├── backend/
│   └── main.py          # FastAPI backend
├── frontend/
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── services/    # API services
│   │   └── App.jsx      # Main app component
│   └── package.json
└── src/                 # Core ML/NLP code
```

## API Endpoints

- `GET /api/status` - Get system status
- `POST /api/upload` - Upload PDF
- `POST /api/ask` - Ask question
- `POST /api/clear` - Clear documents

## Troubleshooting

1. **Backend not starting**: Check if port 8000 is available
2. **Frontend not connecting**: Ensure backend is running on port 8000
3. **CORS errors**: Check that CORS middleware is properly configured
4. **Module not found**: Run `npm install` in frontend directory
