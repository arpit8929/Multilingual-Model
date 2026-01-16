# Complete Setup Guide - Multilingual QnA Assistant

## Prerequisites

1. **Python 3.10+** installed
2. **Node.js 18+** and **npm** installed
3. **Virtual environment** activated (if using one)

## Step-by-Step Setup

### Step 1: Install Python Dependencies

```bash
# Navigate to project root
cd Multilingual-Model

# Activate virtual environment (if using one)
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install Python packages
pip install -r requirements.txt
```

### Step 2: Verify Backend Dependencies

Make sure these are installed:
- fastapi
- uvicorn
- python-multipart
- All ML/NLP dependencies (langchain, chromadb, etc.)

### Step 3: Install Frontend Dependencies

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js packages
npm install
```

This will install:
- React
- Vite
- Tailwind CSS
- Axios
- Lucide React (icons)
- react-markdown

### Step 4: Start the Backend Server

**Open Terminal 1 (PowerShell/Command Prompt):**

```bash
# From project root
cd Multilingual-Model

# Activate virtual environment (if using one)
.venv\Scripts\Activate.ps1

# Start backend (Option 1 - Recommended)
python -m backend.main

# OR (Option 2 - From backend directory)
cd backend
python main.py
```

**Expected output:**
```
✅ Backend initialized successfully
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

The backend will be available at: **http://localhost:8000**

### Step 5: Start the Frontend Server

**Open Terminal 2 (PowerShell/Command Prompt):**

```bash
# Navigate to frontend directory
cd Multilingual-Model\frontend

# Start React development server
npm run dev
```

**Expected output:**
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

The frontend will be available at: **http://localhost:3000**

### Step 6: Access the Application

1. Open your web browser
2. Navigate to: **http://localhost:3000**
3. You should see the Multilingual QnA Assistant interface

## Using the Application

### 1. Upload a PDF

1. Click the **Menu** button (☰) in the top-left
2. In the sidebar, click **"Choose PDF File"**
3. Select your PDF file
4. (Optional) Check **"Clear existing documents before upload"**
5. Click **"Upload & Ingest"**
6. Wait for the success message

### 2. Ask Questions

1. Type your question in the input box at the bottom
2. Press **Enter** or click the **Send** button
3. Wait for the AI to process and respond
4. View source documents by clicking **"View Source Documents"** in the response

### 3. Clear Documents

1. Open the sidebar (Menu button)
2. Click **"Clear All Documents"** button
3. Confirm the action

## Troubleshooting

### Backend Issues

**Problem: ModuleNotFoundError**
```bash
# Solution: Make sure you're running from project root
cd Multilingual-Model
python -m backend.main
```

**Problem: Port 8000 already in use**
```bash
# Solution: Change port in backend/main.py
# Or kill the process using port 8000
```

**Problem: Model file not found**
- Ensure `models/llama-3.2-3b-instruct-q4_K_S.gguf` exists
- Check the path in `src/config.py`

### Frontend Issues

**Problem: npm install fails**
```bash
# Solution: Clear cache and reinstall
npm cache clean --force
npm install
```

**Problem: Port 3000 already in use**
```bash
# Solution: Vite will automatically use next available port
# Or change port in vite.config.js
```

**Problem: Cannot connect to backend**
- Ensure backend is running on port 8000
- Check browser console for CORS errors
- Verify API URL in `frontend/src/services/api.js`

**Problem: Module not found errors**
```bash
# Solution: Reinstall dependencies
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## Development Commands

### Backend
```bash
# Run with auto-reload (development)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Run production server
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
Multilingual-Model/
├── backend/
│   ├── main.py          # FastAPI backend server
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── components/  # React components
│   │   │   ├── ChatInterface.jsx
│   │   │   ├── Message.jsx
│   │   │   ├── ChatInput.jsx
│   │   │   └── Sidebar.jsx
│   │   ├── services/
│   │   │   └── api.js   # API client
│   │   ├── App.jsx      # Main app
│   │   └── main.jsx     # Entry point
│   ├── package.json
│   └── vite.config.js
├── src/                 # Core ML/NLP code
│   ├── ingest.py
│   ├── qa.py
│   ├── vector_store.py
│   └── config.py
├── requirements.txt
└── README.md
```

## API Endpoints

Once backend is running, you can access:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Status**: http://localhost:8000/api/status

## Quick Start Summary

```bash
# Terminal 1 - Backend
cd Multilingual-Model
python -m backend.main

# Terminal 2 - Frontend
cd Multilingual-Model\frontend
npm run dev

# Browser
# Open http://localhost:3000
```

## Next Steps After Setup

1. ✅ Upload a PDF document
2. ✅ Wait for ingestion to complete
3. ✅ Start asking questions
4. ✅ Explore source documents
5. ✅ Try different languages (English, Hindi, Hinglish)

Enjoy your Multilingual QnA Assistant! 🚀
