# Quick Start - Run Everything

## One-Time Setup

### 1. Install Backend Dependencies
```powershell
cd Multilingual-Model
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies
```powershell
cd frontend
npm install
cd ..
```

## Running the Application

### Terminal 1: Backend
```powershell
cd Multilingual-Model
python -m backend.main
```
✅ Backend running on http://localhost:8000

### Terminal 2: Frontend
```powershell
cd Multilingual-Model\frontend
npm run dev
```
✅ Frontend running on http://localhost:3000

### Browser
Open: **http://localhost:3000**

---

## That's it! 🎉

1. Upload a PDF via the sidebar
2. Ask questions in the chat
3. View source documents in responses

For detailed setup, see `SETUP_GUIDE.md`
