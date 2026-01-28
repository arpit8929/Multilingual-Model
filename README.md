## Hindi/English PDF Q&A LLM

End-to-end retrieval-augmented QA for Hindi, English, or Hinglish PDFs, including table extraction. Stack: PyMuPDF, Tesseract OCR (hin+eng), ChromaDB + multilingual MiniLM embeddings, LangChain, LLaMA-3.2-3B-Instruct Q4_K_S (llama.cpp), Streamlit UI.

---

## 🚀 Quick Start Guide for Team Members

> **💡 Quick Setup Option**: For Windows users, you can use the automated setup script:
> ```powershell
> .\setup.ps1
> ```
> This will create the virtual environment and install dependencies automatically. Then download the model file and run `python verify_setup.py` to verify everything is set up correctly.

---

### Step 1: Prerequisites Installation

#### 1.1 Install Python 3.10 or higher
- Download from [python.org](https://www.python.org/downloads/)
- **Important**: Check "Add Python to PATH" during installation
- Verify installation:
  ```powershell
  python --version
  ```

#### 1.2 Install Tesseract OCR (Windows)
- Download installer from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- **Recommended**: Download `tesseract-ocr-w64-setup-5.x.x.exe`
- During installation:
  - ✅ Check "Add to PATH"
  - ✅ Install Hindi language pack (`hin.traineddata`)
- Verify installation:
  ```powershell
  tesseract --version
  ```
- If Tesseract is not in PATH, note the installation directory (usually `C:\Program Files\Tesseract-OCR`)

#### 1.3 Install Visual C++ Build Tools (CRITICAL - Required for llama-cpp-python)

**⚠️ This is REQUIRED for Windows users - do not skip this step!**

- Download from [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
- Install "Desktop development with C++" workload
- Alternative: [Visual Studio Build Tools](https://aka.ms/vs/17/release/vs_buildtools.exe)

**Why is this needed?** The `llama-cpp-python` package must be compiled from source on Windows, which requires C++ build tools.

**Verification**: After installation, restart your terminal and try:
```powershell
cl  # Should show Microsoft C++ compiler version
```

---

### Step 2: Clone the Repository

```powershell
git clone https://github.com/arpit8929/Multilingual-Model.git
cd Multilingual-Model
```

---

### Step 3: Set Up Python Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# For PowerShell:
.venv\Scripts\Activate.ps1

# If you get an execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# For Command Prompt (cmd):
# .venv\Scripts\activate.bat

# Upgrade pip
python -m pip install --upgrade pip
```

---

### Step 4: Install Python Dependencies

```powershell
# Make sure virtual environment is activated (you should see (.venv) in prompt)
pip install -r requirements.txt
```

**Note**: Installation may take 5-10 minutes, especially for packages like `torch`, `llama-cpp-python`, etc.

---

### Step 5: Download the LLaMA Model

The model file is **NOT included in git** (see `.gitignore`). You need to download it separately:

#### Option A: Direct Download (Recommended)
1. Download `llama-3.2-3b-instruct-q4_K_S.gguf` from:
   - [Hugging Face](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) (search for `q4_K_S`)
   - Or use: `https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_S.gguf`

2. Create `models` folder in project root:
   ```powershell
   mkdir models
   ```

3. Place the downloaded file as:
   ```
   models/llama-3.2-3b-instruct-q4_K_S.gguf
   ```

#### Option B: Using Ollama (Alternative)
```powershell
# Install Ollama from https://ollama.ai
ollama pull llama3.2:3b
# Copy model from Ollama's directory to models/
```

**Model Size**: ~1.8 GB (ensure you have enough disk space)

---

### Step 6: Configure Environment (Optional)

Create a `.env` file in the project root if you need custom settings:

```env
MODEL_PATH=models/llama-3.2-3b-instruct-q4_K_S.gguf
CHROMA_DIR=chroma_db
TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
N_THREADS=4
N_CTX=4096
TEMPERATURE=0.1
```

**Note**: Only set `TESSDATA_PREFIX` if Tesseract is installed in a non-standard location.

---

### Step 7: Verify Installation (Optional but Recommended)

Run the verification script to check if everything is set up correctly:

```powershell
python verify_setup.py
```

This will check:
- ✅ Python version
- ✅ Virtual environment
- ✅ Tesseract OCR installation
- ✅ Model file presence
- ✅ Python dependencies
- ✅ ChromaDB directory

---

### Step 8: Run the Application

```powershell
# Make sure virtual environment is activated
streamlit run app.py
```

The application will:
- Open in your default browser (usually `http://localhost:8501`)
- Allow you to upload PDFs via the sidebar
- Process and ingest PDFs automatically
- Answer questions in Hindi/English/Hinglish

---

### Step 9: Using the Application

1. **Upload PDF**: Click "Choose a PDF" in the sidebar and select your PDF file
2. **Wait for Ingestion**: The app will process the PDF (extract text, tables, OCR)
3. **Ask Questions**: Type questions in the chat input at the bottom
4. **View History**: Chat history is saved in `chat_history.json`

---

## 📋 Project Flow

- **Ingest**: Load PDF with PyMuPDF → extract text and tables (`find_tables`) → render page images and OCR with Tesseract (`hin+eng`) for scanned content → normalize and merge sources.
- **Chunk & Embed**: Split text (overlapping chunks) → embed via `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- **Store**: Persist embeddings in ChromaDB.
- **Serve**: Build LangChain RetrievalQA using `LlamaCpp` with the local Q4_K_S model.
- **UI**: Streamlit app to upload a PDF, ingest it, and query in Hindi/English/Hinglish.

---

## 🔧 Troubleshooting

### Issue: Tesseract not found
**Solution**: 
- Verify Tesseract is installed and in PATH: `tesseract --version`
- If not in PATH, set environment variable:
  ```powershell
  $env:TESSDATA_PREFIX="C:\Program Files\Tesseract-OCR\tessdata"
  ```

### Issue: llama-cpp-python installation fails
**Solution**:
- Ensure Visual C++ Build Tools are installed
- Try installing with specific flags:
  ```powershell
  pip install llama-cpp-python --no-cache-dir
  ```

### Issue: Model file not found
**Solution**:
- Verify the model file exists at `models/llama-3.2-3b-instruct-q4_K_S.gguf`
- Check file name matches exactly (case-sensitive)
- Ensure model file is ~1.8 GB in size

### Issue: Out of memory errors
**Solution**:
- Reduce `N_CTX` in `.env` (default: 4096, try 2048)
- Close other applications
- Use a smaller model if available

### Issue: ChromaDB errors
**Solution**:
- Delete `chroma_db` folder and let it recreate
- Ensure write permissions in project directory

---

## 📁 Project Structure

```
BISAG-212/
├── app.py                 # Streamlit main application
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── models/               # Model files (not in git)
│   └── llama-3.2-3b-instruct-q4_K_S.gguf
├── chroma_db/            # Vector database (not in git)
├── src/
│   ├── config.py         # Configuration settings
│   ├── ingest.py         # PDF ingestion logic
│   ├── qa.py             # Q&A chain builder
│   └── vector_store.py   # ChromaDB wrapper
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

---

## 🔑 Environment Variables

- `MODEL_PATH` - Path to GGUF model file (default: `models/llama-3.2-3b-instruct-q4_K_S.gguf`)
- `CHROMA_DIR` - ChromaDB directory (default: `chroma_db`)
- `TESSDATA_PREFIX` - Tesseract data directory (if custom location)
- `N_THREADS` - Number of CPU threads for llama.cpp (default: auto-detect)
- `N_CTX` - Context window size (default: 4096)
- `TEMPERATURE` - LLM temperature (default: 0.1)
- `CHUNK_SIZE` - Text chunk size (default: 800)
- `CHUNK_OVERLAP` - Chunk overlap (default: 200)

---

## 📝 Notes

- **Tables**: Uses PyMuPDF `find_tables`; falls back to OCR text if not structurally found.
- **OCR**: Language pack uses `hin+eng`, enabling Hinglish questions and answers based on retrieved context.
- **Embeddings**: Uses multilingual MiniLM via sentence-transformers (no fastText dependency).
- **Model**: Model files are excluded from git to keep repository size small (~17 KB).

---

## 🆘 Need Help?

If you encounter issues:
1. Check the Troubleshooting section above
2. Verify all prerequisites are installed correctly
3. Ensure virtual environment is activated
4. Check that model file is in the correct location
5. Review error messages in the terminal/console

---

## 📋 Quick Reference

### Daily Usage Commands

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run the application
streamlit run app.py

# Verify setup
python verify_setup.py

# Pre-ingest a PDF from command line
python -m src.ingest --pdf path/to/file.pdf
```

---

## 🔧 Troubleshooting

### "Failed building wheel for llama-cpp-python" Error

**Symptoms**:
```
CMake Error: CMAKE_C_COMPILER not set, after EnableLanguage
CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage
error: subprocess-exited-with-error
× Building wheel for llama-cpp-python (pyproject.toml) did not run successfully.
```

**Solution**:
1. Install Visual C++ Build Tools (see Prerequisites section above)
2. Restart your terminal/command prompt completely
3. Try installing again:
   ```powershell
   pip install llama-cpp-python --no-cache-dir
   ```
4. If still failing, try installing from a different source:
   ```powershell
   pip install llama-cpp-python --index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
   ```

### "huggingface-cli not recognized" Error

**Solution**:
```powershell
pip install huggingface_hub
# Restart terminal, then try again
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF llama-3.2-3b-instruct-q4_K_S.gguf --local-dir models/
```

### "Tesseract not found" Error

**Solution**:
1. Verify Tesseract installation: `tesseract --version`
2. If not found, reinstall from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
3. Make sure "Add to PATH" was checked during installation
4. Or set in `.env`: `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`

### Virtual Environment Issues

**"execution policy" error in PowerShell**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Virtual environment not activating**:
```powershell
# Make sure you're in the project directory
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Model Download Issues

**Slow download or timeout**:
```powershell
# Use a download manager or try:
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF llama-3.2-3b-instruct-q4_K_S.gguf --local-dir models/ --resume-download
```

**Alternative download methods**:
- Direct link: `https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_S.gguf`
- Use browser download or tools like `wget`/`curl`

### Still Having Issues?

Run the verification script with verbose output:
```powershell
python verify_setup.py --verbose
```

Or use auto-fix mode:
```powershell
python verify_setup.py --auto-fix --verbose
```

---

### Common Workflow

1. **First Time Setup**:
   ```powershell
   git clone https://github.com/arpit8929/Multilingual-Model.git
   cd Multilingual-Model
   .\setup.ps1  # or follow manual steps
   # Download model file to models/
   python verify_setup.py
   ```

2. **Daily Usage**:
   ```powershell
   cd Multilingual-Model
   .\.venv\Scripts\Activate.ps1
   streamlit run app.py
   ```

3. **Upload PDF & Ask Questions**:
   - Open browser (usually http://localhost:8501)
   - Upload PDF via sidebar
   - Wait for ingestion to complete
   - Ask questions in chat

---

## 📄 License

[Add your license information here]

