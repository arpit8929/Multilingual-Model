# Installation Check Script

Run these commands to verify your setup:

## Check Python Setup

```powershell
# Check Python version (should be 3.10+)
python --version

# Check if virtual environment is activated
# You should see (.venv) in your prompt

# Check if backend dependencies are installed
pip list | Select-String "fastapi|uvicorn|langchain"
```

## Check Node.js Setup

```powershell
# Check Node.js version (should be 18+)
node --version

# Check npm version
npm --version

# If these fail, install Node.js (see INSTALL_NODEJS.md)
```

## Check Frontend Setup

```powershell
cd Multilingual-Model\frontend

# Check if node_modules exists
Test-Path node_modules

# If false, run: npm install
```

## Check Backend Setup

```powershell
cd Multilingual-Model

# Check if model file exists
Test-Path models\llama-3.2-3b-instruct-q4_K_S.gguf

# Check if chroma_db exists
Test-Path chroma_db
```

## Complete Setup Verification

Run this script to check everything:

```powershell
Write-Host "=== Python Check ===" -ForegroundColor Cyan
python --version
pip --version

Write-Host "`n=== Node.js Check ===" -ForegroundColor Cyan
node --version
npm --version

Write-Host "`n=== Project Files Check ===" -ForegroundColor Cyan
Test-Path "models\llama-3.2-3b-instruct-q4_K_S.gguf"
Test-Path "frontend\node_modules"
Test-Path "backend\main.py"

Write-Host "`n=== Dependencies Check ===" -ForegroundColor Cyan
pip list | Select-String "fastapi|uvicorn|langchain|chromadb"
```
