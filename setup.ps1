# Quick Setup Script for Windows PowerShell
# Run this script to set up the project quickly
# Usage: .\setup.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Multilingual PDF Q&A - Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "[1/5] Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.10+ from python.org" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "[2/5] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "⚠️  Virtual environment already exists. Skipping..." -ForegroundColor Yellow
} else {
    python -m venv .venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "[3/5] Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1
Write-Host "✅ Virtual environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host ""
Write-Host "[4/5] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "✅ pip upgraded" -ForegroundColor Green

# Install dependencies
Write-Host ""
Write-Host "[5/5] Installing dependencies (this may take 5-10 minutes)..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Check model file
Write-Host ""
Write-Host "Checking model file..." -ForegroundColor Yellow
if (Test-Path "models\llama-3.2-3b-instruct-q4_K_S.gguf") {
    Write-Host "✅ Model file found" -ForegroundColor Green
} else {
    Write-Host "⚠️  Model file not found!" -ForegroundColor Yellow
    Write-Host "   Please download from: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF" -ForegroundColor Yellow
    Write-Host "   Place it in: models\llama-3.2-3b-instruct-q4_K_S.gguf" -ForegroundColor Yellow
}

# Final instructions
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Download the model file if not already done" -ForegroundColor White
Write-Host "2. Run verification: python verify_setup.py" -ForegroundColor White
Write-Host "3. Start the app: streamlit run app.py" -ForegroundColor White
Write-Host ""
Write-Host "Note: Make sure to activate the virtual environment before running:" -ForegroundColor Yellow
Write-Host "     .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
