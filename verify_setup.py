"""
Setup Verification Script
Run this script to verify your installation is correct.
Usage: python verify_setup.py
"""

import sys
from pathlib import Path
import subprocess


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (Required: 3.10+)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (Required: 3.10+)")
        return False


def check_tesseract():
    """Check if Tesseract OCR is installed."""
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ Tesseract OCR installed: {version_line}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print("❌ Tesseract OCR not found. Please install from: https://github.com/UB-Mannheim/tesseract/wiki")
    return False


def check_model_file():
    """Check if model file exists."""
    model_path = Path("models/llama-3.2-3b-instruct-q4_K_S.gguf")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model file found: {model_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ Model file not found: {model_path}")
        print("   Download from: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF")
        return False


def check_dependencies():
    """Check if required Python packages are installed."""
    required_packages = [
        "streamlit",
        "langchain",
        "chromadb",
        "sentence_transformers",
        "fitz",  # PyMuPDF
        "pytesseract",
        "llama_cpp",
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == "fitz":
                __import__("fitz")
            elif package == "llama_cpp":
                __import__("llama_cpp")
            else:
                __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")
            missing.append(package)
    
    return len(missing) == 0


def check_virtual_env():
    """Check if running in virtual environment."""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    if in_venv:
        print(f"✅ Running in virtual environment: {sys.prefix}")
        return True
    else:
        print("⚠️  Not running in virtual environment (recommended but not required)")
        return True  # Not critical, just a warning


def check_chroma_db():
    """Check ChromaDB directory."""
    chroma_dir = Path("chroma_db")
    if chroma_dir.exists():
        print(f"✅ ChromaDB directory exists: {chroma_dir}")
    else:
        print(f"ℹ️  ChromaDB directory will be created automatically: {chroma_dir}")
    return True


def main():
    """Run all checks."""
    print("=" * 60)
    print("Setup Verification Script")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_env),
        ("Tesseract OCR", check_tesseract),
        ("Model File", check_model_file),
        ("Python Dependencies", check_dependencies),
        ("ChromaDB Directory", check_chroma_db),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n[{name}]")
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All checks passed! You're ready to run the application.")
        print("   Run: streamlit run app.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above before running the application.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
