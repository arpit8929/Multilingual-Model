"""
Setup Verification Script
Run this script to verify your installation is correct.
Usage: python verify_setup.py [--auto-fix] [--verbose] [--auto-download]
Options:
  --auto-fix, -f: Automatically install missing Python packages
  --verbose, -v: Show detailed error information
  --auto-download, -d: Attempt to automatically download the model file
"""

import sys
import os
from pathlib import Path
import subprocess

# Try to load .env file if dotenv is available
try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False
    def load_dotenv(path):
        # Manual .env parsing if dotenv not available
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()


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
    # Check for custom Tesseract path from .env
    tesseract_cmd = os.environ.get("TESSERACT_CMD")
    tesseract_paths = []
    
    if tesseract_cmd:
        tesseract_paths.append(tesseract_cmd)
    tesseract_paths.append("tesseract")  # Default PATH check
    
    for tesseract_exe in tesseract_paths:
        try:
            result = subprocess.run(
                [tesseract_exe, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                if tesseract_cmd:
                    print(f"✅ Tesseract OCR installed (from .env): {version_line}")
                    print(f"   Path: {tesseract_cmd}")
                else:
                    print(f"✅ Tesseract OCR installed: {version_line}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    print("❌ Tesseract OCR not found.")
    if tesseract_cmd:
        print(f"   ⚠️  TESSERACT_CMD in .env points to: {tesseract_cmd}")
        print(f"   But this path doesn't work. Please verify the path is correct.")
    print("   Windows Installation:")
    print("   1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    print("   2. Install tesseract-ocr-w64-setup-5.x.x.exe")
    print("   3. During installation, check 'Add to PATH'")
    print("   4. Install Hindi language pack (hin.traineddata)")
    print("   5. If not in PATH, add to .env: TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
    print("   6. After installation, restart your terminal and run this script again")
    return False


def check_huggingface_cli():
    """Check if huggingface-cli is available."""
    # Try direct command first
    try:
        result = subprocess.run(
            ["huggingface-cli", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✅ huggingface-cli available")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Try to find it in Python scripts directory
    try:
        import site
        scripts_dirs = site.getsitepackages() + [site.getusersitepackages()]
        for scripts_dir in scripts_dirs:
            scripts_path = Path(scripts_dir) / ".." / "Scripts" / "huggingface-cli.exe"
            if scripts_path.exists():
                try:
                    result = subprocess.run(
                        [str(scripts_path), "--version"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        print("✅ huggingface-cli found in Python Scripts directory")
                        return True
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
    except Exception:
        pass
    
    print("❌ huggingface-cli not found")
    return False


def install_huggingface_cli():
    """Install huggingface-cli if not available."""
    try:
        print("   Installing huggingface-hub (includes huggingface-cli)...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "huggingface_hub", "--no-warn-script-location", "--upgrade"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            # Check for dependency warnings
            if "dependency resolver does not currently take into account" in result.stderr:
                print("   ⚠️  Dependency conflicts detected but installation succeeded")
                print("      This is usually safe to ignore for huggingface packages")
            
            print("   ✅ huggingface-hub installed")
            # Check if cli is now available
            if check_huggingface_cli():
                return True
            else:
                print("   ⚠️  huggingface-hub installed but huggingface-cli not accessible")
                print("   The CLI should be available in your Python Scripts directory")
                print("   You may need to restart your terminal or add Python Scripts to PATH")
                return False
        else:
            print(f"   ❌ Failed to install huggingface-hub")
            stderr_lines = result.stderr.split('\n')
            for line in stderr_lines[-3:]:  # Show last 3 lines of error
                if line.strip():
                    print(f"      {line}")
            return False
    except subprocess.TimeoutExpired:
        print("   ❌ Installation timeout")
        return False
    except Exception as e:
        print(f"   ❌ Error installing huggingface-hub: {str(e)}")
        return False


def download_model_file():
    """Attempt to download the model file automatically."""
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Find the huggingface-cli executable
    cli_cmd = None
    
    # Try direct command first
    try:
        result = subprocess.run(
            ["huggingface-cli", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            cli_cmd = "huggingface-cli"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # If not found, try to find it in Python scripts directory
    if cli_cmd is None:
        try:
            import site
            scripts_dirs = site.getsitepackages() + [site.getusersitepackages()]
            for scripts_dir in scripts_dirs:
                scripts_path = Path(scripts_dir) / ".." / "Scripts" / "huggingface-cli.exe"
                if scripts_path.exists():
                    cli_cmd = str(scripts_path)
                    break
        except Exception:
            pass
    
    if cli_cmd is None:
        print("   ❌ Cannot find huggingface-cli executable")
        return False
    
    try:
        print("   Downloading model file (this may take several minutes)...")
        result = subprocess.run(
            [cli_cmd, "download", "bartowski/Llama-3.2-3B-Instruct-GGUF", 
             "llama-3.2-3b-instruct-q4_K_S.gguf", "--local-dir", "models/"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        if result.returncode == 0:
            print("   ✅ Model file downloaded successfully")
            return True
        else:
            print(f"   ❌ Download failed: {result.stderr[-500:]}")  # Show last 500 chars
            return False
    except subprocess.TimeoutExpired:
        print("   ❌ Download timeout (model is ~1.8 GB, try manual download)")
        return False
    except Exception as e:
        print(f"   ❌ Download error: {str(e)}")
        return False


def check_model_file(auto_download=False):
    """Check if model file exists."""
    model_path = Path("models/llama-3.2-3b-instruct-q4_K_S.gguf")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model file found: {model_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ Model file not found: {model_path}")
        
        if auto_download:
            print("   Attempting automatic download...")
            if not check_huggingface_cli():
                if not install_huggingface_cli():
                    print("   ❌ Cannot download automatically - huggingface-cli not available")
                    auto_download = False
                else:
                    print("   ✅ huggingface-cli ready, proceeding with download...")
            
            if auto_download:
                if download_model_file():
                    return True
        
        print("   Download options:")
        print("   1. Manual: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF")
        print("      Look for: llama-3.2-3b-instruct-q4_K_S.gguf (~1.8 GB)")
        print("   2. Using huggingface-cli (if installed):")
        print("      huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF llama-3.2-3b-instruct-q4_K_S.gguf --local-dir models/")
        print("   3. Create models/ directory and place the downloaded file there")
        print("   4. Run with --auto-download to attempt automatic download:")
        print("      python verify_setup.py --auto-download")
        return False


def install_package(package_name, import_name=None):
    """Install a Python package."""
    if import_name is None:
        import_name = package_name
    
    # Map import names to pip package names
    package_map = {
        "fitz": "pymupdf",
        "llama_cpp": "llama-cpp-python",
    }
    
    pip_name = package_map.get(import_name, import_name)
    
    try:
        print(f"   Installing {pip_name}...")
        # Use --no-warn-script-location to suppress PATH warnings
        # Use --quiet to reduce output, but show errors
        cmd = [sys.executable, "-m", "pip", "install", pip_name, "--no-warn-script-location"]
        
        # For packages with known conflicts, try with --upgrade
        if pip_name in ["huggingface_hub", "transformers"]:
            cmd.append("--upgrade")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            # Check for dependency warnings but don't fail on them
            if "dependency resolver does not currently take into account" in result.stderr:
                print(f"   ⚠️  Dependency conflicts detected but installation succeeded")
                if "huggingface-hub" in result.stderr:
                    print(f"      This is usually safe to ignore for huggingface packages")
            
            # Verify installation
            try:
                if import_name == "fitz":
                    __import__("fitz")
                elif import_name == "llama_cpp":
                    __import__("llama_cpp")
                else:
                    __import__(import_name)
                print(f"   ✅ Successfully installed {pip_name}")
                return True
            except ImportError:
                print(f"   ⚠️  {pip_name} installed but import failed")
                return False
        else:
            print(f"   ❌ Failed to install {pip_name}")
            # Show relevant error information
            stderr_lines = result.stderr.split('\n')
            for line in stderr_lines[-5:]:  # Show last 5 lines of error
                if line.strip():
                    print(f"      {line}")
            
            # Special handling for common errors
            if "llama-cpp-python" in pip_name:
                if "CMAKE_C_COMPILER not set" in result.stderr or "CMAKE_CXX_COMPILER not set" in result.stderr:
                    print(f"      This error indicates missing C++ build tools.")
                    print(f"      SOLUTION: Install Visual C++ Build Tools from:")
                    print(f"      https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022")
                    print(f"      Select 'Desktop development with C++' workload, then restart terminal and try again.")
                elif "Microsoft Visual C++" in result.stderr:
                    print(f"      This indicates a Visual C++ version compatibility issue.")
                    print(f"      Try: pip install llama-cpp-python --no-cache-dir --upgrade")
                else:
                    print(f"      General llama-cpp-python build error.")
                    print(f"      Try: pip install llama-cpp-python --no-cache-dir")
                    print(f"      Or: pip install --upgrade pip setuptools wheel")
            return False
    except subprocess.TimeoutExpired:
        print(f"   ❌ Installation timeout for {pip_name}")
        return False
    except Exception as e:
        print(f"   ❌ Error installing {pip_name}: {str(e)}")
        return False


def check_dependencies(auto_fix=False):
    """Check if required Python packages are installed."""
    required_packages = [
        ("streamlit", "streamlit"),
        ("langchain", "langchain"),
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence_transformers"),
        ("fitz", "fitz"),  # PyMuPDF
        ("pytesseract", "pytesseract"),
        ("llama_cpp", "llama_cpp"),
    ]
    
    missing = []
    for import_name, package_key in required_packages:
        try:
            if import_name == "fitz":
                __import__("fitz")
            elif import_name == "llama_cpp":
                __import__("llama_cpp")
            else:
                __import__(import_name)
            print(f"✅ {import_name} installed")
        except ImportError:
            print(f"❌ {import_name} not installed")
            missing.append((import_name, package_key))
        except Exception as e:
            print(f"❌ Error checking {import_name}: {e}")
            missing.append((import_name, package_key))
    
    if missing and auto_fix:
        print(f"\n🔧 Auto-fixing {len(missing)} missing package(s)...")
        for import_name, package_key in missing:
            try:
                install_package(package_key, import_name)
            except Exception as e:
                print(f"❌ Failed to auto-install {package_key}: {e}")
        # Re-check after installation
        print("\n🔍 Re-checking dependencies...")
        return check_dependencies(auto_fix=False)
    
    return len(missing) == 0


def check_virtual_env():
    """Check if running in virtual environment."""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    # Also check if venv/.venv directory exists
    venv_dirs = [Path("venv"), Path(".venv")]
    venv_exists = any(venv_dir.exists() for venv_dir in venv_dirs)
    
    if in_venv:
        print(f"✅ Running in virtual environment: {sys.prefix}")
        return True
    elif venv_exists:
        venv_dir = next((d for d in venv_dirs if d.exists()), None)
        print(f"⚠️  Virtual environment directory found ({venv_dir}) but not activated")
        print(f"   Activate it with:")
        if sys.platform == "win32":
            print(f"   PowerShell: {venv_dir}\\Scripts\\Activate.ps1")
            print(f"   CMD: {venv_dir}\\Scripts\\activate.bat")
        else:
            print(f"   source {venv_dir}/bin/activate")
        return True  # Not critical, just a warning
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
    # Fix Windows console encoding for emoji support
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            # Fallback for older Python versions
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Load .env file if it exists (before other checks)
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        try:
            if _HAS_DOTENV:
                load_dotenv(env_path)
                print("📄 Loaded .env file (using python-dotenv)")
            else:
                load_dotenv(env_path)  # Use manual parsing
                print("📄 Loaded .env file (manual parsing)")
        except Exception as e:
            print(f"⚠️  Failed to load .env file: {e}")
            print("   Continuing without .env file...")
    
    auto_fix = "--auto-fix" in sys.argv or "-f" in sys.argv
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    auto_download = "--auto-download" in sys.argv or "-d" in sys.argv
    
    print("=" * 60)
    print("Setup Verification Script")
    print("=" * 60)
    if auto_fix:
        print("🔧 Auto-fix mode enabled")
    if verbose:
        print("📝 Verbose mode enabled")
    if auto_download:
        print("⬇️  Auto-download mode enabled")
    print()
    
    # Collect all check results
    check_results = []
    critical_failures = []
    
    # Check Python version first (critical)
    print("[Python Version]")
    try:
        python_ok = check_python_version()
        check_results.append(("Python Version", python_ok, True))  # True = critical
        if not python_ok:
            critical_failures.append("Python Version")
    except Exception as e:
        print(f"❌ Unexpected error checking Python version: {e}")
        check_results.append(("Python Version", False, True))
        critical_failures.append("Python Version")
    
    # Check virtual environment (warning only)
    print("\n[Virtual Environment]")
    try:
        venv_ok = check_virtual_env()
        check_results.append(("Virtual Environment", venv_ok, False))  # False = not critical
    except Exception as e:
        print(f"⚠️  Unexpected error checking virtual environment: {e}")
        check_results.append(("Virtual Environment", False, False))
    
    # Check dependencies (critical, with auto-fix)
    print("\n[Python Dependencies]")
    try:
        deps_ok = check_dependencies(auto_fix=auto_fix)
        check_results.append(("Python Dependencies", deps_ok, True))
        if not deps_ok:
            critical_failures.append("Python Dependencies")
    except Exception as e:
        print(f"❌ Unexpected error checking dependencies: {e}")
        check_results.append(("Python Dependencies", False, True))
        critical_failures.append("Python Dependencies")
    
    # Check Tesseract (critical)
    print("\n[Tesseract OCR]")
    try:
        tesseract_ok = check_tesseract()
        check_results.append(("Tesseract OCR", tesseract_ok, True))
        if not tesseract_ok:
            critical_failures.append("Tesseract OCR")
    except Exception as e:
        print(f"❌ Unexpected error checking Tesseract: {e}")
        check_results.append(("Tesseract OCR", False, True))
        critical_failures.append("Tesseract OCR")
    
    # Check model file (critical)
    print("\n[Model File]")
    try:
        model_ok = check_model_file(auto_download=auto_download)
        check_results.append(("Model File", model_ok, True))
        if not model_ok:
            critical_failures.append("Model File")
    except Exception as e:
        print(f"❌ Unexpected error checking model file: {e}")
        check_results.append(("Model File", False, True))
        critical_failures.append("Model File")
    
    # Check ChromaDB directory (warning only)
    print("\n[ChromaDB Directory]")
    try:
        chroma_ok = check_chroma_db()
        check_results.append(("ChromaDB Directory", chroma_ok, False))
    except Exception as e:
        print(f"⚠️  Unexpected error checking ChromaDB: {e}")
        check_results.append(("ChromaDB Directory", False, False))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_critical_passed = len(critical_failures) == 0
    
    for name, result, is_critical in check_results:
        status = "✅ PASS" if result else ("❌ FAIL" if is_critical else "⚠️  WARN")
        print(f"{status}: {name}")
    
    print()
    if all_critical_passed:
        print("🎉 All critical checks passed! You're ready to run the application.")
        print("   Run: streamlit run app.py")
        return 0
    else:
        print("❌ Critical checks failed:")
        for failure in critical_failures:
            print(f"   • {failure}")
        
        if not auto_fix:
            missing_deps = any(name == "Python Dependencies" and not result 
                             for name, result, _ in check_results)
            if missing_deps:
                print("\n💡 Tip: Run with --auto-fix to automatically install missing Python packages:")
                print("   python verify_setup.py --auto-fix")
        
        if not auto_download:
            missing_model = any(name == "Model File" and not result 
                              for name, result, _ in check_results)
            if missing_model:
                print("\n💡 Tip: Run with --auto-download to attempt automatic model download:")
                print("   python verify_setup.py --auto-download")
        
        print("\n   Please fix the critical issues above before running the application.")
        print("   For detailed error information, run with --verbose:")
        print("   python verify_setup.py --verbose")
        return 1


if __name__ == "__main__":
    sys.exit(main())
