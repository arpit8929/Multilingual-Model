"""
Alternative way to run the backend from project root
Usage: python -m backend.main
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn
    from backend.main import app
    uvicorn.run(app, host="0.0.0.0", port=8000)
