from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Settings:
    """Centralized configuration with env overrides."""

    model_path: Path = Path(os.environ.get("MODEL_PATH", "models/llama-3.2-3b-instruct-q4_K_S.gguf"))
    chroma_dir: Path = Path(os.environ.get("CHROMA_DIR", "chroma_db"))
    chunk_size: int = int(os.environ.get("CHUNK_SIZE", 800))
    chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP", 150))
    ocr_lang: str = os.environ.get("OCR_LANG", "hin+eng")
    n_ctx: int = int(os.environ.get("N_CTX", 4096))
    n_threads: int = int(os.environ.get("N_THREADS", max(os.cpu_count() or 4, 4)))
    temperature: float = float(os.environ.get("TEMPERATURE", 0.1))
    top_k: int = int(os.environ.get("TOP_K", 10))  # Increased slightly for better coverage while staying within context window
    score_threshold: float = float(os.environ.get("SCORE_THRESHOLD", 0.4))
    collection_name: str = os.environ.get("COLLECTION_NAME", "pdf_qa")


settings = Settings()

