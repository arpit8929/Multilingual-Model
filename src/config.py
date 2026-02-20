from dataclasses import dataclass
from pathlib import Path
import os

# Try to load .env file if dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env file if it exists
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # dotenv not installed, will use environment variables directly
    pass


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

    # 🔍 Retrieval settings
    top_k: int = int(os.environ.get("TOP_K", 10))
    score_threshold: float = float(os.environ.get("SCORE_THRESHOLD", 0.4))

    # 🔁 Reranking settings
    use_reranker: bool = os.environ.get("USE_RERANKER", "true").lower() == "true"
    rerank_top_k: int = int(os.environ.get("RERANK_TOP_K", 8))   # docs from retriever to rerank
    final_top_k: int = int(os.environ.get("FINAL_TOP_K", 5))     # docs sent to LLM
    reranker_model: str = os.environ.get(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    collection_name: str = os.environ.get("COLLECTION_NAME", "pdf_qa")


settings = Settings()

