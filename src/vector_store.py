import os
import warnings
from pathlib import Path
from typing import List

# Suppress ChromaDB telemetry warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore", message=".*telemetry.*")
warnings.filterwarnings("ignore", message=".*ClientStartEvent.*")
warnings.filterwarnings("ignore", message=".*ClientCreateCollectionEvent.*")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from src.config import settings


class VectorStore:
    """Wrap ChromaDB with multilingual MiniLM embeddings."""

    def __init__(self, persist_directory: Path | None = None):
        self.persist_path = Path(persist_directory or settings.chroma_dir)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
        self._db = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_path),
        )

    def add_documents(self, docs: List[Document]) -> None:
        self._db.add_documents(docs)

    def as_retriever(self):
        """Get retriever with optimized search parameters."""
        # Standard similarity search with more results for better coverage
        # Increased from 3 to 5 to capture more relevant chunks
        return self._db.as_retriever(
            #search_kwargs={"k": min(settings.top_k + 2, 5)}  # Get 5 instead of 3 for better coverage
            search_type="mmr",
            search_kwargs={
                "k": settings.top_k,          # total chunks returned
                "fetch_k": 20,    # candidate pool
                "lambda_mult": 0.7,
                "score_threshold": settings.score_threshold
            }
        )

    def persist(self) -> None:
        self._db.persist()
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vector store."""
        try:
            result = self._db.get()
            return len(result.get("ids", []))
        except Exception:
            return 0
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        # Delete the collection and recreate it
        try:
            self._db.delete_collection()
        except Exception:
            pass
        # Recreate the database
        self._db = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_path),
        )

