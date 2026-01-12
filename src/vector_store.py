from pathlib import Path
from typing import List

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from src.config import settings


class VectorStore:
    """Wrap ChromaDB with multilingual MiniLM embeddings."""

    def __init__(self, persist_directory: Path | None = None):
        self.persist_path = Path(persist_directory or settings.chroma_dir)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
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
        # Chroma version in this project does not support score_threshold in query kwargs,
        # so we configure only k here. Relevance is primarily controlled via embeddings +
        # better chunking; additional filtering is handled in the prompt.
        return self._db.as_retriever(search_kwargs={"k": settings.top_k})

    def persist(self) -> None:
        self._db.persist()

