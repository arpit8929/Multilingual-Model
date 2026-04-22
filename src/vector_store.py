import os
import warnings
from collections import OrderedDict
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
from sentence_transformers import CrossEncoder



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
        if settings.use_reranker:
            self.reranker = CrossEncoder(settings.reranker_model)
        else:
            self.reranker = None
        self.active_document_language = self._infer_active_document_language()


    def add_documents(self, docs: List[Document]) -> None:
        self._db.add_documents(docs)
        self.set_active_document_language(self._infer_document_language_from_docs(docs))

    def as_retriever(self):
        """Get retriever with optimized search parameters."""
        search_k = max(settings.top_k * settings.retrieval_k_multiplier, settings.final_top_k * 2)
        fetch_k = max(20, search_k * 2)
        return self._db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": search_k,
                "fetch_k": fetch_k,
                "lambda_mult": 0.7,
                "score_threshold": settings.score_threshold
            }
        )

    def as_english_retriever(self):
        """Get a simpler retriever for English documents to preserve legacy behavior."""
        search_k = max(settings.top_k, settings.final_top_k, 6)
        return self._db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": search_k},
        )

    @staticmethod
    def _pair_key(doc: Document) -> str:
        metadata = doc.metadata or {}
        pair_id = metadata.get("pair_id")
        if pair_id:
            return str(pair_id)
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "?")
        doc_type = metadata.get("type", "text")
        return f"{source}:{page}:{doc_type}:{hash(doc.page_content)}"

    @staticmethod
    def _prefer_candidate(candidate: Document, existing: Document) -> bool:
        candidate_variant = (candidate.metadata or {}).get("variant", "original")
        existing_variant = (existing.metadata or {}).get("variant", "original")

        if candidate_variant != existing_variant:
            return candidate_variant == "original"

        candidate_lang = (candidate.metadata or {}).get("lang")
        existing_lang = (existing.metadata or {}).get("lang")
        if candidate_variant == "original" and candidate_lang != existing_lang:
            return candidate_lang == "hi"

        return False

    @staticmethod
    def _should_skip_reranking(docs: List[Document]) -> bool:
        for doc in docs:
            metadata = doc.metadata or {}
            if metadata.get("lang") == "hi" or metadata.get("variant") == "translation":
                return True
        return False

    def _merge_bilingual_results(self, docs: List[Document]) -> List[Document]:
        merged: "OrderedDict[str, Document]" = OrderedDict()
        for doc in docs:
            key = self._pair_key(doc)
            existing = merged.get(key)
            if existing is None or self._prefer_candidate(doc, existing):
                merged[key] = doc
        return list(merged.values())[:settings.final_top_k]

    @staticmethod
    def _metadata_int(metadata: dict, key: str, default: int = -1) -> int:
        value = metadata.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _page_doc_cache_key(self, metadata: dict) -> tuple[str, int, str]:
        return (
            str(metadata.get("source", "")),
            self._metadata_int(metadata, "page", -1),
            str(metadata.get("type", "text")),
        )

    @staticmethod
    def _infer_document_language_from_docs(docs: List[Document]) -> str:
        document_lang_counts = {"hi": 0, "en": 0}
        hi_original = 0
        en_original = 0

        for doc in docs:
            metadata = doc.metadata or {}
            document_lang = metadata.get("document_lang")
            if document_lang in document_lang_counts:
                document_lang_counts[document_lang] += 1
            if metadata.get("variant", "original") != "original":
                continue
            if metadata.get("lang") == "hi":
                hi_original += 1
            elif metadata.get("lang") == "en":
                en_original += 1

        if document_lang_counts["hi"] > 0 and document_lang_counts["hi"] >= document_lang_counts["en"]:
            return "hi"
        if document_lang_counts["en"] > 0:
            return "en"

        if hi_original > 0 and hi_original >= en_original:
            return "hi"
        if en_original > 0:
            return "en"
        return "unknown"

    def _infer_active_document_language(self) -> str:
        try:
            result = self._db.get(include=["metadatas"])
            metadatas = result.get("metadatas", []) or []
            docs = [Document(page_content="", metadata=metadata or {}) for metadata in metadatas]
            return self._infer_document_language_from_docs(docs)
        except Exception:
            return "unknown"

    def set_active_document_language(self, language: str) -> None:
        normalized = (language or "unknown").strip().lower()
        if normalized not in {"hi", "en", "unknown"}:
            normalized = "unknown"
        self.active_document_language = normalized

    def get_active_document_language(self) -> str:
        return getattr(self, "active_document_language", "unknown")

    def _get_page_docs(self, metadata: dict, page_cache: dict) -> List[Document]:
        key = self._page_doc_cache_key(metadata)
        if key in page_cache:
            return page_cache[key]

        source, page, doc_type = key
        if not source or page < 0:
            page_cache[key] = []
            return []

        result = self._db.get(
            where={
                "$and": [
                    {"source": source},
                    {"page": page},
                    {"type": doc_type},
                    {"variant": "original"},
                ]
            },
            include=["documents", "metadatas"],
        )

        documents = result.get("documents", []) or []
        metadatas = result.get("metadatas", []) or []
        page_docs: List[Document] = []
        for doc_text, doc_metadata in zip(documents, metadatas):
            if not doc_text or not doc_metadata:
                continue
            page_docs.append(Document(page_content=doc_text, metadata=doc_metadata))

        page_docs.sort(key=lambda doc: self._metadata_int(doc.metadata or {}, "chunk_index"))
        page_cache[key] = page_docs
        return page_docs

    def _expand_text_neighbors(self, docs: List[Document]) -> List[Document]:
        if settings.page_neighbor_window <= 0:
            return docs

        page_cache: dict = {}
        expanded: "OrderedDict[str, Document]" = OrderedDict()

        for doc in docs:
            expanded[self._pair_key(doc)] = doc
            metadata = doc.metadata or {}
            if metadata.get("type") != "text":
                continue

            chunk_index = self._metadata_int(metadata, "chunk_index")
            if chunk_index < 0:
                continue

            page_docs = self._get_page_docs(metadata, page_cache)
            if not page_docs:
                continue

            for neighbor in page_docs:
                neighbor_index = self._metadata_int(neighbor.metadata or {}, "chunk_index")
                if neighbor_index < 0:
                    continue
                if abs(neighbor_index - chunk_index) <= settings.page_neighbor_window:
                    key = self._pair_key(neighbor)
                    if key not in expanded:
                        expanded[key] = neighbor

        return list(expanded.values())[:settings.max_context_docs]

    def _trim_context_budget(self, docs: List[Document], query: str) -> List[Document]:
        if not docs:
            return []

        remaining_chars = max(settings.max_context_chars - len(query), settings.max_context_chars // 2)
        trimmed: List[Document] = []
        consumed = 0

        for doc in docs:
            content = (doc.page_content or "").strip()
            if not content:
                continue

            available = remaining_chars - consumed
            if available <= 0:
                break

            if len(content) > available:
                if not trimmed:
                    trimmed.append(Document(page_content=content[:available], metadata=doc.metadata))
                break

            trimmed.append(doc)
            consumed += len(content)

            if len(trimmed) >= settings.max_context_docs:
                break

        if len(trimmed) < len(docs):
            print(
                f"[Retrieval] Trimmed context docs from {len(docs)} to {len(trimmed)} "
                f"within ~{remaining_chars} chars"
            )
        return trimmed

    def _retrieve_for_queries(self, queries: List[str]) -> List[Document]:
        retriever = self.as_retriever()
        collected: List[Document] = []
        for query in queries:
            if not query:
                continue
            query_docs = retriever.invoke(query)
            collected.extend(query_docs)

        deduped: "OrderedDict[str, Document]" = OrderedDict()
        for doc in collected:
            key = self._pair_key(doc)
            existing = deduped.get(key)
            if existing is None or self._prefer_candidate(doc, existing):
                deduped[key] = doc
        return list(deduped.values())

    def _retrieve_english(self, query: str) -> List[Document]:
        retriever = self.as_english_retriever()
        docs = retriever.invoke(query)
        docs = docs[:max(settings.final_top_k, 5)]
        return self._trim_context_budget(docs, query)

    def retrieve(self, query: str, queries: List[str] | None = None):
        """
        Retrieve documents with optional reranking.
        """
        if self.get_active_document_language() == "en":
            return self._retrieve_english(query)

        search_queries = queries or [query]
        docs = self._retrieve_for_queries(search_queries)

        if not docs:
            return []

        # If reranker disabled → return directly
        if not settings.use_reranker or self.reranker is None or self._should_skip_reranking(docs):
            merged_docs = self._merge_bilingual_results(docs)
            expanded_docs = self._expand_text_neighbors(merged_docs)
            return self._trim_context_budget(expanded_docs, query)

        # Only rerank top N docs
        candidate_docs = docs[:settings.rerank_top_k]

        pairs = [(query, doc.page_content) for doc in candidate_docs]
        scores = self.reranker.predict(pairs)

        ranked_pairs = sorted(
            zip(scores, candidate_docs),
            key=lambda item: float(item[0]),
            reverse=True,
        )
        ranked_docs = [doc for _, doc in ranked_pairs]

        merged_docs = self._merge_bilingual_results(ranked_docs)
        expanded_docs = self._expand_text_neighbors(merged_docs)
        return self._trim_context_budget(expanded_docs, query)


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
        self.active_document_language = "unknown"

