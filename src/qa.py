from pathlib import Path
from typing import Optional

from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate

from src.config import settings
from src.vector_store import VectorStore


QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant for Hindi/English/Hinglish PDF QA.\n"
        "Use ONLY the provided context to answer. If the answer is not in the context, say: \"I do not know\".\n"
        "Respond in the same language as the question.\n\n"
        "Rules:\n"
        "- Answer exactly what is asked. Match criteria exactly (location, date, category, etc.).\n"
        "- Be precise. Do not guess or assume.\n"
        "- Use bullet points for lists. Use tables for pairs (name | value).\n"
        "- Language: Match the question's language (English/Hindi/Hinglish).\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    ),
)


def load_llm(model_path: Path | str | None = None) -> LlamaCpp:
    path = Path(model_path or settings.model_path)
    if not path.exists():
        raise FileNotFoundError(f"LLM model not found at {path}")
    return LlamaCpp(
        model_path=str(path),
        n_ctx=settings.n_ctx,
        n_threads=settings.n_threads,
        temperature=settings.temperature,
        f16_kv=True,
        verbose=False,
    )


def build_chain(store: Optional[VectorStore] = None) -> RetrievalQA:
    store = store or VectorStore()
    llm = load_llm()
    retriever = store.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )

