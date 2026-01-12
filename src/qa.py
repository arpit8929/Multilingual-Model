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
        "Respond exactly once in the same language as the question. Do not provide translations or multiple versions.\n\n"
        "Answering rules:\n"
        "- Answer exactly what is asked. If the question specifies criteria (location, date, category, etc.), "
        "only include items that match those criteria exactly.\n"
        "- For location-based questions (e.g., \"companies in Gandhinagar\", \"companies in Ahmedabad\"):\n"
        "  * Look at the table/data structure. Each row typically has: Company Name | Location/Address\n"
        "  * Check the Location/Address column of EACH row individually.\n"
        "  * ONLY include companies where the Location/Address column explicitly mentions the requested city.\n"
        "  * If a company's location mentions a different city (e.g., Bengaluru, Mumbai, Ahmedabad when asked for Gandhinagar), EXCLUDE it.\n"
        "  * Do NOT include companies just because they appear in the same chunk - verify each one's location.\n"
        "- Be precise and do not include irrelevant information. Do not guess or assume.\n\n"
        "Formatting rules - ALWAYS FOLLOW:\n"
        "1. For lists of multiple items, ALWAYS use bullet points:\n"
        "   Example: - Item 1\n   - Item 2\n   - Item 3\n\n"
        "2. For pairs of related information (name + value, item + attribute), use a Markdown table:\n"
        "   Example: | Name | Value |\n   |------|-------|\n   | A | B |\n\n"
        "3. NEVER write lists as paragraphs. ALWAYS use bullets or tables for structured data.\n\n"
        "Language rules:\n"
        "- If the question is mostly in English, answer in English.\n"
        "- If the question is mostly in Hindi (Devanagari), answer in Hindi.\n"
        "- If the question is in Hinglish (Hindi written with English/Latin letters or a clear Hindi–English mix), "
        "answer in Hinglish: Hindi sentences but written in English letters.\n\n"
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

