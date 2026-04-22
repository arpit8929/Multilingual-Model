import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from src.config import settings
from src.vector_store import VectorStore
from langchain.schema import BaseRetriever
from typing import List
from langchain.schema import Document
from pydantic import Field


class RerankRetriever(BaseRetriever):
    store: VectorStore = Field() 

    def _get_relevant_documents(self, query: str) -> List[Document]:
        document_language = self.store.get_active_document_language()
        return self.store.retrieve(
            query,
            queries=build_query_variants(query, document_language=document_language),
        )

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        document_language = self.store.get_active_document_language()
        return self.store.retrieve(
            query,
            queries=build_query_variants(query, document_language=document_language),
        )


ENGLISH_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Use the following context to answer the question.\n"
        "If the answer is not in the context, say exactly: Not mentioned in the document.\n"
        "Do not use outside knowledge.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    ),
)


QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an accurate and careful assistant for PDF Question Answering.\n"
        "RULES (follow strictly):\n"
        "- Answer ONLY using the information in the provided context.\n"
        "- Do NOT use outside knowledge.\n"
        "- Do NOT guess.\n"
        "- If the answer is not present in the context, respond exactly with: Not mentioned in the document.\n"
        "- Answer in the same language as the question.\n"
        "- Prefer exact names, dates, numbers, and phrases from the context.\n"
        "- If the context contains multiple points, include all of them.\n"
        "- If the context appears noisy, still use only the readable supported parts.\n"
        "- If the document lists multiple points, you MUST return ALL points.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    ),
)


TRANSLATION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "Translate the following Hindi document text into concise, faithful English.\n"
        "Rules:\n"
        "- Preserve names, numbers, dates, abbreviations, and list items.\n"
        "- Do not explain the text.\n"
        "- Do not summarize.\n"
        "- Return only the English translation.\n\n"
        "Hindi text:\n{text}\n\n"
        "English translation:"
    ),
)


QUERY_TRANSLATION_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "Translate the following user question into natural Hindi for semantic search.\n"
        "Rules:\n"
        "- Preserve names, numbers, dates, abbreviations, and technical terms.\n"
        "- Return a single clean sentence only.\n"
        "- Return only the Hindi translation.\n"
        "- Do not explain anything.\n\n"
        "Question:\n{question}\n\n"
        "Hindi translation:"
    ),
)


QUERY_ENGLISH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "Translate the following user question into clear English for semantic search.\n"
        "Rules:\n"
        "- Preserve names, numbers, dates, abbreviations, and technical terms.\n"
        "- Return a single clean sentence only.\n"
        "- Return only the English translation.\n"
        "- Do not explain anything.\n\n"
        "Question:\n{question}\n\n"
        "English translation:"
    ),
)


ANSWER_REWRITE_PROMPT = PromptTemplate(
    input_variables=["target_language", "answer"],
    template=(
        "Rewrite the following answer in {target_language}.\n"
        "Rules:\n"
        "- Keep the meaning exactly the same.\n"
        "- Preserve names, dates, numbers, and list items.\n"
        "- Do not add new facts.\n"
        "- Do not mention the user, the question, the rewrite process, or the language choice.\n"
        "- Do not include labels such as User, Question, Answer, Rewritten answer, Note, Explanation, or Translation.\n"
        "- Return only one final answer block.\n"
        "- Return only the rewritten answer.\n\n"
        "Answer:\n{answer}\n\n"
        "Rewritten answer:"
    ),
)


HINGLISH_REWRITE_PROMPT = PromptTemplate(
    input_variables=["answer"],
    template=(
        "Rewrite the following answer in natural Hinglish.\n"
        "Rules:\n"
        "- Use only Roman script. Do not use any Devanagari characters.\n"
        "- Mix Hindi and English naturally, like everyday Indian Hinglish.\n"
        "- Keep the meaning exactly the same.\n"
        "- Preserve names, dates, numbers, and list items.\n"
        "- Do not add new facts.\n"
        "- Do not include labels such as User, Question, Answer, Rewritten answer, Note, Explanation, or Translation.\n"
        "- Return only one final answer block.\n\n"
        "Answer:\n{answer}\n\n"
        "Hinglish answer:"
    ),
)


QUESTION_WRAPPER_PROMPT = PromptTemplate(
    input_variables=["language_label", "question"],
    template=(
        "Answer language: {language_label}\n"
        "User question: {question}"
    ),
)


@lru_cache(maxsize=1)
def _get_cached_llm() -> LlamaCpp:
    path = Path(settings.model_path)
    if not path.exists():
        raise FileNotFoundError(f"LLM model not found at {path}")
    return LlamaCpp(
        model_path=str(path),
        n_ctx=settings.n_ctx,
        n_threads=settings.n_threads,
        temperature=settings.temperature,
        f16_kv=True,
        verbose=False,
        stop=["\nUser:", "\nHuman:", "\nQuestion:"]
    )


def load_llm(model_path: Path | str | None = None) -> LlamaCpp:
    path = Path(model_path or settings.model_path)
    if path != Path(settings.model_path):
        if not path.exists():
            raise FileNotFoundError(f"LLM model not found at {path}")
        return LlamaCpp(
            model_path=str(path),
            n_ctx=settings.n_ctx,
            n_threads=settings.n_threads,
            temperature=settings.temperature,
            f16_kv=True,
            verbose=False,
            stop=["\nUser:", "\nHuman:", "\nQuestion:"]
        )
    return _get_cached_llm()


@lru_cache(maxsize=256)
def _translate_with_prompt(prompt_text: str) -> str:
    return load_llm().invoke(prompt_text).strip()


def translate_text_to_english(text: str) -> str:
    if not text or not text.strip():
        return ""

    preview = safe_log_preview(text)
    print(f"[Translation] Running translation for Hindi chunk ({len(text)} chars): {preview}")
    prompt = TRANSLATION_PROMPT.format(text=text[:settings.translation_max_chars])
    raw_translation = _translate_with_prompt(prompt)

    if "English translation:" in raw_translation:
        raw_translation = raw_translation.split("English translation:", 1)[-1].strip()

    lines = []
    for line in raw_translation.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned.lower().startswith("hindi text:"):
            continue
        if cleaned.lower().startswith("translation:"):
            cleaned = cleaned.split(":", 1)[-1].strip()
        lines.append(cleaned)

    translation = "\n".join(lines).strip()
    translated_preview = safe_log_preview(translation)
    print(f"[Translation] Completed translation ({len(translation)} chars): {translated_preview}")
    return translation


def safe_log_preview(text: str) -> str:
    preview = " ".join((text or "").split())[:settings.translation_log_preview]
    encoding = sys.stdout.encoding or "utf-8"
    return preview.encode(encoding, errors="replace").decode(encoding, errors="replace")


def clean_query_translation(raw_translation: str, label: str) -> str:
    cleaned = (raw_translation or "").strip()
    if not cleaned:
        return ""

    if label in cleaned:
        cleaned = cleaned.split(label, 1)[-1].strip()

    cleaned = " ".join(line.strip() for line in cleaned.splitlines() if line.strip())
    leakage_pattern = re.compile(
        r"\b(?:Answer|Note|User question|Question|Rewritten answer|Explanation|Translation|Semantic search)\s*:",
        re.IGNORECASE,
    )
    leakage_match = leakage_pattern.search(cleaned)
    if leakage_match:
        cleaned = cleaned[:leakage_match.start()].strip()

    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -:\n\t\"'")
    sentences = [segment.strip() for segment in re.split(r"(?<=[?.!।])\s+", cleaned) if segment.strip()]
    if len(sentences) > 1 and len(sentences[0]) >= 12:
        cleaned = sentences[0]

    return cleaned.strip()


def is_question_like_query(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False

    lower = cleaned.lower()
    question_markers = [
        "what", "when", "where", "why", "how", "who", "which",
        "kya", "kab", "kaun", "kis", "kitna", "kitni", "kitne", "kaise", "kyon",
        "क्या", "कब", "कौन", "किस", "कितना", "कितनी", "कितने", "कैसे", "क्यों",
    ]
    if any(re.search(rf"\b{re.escape(marker)}\b", lower) for marker in question_markers):
        return True
    if "?" in cleaned or "।" in cleaned:
        return True
    return len(cleaned.split()) <= 8


def translate_query_to_hindi(question: str) -> str:
    if not question or not question.strip():
        return ""

    prompt = QUERY_TRANSLATION_PROMPT.format(question=question.strip())
    raw_translation = _translate_with_prompt(prompt)
    translation = clean_query_translation(raw_translation, "Hindi translation:")
    if translation and not is_question_like_query(translation):
        print("[QueryTranslation] Ignoring Hindi retrieval query because it does not look like a search question")
        return ""
    if translation:
        preview = safe_log_preview(translation)
        print(f"[QueryTranslation] Using Hindi retrieval query: {preview}")
    return translation


def translate_query_to_english(question: str) -> str:
    if not question or not question.strip():
        return ""

    prompt = QUERY_ENGLISH_PROMPT.format(question=question.strip())
    raw_translation = _translate_with_prompt(prompt)
    translation = clean_query_translation(raw_translation, "English translation:")
    if translation and not is_question_like_query(translation):
        print("[QueryTranslation] Ignoring English retrieval query because it does not look like a search question")
        return ""
    if translation:
        preview = safe_log_preview(translation)
        print(f"[QueryTranslation] Using English retrieval query: {preview}")
    return translation


def _append_unique_variant(variants: List[str], candidate: str) -> None:
    cleaned = (candidate or "").strip()
    if not cleaned:
        return

    candidate_key = " ".join(cleaned.lower().split())
    for existing in variants:
        if " ".join(existing.lower().split()) == candidate_key:
            return
    variants.append(cleaned)


def build_query_variants(query: str, document_language: str = "unknown") -> List[str]:
    normalized_query = extract_search_question(query).strip()
    if not normalized_query:
        return []

    query_style = detect_language_style(normalized_query)
    variants: List[str] = []
    _append_unique_variant(variants, normalized_query)
    document_language = (document_language or "unknown").strip().lower()

    if not settings.enable_query_translation:
        return variants

    if document_language == "en":
        print("[QueryTranslation] Skipping query translation for English document")
        return variants

    if query_style == "hindi":
        return variants

    try:
        hindi_query = translate_query_to_hindi(normalized_query)
    except Exception as exc:
        print(f"[QueryTranslation] Failed to translate query: {exc}")
        hindi_query = ""
    _append_unique_variant(variants, hindi_query)

    return variants


def detect_language_style(text: str) -> str:
    normalized = (text or "").strip()
    if not normalized:
        return "english"

    if re.search(r"[\u0900-\u097F]", normalized):
        return "hindi"

    lower = normalized.lower()
    roman_hindi_markers = {
        "kya", "ka", "ki", "ke", "hai", "hain", "ho", "tha", "thi", "the", "hua", "hui",
        "hota", "hoti", "hote", "ko", "se", "par", "tak", "mein", "me", "kaise", "kab",
        "kyu", "kyon", "kitna", "kitni", "kis", "kon", "kaun", "kitne", "bachche",
        "bache", "vidyalay", "school", "bahar", "adhikar", "shiksha", "act", "kanoon",
        "lagu", "umar", "umra", "ayu", "varg", "kitab", "article", "prashikshit",
        "shikshak", "teacher", "student", "ratio", "pratishat", "percent", "cover",
        "hota", "hoti", "honge", "milta", "milti", "pucha", "poocha", "kaunsa", "kaunsi"
    }
    tokens = re.findall(r"[a-zA-Z]+", lower)
    marker_hits = sum(1 for token in tokens if token in roman_hindi_markers)
    if marker_hits >= 2:
        return "hinglish"

    roman_question_patterns = [
        r"\bkya\b.*\b(hai|tha|hua|hota|hoti)\b",
        r"\b(kab|kaise|kyu|kyon|kitne|kitna|kitni|kaun|kis)\b",
        r"\b(age|title|act|law|article|teacher|student|ratio|school)\b.*\b(kya|kaunsa|kitna|kab)\b",
    ]
    if any(re.search(pattern, lower) for pattern in roman_question_patterns):
        return "hinglish"

    return "english"


def language_label(language_style: str) -> str:
    if language_style == "hindi":
        return "Hindi in Devanagari script"
    if language_style == "hinglish":
        return "natural Hinglish in Roman script only"
    return "English"


def format_query_for_chain(question: str) -> str:
    style = detect_language_style(question)
    return QUESTION_WRAPPER_PROMPT.format(
        language_label=language_label(style),
        question=question.strip(),
    )


def extract_search_question(query: str) -> str:
    if not query:
        return ""

    marker = "User question:"
    if marker in query:
        return query.split(marker, 1)[-1].strip()
    return query.strip()


def answer_not_found_message(question: str) -> str:
    style = detect_language_style(question)
    if style == "hindi":
        return "उत्तर संदर्भ में नहीं मिला"
    if style == "hinglish":
        return "Answer context me nahi mila"
    return "Answer not found in context"


def strip_prompt_leakage(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    leakage_markers = [
        "User question:",
        "Question:",
        "User:",
        "User ",
        "Answer:",
        "Rewritten answer:",
        "Note:",
        "Explanation:",
        "Translation:",
        "Translated:",
    ]

    # Drop everything after the first leaked prompt marker that appears after some real answer content.
    for marker in leakage_markers:
        idx = cleaned.find(marker)
        if idx > 0:
            prefix = cleaned[:idx].strip()
            if len(prefix) >= 8:
                cleaned = prefix
                break

    lines = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        prefixed_answer = re.match(
            r"^(Answer|Rewritten answer|Translation|Translated)\s*:\s*(.+)$",
            stripped,
            re.IGNORECASE,
        )
        if prefixed_answer:
            stripped = prefixed_answer.group(2).strip()
            if not stripped:
                continue
            lines.append(stripped)
            continue
        if re.match(r"^(User question|Question|User|Note|Explanation)\b", stripped, re.IGNORECASE):
            break
        lines.append(stripped)

    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def dedupe_repeated_blocks(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", cleaned) if paragraph.strip()]
    unique_paragraphs: List[str] = []
    seen = set()
    for paragraph in paragraphs:
        normalized = " ".join(paragraph.lower().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_paragraphs.append(paragraph)

    return "\n\n".join(unique_paragraphs).strip()


def should_rewrite_answer(answer: str, target_style: str) -> bool:
    if not answer or not answer.strip():
        return False

    if not re.search(r"[A-Za-z\u0900-\u097F]", answer):
        return False

    detected_style = detect_language_style(answer)
    if target_style == "hinglish":
        return detected_style != "hinglish" or bool(re.search(r"[\u0900-\u097F]", answer))
    return detected_style != target_style


def rewrite_answer_to_language(answer: str, target_style: str) -> str:
    if target_style == "hinglish":
        prompt = HINGLISH_REWRITE_PROMPT.format(answer=answer.strip())
    else:
        prompt = ANSWER_REWRITE_PROMPT.format(
            target_language=language_label(target_style),
            answer=answer.strip(),
        )
    rewritten = _translate_with_prompt(prompt)
    if target_style == "hinglish":
        if "Hinglish answer:" in rewritten:
            rewritten = rewritten.split("Hinglish answer:", 1)[-1].strip()
    elif "Rewritten answer:" in rewritten:
        rewritten = rewritten.split("Rewritten answer:", 1)[-1].strip()
    rewritten = strip_prompt_leakage(rewritten)
    rewritten = dedupe_repeated_blocks(rewritten)
    return rewritten.strip()


def enforce_answer_language(answer: str, question: str, document_language: str = "unknown") -> str:
    cleaned_answer = strip_prompt_leakage(answer)
    cleaned_answer = dedupe_repeated_blocks(cleaned_answer)
    target_style = detect_language_style(question)
    document_language = (document_language or "unknown").strip().lower()

    if cleaned_answer in {"Not mentioned in the document.", "NOT_FOUND"}:
        return answer_not_found_message(question)

    if document_language == "en":
        return cleaned_answer

    if not should_rewrite_answer(cleaned_answer, target_style):
        return cleaned_answer

    try:
        print(
            f"[AnswerLanguage] Rewriting answer from {detect_language_style(cleaned_answer)} "
            f"to {target_style}"
        )
        rewritten = rewrite_answer_to_language(cleaned_answer, target_style)
        rewritten = strip_prompt_leakage(rewritten)
        rewritten = dedupe_repeated_blocks(rewritten)
        return rewritten or cleaned_answer
    except Exception as exc:
        print(f"[AnswerLanguage] Failed to rewrite answer: {exc}")
        return cleaned_answer


def clean_answer(answer: str) -> str:
    """
    Post-process answer to improve quality:
    - Remove incomplete sentences at the end
    - Remove excessive repetition
    - Remove question format instructions
    - Remove meta-commentary and explanations
    - Ensure proper sentence endings
    """
    if not answer:
        return answer
    answer = strip_prompt_leakage(answer)
    answer = dedupe_repeated_blocks(answer)
    # Preserve numbered or bullet lists exactly as-is
    if re.search(r'^\s*(\d+\.|-|\*)\s+', answer, re.MULTILINE):
        return answer.strip()
    answer = answer.strip()
    
    # Remove common question format phrases (Hindi and English)
    question_format_phrases = [
        "नीचे दिए गए विकल्पों में से एक चुनें",
        "choose from the options given below",
        "select the correct answer",
        "choose one of the options",
        "नीचे दिए गए विकल्पों में से",
        "from the options below",
        "select from options",
        "option",
        "विकल्प",
    ]
    
    for phrase in question_format_phrases:
        # Remove phrase and everything after it if it appears
        idx = answer.lower().find(phrase.lower())
        if idx != -1:
            # Check if it's followed by option letters (A, B, C, etc.)
            remaining = answer[idx:idx+100].lower()
            if any(marker in remaining for marker in ['a)', 'b)', 'c)', 'd)', 'option a', 'option b']):
                answer = answer[:idx].strip()
                break
    
    # Remove question repetition patterns (Hindi and English)
    # Remove patterns like "प्रश्न: ... उत्तर: ..." or "Question: ... Answer: ..."
    question_answer_patterns = [
        r"प्रश्न:.*?उत्तर:\s*",
        r"Question:.*?Answer:\s*",
        r"Q:.*?A:\s*",
        r"Q\..*?A\.\s*",
    ]
    for pattern in question_answer_patterns:
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove lines that look like multiple choice options (A), B), C), etc.)
    lines = answer.split('\n')
    cleaned_lines = []
    for line in lines:
        line_stripped = line.strip()
        # Skip lines that start with option markers
        if re.match(r'^[A-E]\)\s*', line_stripped, re.IGNORECASE):
            continue
        # Skip lines that are just option letters
        if re.match(r'^[A-E]\s*$', line_stripped, re.IGNORECASE):
            continue
        # Skip lines that repeat "प्रश्न:" or "Question:"
        if re.match(r'^(प्रश्न|Question|Q):', line_stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    answer = '\n'.join(cleaned_lines).strip()
    
    # Remove translation meta-commentary
    translation_patterns = [
        r"Translation:.*$",
        r"Translated:.*$",
        r"The answer is in Hindi.*$",
        r"The answer is in English.*$",
        r"यह उत्तर .*? में है.*$",
    ]
    for pattern in translation_patterns:
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    answer = strip_prompt_leakage(answer)
    answer = dedupe_repeated_blocks(answer)
    
    # Find complete sentences and remove duplicates
    # Hindi/English sentence endings: . ! ? । 
    # Split into sentences, keeping the punctuation
    sentences = []
    current_sentence = ""
    
    for char in answer:
        current_sentence += char
        if char in ['.', '!', '?', '।']:
            sent = current_sentence.strip()
            if sent:
                sentences.append(sent)
            current_sentence = ""
    
    # Add any remaining text
    if current_sentence.strip():
        # Check if it looks like an incomplete sentence (very short, no punctuation)
        if len(current_sentence.strip()) >= 10:
            sentences.append(current_sentence.strip())
    
    # Remove duplicate sentences (normalize for comparison)
    unique_sentences = []
    seen_sentences = set()
    for sent in sentences:
        # Normalize: lowercase, remove extra spaces
        normalized = ' '.join(sent.lower().split())
        if normalized not in seen_sentences and len(sent.strip()) > 5:
            unique_sentences.append(sent)
            seen_sentences.add(normalized)
    
    # Join complete sentences
    if unique_sentences:
        result = '\n'.join(unique_sentences).strip()
    else:
        # Fallback: use original but ensure it ends properly
        result = answer
        if result and result[-1] not in ['.', '!', '?', '।']:
            # Find last space and cut there if sentence seems incomplete
            last_space = result.rfind(' ')
            if last_space > len(result) * 0.8:  # If last word is less than 20% of text
                result = result[:last_space]
            result += '.'
    
    # Remove excessive repetition - check for repeated phrases
    words = result.split()
    if len(words) > 15:
        # Look for 4-word phrases that repeat
        seen_phrases = {}
        for i in range(len(words) - 3):
            phrase = ' '.join(words[i:i+4])
            if phrase in seen_phrases:
                seen_phrases[phrase] += 1
            else:
                seen_phrases[phrase] = 1
        
        # Remove phrases that appear more than twice
        for phrase, count in seen_phrases.items():
            if count > 2:
                # Remove all but first occurrence
                parts = result.split(phrase)
                if len(parts) > 2:
                    result = phrase.join([parts[0]] + parts[2:])
                    break
    
    # Ensure it ends with proper punctuation
    if result and result[-1] not in ['.', '!', '?', '।']:
        result += '.'
    
    result = strip_prompt_leakage(result)
    result = dedupe_repeated_blocks(result)
    return result.strip()



def build_chain(store: Optional[VectorStore] = None) -> RetrievalQA:
    """Build QA chain with reranking retrieval."""

    store = store or VectorStore()
    llm = load_llm()
    document_language = store.get_active_document_language()

    # Use custom reranking retriever
    retriever = RerankRetriever(store=store)
    prompt = ENGLISH_QA_PROMPT if document_language == "en" else QA_PROMPT


    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
        
    )
