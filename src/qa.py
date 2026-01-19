import re
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
        "You are a precise and reliable assistant for PDF Question Answering.\n"
        "Answer questions using ONLY the provided context.\n\n"

        "LANGUAGE RULES (STRICT):\n"
        "- If the question is in Hindi (Devanagari script), answer ONLY in Hindi.\n"
        "- If the question is in English, answer ONLY in English.\n"
        "- If the question is in Hinglish (Hindi words written in English letters or a clear Hindi–English mix), answer in Hinglish.\n"
        "- NEVER mix languages.\n"
        "- Do NOT explain language choice.\n\n"

        "ANSWERING RULES:\n"
        "- Use ONLY information that is explicitly stated in the context.\n"
        "- Do NOT infer, assume, or generalize beyond the context.\n"
        "- If multiple items appear together, include ONLY those that match the question exactly.\n"
        "- Do NOT add background information or explanations.\n"
        "- Do NOT repeat the question.\n"
        "- Do NOT repeat the answer.\n"
        "- Answer ONCE, clearly and concisely.\n\n"

        "GROUNDING & ACCURACY RULES:\n"
        "- Every statement in the answer must be directly supported by the context.\n"
        "- If the context mentions something without giving specific details, do NOT invent details.\n"
        "- For date-based or category-based questions, include only the information explicitly listed under that date or category.\n\n"

        "WHEN ANSWER IS NOT FOUND:\n"
        "- If the answer is NOT clearly stated anywhere in the context, reply exactly with:\n"
        "NOT_FOUND\n\n"

        "FORMATTING RULES:\n"
        "- Use bullet points for lists.\n"
        "- Use tables for structured or paired information (e.g., name | value).\n"
        "- Do NOT write long paragraphs for lists.\n"
        "- Keep answers short, factual, and to the point.\n\n"

        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
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
    
    # Find complete sentences and remove duplicates
    # Hindi/English sentence endings: . ! ? । (Devanagari full stop)
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
        result = ' '.join(unique_sentences).strip()
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
    
    return result.strip()


def build_chain(store: Optional[VectorStore] = None) -> RetrievalQA:
    """Build QA chain with improved retrieval."""
    store = store or VectorStore()
    llm = load_llm()
    # Get retriever with increased document count for better coverage
    retriever = store.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )

