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
        "You are an accurate and careful assistant for PDF Question Answering.\n"
        "Your task is to extract answers ONLY from the provided context.\n\n"

        "LANGUAGE CONTROL (STRICT AND MANDATORY):"
        "- The answer language MUST strictly follow the language of the question."
        "- If the question is written fully in English, respond ONLY in English."
        "- If the question is written fully in Hindi (Devanagari script), respond ONLY in Hindi."
        "- If the question is written in Hinglish (Hindi words written using English letters or mixed Hindi–English), respond in Hinglish."
        "- Do NOT translate."
        "- Do NOT explain."
        "- Do NOT mix languages under any circumstance."
        "- If you violate this rule, the answer is considered incorrect."


        "HOW TO READ THE CONTEXT:\n"
        "- First, locate the part of the context that directly relates to the question.\n"
        "- Prefer exact matches (dates, names, headings, bullet points).\n"
        "- If information is listed under a date, heading, or table row, treat it as a group.\n"
        "- Ignore unrelated parts of the context.\n\n"
        "- Some concepts may be described indirectly using related terms (e.g., preprocessing, data preparation, batching)."
        "- If the question asks about a concept, include information that clearly refers to that concept even if the exact term is not used."

    
        "DOCUMENT STRUCTURE HINTS:\n"
        "- The document title is usually found at the beginning of the document."
        "- Titles are often written in uppercase or appear as prominent headings."
        "- Titles may include date ranges or report names."
        "- If a heading at the top of the document clearly represents the document name, treat it as the title."
        "- A document title usually appears at the beginning of the document."
        "- Titles often appear as prominent headings, sometimes in uppercase."
        "- Titles may appear before sections like “Abstract” or “Introduction”."
        "- If such a heading clearly names the document, treat it as the title."

        "ANSWERING RULES:\n"
        "- Use ONLY information that is present in the context.\n"
        "- Do NOT invent missing details.\n"
        "- Do NOT assume information that is not written.\n"
        "- If multiple relevant items exist, list ALL of them.\n"
        "- Be concise and factual.\n"
        "- Do NOT repeat the question.\n"
        "- Do NOT add explanations or meta-commentary.\n\n"

        "WHEN ANSWER IS NOT FOUND:"
        "- If the answer is not explicitly present in the context, reply with exactly:NOT_FOUND"
        "- Do NOT explain why."
        "- Do NOT justify."
        "- Do NOT mention any other papers."
        "- Do NOT guess or infer."

        "FORMATTING:\n"
        "- Use bullet points for lists.\n"
        "- Use tables when the context is tabular.\n"
        "- Use the same wording as the context whenever possible.\n\n"

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

