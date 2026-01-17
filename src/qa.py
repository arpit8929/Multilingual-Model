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
        "You are a helpful assistant for PDF Q&A. Answer questions based ONLY on the provided context.\n\n"
        "CRITICAL LANGUAGE RULE:\n"
        "- If the question is in Hindi (Devanagari script), answer ONLY in Hindi.\n"
        "- If the question is in English, answer ONLY in English.\n"
        "- If the question is in Hinglish (Hindi words in English script), answer in Hinglish.\n"
        "- NEVER mix languages. NEVER add English explanations to Hindi answers.\n"
        "- Answer DIRECTLY without meta-commentary or explanations about the answer.\n"
        "- NEVER repeat the question in your answer. Just give the answer.\n"
        "- NEVER repeat your answer multiple times. Answer ONCE only.\n\n"
        "CRITICAL RULES FOR DATASET QUESTIONS:\n"
        "- When asked about the dataset used in THIS paper, look for mentions in:\n"
        "  * Abstract or introduction sections\n"
        "  * Methodology sections that describe the CURRENT study\n"
        "  * Phrases like 'we used', 'we worked with', 'this research uses', 'the proposed model'\n"
        "- IGNORE datasets mentioned only in:\n"
        "  * Reference citations [28], [39], etc.\n"
        "  * Literature review sections about OTHER papers\n"
        "  * Comparison tables mentioning multiple datasets\n"
        "- For THIS paper, the dataset is CICIDS-2017 (or CIC-IDS-2017). Only mention this if explicitly stated in the current study sections.\n\n"
        "CRITICAL RULES:\n"
        "1. Carefully read through ALL the provided context to find the answer.\n"
        "2. Distinguish between information about THIS paper vs OTHER papers mentioned in references.\n"
        "3. Look for specific names, datasets, methods, numbers, or technical terms mentioned in the context.\n"
        "4. If you find the answer anywhere in the context, provide it DIRECTLY. Do not add explanations.\n"
        "5. If the context only mentions that something EXISTS (e.g., 'limitations were identified') but does NOT list specific details, you MUST say: \"उत्तर संदर्भ में नहीं मिला\" (Hindi) or \"Answer not found in context\" (English).\n"
        "6. DO NOT make up or guess answers. DO NOT provide generic information not explicitly stated in the context.\n"
        "7. DO NOT infer or assume details that are not explicitly written in the context.\n"
        "8. Answer exactly what is asked. Be precise and include specific details (names, numbers, dates) when available.\n"
        "9. DO NOT add meta-commentary like 'The answer is...' or 'According to the context...'. Just provide the answer directly.\n"
        "10. DO NOT repeat the question or your answer. Give the answer ONCE only.\n\n"
        "IMPORTANT: Before saying 'I do not know', carefully re-read the context. Look for:\n"
        "- Dataset names (e.g., CICIDS-2017, CIC-IDS-2017, NSL-KDD)\n"
        "- Method names (e.g., LIME, SHAP, Random Forest)\n"
        "- Numbers and percentages\n"
        "- Technical terms related to the question\n\n"
        "Formatting:\n"
        "- Use bullet points for lists\n"
        "- Use tables for structured data (name | value)\n"
        "- Be specific: include exact names, numbers, and technical terms\n"
        "- Keep answers concise and direct (one sentence if possible)\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
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
    
    import re
    
    # Remove meta-commentary patterns (English and Hindi) - but be careful not to remove too much
    meta_patterns = [
        r"^The answer is\s*[:\-]?\s*",
        r"^According to the context[,\s]*",
        r"^Based on the provided context[,\s]*",
        r"^The information provided indicates that\s*",
        r"^Note:\s*",
        r"^Note that\s*",
        r"^उत्तर यह है\s*[:\-]?\s*",
        r"^संदर्भ के अनुसार[,\s]*",
        r"^प्रदान किए गए संदर्भ के आधार पर[,\s]*",
        r"^सूचना से पता चलता है कि\s*",
        r"^ध्यान दें:\s*",
        r"^This work proposes\s*",
        r"^The question asks\s*",
        r"^The answer provided is\s*",
    ]
    
    for pattern in meta_patterns:
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove explanations that start with "The answer is found in..." or similar (only at start)
    explanation_patterns = [
        r"^The answer is found in.*?\.\s*",
        r"^The question asks about.*?\.\s*",
        r"^Note:.*?\.\s*",
        r"^Note that.*?\.\s*",
    ]
    for pattern in explanation_patterns:
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    
    # Remove trailing explanations
    trailing_explanations = [
        r"\s*The answer is found in Section \d+.*$",
        r"\s*Note:.*$",
        r"\s*Note that.*$",
    ]
    for pattern in trailing_explanations:
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE | re.DOTALL)
    
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

