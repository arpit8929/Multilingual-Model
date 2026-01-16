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
        "You are a helpful assistant for Hindi/English/Hinglish PDF QA.\n"
        "Use ONLY the provided context to answer. If the answer is not in the context, say: \"I do not know\".\n"
        "Respond exactly once in the same language as the question. Do not provide translations or multiple versions.\n\n"
        "Answering rules:\n"
        "- Answer exactly what is asked. If the question specifies criteria (location, date, category, etc.), "
        "only include items that match those criteria exactly.\n"
        "- If the context contains unrelated or off-topic text, ignore it. Do NOT bring in topics unrelated to the question.\n"
        "- For \"how\" questions (e.g., \"कैसे\", \"how\", \"kaise\"), explain the process, mechanism, or method. "
        "Do NOT just state that something happens - explain HOW it happens.\n"
        "- For \"why\" questions, provide reasons, causes, or explanations.\n"
        "- For \"what\" questions, provide definitions, descriptions, or explanations.\n"
        "- NEVER copy question instructions, multiple-choice options, or question format text from the context. "
        "Only extract and explain the actual content/information.\n"
        "- Do NOT include phrases like \"नीचे दिए गए विकल्पों में से एक चुनें\", \"choose from options\", "
        "\"select the correct answer\", or similar question format instructions in your answer.\n"
        "- For location-based questions (e.g., \"companies in Gandhinagar\", \"companies in Ahmedabad\"):\n"
        "  * Look at the table/data structure. Each row typically has: Company Name | Location/Address\n"
        "  * Check the Location/Address column of EACH row individually.\n"
        "  * ONLY include companies where the Location/Address column explicitly mentions the requested city.\n"
        "  * If a company's location mentions a different city (e.g., Bengaluru, Mumbai, Ahmedabad when asked for Gandhinagar), EXCLUDE it.\n"
        "  * Do NOT include companies just because they appear in the same chunk - verify each one's location.\n"
        "- Be precise and do not include irrelevant information. Do not guess or assume.\n\n"
        "Quality rules - CRITICAL:\n"
        "- Provide COMPLETE answers. Ensure your answer ends with a complete sentence, not mid-sentence.\n"
        "- Avoid repetition. Do not repeat the same information multiple times in different ways.\n"
        "- Be concise but comprehensive. Cover all key points from the context without being verbose.\n"
        "- Structure your answer logically:\n"
        "  1. Start with a direct answer to the question\n"
        "  2. Provide supporting details and explanations\n"
        "  3. Conclude with a summary if the answer is long\n"
        "- When explaining any concept/term or answering \"how\" questions, include:\n"
        "  * A clear definition or meaning (if applicable)\n"
        "  * The process, mechanism, or method (for \"how\" questions)\n"
        "  * Key characteristics or components\n"
        "  * Examples or specific ways it happens (if mentioned in context)\n"
        "  * Why it matters (if mentioned in the context)\n"
        "- Keep answers concise (ideally 1-3 sentences). If the answer is not explicitly supported by context, say \"I do not know\".\n"
        "- Do NOT cut off mid-sentence. Always complete your thoughts.\n"
        "- Provide a natural, flowing answer as if explaining to someone, not copying from a question paper.\n\n"
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


def clean_answer(answer: str) -> str:
    """
    Post-process answer to improve quality:
    - Remove incomplete sentences at the end
    - Remove excessive repetition
    - Remove question format instructions
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
        cleaned_lines.append(line)
    answer = '\n'.join(cleaned_lines).strip()
    
    # Find the last complete sentence by looking for sentence endings
    # Hindi/English sentence endings: . ! ? । (Devanagari full stop)
    # Split into sentences, keeping the punctuation
    sentences = []
    current_sentence = ""
    
    for char in answer:
        current_sentence += char
        if char in ['.', '!', '?', '।']:
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    # Add any remaining text
    if current_sentence.strip():
        # Check if it looks like an incomplete sentence (very short, no punctuation)
        if len(current_sentence.strip()) < 10 or not any(c in current_sentence for c in ['।', '.', '!', '?']):
            # Likely incomplete, don't include it
            pass
        else:
            sentences.append(current_sentence.strip())
    
    # Join complete sentences
    if sentences:
        result = ' '.join(sentences).strip()
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

