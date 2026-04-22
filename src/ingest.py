import argparse
import hashlib
import os
import re
import warnings
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import List, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from tqdm import tqdm

from src.config import settings
from src.qa import translate_text_to_english
from src.vector_store import VectorStore

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


# Suppress warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", category=UserWarning)


_PADDLE_OCR_EN = None
_PADDLE_OCR_HI = None

def _load_paddle_ocr():
    global _PADDLE_OCR_EN, _PADDLE_OCR_HI

    if _PADDLE_OCR_EN is None:
        _PADDLE_OCR_EN = PaddleOCR(use_angle_cls=True, lang="en")

    if _PADDLE_OCR_HI is None:
        _PADDLE_OCR_HI = PaddleOCR(use_angle_cls=True, lang="hi")

    return _PADDLE_OCR_EN, _PADDLE_OCR_HI

def extract_title_from_first_page(doc):
    page = doc[0]
    blocks = page.get_text("blocks")

    # Sort text blocks top-to-bottom
    blocks = sorted(blocks, key=lambda b: b[1])

    for block in blocks:
        text = block[4].strip()

        # Skip very small text
        if len(text) < 10:
            continue

        # Heuristic 1: ALL CAPS titles
        if text.isupper():
            return text

        # Heuristic 2: First large meaningful line
        if len(text) > 20:
            return text

    return None

@dataclass
class PageExtraction:
    text: str
    tables: List[str]
    ocr_text: str


@dataclass
class ChunkSeed:
    page: int
    doc_type: str
    content: str


@dataclass
class OCRCandidate:
    source: str
    text: str

def _extract_tables(page: fitz.Page) -> List[pd.DataFrame]:
    try:
        tables = page.find_tables()
    except Exception:
        return []
    dfs: List[pd.DataFrame] = []
    for table in tables or []:
        try:
            dfs.append(pd.DataFrame(table.extract()))
        except Exception:
            continue
    return dfs


def normalize_extracted_text(text: str) -> str:
    if not text:
        return ""

    normalized = text.replace("\ufeff", "").replace("\u200c", "").replace("\u200d", "")
    normalized = normalized.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def count_hindi_chars(text: str) -> int:
    return sum(1 for char in text if "\u0900" <= char <= "\u097F")


def count_latin_chars(text: str) -> int:
    return sum(1 for char in text if ("a" <= char.lower() <= "z"))


def contains_devanagari(text: str) -> bool:
    return bool(text) and bool(re.search(r"[\u0900-\u097F]", text))


def is_hindi_dominant_text(text: str) -> bool:
    if not text:
        return False

    hindi_chars = count_hindi_chars(text)
    if hindi_chars < settings.hindi_char_threshold:
        return False

    alpha_chars = sum(1 for char in text if char.isalpha())
    if alpha_chars == 0:
        return hindi_chars > 0

    return (hindi_chars / alpha_chars) >= settings.hindi_ratio_threshold


def legacy_hindi_marker_count(text: str) -> int:
    if not text:
        return 0

    tokens = re.findall(r"[A-Za-z][A-Za-z'`-]{2,}", text)
    marker_count = 0
    marker_patterns = ("kk", "vk", "ij", "jh", "dh", "ds", "dks", "fn", "iz", "v'", "ks", "rk")

    for token in tokens:
        lowered = token.lower()
        if any(pattern in lowered for pattern in marker_patterns):
            marker_count += 1
        elif re.search(r"[A-Z]{2,}[a-z]+|[a-z]+[A-Z]{2,}", token):
            marker_count += 1

    return marker_count


def looks_like_legacy_hindi_text(text: str) -> bool:
    if not text:
        return False
    if count_hindi_chars(text) > 0:
        return False

    latin_chars = count_latin_chars(text)
    if latin_chars < 30:
        return False

    markers = legacy_hindi_marker_count(text)
    return markers >= 2


def text_quality_score(text: str) -> int:
    normalized = normalize_extracted_text(text)
    if not normalized:
        return 0

    hindi_chars = count_hindi_chars(normalized)
    latin_chars = count_latin_chars(normalized)
    digit_chars = sum(1 for char in normalized if char.isdigit())
    word_count = len(normalized.split())
    weird_chars = sum(1 for char in normalized if char in "{}[]^|~")
    legacy_penalty = legacy_hindi_marker_count(normalized) * 4 if looks_like_legacy_hindi_text(normalized) else 0

    return (hindi_chars * 4) + min(word_count, 120) + digit_chars - weird_chars - legacy_penalty - (latin_chars // 6)


def choose_best_text_candidate(extracted_text: str, ocr_text: str) -> tuple[str, str]:
    extracted = normalize_extracted_text(extracted_text)
    ocr = normalize_extracted_text(ocr_text)

    if not extracted:
        return ocr, "ocr_only" if ocr else "empty"
    if not ocr:
        return extracted, "text_only"

    extracted_score = text_quality_score(extracted)
    ocr_score = text_quality_score(ocr)
    extracted_hindi = count_hindi_chars(extracted)
    ocr_hindi = count_hindi_chars(ocr)

    if looks_like_legacy_hindi_text(extracted) and (
        ocr_hindi >= settings.ocr_min_hindi_chars or ocr_score >= extracted_score + 2
    ):
        return ocr, "ocr_replaced_legacy_text"

    if extracted_hindi == 0 and ocr_hindi >= settings.ocr_min_hindi_chars:
        return ocr, "ocr_replaced_non_unicode_text"

    if ocr_score >= extracted_score + settings.ocr_quality_margin:
        return ocr, "ocr_higher_quality"

    if extracted_score >= ocr_score + settings.ocr_quality_margin:
        return extracted, "text_higher_quality"

    if ocr_hindi > extracted_hindi:
        return normalize_extracted_text(f"{extracted}\n\n{ocr}"), "combined_preferring_ocr"

    return normalize_extracted_text(f"{extracted}\n\n{ocr}"), "combined"


def is_hindi_dominant_document(pages: List[PageExtraction]) -> bool:
    samples: List[str] = []
    legacy_pages = 0
    for page in pages:
        if page.text:
            samples.append(page.text[:1200])
            if looks_like_legacy_hindi_text(page.text):
                legacy_pages += 1
        if page.ocr_text:
            samples.append(page.ocr_text[:1200])
            if looks_like_legacy_hindi_text(page.ocr_text):
                legacy_pages += 1
        if sum(len(sample) for sample in samples) >= 6000:
            break

    combined = "\n".join(samples)
    if is_hindi_dominant_text(combined):
        return True

    total_legacy_markers = legacy_hindi_marker_count(combined)
    return legacy_pages >= 2 or total_legacy_markers >= 8


def detect_document_language(pages: List[PageExtraction], source_name: str = "") -> str:
    if contains_devanagari(source_name):
        return "hi"
    return "hi" if is_hindi_dominant_document(pages) else "en"


def build_pair_id(source: str, page: int, doc_type: str, item_index: int, content: str) -> str:
    raw = f"{source}|{page}|{doc_type}|{item_index}|{content[:160]}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:20]


def build_bilingual_documents(
    content: str,
    *,
    source: str,
    page: int,
    doc_type: str,
    item_index: int,
    translate_hindi: bool,
    document_language: str,
    extra_metadata: dict | None = None,
) -> List[Document]:
    normalized = normalize_extracted_text(content)
    if not normalized:
        return []

    original_lang = "hi" if is_hindi_dominant_text(normalized) else "en"
    pair_id = build_pair_id(source, page, doc_type, item_index, normalized)
    metadata = {
        "source": source,
        "page": page,
        "type": doc_type,
        "lang": original_lang,
        "document_lang": document_language,
        "variant": "original",
        "pair_id": pair_id,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    documents = [
        Document(
            page_content=normalized,
            metadata=metadata,
        )
    ]

    if translate_hindi and settings.enable_hindi_translation and original_lang == "hi":
        try:
            translation = normalize_extracted_text(translate_text_to_english(normalized))
        except Exception as exc:
            print(f"[Translation] Failed for page {page}: {exc}")
            translation = ""

        if translation and translation.lower() != normalized.lower():
            preview = " ".join(normalized.split())[:settings.translation_log_preview]
            print(f"[Translation] Stored bilingual pair for page {page} ({doc_type}, item {item_index}): {preview}")
            translation_metadata = dict(metadata)
            translation_metadata.update({
                "lang": "en",
                "variant": "translation",
            })
            documents.append(
                Document(
                    page_content=translation,
                    metadata=translation_metadata,
                )
            )

    return documents

def _preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    # Convert to grayscale if needed
    if img.mode != "L":
        img = img.convert("L")
    
    # Enhance contrast for better text visibility
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)  # Increase contrast by 50%
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)  # Increase sharpness by 30%
    
    # Apply slight denoising (median filter)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    return img


def _ocr_with_tesseract(img: Image.Image) -> str:
    try:
        config = "--psm 6"
        text = pytesseract.image_to_string(img, lang=settings.tesseract_lang, config=config)
        text = normalize_extracted_text(text)
        if text:
            print("[OCR] Tesseract used")
        return text
    except Exception as exc:
        print("[OCR] Tesseract error:", exc)
        return ""


def _should_use_english_only_ocr(extracted_text_hint: str, paddle_en_text: str) -> bool:
    hint = normalize_extracted_text(extracted_text_hint)
    if looks_like_legacy_hindi_text(hint) or looks_like_legacy_hindi_text(paddle_en_text):
        return False

    common_english_words = {
        "the", "and", "for", "with", "from", "this", "that", "are", "was", "were", "will",
        "shall", "should", "must", "child", "children", "education", "right", "school",
        "teacher", "student", "section", "article", "act", "page", "document", "class",
        "name", "date", "address", "government", "state", "under", "into", "between",
    }

    def looks_like_readable_english(text: str) -> bool:
        normalized = normalize_extracted_text(text).lower()
        if not normalized:
            return False
        tokens = re.findall(r"[a-z]{2,}", normalized)
        if len(tokens) < 6:
            return False
        common_hits = sum(1 for token in tokens if token in common_english_words)
        return common_hits >= 3

    english_hint = (
        bool(hint)
        and not is_hindi_dominant_text(hint)
        and count_latin_chars(hint) >= max(20, count_hindi_chars(hint) * 2)
        and looks_like_readable_english(hint)
    )
    english_ocr = (
        bool(paddle_en_text)
        and count_hindi_chars(paddle_en_text) == 0
        and count_latin_chars(paddle_en_text) >= 30
        and text_quality_score(paddle_en_text) >= 25
        and looks_like_readable_english(paddle_en_text)
    )
    return english_hint or english_ocr


def _ocr_page(
    page: fitz.Page,
    extracted_text_hint: str = "",
    scale: float = 3.0,
    use_preprocessing: bool = True,
) -> str:
    ocr_en, ocr_hi = _load_paddle_ocr()

    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    if use_preprocessing:
        img = _preprocess_image_for_ocr(img)

    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    texts = []
    english_texts = []
    hindi_texts = []

    # English OCR
    try:
        result_en = ocr_en.ocr(img_np)
        if result_en and len(result_en) > 0:
            for line in result_en[0]:
                english_texts.append(line[1][0])
        print("[OCR] PaddleOCR English used")
    except Exception as e:
        print("[OCR] English OCR error:", e)

    paddle_en_text = normalize_extracted_text("\n".join(english_texts))
    if _should_use_english_only_ocr(extracted_text_hint, paddle_en_text):
        print("[OCR] English-only OCR path selected")
        print("\n=== OCR TEXT SAMPLE ===")
        print(paddle_en_text[:500])
        print("======================\n")
        return paddle_en_text

    # Hindi OCR
    try:
        result_hi = ocr_hi.ocr(img_np)
        if result_hi and len(result_hi) > 0:
            for line in result_hi[0]:
                hindi_texts.append(line[1][0])
        print("[OCR] PaddleOCR Hindi used")
    except Exception as e:
        print("[OCR] Hindi OCR error:", e)

    paddle_hi_text = normalize_extracted_text("\n".join(hindi_texts))
    tesseract_text = _ocr_with_tesseract(img)

    candidates = [
        OCRCandidate(source="paddle_en", text=paddle_en_text),
        OCRCandidate(source="paddle_hi", text=paddle_hi_text),
        OCRCandidate(source="tesseract", text=tesseract_text),
    ]
    candidate_scores = {candidate.source: text_quality_score(candidate.text) for candidate in candidates}
    best_candidate = max(candidates, key=lambda candidate: candidate_scores[candidate.source])

    texts.extend(filter(None, [paddle_en_text, paddle_hi_text, tesseract_text]))
    combined_text = normalize_extracted_text("\n".join(texts))
    combined_score = text_quality_score(combined_text)
    if combined_score >= candidate_scores[best_candidate.source] + settings.ocr_quality_margin:
        best_candidate = OCRCandidate(source="combined", text=combined_text)
        best_score = combined_score
    else:
        best_score = candidate_scores[best_candidate.source]

    print(f"[OCR] Selected {best_candidate.source} text (score={best_score}, hindi_chars={count_hindi_chars(best_candidate.text)})")
    print("\n=== OCR TEXT SAMPLE ===")
    print(best_candidate.text[:500])
    print("======================\n")

    return best_candidate.text


def extract_pdf(path: Path) -> List[PageExtraction]:
    doc = fitz.open(path)
    pages: List[PageExtraction] = []
    
    title = extract_title_from_first_page(doc)

    for idx, page in enumerate(tqdm(doc, desc=f"Extracting {path.name}", unit="page")):
        text = normalize_extracted_text(page.get_text("text") or "")
        table_dfs = _extract_tables(page)
        tables_csv = [normalize_extracted_text(df.to_csv(index=False)) for df in table_dfs]
        ocr_text = normalize_extracted_text(_ocr_page(page, extracted_text_hint=text))

        if idx == 0 and title:
            text = normalize_extracted_text(f"Document Title: {title}\n\n{text}")

        pages.append(
            PageExtraction(
                text=text,
                tables=tables_csv,
                ocr_text=ocr_text
            )
        )

    return pages

def build_documents(pages: List[PageExtraction], source: str, document_language: str | None = None) -> List[Document]:
    text_blocks: List[ChunkSeed] = []
    docs: List[Document] = []
    translate_hindi = is_hindi_dominant_document(pages)
    document_language = document_language or detect_document_language(pages, source)
    translate_hindi = translate_hindi or document_language == "hi"
    item_index = 0

    for idx, page in enumerate(pages):
        page_text, page_text_strategy = choose_best_text_candidate(page.text, page.ocr_text)
        print(
            f"[Ingest] Page {idx + 1}: strategy={page_text_strategy}, "
            f"text_score={text_quality_score(page.text)}, ocr_score={text_quality_score(page.ocr_text)}, "
            f"text_hi={count_hindi_chars(page.text)}, ocr_hi={count_hindi_chars(page.ocr_text)}"
        )
        
        if page_text.strip():
            text_blocks.append(ChunkSeed(page=idx + 1, doc_type="text", content=page_text))

        # Table rows as dedicated documents to avoid mixing cities/rows
        for table_idx, table_csv in enumerate(page.tables):
            try:
                df = pd.read_csv(StringIO(table_csv))
            except Exception:
                # Fallback: store whole table as one doc
                docs.extend(
                    build_bilingual_documents(
                        table_csv,
                        source=source,
                        page=idx + 1,
                        doc_type="table",
                        item_index=item_index,
                        translate_hindi=translate_hindi,
                        document_language=document_language,
                        extra_metadata={"table_index": table_idx, "row_index": -1},
                    )
                )
                item_index += 1
                continue
            for row_idx, (_, row) in enumerate(df.iterrows()):
                row_text = " | ".join(str(v) for v in row.tolist())
                docs.extend(
                    build_bilingual_documents(
                        row_text,
                        source=source,
                        page=idx + 1,
                        doc_type="table_row",
                        item_index=item_index,
                        translate_hindi=translate_hindi,
                        document_language=document_language,
                        extra_metadata={"table_index": table_idx, "row_index": row_idx},
                    )
                )
                item_index += 1

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    for block in text_blocks:
        split_chunks = splitter.split_text(block.content)
        total_chunks = len(split_chunks)
        for chunk_index, chunk in enumerate(split_chunks):
            docs.extend(
                build_bilingual_documents(
                    chunk,
                    source=source,
                    page=block.page,
                    doc_type=block.doc_type,
                    item_index=item_index,
                    translate_hindi=translate_hindi,
                    document_language=document_language,
                    extra_metadata={
                        "chunk_index": chunk_index,
                        "page_chunk_count": total_chunks,
                    },
                )
            )  
            item_index += 1

    translation_docs = sum(1 for doc in docs if (doc.metadata or {}).get("variant") == "translation")
    original_docs = sum(1 for doc in docs if (doc.metadata or {}).get("variant") == "original")
    print(
        f"[Ingest] Source={source}, hindi_dominant={translate_hindi}, document_language={document_language}, "
        f"chunk_translation_enabled={settings.enable_hindi_translation}, "
        f"original_docs={original_docs}, translation_docs={translation_docs}"
    )
    return docs


def ingest_file(
    pdf_path: Path,
    store: VectorStore | None = None,
    source_name: str | None = None,
) -> Tuple[int, Path]:
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    store = store or VectorStore()
    pages = extract_pdf(pdf_path)
    document_language = detect_document_language(pages, source_name or pdf_path.name)
    docs = build_documents(
        pages,
        source=source_name or pdf_path.name,
        document_language=document_language,
    )
    store.add_documents(docs)
    store.set_active_document_language(document_language)
    print(f"[Ingest] Active document language set to {document_language}")
    store.persist()
    return len(docs), store.persist_path


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into ChromaDB")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to PDF file")
    args = parser.parse_args()
    count, path = ingest_file(args.pdf)
    print(f"Ingested {count} chunks into {path}")


if __name__ == "__main__":
    main()

