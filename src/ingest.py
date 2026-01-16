import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.config import settings
from src.vector_store import VectorStore


@dataclass
class PageExtraction:
    text: str
    tables: List[str]
    ocr_text: str


def _extract_tables(page: fitz.Page) -> List[pd.DataFrame]:
    """Extract tables using PyMuPDF structured detection as DataFrames."""
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


def _preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """Preprocess image for better OCR, especially for handwritten text."""
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


def _ocr_page(page: fitz.Page, scale: float = 3.0, use_preprocessing: bool = True) -> str:
    """OCR a page image with enhanced preprocessing for handwritten/scanned PDFs.
    
    Args:
        page: PyMuPDF page object
        scale: Resolution scale (higher = better quality, slower). Use 3.0+ for handwritten text.
        use_preprocessing: Apply image enhancement for better OCR accuracy
    """
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    if use_preprocessing:
        img = _preprocess_image_for_ocr(img)
    
    # Use Tesseract with Hindi+English and better config for handwritten text
    custom_config = r'--oem 3 --psm 6'  # OCR Engine Mode 3, Page Segmentation Mode 6 (uniform block)
    return pytesseract.image_to_string(img, lang=settings.ocr_lang, config=custom_config)


def extract_pdf(path: Path) -> List[PageExtraction]:
    doc = fitz.open(path)
    pages: List[PageExtraction] = []
    for page in tqdm(doc, desc=f"Extracting {path.name}", unit="page"):
        text = page.get_text("text") or ""
        table_dfs = _extract_tables(page)
        tables_csv = [df.to_csv(index=False) for df in table_dfs]
        ocr_text = _ocr_page(page)
        pages.append(PageExtraction(text=text, tables=tables_csv, ocr_text=ocr_text))
    return pages


def build_documents(pages: List[PageExtraction], source: str) -> List[Document]:
    text_blocks: List[str] = []
    docs: List[Document] = []

    for idx, page in enumerate(pages):
        # For scanned/handwritten PDFs, prioritize OCR text
        # If extracted text is very short (likely empty from scanned PDF), use only OCR
        extracted_text_len = len(page.text.strip())
        ocr_text_len = len(page.ocr_text.strip()) if page.ocr_text else 0
        
        if extracted_text_len < 50 and ocr_text_len > 0:
            # Likely a scanned PDF - use OCR text primarily
            page_text = page.ocr_text
        elif page.text.strip() and page.ocr_text:
            # Both available - combine them
            page_text = f"{page.text}\n\n{page.ocr_text}"
        elif page.text.strip():
            # Only extracted text available
            page_text = page.text
        elif page.ocr_text:
            # Only OCR text available
            page_text = page.ocr_text
        else:
            # Empty page
            page_text = ""
        
        if page_text.strip():
            text_blocks.append(page_text)

        # Table rows as dedicated documents to avoid mixing cities/rows
        for table_csv in page.tables:
            try:
                df = pd.read_csv(pd.compat.StringIO(table_csv))
            except Exception:
                # Fallback: store whole table as one doc
                docs.append(
                    Document(
                        page_content=table_csv,
                        metadata={"source": source, "page": idx + 1, "type": "table"},
                    )
                )
                continue
            for _, row in df.iterrows():
                row_text = " | ".join(str(v) for v in row.tolist())
                docs.append(
                    Document(
                        page_content=row_text,
                        metadata={"source": source, "page": idx + 1, "type": "table_row"},
                    )
                )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    for i, block in enumerate(text_blocks):
        for chunk in splitter.split_text(block):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source": source, "page": i + 1, "type": "text"},
                )
            )
    return docs


def ingest_file(pdf_path: Path, store: VectorStore | None = None) -> Tuple[int, Path]:
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    store = store or VectorStore()
    pages = extract_pdf(pdf_path)
    docs = build_documents(pages, source=str(pdf_path))
    store.add_documents(docs)
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

