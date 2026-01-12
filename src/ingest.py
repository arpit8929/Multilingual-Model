import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
import pandas as pd
import pytesseract
from PIL import Image
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


def _ocr_page(page: fitz.Page, scale: float = 2.0) -> str:
    """OCR a page image as a fallback for scanned PDFs."""
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img, lang=settings.ocr_lang)


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
        # Plain text + OCR combined
        combined = [page.text]
        if page.ocr_text:
            combined.append(page.ocr_text)
        page_text = "\n\n".join(filter(None, combined))
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

