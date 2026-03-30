"""
core/retrieval/pdf_processor.py
Fix: removed unused PyPDF2 import (pdfplumber handles everything).
"""

from pathlib import Path
from typing import List, Dict, Optional, BinaryIO

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFProcessor:
    """Extract text from PDFs and prepare LangChain Documents for RAG indexing."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    # ── Text extraction ───────────────────────────────────────────────────────
    def extract_text_from_pdf(self, pdf_file: BinaryIO) -> Dict:
        """Extract full text, per-page text, and tables from a PDF file object."""
        text_by_page = []
        full_text_parts = []
        tables = []
        metadata = {}

        with pdfplumber.open(pdf_file) as pdf:
            metadata = {
                "pages":    len(pdf.pages),
                "metadata": pdf.metadata or {},
            }
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                text_by_page.append({"page": page_num, "text": page_text})
                full_text_parts.append(page_text)
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)

        return {
            "full_text":   "\n\n".join(full_text_parts),
            "text_by_page": text_by_page,
            "metadata":    metadata,
            "tables":      tables,
            "total_pages": metadata["pages"],
        }

    # ── Process for RAG ───────────────────────────────────────────────────────
    def process_uploaded_pdf(
        self, pdf_file: BinaryIO, filename: str
    ) -> List[Document]:
        """Extract text, split into chunks, return LangChain Documents."""
        extracted = self.extract_text_from_pdf(pdf_file)
        doc = Document(
            page_content=extracted["full_text"],
            metadata={
                "source":      filename,
                "type":        "user_uploaded_pdf",
                "total_pages": extracted["total_pages"],
                "has_tables":  len(extracted["tables"]) > 0,
            },
        )
        chunks = self.text_splitter.split_documents([doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"]   = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        return chunks

    # ── Table extraction ──────────────────────────────────────────────────────
    def extract_tables_as_markdown(self, pdf_file: BinaryIO) -> str:
        """Extract all tables from a PDF and return them as Markdown."""
        markdown_tables = []
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                for table in page.extract_tables() or []:
                    if table:
                        md = self._table_to_markdown(table)
                        markdown_tables.append(f"<!-- Page {page_num} -->\n{md}")
        return "\n\n".join(markdown_tables)

    @staticmethod
    def _table_to_markdown(table: List[List]) -> str:
        if not table:
            return ""
        headers = [str(cell or "") for cell in table[0]]
        separator = "|" + "|".join(" --- " for _ in headers) + "|"
        header_line = "| " + " | ".join(headers) + " |"
        body_lines = []
        for row in table[1:]:
            cells = [str(cell or "") for cell in row]
            while len(cells) < len(headers):
                cells.append("")
            body_lines.append("| " + " | ".join(cells) + " |")
        return "\n".join([header_line, separator] + body_lines)