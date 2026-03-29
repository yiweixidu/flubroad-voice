"""
PDF 文档处理模块
支持：用户上传 PDF、文本提取、分块与索引
"""
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, BinaryIO

import pdfplumber
import PyPDF2
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFProcessor:
    """PDF 处理器 - 提取文本并准备 RAG 索引 [citation:6]"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_file: BinaryIO) -> Dict:
        """
        从 PDF 文件提取文本
        使用 pdfplumber 获得更好的布局保留 [citation:6]
        """
        text_by_page = []
        full_text = []
        metadata = {}
        
        with pdfplumber.open(pdf_file) as pdf:
            # 提取元数据
            metadata = {
                "pages": len(pdf.pages),
                "metadata": pdf.metadata
            }
            
            # 逐页提取
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                text_by_page.append({
                    "page": page_num,
                    "text": page_text
                })
                full_text.append(page_text)
            
            # 尝试提取表格
            tables = []
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)
        
        return {
            "full_text": "\n\n".join(full_text),
            "text_by_page": text_by_page,
            "metadata": metadata,
            "tables": tables,
            "total_pages": metadata["pages"]
        }
    
    def process_uploaded_pdf(self, pdf_file: BinaryIO, filename: str) -> List[Document]:
        """
        处理上传的 PDF，生成 LangChain Document 对象
        用于后续 RAG 知识库构建
        """
        # 提取文本
        extracted = self.extract_text_from_pdf(pdf_file)
        
        # 创建 Document
        doc = Document(
            page_content=extracted["full_text"],
            metadata={
                "source": filename,
                "type": "user_uploaded_pdf",
                "total_pages": extracted["total_pages"],
                "has_tables": len(extracted["tables"]) > 0
            }
        )
        
        # 分块
        chunks = self.text_splitter.split_documents([doc])
        
        # 为每个块添加页面位置信息
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        return chunks
    
    def extract_tables_as_markdown(self, pdf_file: BinaryIO) -> str:
        """提取表格并转换为 Markdown 格式"""
        markdown_tables = []
        
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 0:
                        # 转换为 Markdown 表格
                        md_table = self._table_to_markdown(table)
                        markdown_tables.append(f"<!-- Page {page_num} -->\n{md_table}")
        
        return "\n\n".join(markdown_tables)
    
    def _table_to_markdown(self, table: List[List]) -> str:
        """将表格转换为 Markdown 格式"""
        if not table or len(table) == 0:
            return ""
        
        headers = [str(cell or "") for cell in table[0]]
        rows = table[1:]
        
        # 构建 Markdown 表格
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "|" + "|".join([" --- " for _ in headers]) + "|"
        body_lines = []
        
        for row in rows:
            cells = [str(cell or "") for cell in row]
            # 确保列数匹配
            while len(cells) < len(headers):
                cells.append("")
            body_lines.append("| " + " | ".join(cells) + " |")
        
        return "\n".join([header_line, separator_line] + body_lines)