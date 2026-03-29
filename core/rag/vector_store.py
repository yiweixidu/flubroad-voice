import os
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class FluBroadRAG:
    """RAG知识库（开源核心）"""

    def __init__(self, collection_name: str, persist_directory: str):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        # 使用本地嵌入模型（BAAI/bge-large-en-v1.5 适合英文）
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cpu'},  # 可改为 'cuda' 如果有 GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None

    def build(self, articles: List[Dict]):
        """从文献构建向量库"""
        docs = []
        for art in articles:
            content = f"Title: {art['title']}\nAbstract: {art['abstract']}\nPMID: {art['pmid']}"
            metadata = {"pmid": art["pmid"], "title": art["title"], "year": art["year"]}
            docs.append(Document(page_content=content, metadata=metadata))

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()

    def load(self):
        """加载已存在的向量库"""
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def similarity_search(self, query: str, k: int = 5):
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call build() or load() first.")
        return self.vectorstore.similarity_search(query, k=k)