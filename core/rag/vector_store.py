import os
from typing import List, Dict, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

class FluBroadRAG:
    def __init__(self, collection_name: str, persist_directory: str,
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 device: str = "cpu"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None

    def build(self, articles: List[Dict]):
        if not articles:
            raise ValueError("No articles provided to build vector store.")
        print(f"Building vector store from {len(articles)} articles...")
        docs = []
        for art in articles:
            content = f"Title: {art['title']}\nAbstract: {art['abstract']}\nPMID: {art['pmid']}"
            metadata = {"pmid": art["pmid"], "title": art["title"], "year": art.get("year", "")}
            docs.append(Document(page_content=content, metadata=metadata))

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        print(f"Split into {len(splits)} chunks. Generating embeddings...")

        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()
        print(f"Vector store built and persisted to {self.persist_directory}")

    def build_from_fulltext(self, articles: List[Dict]) -> None:
        if not articles:
            raise ValueError("No articles provided to build vector store.")
        docs = []
        for art in articles:
            # 优先使用全文，其次使用摘要
            if art.get("fulltext_available") and art.get("fulltext_content"):
                content = art["fulltext_content"]
                source_type = "fulltext"
            else:
                content = f"Title: {art['title']}\nAbstract: {art['abstract']}"
                source_type = "abstract"
            
            # 跳过空内容
            if not content or not content.strip():
                print(f"Warning: Empty content for article {art.get('pmid', 'unknown')}. Skipping.")
                continue

            doc = Document(
                page_content=content,
                metadata={
                    "pmid": art.get("pmid", ""),
                    "title": art.get("title", ""),
                    "year": art.get("year", ""),
                    "source_type": source_type,
                    "fulltext_source": art.get("fulltext_source"),
                    "has_fulltext": art.get("fulltext_available", False)
                }
            )
            docs.append(doc)
        
        if not docs:
            raise ValueError("No valid documents after filtering.")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()
    
    def add_user_pdf(self, pdf_chunks: List[Document]) -> None:
        if not self.vectorstore:
            try:
                self.load()
            except Exception as e:
                raise RuntimeError(
                    "No existing vectorstore found. Please call build() or build_from_fulltext() first."
                ) from e
        self.vectorstore.add_documents(pdf_chunks)
        self.vectorstore.persist()

    def load(self):
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def similarity_search(self, query: str, k: int = 5, return_scores: bool = False):
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call build() or load() first.")
        if return_scores:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        return self.vectorstore.similarity_search(query, k=k)