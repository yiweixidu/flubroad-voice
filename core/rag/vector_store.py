import os
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

class FluBroadRAG:
    def __init__(self, collection_name: str, persist_directory: str,
                 embedding_model: str = "BAAI/bge-large-en-v1.5"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None

    def build(self, articles: List[Dict]):
        print(f"Building vector store from {len(articles)} articles...")
        docs = []
        for art in articles:
            content = f"Title: {art['title']}\nAbstract: {art['abstract']}\nPMID: {art['pmid']}"
            metadata = {"pmid": art["pmid"], "title": art["title"], "year": art["year"]}
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

    def load(self):
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def similarity_search(self, query: str, k: int = 5):
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call build() or load() first.")
        return self.vectorstore.similarity_search(query, k=k)