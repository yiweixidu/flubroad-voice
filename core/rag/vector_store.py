"""
core/rag/vector_store.py  —  fixed & complete

Key fixes vs original:
  1. build() now prefers fulltext_content over abstract when available,
     so crawled full text is actually used in the RAG index.
  2. All dict access uses .get() with safe defaults — no more KeyError
     on articles that have no abstract or pmid.
  3. build_from_fulltext() removed — build() now does the same job,
     so there is a single code path for both abstract and fulltext articles.
  4. Metadata is richer: source_type, fulltext_source, has_fulltext are stored
     on every chunk so you can filter or inspect them later.
"""

import os
from typing import List, Dict, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


class FluBroadRAG:
    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        device: str = "cpu",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = None

    # ── Build ──────────────────────────────────────────────────────────────────
    def build(self, articles: List[Dict]) -> None:
        """
        Build the Chroma vector store from a list of article dicts.

        Content priority for each article:
          1. fulltext_content  (if fulltext_available is True and content is non-empty)
          2. Title + Abstract  (fallback)

        Articles with no usable content at all are skipped with a warning.
        """
        if not articles:
            raise ValueError("No articles provided to build vector store.")

        print(f"Building vector store from {len(articles)} articles...")

        docs: List[Document] = []
        skipped = 0

        for art in articles:
            # ── Choose content ────────────────────────────────────────────────
            if art.get("fulltext_available") and art.get("fulltext_content"):
                content = art["fulltext_content"]
                source_type = "fulltext"
            else:
                title    = art.get("title") or ""
                abstract = art.get("abstract") or ""
                pmid     = art.get("pmid") or ""
                content  = f"Title: {title}\nAbstract: {abstract}\nPMID: {pmid}"
                source_type = "abstract"

            # ── Skip empty content ────────────────────────────────────────────
            if not content.strip():
                print(
                    f"  Warning: no usable content for PMID "
                    f"{art.get('pmid', 'unknown')} — skipping."
                )
                skipped += 1
                continue

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "pmid":            art.get("pmid", ""),
                        "title":           art.get("title", ""),
                        "year":            art.get("year", ""),
                        "source_type":     source_type,
                        "fulltext_source": art.get("fulltext_source") or "",
                        "has_fulltext":    bool(art.get("fulltext_available")),
                    },
                )
            )

        if not docs:
            raise ValueError("No valid documents after filtering — nothing to index.")

        if skipped:
            print(f"  Skipped {skipped} articles with empty content.")

        # ── Split ─────────────────────────────────────────────────────────────
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = splitter.split_documents(docs)
        print(f"  Split into {len(splits)} chunks. Generating embeddings...")

        # ── Embed + persist ───────────────────────────────────────────────────
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )
        self.vectorstore.persist()
        print(f"  Vector store persisted to {self.persist_directory}")

        # Summary
        ft_docs = sum(1 for d in docs if d.metadata["source_type"] == "fulltext")
        print(
            f"  Indexed: {len(docs)} articles "
            f"({ft_docs} full text, {len(docs) - ft_docs} abstract only)"
        )

    # ── Add user PDFs ──────────────────────────────────────────────────────────
    def add_user_pdf(self, pdf_chunks: List[Document]) -> None:
        """Add chunks from a user-uploaded PDF to the existing vector store."""
        if not self.vectorstore:
            try:
                self.load()
            except Exception as exc:
                raise RuntimeError(
                    "No existing vector store found. "
                    "Call build() first, then add PDFs."
                ) from exc
        self.vectorstore.add_documents(pdf_chunks)
        self.vectorstore.persist()
        print(f"  Added {len(pdf_chunks)} PDF chunks to vector store.")

    # ── Load ───────────────────────────────────────────────────────────────────
    def load(self) -> None:
        """Load an existing persisted Chroma collection from disk."""
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

    # ── Search ─────────────────────────────────────────────────────────────────
    def similarity_search(
        self, query: str, k: int = 5, return_scores: bool = False
    ):
        if not self.vectorstore:
            raise ValueError(
                "Vector store not initialised. Call build() or load() first."
            )
        if return_scores:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        return self.vectorstore.similarity_search(query, k=k)