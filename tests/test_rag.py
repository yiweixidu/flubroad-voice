import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from core.retrieval.pubmed import PubMedFetcher
from core.rag.vector_store import FluBroadRAG

load_dotenv()

def test_rag():
    # 1. 检索文献
    email = os.getenv("EMAIL")
    fetcher = PubMedFetcher(email=email)
    pmids = fetcher.search("broadly neutralizing antibody influenza", max_results=5)
    articles = fetcher.fetch_details(pmids)
    print(f"Retrieved {len(articles)} articles")

    # 2. 构建知识库
    rag = FluBroadRAG(
        collection_name="test_flu",
        persist_directory="./data/test_vector_db"
    )
    rag.build(articles)
    print("Vector store built and persisted.")

    # 3. 相似性搜索测试
    query = "What are the key antibodies against influenza HA stem?"
    results = rag.similarity_search(query, k=2)
    for doc in results:
        print(f"\n--- Result ---\n{doc.page_content[:300]}...\nMetadata: {doc.metadata}")

if __name__ == "__main__":
    test_rag()