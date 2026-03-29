import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag.vector_store import FluBroadRAG

def main():
    # 加载已有向量库
    rag = FluBroadRAG(
        collection_name="flu_bnabs_full",
        persist_directory="./data/vector_db"
    )
    rag.load()
    
    # 示例查询
    queries = [
        "broadly neutralizing antibodies targeting HA stem",
        "cross-reactive antibodies against H1N1 and H3N2",
        "clinical trials of bnAbs for influenza"
    ]
    
    for q in queries:
        print(f"\n=== Query: {q} ===")
        results = rag.similarity_search(q, k=3)
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Title: {doc.metadata.get('title', 'N/A')}")
            print(f"PMID: {doc.metadata.get('pmid', 'N/A')}")
            print(f"Snippet: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main()