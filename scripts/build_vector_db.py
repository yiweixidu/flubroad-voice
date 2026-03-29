import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag.vector_store import FluBroadRAG

def main():
    # 加载文献 JSON
    with open("data/flu_bnabs_all_articles.json", "r") as f:
        articles = json.load(f)
    print(f"Loaded {len(articles)} articles")

    # 初始化 RAG 实例（使用本地嵌入模型）
    rag = FluBroadRAG(
        collection_name="flu_bnabs_full",
        persist_directory="./data/vector_db",
        embedding_model="BAAI/bge-large-en-v1.5"   # 或其他本地模型
    )
    
    # 构建向量库
    rag.build(articles)
    print("Vector database built successfully.")

if __name__ == "__main__":
    main()