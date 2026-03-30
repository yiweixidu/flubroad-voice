import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag.vector_store import FluBroadRAG
from core.narrative.generator import NarrativeGenerator   # 复用已有的生成器

def main():
    # 加载向量库
    rag = FluBroadRAG(
        collection_name="flu_bnabs_full",
        persist_directory="./data/vector_db"
    )
    rag.load()
    
    # 初始化 LLM（Ollama）
    llm = NarrativeGenerator(model="llama3.2:3b", temperature=0.1, llm_type="ollama")
    
    # 检索增强的问答
    question = "What are the most promising broadly neutralizing antibodies for influenza?"
    retrieved_docs = rag.similarity_search(question, k=5)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = f"""Answer the following question based only on the provided context.
If the answer is not in the context, say "I don't have enough information".

Context:
{context}

Question: {question}

Answer:"""
    
    from langchain_core.messages import HumanMessage
    response = llm.llm.invoke([HumanMessage(content=prompt)])

    print(f"Question: {question}\n")
    print(f"Answer: {response}")
    
    # 打印引用来源
    print("\nReferences:")
    for i, doc in enumerate(retrieved_docs):
        print(f"{i+1}. PMID {doc.metadata.get('pmid')}: {doc.metadata.get('title')}")

if __name__ == "__main__":
    main()