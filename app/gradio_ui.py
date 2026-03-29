import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import json
from dotenv import load_dotenv
from app.orchestrator import FluBroadOrchestrator
from core.rag.vector_store import FluBroadRAG
from core.narrative.generator import NarrativeGenerator

load_dotenv()

config = {
    "email": "yiweixidu@gmail.com",   # 或 os.getenv("EMAIL")
    "pubmed_api_key": "0c2a57b91a4503e808d45c2619b978b36808",  # 或 os.getenv("PUBMED_API_KEY")
    "collection_name": os.getenv("COLLECTION_NAME", "flu_bnabs"),
    "persist_dir": os.getenv("PERSIST_DIR", "./data/vector_db"),
    "ppt_template": os.getenv("PPT_TEMPLATE", "./templates/lab_template.pptx"),
    "output_dir": os.getenv("OUTPUT_DIR", "./output"),
    "llm_type": os.getenv("LLM_TYPE", "ollama"),
    "llm_model": os.getenv("LLM_MODEL", "llama3.2:3b"),
    "llm_temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
}

orchestrator = FluBroadOrchestrator(config)

# 预加载已爬取的文献（543篇）
with open("data/flu_bnabs_all_articles.json", "r") as f:
    all_articles = json.load(f)
print(f"Loaded {len(all_articles)} articles from local JSON")

def generate_report(query, max_papers):
    # 此处使用本地 JSON 数据，忽略 query 和 max_papers，或者根据 query 过滤
    # 简单起见，直接使用全部文献的前 max_papers 篇
    articles = all_articles[:max_papers]
    result = orchestrator.run_from_articles(articles)
    review_text = result["review"]
    antibodies = result["antibodies"]
    ppt_file = result["ppt_file"]
    video_file = result["video_file"]
    
    if antibodies:
        import pandas as pd
        df = pd.DataFrame(antibodies)
        table_html = df.to_html(classes="table table-striped")
    else:
        table_html = "No antibodies extracted."
    return review_text, table_html, ppt_file, video_file

# 初始化 RAG 和 LLM 用于问答（全局）
rag = FluBroadRAG(collection_name="flu_bnabs_full", persist_directory="./data/vector_db")
rag.load()
llm = NarrativeGenerator(model="llama3.2:3b", temperature=0.1, llm_type="ollama")

def answer_question(question, history):
    docs = rag.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Based on the following context, answer the question.\nContext: {context}\nQuestion: {question}\nAnswer:"
    response = llm.llm.invoke(prompt)
    # 提取文本内容
    answer = response.content if hasattr(response, 'content') else str(response)
    refs = "\n\n**References:**\n" + "\n".join([f"- {d.metadata.get('title', 'N/A')} (PMID: {d.metadata.get('pmid', 'N/A')})" for d in docs])
    return answer + refs

# 创建带 Tab 的界面
with gr.Blocks(title="FluBroad-Voice AI Presentation Agent") as demo:
    gr.Markdown("# FluBroad-Voice")
    gr.Markdown("AI agent for broadly neutralizing antibody research")
    
    with gr.Tabs():
        with gr.TabItem("Generate Report"):
            with gr.Row():
                with gr.Column(scale=4):
                    query_input = gr.Textbox(label="Search Query (not used with local data)", value="broadly neutralizing antibody influenza", lines=2)
                    max_papers = gr.Slider(label="Max Papers (from local 543 papers)", minimum=5, maximum=543, value=20, step=5)
                    submit_btn = gr.Button("Generate Presentation")
                with gr.Column(scale=1):
                    gr.Markdown("### About")
                    gr.Markdown("This agent uses 543 pre-fetched PubMed papers to generate a review, extract antibodies, and produce a PPT.")
            with gr.Row():
                with gr.Column():
                    review_output = gr.Textbox(label="Generated Review", lines=20, interactive=False)
                with gr.Column():
                    antibodies_output = gr.HTML(label="Extracted Antibodies")
            with gr.Row():
                ppt_download = gr.File(label="Download PPT", interactive=False)
                video_download = gr.File(label="Download Video", interactive=False)
            submit_btn.click(
                fn=generate_report,
                inputs=[query_input, max_papers],
                outputs=[review_output, antibodies_output, ppt_download, video_download]
            )
        
        with gr.TabItem("RAG Q&A"):
            gr.ChatInterface(fn=answer_question, title="Ask questions about the literature")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)