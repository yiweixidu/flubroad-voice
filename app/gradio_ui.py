import gradio as gr
import os
from dotenv import load_dotenv
from app.orchestrator import FluBroadOrchestrator
import tempfile

load_dotenv()

# 读取配置
config = {
    "email": os.getenv("EMAIL"),
    "pubmed_api_key": os.getenv("PUBMED_API_KEY"),
    "collection_name": os.getenv("COLLECTION_NAME", "flu_bnabs"),
    "persist_dir": os.getenv("PERSIST_DIR", "./data/vector_db"),
    "ppt_template": os.getenv("PPT_TEMPLATE", "./templates/lab_template.pptx"),
    "output_dir": os.getenv("OUTPUT_DIR", "./output"),
    "llm_type": os.getenv("LLM_TYPE", "ollama"),
    "llm_model": os.getenv("LLM_MODEL", "llama3.2:3b"),
    "llm_temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
}

orchestrator = FluBroadOrchestrator(config)

def generate_report(query, max_papers):
    # 运行编排器
    result = orchestrator.run(query, max_papers)
    # 返回结果
    review_text = result["review"]
    antibodies = result["antibodies"]
    ppt_file = result["ppt_file"]
    video_file = result["video_file"]

    # 构建抗体表格HTML
    if antibodies:
        import pandas as pd
        df = pd.DataFrame(antibodies)
        table_html = df.to_html(classes="table table-striped")
    else:
        table_html = "No antibodies extracted."

    return review_text, table_html, ppt_file, video_file

with gr.Blocks(title="FluBroad-Voice AI Presentation Agent") as demo:
    gr.Markdown("# FluBroad-Voice")
    gr.Markdown("AI agent for broadly neutralizing antibody research")
    
    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(label="Search Query", value="broadly neutralizing antibody influenza", lines=2)
            max_papers = gr.Slider(label="Max Papers", minimum=5, maximum=100, value=20, step=5)
            submit_btn = gr.Button("Generate Presentation")
        with gr.Column(scale=1):
            # 可以放一些说明
            gr.Markdown("### About")
            gr.Markdown("This agent retrieves papers, builds a knowledge base, generates a review, and produces a narrated presentation.")

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

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)