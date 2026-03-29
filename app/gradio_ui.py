import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import json
import requests
from dotenv import load_dotenv
from app.orchestrator import FluBroadOrchestrator
from core.rag.vector_store import FluBroadRAG
from core.narrative.generator import NarrativeGenerator
from core.retrieval.pdf_processor import PDFProcessor   # your existing PDF processor

load_dotenv()

config = {
    "email": "yiweixidu@gmail.com",
    "pubmed_api_key": "0c2a57b91a4503e808d45c2619b978b36808",
    "collection_name": os.getenv("COLLECTION_NAME", "flu_bnabs"),
    "persist_dir": os.getenv("PERSIST_DIR", "./data/vector_db"),
    "ppt_template": os.getenv("PPT_TEMPLATE", "./templates/lab_template.pptx"),
    "output_dir": os.getenv("OUTPUT_DIR", "./output"),
    "llm_type": os.getenv("LLM_TYPE", "ollama"),
    "llm_model": os.getenv("LLM_MODEL", "llama3.2:3b"),
    "llm_temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
}

orchestrator = FluBroadOrchestrator(config)

# Pre‑load cached articles (543 papers)
with open("data/flu_bnabs_all_articles.json", "r") as f:
    all_articles = json.load(f)
print(f"Loaded {len(all_articles)} articles from local JSON")

# ---------- Report generation ----------
def generate_report(query, max_papers):
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

# ---------- RAG Q&A ----------
rag = FluBroadRAG(collection_name="flu_bnabs_full", persist_directory="./data/vector_db")
rag.load()
llm = NarrativeGenerator(model="llama3.2:3b", temperature=0.1, llm_type="ollama")

def answer_question(question, history):
    docs = rag.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Based on the following context, answer the question.\nContext: {context}\nQuestion: {question}\nAnswer:"
    response = llm.llm.invoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)
    refs = "\n\n**References:**\n" + "\n".join([f"- {d.metadata.get('title', 'N/A')} (PMID: {d.metadata.get('pmid', 'N/A')})" for d in docs])
    return answer + refs

# ---------- PDF upload handling ----------
def process_uploaded_pdfs(files):
    """Process uploaded PDFs and add them to the vector store."""
    if not files:
        return "No files uploaded."

    pdf_processor = PDFProcessor()
    # Use the same RAG instance that powers Q&A
    rag_for_pdf = FluBroadRAG(
        collection_name="flu_bnabs_full",
        persist_directory="./data/vector_db"
    )
    try:
        rag_for_pdf.load()
    except Exception as e:
        return f"Error loading existing knowledge base: {str(e)}. Please run a literature search first."

    all_chunks = []
    for file in files:
        # Gradio provides file objects with a `.name` attribute
        with open(file.name, 'rb') as f:
            chunks = pdf_processor.process_uploaded_pdf(f, file.name)
            all_chunks.extend(chunks)

    if not all_chunks:
        return "No text could be extracted from the uploaded PDF(s)."

    rag_for_pdf.add_user_pdf(all_chunks)
    return f"Successfully processed {len(files)} PDF(s), added {len(all_chunks)} chunks to knowledge base."

# ---------- Unpaywall OA lookup ----------
def check_open_access(doi_string):
    """Check open access status for one or more DOIs using Unpaywall."""
    if not doi_string:
        return [["No DOI provided", False, "", ""]]
    dois = [d.strip() for d in doi_string.split(",") if d.strip()]
    results = []
    for doi in dois:
        url = f"https://api.unpaywall.org/v2/{doi}?email=yiweixidu@gmail.com"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                is_oa = data.get("is_oa", False)
                best_oa = data.get("best_oa_location", {})
                pdf_url = best_oa.get("url_for_pdf") or best_oa.get("url") or ""
                license_val = best_oa.get("license") or ""
                results.append([doi, is_oa, pdf_url, license_val])
            else:
                results.append([doi, False, f"API error {resp.status_code}", ""])
        except Exception as e:
            results.append([doi, False, str(e), ""])
    return results

# ---------- Gradio UI ----------
with gr.Blocks(title="FluBroad-Voice AI Presentation Agent") as demo:
    gr.Markdown("# FluBroad-Voice")
    gr.Markdown("AI agent for broadly neutralizing antibody research")

    with gr.Tabs():
        # Tab 1: Generate Report
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

        # Tab 2: RAG Q&A
        with gr.TabItem("RAG Q&A"):
            gr.ChatInterface(fn=answer_question, title="Ask questions about the literature")

        # Tab 3: Upload full‑text PDFs
        with gr.TabItem("📄 Upload PDFs"):
            gr.Markdown("### Upload PDF Full‑Text Articles")
            gr.Markdown("Add your own PDFs to the knowledge base. Extracted text will be used in Q&A and future reports.")
            pdf_upload = gr.File(
                label="Select PDF files",
                file_types=[".pdf"],
                file_count="multiple"
            )
            with gr.Row():
                upload_btn = gr.Button("Process & Add to Knowledge Base")
                clear_btn = gr.Button("Clear")
            upload_status = gr.Textbox(label="Processing Status", lines=3)
            upload_btn.click(
                fn=process_uploaded_pdfs,
                inputs=[pdf_upload],
                outputs=[upload_status]
            )
            clear_btn.click(lambda: (None, ""), inputs=[], outputs=[pdf_upload, upload_status])

        # Tab 4: Unpaywall OA lookup
        with gr.TabItem("🔓 Open Access Lookup"):
            gr.Markdown("### Check Open Access Status via Unpaywall")
            oa_doi_input = gr.Textbox(label="DOI(s) (comma separated)")
            oa_search_btn = gr.Button("Check Status")
            oa_results = gr.Dataframe(
                label="Open Access Information",
                headers=["DOI", "Is OA", "PDF URL", "License"]
            )
            oa_search_btn.click(
                fn=check_open_access,
                inputs=[oa_doi_input],
                outputs=[oa_results]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)