"""
app/gradio_ui.py  —  final version
Tabs:
  1. Generate Report  — use local cache to generate review + PPT
  2. Fetch Literature — crawl PubMed, enrich full text, update local cache
  3. RAG Q&A          — ask questions about the literature
  4. Upload PDFs      — add user PDFs to the knowledge base
  5. Open Access      — Unpaywall DOI lookup
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import threading
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from app.orchestrator import FluBroadOrchestrator
from core.rag.vector_store import FluBroadRAG
from core.narrative.generator import NarrativeGenerator
from core.retrieval.pdf_processor import PDFProcessor
from core.retrieval.pmc_fulltext import UnpaywallFetcher

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
config = {
    "email":           os.getenv("EMAIL", "yiweixidu@gmail.com"),
    "pubmed_api_key":  os.getenv("PUBMED_API_KEY"),
    "collection_name": os.getenv("COLLECTION_NAME", "flu_bnabs"),
    "persist_dir":     os.getenv("PERSIST_DIR", "./data/vector_db"),
    "ppt_template":    os.getenv("PPT_TEMPLATE", "./templates/lab_template.pptx"),
    "output_dir":      os.getenv("OUTPUT_DIR", "./output"),
    "llm_type":        os.getenv("LLM_TYPE",  "openai"),
    "llm_model":       os.getenv("LLM_MODEL", "gpt-4o-mini"),
    "llm_temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
    "pmc_delay":       0.5,
    "unpaywall_delay": 0.5,
}

CACHE_FILE = os.getenv("CACHE_FILE", "data/flu_bnabs_all_articles.json")

orchestrator = FluBroadOrchestrator(config)

# Load cached articles at startup
def _load_cache() -> list:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            articles = json.load(f)
        ft = sum(1 for a in articles if a.get("fulltext_available"))
        print(f"Loaded {len(articles)} articles ({ft} with full text) from {CACHE_FILE}")
        return articles
    print(f"No cache found at {CACHE_FILE}. Use 'Fetch Literature' tab to crawl.")
    return []

all_articles = _load_cache()


# ── Tab 1: Generate Report ────────────────────────────────────────────────────
def generate_report(max_papers: int):
    articles = all_articles[:int(max_papers)]
    if not articles:
        return (
            "No articles loaded. Please use the 'Fetch Literature' tab first.",
            "<p>No data.</p>", None, None, "No articles."
        )
    result = orchestrator.run_from_articles(articles)
    review_text = result["review"]
    antibodies  = result["antibodies"]
    ppt_file    = result["ppt_file"]
    stats       = result.get("stats", {})

    if antibodies:
        import pandas as pd
        table_html = pd.DataFrame(antibodies).to_html(
            classes="table table-striped", index=False
        )
    else:
        table_html = "<p>No antibodies extracted.</p>"

    stats_str = (
        f"Papers used: {stats.get('after_filter','?')} / {stats.get('total_input','?')} | "
        f"Full text: {stats.get('fulltext_count','?')} | "
        f"Antibodies: {stats.get('antibodies_found','?')}"
    )
    return review_text, table_html, ppt_file, None, stats_str


# ── Tab 2: Fetch Literature ───────────────────────────────────────────────────
_crawl_log: list = []
_crawl_lock = threading.Lock()

def fetch_literature(query: str, max_papers: int, fulltext: bool):
    """
    Crawl PubMed, optionally enrich with full text, save cache.
    Runs in the Gradio thread; yields log lines for the UI textbox.
    """
    global all_articles
    log_lines = []

    def log(msg: str):
        log_lines.append(msg)
        return "\n".join(log_lines)

    yield log(f"Starting crawl: {query!r}, max={max_papers}, full_text={fulltext}")

    try:
        if fulltext:
            # Full pipeline: PubMed + PMC + Unpaywall
            progress_msgs = []
            def progress_cb(current, total, pmid):
                progress_msgs.append(
                    f"  [{current}/{total}] PMID {pmid}"
                )
            articles = orchestrator.lit_fetcher.fetch_and_cache(
                query=query,
                max_papers=int(max_papers),
                cache_file=CACHE_FILE,
                progress_callback=progress_cb,
            )
            for msg in progress_msgs[-10:]:   # show last 10
                log_lines.append(msg)
        else:
            # Abstracts only — faster
            from core.retrieval.pubmed import PubMedFetcher
            pubmed = PubMedFetcher(
                email=config["email"],
                api_key=config["pubmed_api_key"],
            )
            yield log("Searching PubMed...")
            pmids = pubmed.search(query, max_results=int(max_papers))
            yield log(f"Found {len(pmids)} PMIDs. Fetching abstracts...")
            articles = pubmed.fetch_details(pmids)
            cache_dir = os.path.dirname(CACHE_FILE) or "."
            os.makedirs(cache_dir, exist_ok=True)
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)

        ft = sum(1 for a in articles if a.get("fulltext_available"))
        all_articles = articles
        yield log(
            f"\nDone!\n"
            f"  Total articles : {len(articles)}\n"
            f"  With full text : {ft}\n"
            f"  Abstract only  : {len(articles) - ft}\n"
            f"  Saved to       : {CACHE_FILE}\n\n"
            f"Switch to 'Generate Report' tab to build the review."
        )
    except Exception as e:
        yield log(f"\nERROR: {e}")


# ── Tab 3: RAG Q&A ────────────────────────────────────────────────────────────
rag = FluBroadRAG(
    collection_name="flu_bnabs_full",
    persist_directory="./data/vector_db",
)
try:
    rag.load()
except Exception:
    pass  # will be built on first generate

llm = NarrativeGenerator(
    model=config["llm_model"],
    temperature=config["llm_temperature"],
    llm_type=config["llm_type"],
)

def answer_question(question, history):
    docs = rag.similarity_search(question, k=4)
    context = "\n\n".join(d.page_content for d in docs)
    messages = [
        SystemMessage(content="You are a helpful virology research assistant."),
        HumanMessage(
            content=(
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\nAnswer:"
            )
        ),
    ]
    response = llm.llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)
    refs = "\n\n**References:**\n" + "\n".join(
        f"- {d.metadata.get('title','N/A')} (PMID: {d.metadata.get('pmid','N/A')})"
        for d in docs
    )
    return answer + refs


# ── Tab 4: Upload PDFs ────────────────────────────────────────────────────────
def process_uploaded_pdfs(files):
    if not files:
        return "No files uploaded."
    pdf_processor = PDFProcessor()
    rag_for_pdf = FluBroadRAG(
        collection_name="flu_bnabs_full",
        persist_directory="./data/vector_db",
    )
    try:
        rag_for_pdf.load()
    except Exception as e:
        return f"Error loading knowledge base: {e}. Run 'Generate Report' first."
    all_chunks = []
    for file in files:
        with open(file.name, "rb") as f:
            chunks = pdf_processor.process_uploaded_pdf(f, file.name)
            all_chunks.extend(chunks)
    if not all_chunks:
        return "No text extracted from PDFs."
    rag_for_pdf.add_user_pdf(all_chunks)
    return (
        f"Processed {len(files)} PDF(s). "
        f"Added {len(all_chunks)} chunks to knowledge base."
    )


# ── Tab 5: Open Access Lookup ─────────────────────────────────────────────────
_unpaywall = UnpaywallFetcher(email=config["email"])

def check_open_access(doi_string: str):
    if not doi_string.strip():
        return [["No DOI provided", False, "", ""]]
    dois = [d.strip() for d in doi_string.split(",") if d.strip()]
    rows = []
    for doi in dois:
        try:
            data = _unpaywall.get_oa_info(doi)
            if not data:
                rows.append([doi, False, "Not found", ""])
                continue
            is_oa   = data.get("is_oa", False)
            best    = data.get("best_oa_location") or {}
            pdf_url = best.get("url_for_pdf") or best.get("url") or ""
            lic     = best.get("license") or ""
            rows.append([doi, is_oa, pdf_url, lic])
        except Exception as e:
            rows.append([doi, False, str(e), ""])
    return rows


# ── Gradio UI ─────────────────────────────────────────────────────────────────
cache_info = (
    f"{len(all_articles)} articles loaded"
    if all_articles else "No cache — use Fetch Literature tab"
)

with gr.Blocks(title="FluBroad-Voice") as demo:
    gr.Markdown("# FluBroad-Voice")
    gr.Markdown(
        f"AI agent for broadly neutralizing antibody research · "
        f"**{config['llm_model']}** · Cache: *{cache_info}*"
    )

    with gr.Tabs():

        # ── Tab 1: Generate Report ────────────────────────────────────────────
        with gr.TabItem("Generate Report"):
            gr.Markdown(
                "Generates a 6-section PMRC review from the local article cache, "
                "extracts antibodies, and builds a PPT."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    max_papers = gr.Slider(
                        label="Articles to use from cache",
                        minimum=10, maximum=max(543, len(all_articles)),
                        value=min(200, len(all_articles) or 200), step=10,
                    )
                    gen_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=1):
                    stats_box = gr.Textbox(label="Run stats", interactive=False)

            with gr.Row():
                review_box = gr.Textbox(
                    label="Generated Review", lines=25, interactive=False
                )
                ab_html = gr.HTML(label="Extracted Antibodies")

            with gr.Row():
                ppt_dl   = gr.File(label="Download PPT",   interactive=False)
                video_dl = gr.File(label="Download Video", interactive=False)

            gen_btn.click(
                fn=generate_report,
                inputs=[max_papers],
                outputs=[review_box, ab_html, ppt_dl, video_dl, stats_box],
            )

        # ── Tab 2: Fetch Literature ───────────────────────────────────────────
        with gr.TabItem("Fetch Literature"):
            gr.Markdown(
                "Crawl PubMed with the query below, optionally enrich with PMC "
                "full-text XML and Unpaywall OA PDFs, then save to local cache. "
                "**This overwrites the existing cache.**"
            )
            with gr.Row():
                fetch_query = gr.Textbox(
                    label="PubMed query",
                    value=(
                        "(broadly neutralizing antibody OR bnab OR broadly reactive antibody) "
                        "AND (influenza OR hemagglutinin OR neuraminidase)"
                    ),
                    lines=3,
                )
            with gr.Row():
                fetch_max = gr.Slider(
                    label="Max papers", minimum=10, maximum=2000,
                    value=600, step=50,
                )
                fetch_fulltext = gr.Checkbox(
                    label="Fetch full text (PMC + Unpaywall PDF) — slower but richer",
                    value=True,
                )
            fetch_btn = gr.Button("Start Crawl", variant="primary")
            fetch_log = gr.Textbox(
                label="Crawl log", lines=20, interactive=False
            )
            fetch_btn.click(
                fn=fetch_literature,
                inputs=[fetch_query, fetch_max, fetch_fulltext],
                outputs=[fetch_log],
            )

        # ── Tab 3: RAG Q&A ────────────────────────────────────────────────────
        with gr.TabItem("RAG Q&A"):
            gr.ChatInterface(
                fn=answer_question,
                title="Ask questions about the literature",
            )

        # ── Tab 4: Upload PDFs ────────────────────────────────────────────────
        with gr.TabItem("Upload PDFs"):
            gr.Markdown(
                "Add your own full-text PDFs to the knowledge base. "
                "Run 'Generate Report' first so the vector store exists."
            )
            pdf_upload = gr.File(
                label="Select PDFs",
                file_types=[".pdf"],
                file_count="multiple",
            )
            with gr.Row():
                upload_btn = gr.Button("Process & Add to KB")
                clear_btn  = gr.Button("Clear")
            upload_status = gr.Textbox(label="Status", lines=3)
            upload_btn.click(
                fn=process_uploaded_pdfs,
                inputs=[pdf_upload],
                outputs=[upload_status],
            )
            clear_btn.click(
                lambda: (None, ""),
                inputs=[],
                outputs=[pdf_upload, upload_status],
            )

        # ── Tab 5: Open Access Lookup ─────────────────────────────────────────
        with gr.TabItem("Open Access Lookup"):
            gr.Markdown("Check OA status via Unpaywall (comma-separated DOIs).")
            oa_input = gr.Textbox(label="DOI(s)")
            oa_btn   = gr.Button("Check")
            oa_table = gr.Dataframe(
                headers=["DOI", "Is OA", "PDF URL", "License"]
            )
            oa_btn.click(
                fn=check_open_access,
                inputs=[oa_input],
                outputs=[oa_table],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)