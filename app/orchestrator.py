"""
app/orchestrator.py  —  final integrated version
Two entry points:
  run_from_articles()  — fast path, uses local JSON cache
  run_with_crawl()     — slow path, crawls PubMed + enriches full text first
"""

import os
import subprocess
import tempfile
from typing import Callable, Dict, List, Optional

from core.retrieval.enhanced_fetcher import EnhancedLiteratureFetcher
from core.rag.vector_store import FluBroadRAG
from core.narrative.generator import NarrativeGenerator
from core.presentation.ppt_generator import PPTGenerator
from core.presentation.speech_synthesizer import SpeechSynthesizer
from domain.virology.schemas.antibody_schema import antibody_schema


SECTION_QUERIES = {
    "problem":    "antigenic drift shift influenza vaccine mismatch pandemic",
    "motivation": "conserved epitope HA stem broadly neutralizing antibody strategy",
    "results":    "broadly neutralizing antibody influenza neutralization spectrum IC50 clinical",
    "mechanisms": "Fc effector function ADCC broadly neutralizing antibody mechanism",
    "challenges": "immunodominance germline targeting epitope accessibility bnAb challenge",
    "future":     "mRNA vaccine immunogen design deep mutational scanning universal influenza",
}

TOPIC_KEYWORDS = [
    "influenza", "hemagglutinin", "neuraminidase", "bnab", "broadly neutralizing",
    "sars-cov-2", "coronavirus", "antibody", "vaccine", "epitope", "stem",
]


def is_relevant(article: Dict, min_hits: int = 1) -> bool:
    text = (
        (article.get("title") or "") + " " + (article.get("abstract") or "")
    ).lower()
    return sum(1 for kw in TOPIC_KEYWORDS if kw in text) >= min_hits


class FluBroadOrchestrator:

    def __init__(self, config: Dict):
        self.config = config
        self.lit_fetcher = EnhancedLiteratureFetcher(config)
        self.rag = FluBroadRAG(
            collection_name=config["collection_name"],
            persist_directory=config["persist_dir"],
        )
        self.generator = NarrativeGenerator(
            model=config.get("llm_model", "gpt-4o-mini"),
            temperature=config.get("llm_temperature", 0.1),
            llm_type=config.get("llm_type", "openai"),
        )
        self.ppt_gen = PPTGenerator(template_path=config.get("ppt_template"))
        self.tts = SpeechSynthesizer()

    # ── Fast path: cached articles ────────────────────────────────────────────
    def run_from_articles(self, articles: List[Dict]) -> Dict:
        if not articles:
            return {"review": "No articles provided.", "antibodies": [],
                    "ppt_file": None, "video_file": None, "stats": {}}

        filtered = [a for a in articles if is_relevant(a)]
        skipped = len(articles) - len(filtered)
        if skipped:
            print(f"[Orchestrator] Filtered {skipped} irrelevant articles")
        print(f"[Orchestrator] Using {len(filtered)} articles")

        self.rag.build(filtered)
        review = self._generate_review_by_sections()
        antibodies = self.generator.extract_antibodies(review, antibody_schema)
        print(f"[Orchestrator] Extracted {len(antibodies)} antibodies")
        ppt_file = self._build_ppt(review, antibodies, filtered)

        return {
            "review": review,
            "antibodies": antibodies,
            "ppt_file": ppt_file,
            "video_file": None,
            "stats": {
                "total_input":     len(articles),
                "after_filter":    len(filtered),
                "antibodies_found": len(antibodies),
                "fulltext_count":  sum(
                    1 for a in filtered if a.get("fulltext_available")
                ),
            },
        }

    # ── Slow path: crawl → cache → report ────────────────────────────────────
    def run_with_crawl(
        self,
        query: str,
        max_papers: int = 200,
        cache_file: str = "data/flu_bnabs_all_articles.json",
        days_back: int = 3650,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict:
        print(f"[Orchestrator] Crawling PubMed: {query!r}")
        articles = self.lit_fetcher.fetch_and_cache(
            query=query,
            max_papers=max_papers,
            cache_file=cache_file,
            days_back=days_back,
            progress_callback=progress_callback,
        )
        print(f"[Orchestrator] Crawl done: {len(articles)} articles")
        return self.run_from_articles(articles)

    # ── Section-by-section generation ────────────────────────────────────────
    def _generate_review_by_sections(self) -> str:
        sections: Dict[str, str] = {}
        for name, query in SECTION_QUERIES.items():
            print(f"  [Review] Generating: {name}")
            docs = self.rag.similarity_search(query, k=8)
            context = "\n\n".join(
                f"[PMID: {d.metadata.get('pmid', 'N/A')}] {d.page_content}"
                for d in docs
            )
            sections[name] = self.generator.generate_section(
                section_name=name,
                context=context,
                all_section_texts=sections,
            )
        return self._assemble_review(sections)

    @staticmethod
    def _assemble_review(sections: Dict[str, str]) -> str:
        order = {
            "problem":    "## Problem",
            "motivation": "## Motivation & Rationale",
            "results":    "## Key Results & Broadly Neutralizing Antibodies",
            "mechanisms": "## Mechanisms of Action",
            "challenges": "## Technical Challenges",
            "future":     "## Future Directions",
        }
        return "\n\n---\n\n".join(
            f"{heading}\n\n{sections[key]}"
            for key, heading in order.items()
            if key in sections
        )

    # ── PPT ───────────────────────────────────────────────────────────────────
    def _build_ppt(self, review: str, antibodies: List[Dict],
                   articles: List[Dict]) -> str:
        ft = sum(1 for a in articles if a.get("fulltext_available"))
        self.ppt_gen.add_title_slide(
            "Broadly Neutralizing Antibodies Against Influenza",
            "AI-generated literature review",
        )
        self.ppt_gen.add_content_slide("Background", [
            "Influenza viruses evolve via antigenic drift and shift",
            "bnAbs target conserved epitopes (HA stem, NA active site, fusion peptide)",
            f"Literature: {len(articles)} articles ({ft} full text, "
            f"{len(articles)-ft} abstract only)",
        ])
        self.ppt_gen.add_content_slide("Key Findings", [review[:800]])
        if antibodies:
            valid = [ab for ab in antibodies if isinstance(ab, dict)]
            if valid:
                headers = ["Name", "Target", "Epitope", "Gene", "Spectrum", "Phase"]
                rows = [[
                    ab.get("antibody_name", ""),
                    ab.get("target_protein", ""),
                    ab.get("epitope_region", ""),
                    ab.get("gene_usage", ""),
                    ab.get("neutralization_spectrum", ""),
                    ab.get("clinical_phase", ""),
                ] for ab in valid[:10]]
                self.ppt_gen.add_table_slide(
                    "Broadly Neutralizing Antibodies", headers, rows
                )
        output_dir = self.config.get("output_dir", "./output")
        ppt_file = os.path.join(output_dir, "output.pptx")
        os.makedirs(output_dir, exist_ok=True)
        self.ppt_gen.save(ppt_file)
        return ppt_file

    def _export_ppt_to_images(self, pptx_path: str, output_dir: str,
                               dpi: int = 200) -> List[str]:
        if not os.path.exists(pptx_path):
            return []
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                subprocess.run(
                    ["libreoffice", "--headless", "--convert-to", "pdf",
                     "--outdir", tmpdir, pptx_path],
                    check=True, capture_output=True, text=True,
                )
                import glob
                pdfs = glob.glob(os.path.join(tmpdir, "*.pdf"))
                if not pdfs:
                    raise FileNotFoundError("No PDF generated")
                pdf_path = pdfs[0]
            except Exception as e:
                print(f"LibreOffice failed: {e}")
                return []
            os.makedirs(output_dir, exist_ok=True)
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(pdf_path, dpi=dpi)
                paths = []
                for i, img in enumerate(images):
                    p = os.path.join(output_dir, f"slide_{i+1:03d}.png")
                    img.save(p, "PNG")
                    paths.append(p)
                return paths
            except Exception as e:
                print(f"pdf2image failed: {e}")
                return []

    async def _build_video(self, review: str,
                           slide_images: List[str]) -> Optional[str]:
        from moviepy.editor import AudioFileClip
        script = review.replace("\n", " ").strip()
        if not script or not slide_images:
            return None
        output_dir = self.config.get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)
        audio_path = os.path.join(output_dir, "audio.mp3")
        await self.tts.synthesize(script, audio_path)
        try:
            audio = AudioFileClip(audio_path)
            duration = audio.duration
            audio.close()
        except Exception as e:
            print(f"Audio read failed: {e}")
            return None
        n = len(slide_images)
        timings = [i * duration / n for i in range(n)]
        video_path = os.path.join(output_dir, "presentation.mp4")
        self.tts.create_video(
            audio_path=audio_path,
            slide_images=slide_images,
            timings=timings,
            output_video=video_path,
        )
        return video_path