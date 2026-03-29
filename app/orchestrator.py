import os
import asyncio
import subprocess
import tempfile
from typing import Dict, List, Optional
from pdf2image import convert_from_path
from moviepy.editor import AudioFileClip

from core.retrieval.pubmed import PubMedFetcher
from core.retrieval.biorxiv import BioRxivFetcher
from core.rag.vector_store import FluBroadRAG
from core.narrative.generator import NarrativeGenerator
from core.presentation.ppt_generator import PPTGenerator
from core.presentation.speech_synthesizer import SpeechSynthesizer
from domain.virology.prompts.pmrc_templates import get_template
from domain.virology.schemas.antibody_schema import antibody_schema


class FluBroadOrchestrator:
    """协调各模块完成完整流程"""

    def __init__(self, config: dict):
        self.config = config
        self.fetchers = []

        # 初始化 PubMed 检索器
        if config.get("email"):
            self.pubmed_fetcher = PubMedFetcher(
                email=config["email"],
                api_key=config.get("pubmed_api_key")
            )
            self.fetchers.append(self.pubmed_fetcher)

        # 可选初始化 bioRxiv 检索器（默认禁用，因为可能返回403）
        if config.get("enable_biorxiv", False):
            self.fetchers.append(BioRxivFetcher())

        # 可继续添加 EuropePMC 等

        self.rag = FluBroadRAG(
            collection_name=config["collection_name"],
            persist_directory=config["persist_dir"]
        )
        self.generator = NarrativeGenerator(
            model=config.get("llm_model", "llama3.2:3b"),
            temperature=config.get("llm_temperature", 0.1),
            llm_type=config.get("llm_type", "ollama")
        )
        self.ppt_gen = PPTGenerator(template_path=config.get("ppt_template"))
        self.tts = SpeechSynthesizer()

    def run(self, query: str, max_papers: int = 50, fetch_all: bool = False) -> Dict:
        """
        运行完整流程（实时检索模式）。
        :param query: 检索式
        :param max_papers: 最大文献数量（当 fetch_all=False 时生效）
        :param fetch_all: 是否全量获取所有相关文献（使用 PubMed 批量模式）
        """
        print(f"Starting pipeline for query: {query}")

        # 1. 获取文献
        all_articles = self._fetch_articles(query, max_papers, fetch_all)
        if not all_articles:
            print("No articles retrieved. Exiting.")
            return {
                "review": "No articles found for the query.",
                "antibodies": [],
                "ppt_file": None,
                "video_file": None
            }

        print(f"Total unique articles: {len(all_articles)}")

        # 2. 构建知识库
        self.rag.build(all_articles)
        print("Knowledge base built")

        # 3. 生成综述（使用前10篇作为上下文，可调整）
        context = "\n\n".join(
            [f"{a['title']}\n{a['abstract']}" for a in all_articles[:10]]
        )
        pmrc_template = get_template("flu_bnabs")
        review = self.generator.generate_review(context, pmrc_template)
        print("Review generated")

        # 4. 抽取抗体表格
        antibodies = self.generator.extract_antibodies(review, antibody_schema)
        print(f"Extracted {len(antibodies)} antibodies")

        # 5. 生成 PPT
        ppt_file = self._build_ppt(review, antibodies, all_articles)
        print(f"PPT saved: {ppt_file}")

        # 6. 将 PPT 导出为图片（用于视频合成）
        output_dir = self.config.get('output_dir', './output')
        slide_images = self._export_ppt_to_images(ppt_file, output_dir)

        # 7. 生成视频（如果图片导出成功）
        video_file = None
        if slide_images:
            try:
                video_file = asyncio.run(self._build_video(review, slide_images))
            except Exception as e:
                print(f"Video generation failed: {e}")
        else:
            print("Skipping video generation (no slide images)")

        return {
            "review": review,
            "antibodies": antibodies,
            "ppt_file": ppt_file,
            "video_file": video_file
        }

    def run_from_articles(self, articles: List[Dict]) -> Dict:
        """
        直接使用已有的文献列表生成报告（跳过检索步骤）
        """
        if not articles:
            print("No articles provided.")
            return {
                "review": "No articles provided.",
                "antibodies": [],
                "ppt_file": None,
                "video_file": None
            }

        # 1. 构建知识库（使用传入的文献）
        self.rag.build(articles)
        print(f"Knowledge base built with {len(articles)} articles")

        # 2. 生成综述（使用前10篇作为上下文，可根据需要调整）
        context = "\n\n".join(
            [f"{a['title']}\n{a['abstract']}" for a in articles[:10] if a.get('abstract')]
        )
        pmrc_template = get_template("flu_bnabs")
        review = self.generator.generate_review(context, pmrc_template)
        print("Review generated")

        # 3. 抽取抗体表格
        antibodies = self.generator.extract_antibodies(review, antibody_schema)
        print(f"Extracted {len(antibodies)} antibodies")

        # 4. 生成 PPT
        ppt_file = self._build_ppt(review, antibodies, articles)
        print(f"PPT saved: {ppt_file}")

        # 5. 视频生成（可选，如需可调用 _build_video，但需先导出 PPT 图片）
        video_file = None
        # 如果需要视频，可以取消注释以下代码：
        # slide_images = self._export_ppt_to_images(ppt_file, self.config.get('output_dir', './output'))
        # if slide_images:
        #     video_file = asyncio.run(self._build_video(review, slide_images))

        return {
            "review": review,
            "antibodies": antibodies,
            "ppt_file": ppt_file,
            "video_file": video_file
        }

    def _fetch_articles(self, query: str, max_papers: int, fetch_all: bool) -> List[Dict]:
        """从所有检索器获取文献，去重后返回"""
        all_articles = []
        seen_ids = set()

        for fetcher in self.fetchers:
            # 对于 PubMed 且需要全量获取时，使用批量模式
            if fetch_all and isinstance(fetcher, PubMedFetcher):
                print(f"Fetching all papers from {fetcher.source_name} (batch mode)")
                articles = fetcher.fetch_all(
                    query=query,
                    max_results=None,          # 全部
                    days_back=3650,            # 近10年
                    use_batch=True,
                    checkpoint_file=f"data/checkpoints/{query.replace(' ', '_')}.json"
                )
            else:
                # 普通模式：先搜索 ID 再获取详情
                if hasattr(fetcher, 'search') and 'advanced' in fetcher.search.__code__.co_varnames:
                    ids = fetcher.search(query, max_results=max_papers, advanced=True)
                else:
                    ids = fetcher.search(query, max_results=max_papers)
                print(f"Retrieved {len(ids)} IDs from {fetcher.source_name}")
                articles = fetcher.fetch_details(ids)
                print(f"Fetched {len(articles)} articles from {fetcher.source_name}")

            # 去重
            for art in articles:
                uid = art.get("pmid") or art.get("doi")
                if uid and uid not in seen_ids:
                    seen_ids.add(uid)
                    all_articles.append(art)

        return all_articles

    def _build_ppt(self, review: str, antibodies: List[Dict], articles: List[Dict]) -> str:
        """构建 PPT 并保存，返回文件路径"""
        self.ppt_gen.add_title_slide("流感广谱中和抗体研究进展", "AI生成的综述报告")
        self.ppt_gen.add_content_slide("背景", [
            "流感病毒变异快，现有疫苗覆盖率有限",
            "广谱中和抗体靶向保守表位是突破方向"
        ])
        self.ppt_gen.add_content_slide("Key Findings", [review[:500]])

        if antibodies:
            # 过滤有效抗体（确保是字典）
            valid_abs = [ab for ab in antibodies if isinstance(ab, dict)]
            if valid_abs:
                headers = ["Name", "Target", "Epitope", "Gene", "Spectrum", "Phase"]
                rows = []
                for ab in valid_abs[:10]:
                    rows.append([
                        ab.get("antibody_name", ""),
                        ab.get("target_protein", ""),
                        ab.get("epitope_region", ""),
                        ab.get("gene_usage", ""),
                        ab.get("neutralization_spectrum", ""),
                        ab.get("clinical_phase", "")
                    ])
                self.ppt_gen.add_table_slide("Broadly Neutralizing Antibodies", headers, rows)

        ppt_file = self.config.get('output_dir', './output') + "/output.pptx"
        os.makedirs(os.path.dirname(ppt_file), exist_ok=True)
        self.ppt_gen.save(ppt_file)
        return ppt_file

    def _export_ppt_to_images(self, pptx_path: str, output_dir: str, dpi: int = 200) -> List[str]:
        """
        将 PPTX 文件导出为图片列表。
        使用 LibreOffice 命令行将 PPTX 转为 PDF，再用 pdf2image 转为 PNG。
        若 LibreOffice 不可用，则打印错误并返回空列表。
        """
        if not os.path.exists(pptx_path):
            print(f"PPT file not found: {pptx_path}")
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "slides.pdf")
            try:
                subprocess.run(
                    ["libreoffice", "--headless", "--convert-to", "pdf",
                     "--outdir", tmpdir, pptx_path],
                    check=True,
                    capture_output=True,
                    text=True
                )
                base_name = os.path.splitext(os.path.basename(pptx_path))[0]
                pdf_path = os.path.join(tmpdir, f"{base_name}.pdf")
                if not os.path.exists(pdf_path):
                    for f in os.listdir(tmpdir):
                        if f.endswith(".pdf"):
                            pdf_path = os.path.join(tmpdir, f)
                            break
                    else:
                        raise FileNotFoundError("PDF not generated")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"LibreOffice conversion failed: {e}")
                return []

            os.makedirs(output_dir, exist_ok=True)
            try:
                images = convert_from_path(pdf_path, dpi=dpi)
                image_paths = []
                for i, img in enumerate(images):
                    img_path = os.path.join(output_dir, f"slide_{i+1:03d}.png")
                    img.save(img_path, "PNG")
                    image_paths.append(img_path)
                print(f"Exported {len(image_paths)} slides to images in {output_dir}")
                return image_paths
            except Exception as e:
                print(f"PDF to image conversion failed: {e}")
                return []

    async def _build_video(self, review: str, slide_images: List[str]) -> Optional[str]:
        """异步生成演讲视频"""
        script = review.replace('\n', ' ').replace('\r', ' ').strip()
        if not script:
            print("警告：讲稿为空，无法生成视频")
            return None

        output_dir = self.config.get('output_dir', './output')
        os.makedirs(output_dir, exist_ok=True)

        audio_path = os.path.join(output_dir, 'audio.mp3')
        await self.tts.synthesize(script, audio_path)
        print(f"音频已生成: {audio_path}")

        try:
            audio = AudioFileClip(audio_path)
            total_duration = audio.duration
            audio.close()
        except Exception as e:
            print(f"Failed to read audio duration: {e}")
            return None

        num_slides = len(slide_images)
        if num_slides == 0:
            print("警告：无幻灯片图片，无法生成视频")
            return None

        slide_duration = total_duration / num_slides
        timings = [i * slide_duration for i in range(num_slides)]

        video_path = os.path.join(output_dir, 'presentation.mp4')
        self.tts.create_video(
            audio_path=audio_path,
            slide_images=slide_images,
            timings=timings,
            output_video=video_path
        )
        print(f"视频生成完成: {video_path}")
        return video_path