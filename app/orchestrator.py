import os
import asyncio
import subprocess
import tempfile
from typing import Dict, List, Optional
from pdf2image import convert_from_path
from pptx import Presentation as PptxPresentation
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
        # 初始化PubMed检索器
        if config.get("email"):
            self.fetchers.append(PubMedFetcher(
                email=config["email"],
                api_key=config.get("pubmed_api_key")
            ))
        # 可选初始化bioRxiv检索器
        if config.get("enable_biorxiv", True):
            self.fetchers.append(BioRxivFetcher())
        # 可继续添加EuropePMC等
        
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

    def run(self, query: str, max_papers: int = 50) -> Dict:
        """同步运行完整流程，返回结果字典"""
        print(f"Starting pipeline for query: {query}")

        # 1. 并行检索所有数据源（为简化，仍串行，可改为并行）
        all_articles = []
        seen_ids = set()  # 用于去重

        for fetcher in self.fetchers:
            # 对于支持 advanced 参数的检索器，传递 advanced=True
            if hasattr(fetcher.search, '__code__') and 'advanced' in fetcher.search.__code__.co_varnames:
                ids = fetcher.search(query, max_results=max_papers, advanced=True)
            else:
                ids = fetcher.search(query, max_results=max_papers)
            print(f"Retrieved {len(ids)} IDs from {fetcher.source_name}")
            
            articles = fetcher.fetch_details(ids)
            print(f"Fetched {len(articles)} articles from {fetcher.source_name}")
            
            # 去重：使用 pmid（或 doi）作为唯一标识
            for art in articles:
                uid = art.get("pmid") or art.get("doi")
                if uid and uid not in seen_ids:
                    seen_ids.add(uid)
                    all_articles.append(art)

        if not all_articles:  # 修复：检查总结果
            print("No articles retrieved. Exiting.")
            return {
                "review": "No articles found for the query.",
                "antibodies": [],
                "ppt_file": None,
                "video_file": None
            }

        # 2. 构建知识库
        self.rag.build(all_articles)
        print("Knowledge base built")

        # 3. 生成综述
        context = "\n\n".join([f"{a['title']}\n{a['abstract']}" for a in all_articles[:10]])
        pmrc_template = get_template("flu_bnabs")
        review = self.generator.generate_review(context, pmrc_template)
        print("Review generated")

        # 4. 抽取抗体表格（使用增强版 schema）
        antibodies = self.generator.extract_antibodies(review, antibody_schema)
        print(f"Extracted {len(antibodies)} antibodies")

        # 5. 生成 PPT（内部已包含图表生成）
        self._build_ppt(review, antibodies, all_articles)
        ppt_file = "output.pptx"
        self.ppt_gen.save(ppt_file)
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

    def _build_ppt(self, review: str, antibodies: List[Dict], articles: List[Dict]):
        """构建PPT内容"""
        self.ppt_gen.add_title_slide("流感广谱中和抗体研究进展", "AI生成的综述报告")
        self.ppt_gen.add_content_slide("背景", [
            "流感病毒变异快，现有疫苗覆盖率有限",
            "广谱中和抗体靶向保守表位是突破方向"
        ])
        self.ppt_gen.add_content_slide("Key Findings", [review[:500]])
        valid_antibodies = [ab for ab in antibodies if isinstance(ab, dict)]
        if antibodies:
            headers = ["Name", "Target", "Epitope", "Gene", "Spectrum", "Phase"]
            rows = [[ab.get("antibody_name",""), ab.get("target_protein",""), ab.get("epitope_region",""),
                    ab.get("gene_usage",""), ab.get("neutralization_spectrum",""), ab.get("clinical_phase","")]
                    for ab in antibodies[:10]]
            self.ppt_gen.add_table_slide("Broadly Neutralizing Antibodies", headers, rows)
            # 如果有IC50数据，生成热图
            if any(ab.get("ic50") for ab in antibodies):
                # 构建热图数据：假设抗体名和亚型IC50
                matrix = self._build_neutralization_matrix(antibodies)  # 需实现
                if matrix:
                    self.ppt_gen.add_neutralization_heatmap(matrix, title="Neutralization IC50 (µg/mL)")
        self.ppt_gen.save("output.pptx")

    def _export_ppt_to_images(self, pptx_path: str, output_dir: str, dpi: int = 200) -> List[str]:
        """
        将 PPTX 文件导出为图片列表。
        使用 LibreOffice 命令行将 PPTX 转为 PDF，再用 pdf2image 转为 PNG。
        若 LibreOffice 不可用，则打印错误并返回空列表。
        """
        if not os.path.exists(pptx_path):
            print(f"PPT file not found: {pptx_path}")
            return []

        # 创建临时目录存放 PDF
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "slides.pdf")
            # 尝试用 LibreOffice 转换
            try:
                subprocess.run(
                    ["libreoffice", "--headless", "--convert-to", "pdf",
                     "--outdir", tmpdir, pptx_path],
                    check=True,
                    capture_output=True,
                    text=True
                )
                # LibreOffice 生成的 PDF 名称与 PPTX 同名，需定位
                base_name = os.path.splitext(os.path.basename(pptx_path))[0]
                pdf_path = os.path.join(tmpdir, f"{base_name}.pdf")
                if not os.path.exists(pdf_path):
                    # 尝试寻找生成的 PDF（有时扩展名变化）
                    for f in os.listdir(tmpdir):
                        if f.endswith(".pdf"):
                            pdf_path = os.path.join(tmpdir, f)
                            break
                    else:
                        raise FileNotFoundError("PDF not generated")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"LibreOffice conversion failed: {e}")
                return []

            # 使用 pdf2image 将 PDF 转为 PNG
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
        """
        异步生成演讲视频
        :param review: 综述文本（作为讲稿）
        :param slide_images: 幻灯片图片列表
        :return: 视频文件路径，失败时返回 None
        """
        # 1. 预处理讲稿
        script = review.replace('\n', ' ').replace('\r', ' ').strip()
        if not script:
            print("警告：讲稿为空，无法生成视频")
            return None

        # 2. 输出目录
        output_dir = self.config.get('output_dir', './output')
        os.makedirs(output_dir, exist_ok=True)

        # 3. 生成音频
        audio_path = os.path.join(output_dir, 'audio.mp3')
        await self.tts.synthesize(script, audio_path)
        print(f"音频已生成: {audio_path}")

        # 4. 获取音频总时长，计算均匀分配的时间点
        from moviepy.editor import AudioFileClip
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

        # 5. 合成视频
        video_path = os.path.join(output_dir, 'presentation.mp4')
        self.tts.create_video(
            audio_path=audio_path,
            slide_images=slide_images,
            timings=timings,
            output_video=video_path
        )
        print(f"视频生成完成: {video_path}")
        return video_path