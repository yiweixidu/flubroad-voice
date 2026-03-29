import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag.vector_store import FluBroadRAG
from core.narrative.generator import NarrativeGenerator
from core.presentation.ppt_generator import PPTGenerator
from core.presentation.speech_synthesizer import SpeechSynthesizer
from domain.virology.prompts.pmrc_templates import get_template
from domain.virology.schemas.antibody_schema import antibody_schema

def main():
    # 加载已爬取的文献
    with open("data/flu_bnabs_all_articles.json", "r") as f:
        articles = json.load(f)
    print(f"Loaded {len(articles)} articles")

    # 配置 RAG（使用本地持久化目录）
    rag = FluBroadRAG(
        collection_name="flu_bnabs_full",
        persist_directory="./data/vector_db"
    )
    # 构建向量库（会进行 embedding，需要 Ollama 嵌入模型？）
    # 注意：FluBroadRAG 默认使用 OpenAIEmbeddings，需要改用本地嵌入模型
    # 我们稍后调整，或者暂时跳过 RAG，直接用前10篇生成综述
    # 这里先直接生成综述（不经过 RAG）
    
    # 初始化生成器（使用 Ollama）
    generator = NarrativeGenerator(
        model="llama3.2:3b",
        temperature=0.1,
        llm_type="ollama"   # 确保 generator.py 支持 ollama
    )
    
    # 取前20篇作为上下文（可根据需要调整）
    context = "\n\n".join(
        [f"Title: {a['title']}\nAbstract: {a['abstract']}" for a in articles[:20] if a.get('abstract')]
    )
    pmrc_template = get_template("flu_bnabs")
    review = generator.generate_review(context, pmrc_template)
    print("=== Generated Review ===\n", review)
    
    # 抽取抗体
    antibodies = generator.extract_antibodies(review, antibody_schema)
    print(f"Extracted {len(antibodies)} antibodies")
    
    # 生成 PPT
    ppt_gen = PPTGenerator(template_path=None)  # 使用默认模板
    ppt_gen.add_title_slide("流感广谱中和抗体研究进展", "基于543篇PubMed文献的AI综述")
    ppt_gen.add_content_slide("背景与意义", [
        "流感病毒持续变异，现有疫苗保护有限",
        "广谱中和抗体靶向保守表位是突破方向"
    ])
    ppt_gen.add_content_slide("核心发现", [review[:800]])
    if antibodies:
        headers = ["Antibody", "Target", "Epitope", "Gene", "Spectrum", "Phase"]
        rows = [[
            ab.get("antibody_name", ""),
            ab.get("target_protein", ""),
            ab.get("epitope_region", ""),
            ab.get("gene_usage", ""),
            ab.get("neutralization_spectrum", ""),
            ab.get("clinical_phase", "")
        ] for ab in antibodies[:10]]
        ppt_gen.add_table_slide("Key Broadly Neutralizing Antibodies", headers, rows)
    
    os.makedirs("./output", exist_ok=True)
    ppt_file = "./output/full_review.pptx"
    ppt_gen.save(ppt_file)
    print(f"PPT saved to {ppt_file}")
    
    # 可选：生成语音视频（需要先导出 PPT 为图片，依赖 LibreOffice）
    # 这里先略过，后续可补充

if __name__ == "__main__":
    main()