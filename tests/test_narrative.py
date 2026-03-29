import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from core.retrieval.pubmed import PubMedFetcher
from core.narrative.generator import NarrativeGenerator
from domain.virology.prompts.pmrc_templates import get_template

load_dotenv()

def test_generate_review():
    email = os.getenv("EMAIL")
    if not email:
        print("请设置 EMAIL 环境变量（在 .env 文件中）")
        return

    # 1. 检索少量文献作为测试
    fetcher = PubMedFetcher(email=email)
    pmids = fetcher.search("broadly neutralizing antibody influenza", max_results=3)
    articles = fetcher.fetch_details(pmids)
    if not articles:
        print("未检索到文献，请检查网络或关键词")
        return

    # 2. 准备上下文（只取标题和摘要）
    context = "\n\n".join([f"Title: {a['title']}\nAbstract: {a['abstract']}" for a in articles])
    print(f"准备上下文，共 {len(articles)} 篇文献\n")

    # 3. 获取 PMRC 模板（目前只有 flu_bnabs）
    pmrc = get_template("flu_bnabs")
    print("PMRC 模板：", pmrc[:100], "...\n")

    # 4. 生成综述
    generator = NarrativeGenerator()  # 会从环境变量读取配置
    print("正在生成综述，请稍候...")
    review = generator.generate_review(context, pmrc)

    print("\n" + "="*50)
    print("生成的综述：")
    print("="*50)
    print(review)

if __name__ == "__main__":
    test_generate_review()