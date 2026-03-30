from core.retrieval.pubmed import PubMedBatchFetcher
import os

def main():
    email = os.getenv("EMAIL", "yiweixidu@gmail.com")
    api_key = os.getenv("PUBMED_API_KEY")
    
    # 使用已经验证的查询（不带字段限定符，日期过滤将由 search_all_requests 自动添加）
    query = '("broadly neutralizing antibody" OR bnAb OR "cross-reactive antibody") AND (influenza OR flu OR H1N1 OR H3N2 OR hemagglutinin)'
    
    fetcher = PubMedBatchFetcher(email=email, api_key=api_key)
    
    # 全量爬取（requests 版本）
    articles = fetcher.search_all_requests(
        raw_query=query,
        max_results=None,          # 全部
        days_back=3650,            # 近10年
        checkpoint_file="data/checkpoints/flu_bnabs_full.json"
    )
    
    print(f"成功获取 {len(articles)} 篇文献")
    
    # 保存结果
    import json
    with open("data/flu_bnabs_all_articles.json", "w") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()