import os
from dotenv import load_dotenv
from core.retrieval.pubmed import PubMedFetcher

load_dotenv()

def test_search():
    email = os.getenv("EMAIL")
    if not email:
        print("请设置EMAIL环境变量")
        return
    fetcher = PubMedFetcher(email=email)
    pmids = fetcher.search("broadly neutralizing antibody influenza", max_results=5)
    print(f"Found {len(pmids)} PMIDs: {pmids}")
    articles = fetcher.fetch_details(pmids)
    for art in articles:
        print(f"{art['pmid']}: {art['title']}")

if __name__ == "__main__":
    test_search()