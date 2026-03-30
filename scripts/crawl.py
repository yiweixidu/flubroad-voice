"""
scripts/crawl.py
Standalone script to crawl PubMed, enrich with full text, and save to cache.

Usage:
    python scripts/crawl.py
    python scripts/crawl.py --query "broadly neutralizing antibody influenza" --max 600
    python scripts/crawl.py --max 600 --batch   # use batch mode for >1000 articles
"""

import argparse
import json
import os
import sys

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from core.retrieval.enhanced_fetcher import EnhancedLiteratureFetcher


DEFAULT_QUERY = (
    '(broadly neutralizing antibody OR bnab OR broadly reactive antibody) '
    'AND (influenza OR hemagglutinin OR neuraminidase)'
)
DEFAULT_CACHE = "data/flu_bnabs_all_articles.json"


def main():
    parser = argparse.ArgumentParser(description="Crawl PubMed and cache articles locally")
    parser.add_argument("--query",  default=DEFAULT_QUERY, help="PubMed search query")
    parser.add_argument("--max",    type=int, default=600,  help="Max articles to fetch")
    parser.add_argument("--days",   type=int, default=3650, help="Days back to search")
    parser.add_argument("--cache",  default=DEFAULT_CACHE,  help="Output JSON path")
    parser.add_argument("--batch",  action="store_true",    help="Use batch mode (for >1000)")
    parser.add_argument("--no-fulltext", action="store_true",
                        help="Skip full-text enrichment (faster, abstracts only)")
    args = parser.parse_args()

    config = {
        "email":          os.getenv("EMAIL", "yiweixidu@gmail.com"),
        "pubmed_api_key": os.getenv("PUBMED_API_KEY"),
        "pmc_delay":      0.5,
        "unpaywall_delay": 0.5,
    }

    fetcher = EnhancedLiteratureFetcher(config)

    if args.no_fulltext:
        # Abstract-only mode: just PubMed, no enrichment
        from core.retrieval.pubmed import PubMedFetcher
        pubmed = PubMedFetcher(
            email=config["email"],
            api_key=config["pubmed_api_key"],
        )
        pmids = pubmed.search(args.query, max_results=args.max, days_back=args.days)
        print(f"Found {len(pmids)} PMIDs")
        articles = pubmed.fetch_details(pmids)
        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        with open(args.cache, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(articles)} articles (abstracts only) → {args.cache}")
    else:
        articles = fetcher.fetch_and_cache(
            query=args.query,
            max_papers=args.max,
            cache_file=args.cache,
            days_back=args.days,
            use_batch=args.batch,
        )

    # Summary
    fulltext_count = sum(1 for a in articles if a.get("fulltext_available"))
    print(f"\n{'='*50}")
    print(f"Total cached  : {len(articles)}")
    print(f"With full text: {fulltext_count}")
    print(f"Abstract only : {len(articles) - fulltext_count}")
    print(f"Saved to      : {args.cache}")
    print(f"{'='*50}")
    print("\nNext step: python app/gradio_ui.py")


if __name__ == "__main__":
    main()