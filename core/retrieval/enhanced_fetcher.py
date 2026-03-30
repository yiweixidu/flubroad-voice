"""
core/retrieval/enhanced_fetcher.py  —  complete rewrite
This is the single module responsible for producing the local article cache
(data/flu_bnabs_all_articles.json).

Pipeline:
  1. PubMed search  → PMIDs
  2. PubMed efetch  → abstracts + DOI + PMCID
  3. For each article
       a. PMCID present? → Europe PMC full-text XML   (PMCFulltextFetcher)
       b. DOI present?   → Unpaywall OA PDF download  (UnpaywallFetcher)
       c. else           → keep abstract only
  4. Save enriched list to a local JSON file

Usage:
    from core.retrieval.enhanced_fetcher import EnhancedLiteratureFetcher
    fetcher = EnhancedLiteratureFetcher(config)
    articles = fetcher.fetch_and_cache(
        query="broadly neutralizing antibody influenza",
        max_papers=600,
        cache_file="data/flu_bnabs_all_articles.json",
    )
"""

import json
import os
import time
from typing import Callable, Dict, List, Optional

from .pubmed import PubMedFetcher
from .pmc_fulltext import PMCFulltextFetcher, UnpaywallFetcher


class EnhancedLiteratureFetcher:
    """
    End-to-end fetcher: PubMed → full-text enrichment → local JSON cache.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.pubmed = PubMedFetcher(
            email=config["email"],
            api_key=config.get("pubmed_api_key"),
        )
        self.pmc = PMCFulltextFetcher(
            email=config["email"],
            delay=config.get("pmc_delay", 0.5),
        )
        self.unpaywall = UnpaywallFetcher(
            email=config["email"],
            delay=config.get("unpaywall_delay", 0.5),
        )

    # ── Main public method ────────────────────────────────────────────────────
    def fetch_and_cache(
        self,
        query: str,
        max_papers: int = 600,
        cache_file: str = "data/flu_bnabs_all_articles.json",
        days_back: int = 3650,
        use_batch: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Dict]:
        """
        Run the full pipeline and write results to `cache_file`.

        progress_callback(current, total, stage) — called after each article
        is enriched so the UI can update a progress bar.

        Returns the enriched article list.
        """
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        # ── Step 1: PubMed retrieval ──────────────────────────────────────────
        print(f"[Fetcher] Searching PubMed: {query!r} (max {max_papers})")
        if use_batch:
            checkpoint = cache_file.replace(".json", "_checkpoint.json")
            articles = self.pubmed.fetch_all(
                query=query,
                max_results=max_papers,
                days_back=days_back,
                checkpoint_file=checkpoint,
            )
        else:
            pmids = self.pubmed.search(
                query, max_results=max_papers, days_back=days_back
            )
            print(f"[Fetcher] Retrieved {len(pmids)} PMIDs")
            articles = self.pubmed.fetch_details(pmids)

        print(f"[Fetcher] Fetched {len(articles)} article records from PubMed")

        # ── Step 2: Full-text enrichment ──────────────────────────────────────
        total = len(articles)
        pmc_hits = 0
        oa_hits  = 0

        for i, article in enumerate(articles):
            pmcid = article.get("pmcid", "")
            doi   = article.get("doi", "")

            # If PubMed XML didn't give us a PMCID, ask Europe PMC
            if not pmcid and article.get("pmid"):
                pmcid_found = self.pmc.lookup_pmcid(article["pmid"])
                if pmcid_found:
                    article["pmcid"] = pmcid_found
                    pmcid = pmcid_found

            # Priority 1: PMC full-text XML
            if pmcid and not article.get("fulltext_available"):
                print(f"  [{i+1}/{total}] PMC fulltext: {pmcid}")
                self.pmc.enrich_article(article)
                if article.get("fulltext_available"):
                    pmc_hits += 1

            # Priority 2: Unpaywall OA PDF
            if doi and not article.get("fulltext_available"):
                print(f"  [{i+1}/{total}] Unpaywall: {doi}")
                self.unpaywall.enrich_article(article)
                if article.get("fulltext_available"):
                    oa_hits += 1

            if not article.get("fulltext_available"):
                print(f"  [{i+1}/{total}] Abstract only: PMID {article.get('pmid')}")

            if progress_callback:
                progress_callback(i + 1, total, article.get("pmid", ""))

        print(
            f"\n[Fetcher] Enrichment complete:\n"
            f"  Total articles : {total}\n"
            f"  PMC full text  : {pmc_hits}\n"
            f"  Unpaywall PDF  : {oa_hits}\n"
            f"  Abstract only  : {total - pmc_hits - oa_hits}"
        )

        # ── Step 3: Save to local cache ───────────────────────────────────────
        self._save_cache(articles, cache_file)
        return articles

    # ── Cache helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _save_cache(articles: List[Dict], path: str) -> None:
        """Write articles to JSON, stripping fulltext_content to keep file small
        if desired. Here we keep it — the content is needed for RAG."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        size_mb = os.path.getsize(path) / 1_048_576
        print(f"[Fetcher] Saved {len(articles)} articles → {path} ({size_mb:.1f} MB)")

    @staticmethod
    def load_cache(path: str) -> List[Dict]:
        """Load articles from a previously saved JSON cache."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cache not found: {path}")
        with open(path, encoding="utf-8") as f:
            articles = json.load(f)
        print(f"[Fetcher] Loaded {len(articles)} articles from {path}")
        return articles

    # ── Convenience: fetch without caching ───────────────────────────────────
    def fetch_with_fulltext(
        self,
        query: str,
        max_papers: int = 50,
        days_back: int = 3650,
    ) -> List[Dict]:
        """
        Like fetch_and_cache() but does not write to disk.
        Useful for on-demand queries from the UI.
        """
        pmids = self.pubmed.search(query, max_results=max_papers, days_back=days_back)
        articles = self.pubmed.fetch_details(pmids)
        total = len(articles)
        for i, article in enumerate(articles):
            pmcid = article.get("pmcid", "")
            doi   = article.get("doi", "")
            if not pmcid and article.get("pmid"):
                found = self.pmc.lookup_pmcid(article["pmid"])
                if found:
                    article["pmcid"] = found
                    pmcid = found
            if pmcid:
                self.pmc.enrich_article(article)
            if doi and not article.get("fulltext_available"):
                self.unpaywall.enrich_article(article)
            print(
                f"  [{i+1}/{total}] {article.get('pmid')} — "
                f"{'fulltext' if article.get('fulltext_available') else 'abstract'}"
                f" ({article.get('fulltext_source', 'N/A')})"
            )
        return articles