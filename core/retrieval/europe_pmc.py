"""
core/retrieval/europe_pmc.py
fetch_details was a stub (just `pass`) — now implemented via the
Europe PMC search API.
"""

import time
import requests
from typing import List, Dict, Optional
from .base import BaseFetcher


class EuropePMCFetcher(BaseFetcher):
    """Europe PMC fetcher — covers PubMed and preprints."""

    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"

    def __init__(self, email: Optional[str] = None, delay: float = 0.3):
        self.email = email
        self.delay = delay
        self.session = requests.Session()
        if email:
            self.session.headers.update(
                {"User-Agent": f"FluBroadVoice/1.0 ({email})"}
            )

    @property
    def source_name(self) -> str:
        return "EuropePMC"

    # ── Search ────────────────────────────────────────────────────────────────
    def search(self, query: str, max_results: int = 50, **kwargs) -> List[str]:
        """Return a list of PMIDs (or DOIs for preprints)."""
        params = {
            "query": query,
            "format": "json",
            "pageSize": min(max_results, 1000),
            "resultType": "core",
        }
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/search", params=params, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            ids = []
            for item in data.get("resultList", {}).get("result", []):
                pid = item.get("pmid") or item.get("doi")
                if pid:
                    ids.append(str(pid))
            return ids
        except Exception as e:
            print(f"[EuropePMC] search failed: {e}")
            return []

    # ── Fetch details ─────────────────────────────────────────────────────────
    def fetch_details(self, ids: List[str]) -> List[Dict]:
        """
        Fetch article metadata for a list of PMIDs or DOIs.
        Uses the Europe PMC /search endpoint one ID at a time
        (no bulk POST needed for typical batch sizes).
        """
        articles = []
        for pid in ids:
            time.sleep(self.delay)
            # Decide query field: DOIs contain "/"
            if "/" in str(pid):
                query = f'DOI:"{pid}"'
            else:
                query = f"EXT_ID:{pid} AND SRC:MED"
            params = {
                "query": query,
                "format": "json",
                "pageSize": 1,
                "resultType": "core",
            }
            try:
                resp = self.session.get(
                    f"{self.BASE_URL}/search", params=params, timeout=15
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("resultList", {}).get("result", [])
                if not results:
                    continue
                item = results[0]
                articles.append({
                    "pmid":     item.get("pmid", ""),
                    "doi":      item.get("doi", ""),
                    "pmcid":    item.get("pmcid", ""),
                    "title":    item.get("title", ""),
                    "abstract": item.get("abstractText", ""),
                    "journal":  item.get("journalTitle", ""),
                    "year":     str(item.get("pubYear", "")),
                    "fulltext_available": False,
                    "fulltext_content":   None,
                    "fulltext_source":    None,
                })
            except Exception as e:
                print(f"[EuropePMC] fetch_details failed for {pid}: {e}")
                continue
        return articles