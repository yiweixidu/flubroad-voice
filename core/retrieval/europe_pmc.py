# core/retrieval/europe_pmc.py
import requests
from typing import List, Dict
from .base import BaseFetcher

class EuropePMCFetcher(BaseFetcher):
    """Europe PMC检索器（支持PubMed、预印本）"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest"

    @property
    def source_name(self) -> str:
        return "EuropePMC"

    def search(self, query: str, max_results: int = 50, **kwargs) -> List[str]:
        """
        返回文献ID列表（PMIDs或DOI）
        """
        params = {
            "query": query,
            "format": "json",
            "pageSize": max_results,
            "resultType": "core"  # 包含PubMed和预印本
        }
        url = f"{self.base_url}/search"
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            ids = []
            for item in data.get("resultList", {}).get("result", []):
                # 优先使用pmid，若无则用doi
                pid = item.get("pmid") or item.get("doi")
                if pid:
                    ids.append(pid)
            return ids
        except Exception as e:
            print(f"EuropePMC search failed: {e}")
            return []

    def fetch_details(self, ids: List[str]) -> List[Dict]:
        """批量获取文献详情"""
        articles = []
        # 实际Europe PMC支持批量检索，这里简化处理
        for pid in ids:
            # 构造检索单个文献的请求（或使用POST批量）
            # 此处省略详细实现，可参考其API文档
            pass
        return articles