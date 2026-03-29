# core/retrieval/biorxiv.py
import requests
import time
from typing import List, Dict, Optional
from .base import BaseFetcher

class BioRxivFetcher(BaseFetcher):
    """bioRxiv预印本检索器"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.biorxiv.org"

    @property
    def source_name(self) -> str:
        return "bioRxiv"

    def search(self, query: str, max_results: int = 50, **kwargs) -> List[str]:
        """
        返回DOI列表
        bioRxiv API: /details/biorxiv/{query}/{cursor}/{format}
        这里简化为关键词检索，实际需处理分页
        """
        # 注意：bioRxiv API不支持复杂布尔查询，此处做简单关键词替换
        # 真实场景需使用其高级搜索或使用Europe PMC统一检索
        # 为演示，我们只取前max_results条
        params = {
            "cursor": 0,
            "format": "json"
        }
        # 构建URL
        url = f"{self.base_url}/details/biorxiv/{query}/0/json"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            # 提取DOI列表
            dois = []
            for item in data.get("collection", [])[:max_results]:
                doi = item.get("doi")
                if doi:
                    dois.append(doi)
            return dois
        except Exception as e:
            print(f"BioRxiv search failed: {e}")
            return []

    def fetch_details(self, dois: List[str]) -> List[Dict]:
        """根据DOI获取文献详情"""
        articles = []
        for doi in dois:
            # 为避免请求过快，适当延迟
            time.sleep(0.5)
            url = f"{self.base_url}/details/biorxiv/{doi}/json"
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                items = data.get("collection", [])
                if items:
                    item = items[0]
                    articles.append({
                        "doi": doi,
                        "title": item.get("title", ""),
                        "abstract": item.get("abstract", ""),
                        "journal": "bioRxiv",
                        "year": item.get("date", "")[:4] if item.get("date") else "",
                        "source": "bioRxiv"
                    })
            except Exception as e:
                print(f"Failed to fetch details for {doi}: {e}")
                continue
        return articles