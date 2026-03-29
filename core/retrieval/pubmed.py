import time
from typing import List, Dict, Optional
from Bio import Entrez
import requests
import xml.etree.ElementTree as ET
from .base import BaseFetcher
from ..utils.query_builder import build_pubmed_query

class PubMedFetcher:
    """PubMed文献检索器（开源核心）"""

    def __init__(self, email: str, api_key: Optional[str] = None, tool: str = "FluBroadVoice"):
        Entrez.email = email
        self.api_key = api_key
        self.tool = tool
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    @property
    def source_name(self) -> str:
        return "PubMed"

    def search(self, query: str, max_results: int = 50, days_back: int = 3650, **kwargs) -> List[str]:
        """返回PMID列表"""
        #if days_back:
        #    from datetime import datetime, timedelta
        #    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
        #    query += f' AND ("{cutoff}"[Date - Publication] : "3000"[Date - Publication])'

        full_query = build_pubmed_query(query, days_back, **kwargs)
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, api_key=self.api_key if self.api_key else None)
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]

    def fetch_details(self, pmids: List[str]) -> List[Dict]:
        """获取文献详情，逐个获取并跳过无效PMID"""
        if not pmids:
            return []

        # 清理并验证 PMID
        clean_pmids = []
        for p in pmids:
            p = str(p).strip()
            if p.isdigit():
                clean_pmids.append(p)
            else:
                print(f"Warning: Skipping invalid PMID: {p}")

        if not clean_pmids:
            return []

        all_articles = []
        for pmid in clean_pmids:
            for attempt in range(3):
                try:
                    time.sleep(0.34)  # 遵守速率限制
                    # 构建请求参数，如果 api_key 为 None 则省略
                    kwargs = {
                        "db": "pubmed",
                        "id": pmid,
                        "retmode": "xml"
                    }
                    if self.api_key:
                        kwargs["api_key"] = self.api_key
                    handle = Entrez.efetch(**kwargs)
                    xml_data = handle.read()
                    handle.close()
                    articles = self._parse_pubmed_xml(xml_data)
                    all_articles.extend(articles)
                    break  # 成功则跳出重试循环
                except Exception as e:
                    print(f"Failed to fetch PMID {pmid} (attempt {attempt+1}): {e}")
                    if attempt == 2:
                        print(f"Giving up on PMID {pmid}")
                    else:
                        wait = 2 ** attempt
                        time.sleep(wait)
        return all_articles

    def _parse_pubmed_xml(self, xml_text: str) -> List[Dict]:
        """解析PubMed XML"""
        articles = []
        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            try:
                pmid = article.find(".//PMID").text
                title = article.find(".//ArticleTitle").text
                abstract = " ".join([e.text for e in article.findall(".//AbstractText") if e.text])
                year = article.find(".//PubDate/Year")
                year = year.text if year is not None else ""
                journal = article.find(".//Journal/Title")
                journal = journal.text if journal is not None else ""
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "year": year,
                })
            except Exception:
                continue
        return articles