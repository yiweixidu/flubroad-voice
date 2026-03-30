"""
core/retrieval/pubmed.py  —  enhanced
Key change: _parse_pubmed_xml / _parse_efetch_xml now extract DOI and PMCID
from <ArticleIdList> so downstream fulltext fetching can use them.
"""

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import requests
from Bio import Entrez

from ..utils.query_builder import build_pubmed_query


class PubMedFetcher:
    """PubMed literature fetcher."""

    def __init__(self, email: str, api_key: Optional[str] = None,
                 tool: str = "FluBroadVoice"):
        Entrez.email = email
        self.api_key = api_key
        self.tool = tool

    @property
    def source_name(self) -> str:
        return "PubMed"

    def search(self, query: str, max_results: int = 200,
               days_back: int = 3650, **kwargs) -> List[str]:
        full_query = build_pubmed_query(base_query=query, days_back=days_back)
        print(f"[PubMed] Query: {full_query}")
        handle = Entrez.esearch(
            db="pubmed", term=full_query, retmax=max_results,
            api_key=self.api_key or None,
        )
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]

    def fetch_details(self, pmids: List[str]) -> List[Dict]:
        """Fetch abstracts + DOI + PMCID for a list of PMIDs."""
        if not pmids:
            return []
        clean = [str(p).strip() for p in pmids if str(p).strip().isdigit()]
        articles = []
        for pmid in clean:
            for attempt in range(3):
                try:
                    time.sleep(0.11 if self.api_key else 0.35)
                    handle = Entrez.efetch(
                        db="pubmed", id=pmid, retmode="xml",
                        api_key=self.api_key or None,
                    )
                    xml_data = handle.read()
                    handle.close()
                    articles.extend(self._parse_pubmed_xml(xml_data))
                    break
                except Exception as e:
                    print(f"  PMID {pmid} attempt {attempt+1} failed: {e}")
                    if attempt == 2:
                        print(f"  Giving up on PMID {pmid}")
                    else:
                        time.sleep(2 ** attempt)
        return articles

    def fetch_all(self, query: str, max_results: Optional[int] = None,
                  days_back: int = 3650, use_batch: bool = True,
                  checkpoint_file: str = "data/checkpoints/pubmed_progress.json",
                  ) -> List[Dict]:
        """Large-scale batch retrieval with checkpoint support."""
        fetcher = PubMedBatchFetcher(
            email=Entrez.email,
            api_key=self.api_key,
            tool=self.tool,
        )
        return fetcher.search_all_requests(
            raw_query=query,
            max_results=max_results,
            days_back=days_back,
            checkpoint_file=checkpoint_file,
            progress_callback=lambda c, t: print(
                f"  [{c}/{t}] {100*c/t:.1f}%"
            ),
        )

    # ── XML parsers ────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_ids(article_elem: ET.Element) -> Dict[str, str]:
        """Pull pmid, doi, pmcid from <ArticleIdList>."""
        ids: Dict[str, str] = {}
        for aid in article_elem.findall(".//ArticleId"):
            id_type = aid.get("IdType", "").lower()
            text = (aid.text or "").strip()
            if id_type == "pubmed" and text:
                ids["pmid"] = text
            elif id_type == "doi" and text:
                ids["doi"] = text
            elif id_type == "pmc" and text:
                # Normalise: store as "PMC12345678"
                ids["pmcid"] = text if text.startswith("PMC") else f"PMC{text}"
        return ids

    def _parse_pubmed_xml(self, xml_text) -> List[Dict]:
        """Parse PubMed efetch XML (single or batch)."""
        articles = []
        if isinstance(xml_text, bytes):
            xml_text = xml_text.decode("utf-8", errors="replace")
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            print(f"  XML parse error: {e}")
            return articles

        for art in root.findall(".//PubmedArticle"):
            try:
                # IDs
                extra = self._extract_ids(art)
                pmid_elem = art.find(".//PMID")
                pmid = extra.get("pmid") or (pmid_elem.text if pmid_elem is not None else "")

                title_elem = art.find(".//ArticleTitle")
                title = "".join(title_elem.itertext()) if title_elem is not None else ""

                abstract = " ".join(
                    e.text for e in art.findall(".//AbstractText") if e.text
                )

                year_elem = art.find(".//PubDate/Year")
                year = year_elem.text if year_elem is not None else ""

                journal_elem = art.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else ""

                articles.append({
                    "pmid":   pmid,
                    "doi":    extra.get("doi", ""),
                    "pmcid":  extra.get("pmcid", ""),
                    "title":  title,
                    "abstract": abstract,
                    "journal":  journal,
                    "year":     year,
                    "fulltext_available": False,
                    "fulltext_content":   None,
                    "fulltext_source":    None,
                })
            except Exception as exc:
                print(f"  Parse error for one article: {exc}")
                continue
        return articles


class PubMedBatchFetcher:
    """Large-scale batch fetcher using the E-utilities REST API directly."""

    def __init__(self, email: str, api_key: Optional[str] = None,
                 tool: str = "FluBroadVoice"):
        Entrez.email = email
        self.api_key = api_key
        self.tool = tool
        self.batch_size = 500

    # ── Public API ─────────────────────────────────────────────────────────────
    def search_all_requests(
        self,
        raw_query: str,
        max_results: Optional[int] = None,
        days_back: int = 3650,
        checkpoint_file: str = "data/checkpoints/pubmed_progress.json",
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """Batch-fetch all results with checkpoint/resume support."""
        # Build date-filtered query
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        date_filter = (
            f' AND ("{start_date.strftime("%Y/%m/%d")}"[Date - Publication] '
            f': "{end_date.strftime("%Y/%m/%d")}"[Date - Publication])'
        )
        full_query = f"({raw_query}){date_filter}" if days_back else raw_query
        print(f"[BatchFetcher] Query: {full_query}")

        # Initial esearch for total count + WebEnv
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params: Dict = {
            "db": "pubmed", "term": full_query,
            "retmax": 0, "usehistory": "y",
            "tool": self.tool, "email": Entrez.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        resp = requests.get(esearch_url, params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        total_count = int(root.findtext("Count", "0"))
        webenv     = root.findtext("WebEnv", "")
        query_key  = root.findtext("QueryKey", "")
        print(f"[BatchFetcher] Total: {total_count} articles")

        if max_results and max_results < total_count:
            total_count = max_results
            print(f"[BatchFetcher] Capped at {total_count}")

        # Resume from checkpoint
        processed = 0
        articles: List[Dict] = []
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file) as f:
                ckpt = json.load(f)
            processed = ckpt.get("processed", 0)
            articles  = ckpt.get("articles", [])
            print(f"[BatchFetcher] Resuming from {processed}/{total_count}")

        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        for start in range(processed, total_count, self.batch_size):
            batch_size = min(self.batch_size, total_count - start)
            ep: Dict = {
                "db": "pubmed", "rettype": "xml", "retmode": "xml",
                "retstart": start, "retmax": batch_size,
                "query_key": query_key, "WebEnv": webenv,
                "tool": self.tool, "email": Entrez.email,
            }
            if self.api_key:
                ep["api_key"] = self.api_key
            try:
                r = requests.get(efetch_url, params=ep, timeout=60)
                r.raise_for_status()
                batch = self._parse_efetch_xml(r.content)
                articles.extend(batch)
                processed = start + len(batch)
                if progress_callback:
                    progress_callback(processed, total_count)
                # Save checkpoint
                with open(checkpoint_file, "w") as f:
                    json.dump({
                        "processed": processed,
                        "articles": articles,
                        "query": full_query,
                        "last_update": datetime.now().isoformat(),
                    }, f, indent=2)
                time.sleep(0.11 if self.api_key else 0.35)
            except Exception as exc:
                print(f"[BatchFetcher] Error at start={start}: {exc}")
                print(f"[BatchFetcher] Progress saved to {checkpoint_file}")
                raise

        if os.path.exists(checkpoint_file) and processed >= total_count:
            os.remove(checkpoint_file)
            print("[BatchFetcher] Done. Checkpoint cleaned up.")
        return articles

    # ── Parser ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_ids(article_elem: ET.Element) -> Dict[str, str]:
        ids: Dict[str, str] = {}
        for aid in article_elem.findall(".//ArticleId"):
            id_type = aid.get("IdType", "").lower()
            text = (aid.text or "").strip()
            if id_type == "pubmed" and text:
                ids["pmid"] = text
            elif id_type == "doi" and text:
                ids["doi"] = text
            elif id_type == "pmc" and text:
                ids["pmcid"] = text if text.startswith("PMC") else f"PMC{text}"
        return ids

    def _parse_efetch_xml(self, xml_bytes: bytes) -> List[Dict]:
        articles = []
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as e:
            print(f"  XML parse error: {e}")
            return articles

        for art in root.findall(".//PubmedArticle"):
            try:
                extra = self._extract_ids(art)
                pmid_elem = art.find(".//PMID")
                pmid = extra.get("pmid") or (
                    pmid_elem.text if pmid_elem is not None else ""
                )
                title_elem = art.find(".//ArticleTitle")
                title = "".join(title_elem.itertext()) if title_elem is not None else ""
                abstract = " ".join(
                    e.text for e in art.findall(".//AbstractText") if e.text
                )
                year_elem = art.find(".//PubDate/Year")
                year = year_elem.text if year_elem is not None else ""
                journal_elem = art.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else ""

                articles.append({
                    "pmid":   pmid,
                    "doi":    extra.get("doi", ""),
                    "pmcid":  extra.get("pmcid", ""),
                    "title":  title,
                    "abstract": abstract,
                    "journal":  journal,
                    "year":     year,
                    "fulltext_available": False,
                    "fulltext_content":   None,
                    "fulltext_source":    None,
                })
            except Exception as exc:
                print(f"  Parse error: {exc}")
                continue
        return articles