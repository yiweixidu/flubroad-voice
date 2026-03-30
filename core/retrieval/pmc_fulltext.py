"""
core/retrieval/pmc_fulltext.py  —  complete rewrite
Dependencies: requests, pdfplumber (already in requirements)
No pyeuropepmc / pmc_downloader / unpywall needed.

Priority chain for each article:
  1. PMC full-text XML  (via Europe PMC REST API — free, no auth)
  2. Unpaywall OA PDF   (download + pdfplumber text extraction)
  3. Abstract only      (fallback, always available)
"""

import io
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import pdfplumber
import requests


# ── Europe PMC full-text ───────────────────────────────────────────────────────

class PMCFulltextFetcher:
    """
    Fetches article full text from Europe PMC's REST API.

    API endpoint (no auth required for open-access articles):
      GET https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML
    """

    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
    SEARCH_URL = f"{BASE_URL}/search"

    def __init__(self, email: str, delay: float = 0.5):
        self.email = email
        self.delay = delay          # polite crawl delay between requests
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"FluBroadVoice/1.0 ({email})"})

    # ── Main entry: enrich a single article dict ──────────────────────────────
    def enrich_article(self, article: Dict) -> Dict:
        """
        Try to fetch full text for `article`.
        Returns the article dict with fulltext_* fields populated.
        Modifies in-place and also returns the dict for convenience.
        """
        pmcid = article.get("pmcid", "")
        if pmcid:
            try:
                plaintext = self.fetch_fulltext_by_pmcid(pmcid)
                if plaintext:
                    article["fulltext_available"] = True
                    article["fulltext_content"]   = plaintext
                    article["fulltext_source"]    = "pmc"
                    return article
            except Exception as e:
                print(f"    [PMC] {pmcid} failed: {e}")
        return article

    # ── Fetch + parse PMC XML ─────────────────────────────────────────────────
    def fetch_fulltext_by_pmcid(self, pmcid: str) -> Optional[str]:
        """
        Download the NLM/JATS XML from Europe PMC and extract plain text.
        Returns None if the article is not available as open-access full text.
        """
        # Normalise: Europe PMC wants bare numeric ID or "PMCxxxxxxx"
        pmcid_clean = pmcid.replace("PMC", "")
        url = f"{self.BASE_URL}/PMC{pmcid_clean}/fullTextXML"
        time.sleep(self.delay)
        resp = self.session.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return self._xml_to_plaintext(resp.content)

    @staticmethod
    def _xml_to_plaintext(xml_bytes: bytes) -> str:
        """
        Extract readable text from JATS/NLM XML.
        Pulls text from <abstract>, <body>, and <sec> elements,
        skipping reference lists and figure captions.
        """
        SKIP_TAGS = {"ref-list", "fig", "table-wrap", "supplementary-material"}
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError:
            return ""

        parts: List[str] = []

        def walk(elem: ET.Element, depth: int = 0) -> None:
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag in SKIP_TAGS:
                return
            # Collect direct text
            if elem.text and elem.text.strip():
                parts.append(elem.text.strip())
            for child in elem:
                walk(child, depth + 1)
            if elem.tail and elem.tail.strip():
                parts.append(elem.tail.strip())

        # Focus on abstract + body; fall back to full document
        abstract = root.find(".//{*}abstract")
        body      = root.find(".//{*}body")
        targets   = [t for t in [abstract, body] if t is not None]
        if not targets:
            targets = [root]
        for target in targets:
            walk(target)

        return " ".join(parts)

    # ── Lookup PMCID from PubMed article (Europe PMC search) ─────────────────
    def lookup_pmcid(self, pmid: str) -> Optional[str]:
        """
        Ask Europe PMC for the PMCID corresponding to a PubMed ID.
        Useful if PubMed XML didn't include it.
        """
        params = {
            "query": f"EXT_ID:{pmid} AND SRC:MED",
            "format": "json",
            "pageSize": 1,
            "resultType": "core",
        }
        try:
            time.sleep(self.delay)
            resp = self.session.get(self.SEARCH_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("resultList", {}).get("result", [])
            if results and results[0].get("pmcid"):
                return results[0]["pmcid"]
        except Exception as e:
            print(f"    [PMC lookup] PMID {pmid}: {e}")
        return None


# ── Unpaywall OA PDF fetcher ───────────────────────────────────────────────────

class UnpaywallFetcher:
    """
    Uses the Unpaywall REST API (no library needed) to find an OA PDF URL,
    then downloads and extracts text with pdfplumber.
    """

    API_URL = "https://api.unpaywall.org/v2/{doi}"

    def __init__(self, email: str, delay: float = 0.5):
        self.email = email
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"FluBroadVoice/1.0 ({email})"})

    # ── Main entry ────────────────────────────────────────────────────────────
    def enrich_article(self, article: Dict) -> Dict:
        """
        Try to get full text via Unpaywall for `article`.
        Only called when PMC full text is unavailable.
        """
        doi = article.get("doi", "")
        if not doi:
            return article
        try:
            pdf_url = self.get_pdf_url(doi)
            if pdf_url:
                text = self.download_and_extract(pdf_url)
                if text and len(text) > 500:   # sanity: at least a paragraph
                    article["fulltext_available"] = True
                    article["fulltext_content"]   = text
                    article["fulltext_source"]    = "unpaywall_pdf"
                    article["oa_url"]             = pdf_url
        except Exception as e:
            print(f"    [Unpaywall] DOI {doi}: {e}")
        return article

    def get_oa_info(self, doi: str) -> Dict:
        """Return raw Unpaywall JSON for a DOI (useful for the OA lookup tab)."""
        time.sleep(self.delay)
        url = self.API_URL.format(doi=doi)
        resp = self.session.get(url, params={"email": self.email}, timeout=15)
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return resp.json()

    def get_pdf_url(self, doi: str) -> Optional[str]:
        """Return the best OA PDF URL for a DOI, or None."""
        data = self.get_oa_info(doi)
        if not data or not data.get("is_oa"):
            return None
        best = data.get("best_oa_location") or {}
        return best.get("url_for_pdf") or best.get("url")

    def download_and_extract(self, pdf_url: str, timeout: int = 30) -> str:
        """Download a PDF and extract its text with pdfplumber."""
        time.sleep(self.delay)
        resp = self.session.get(pdf_url, timeout=timeout, stream=True)
        resp.raise_for_status()
        # Check content type
        ct = resp.headers.get("Content-Type", "")
        if "pdf" not in ct.lower() and not pdf_url.lower().endswith(".pdf"):
            # Might be HTML — skip
            return ""
        pdf_bytes = resp.content
        parts: List[str] = []
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        parts.append(text)
        except Exception as e:
            print(f"    [pdfplumber] extraction failed: {e}")
        return "\n\n".join(parts)