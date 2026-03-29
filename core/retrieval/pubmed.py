import json
import os
from datetime import datetime
import time
from typing import List, Dict, Optional, Callable
from Bio import Entrez
import xml.etree.ElementTree as ET
from ..utils.query_builder import build_pubmed_query
import requests
from urllib.parse import quote

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
        full_query = build_pubmed_query(base_query=query, days_back=days_back)
        print(f"[PubMed] 检索式: {full_query}")
        handle = Entrez.esearch(
            db="pubmed",
            term=full_query,
            retmax=max_results,
            api_key=self.api_key if self.api_key else None
        )
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]

    def fetch_all(self, query: str, max_results: Optional[int] = None,
                  days_back: int = 3650, use_batch: bool = True,
                  checkpoint_file: str = "pubmed_progress.json") -> List[Dict]:
        """
        获取全部文献（支持超10,000篇的大规模检索）
        """
        if not use_batch:
            limit = max_results if max_results else 10000
            pmids = self.search(query, max_results=limit, days_back=days_back)
            return self.fetch_details(pmids)

        batch_fetcher = PubMedBatchFetcher(
            email=Entrez.email,
            api_key=self.api_key,
            tool=self.tool
        )

        def progress_callback(current, total):
            print(f"进度: {current}/{total} ({100 * current / total:.1f}%)")

        return batch_fetcher.search_all(
            query=query,
            max_results=max_results,
            days_back=days_back,
            checkpoint_file=checkpoint_file,
            progress_callback=progress_callback
        )

    def fetch_details(self, pmids: List[str]) -> List[Dict]:
        """获取文献详情，逐个获取并跳过无效PMID"""
        if not pmids:
            return []

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
                    time.sleep(0.1 if self.api_key else 0.34)
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
                    break
                except Exception as e:
                    print(f"Failed to fetch PMID {pmid} (attempt {attempt + 1}): {e}")
                    if attempt == 2:
                        print(f"Giving up on PMID {pmid}")
                    else:
                        wait = 2 ** attempt
                        time.sleep(wait)
        return all_articles

    def _parse_pubmed_xml(self, xml_text: str) -> List[Dict]:
        """解析PubMed XML（单篇或批量）"""
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


class PubMedBatchFetcher:
    """全量文献批量检索器（支持字段限定符自动添加）"""

    def __init__(self, email: str, api_key: Optional[str] = None,
                 tool: str = "FluBroadVoice"):
        Entrez.email = email
        self.api_key = api_key
        Entrez.api_key = api_key
        self.tool = tool
        self.batch_size = 500

    def _qualify_query(self, raw_query: str, field: str = "Title/Abstract") -> str:
        """将原始查询中的每个检索词（短语）加上字段限定符"""
        import re
        def replacer(match):
            word = match.group(0)
            if word.upper() in ('AND', 'OR', 'NOT'):
                return word
            if f'[{field}]' in word:
                return word
            return f'{word}[{field}]'
        pattern = r'"[^"]+"|\b\w+\b'
        return re.sub(pattern, replacer, raw_query)

    def get_total_count_requests(self, raw_query: str, days_back: int = 3650) -> int:
        """使用 requests 直接调用 PubMed API"""
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": raw_query,
            "retmax": 0,
            "tool": self.tool,
            "email": Entrez.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        # 注意：不要自己编码，让 requests 处理
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        # 解析 XML 响应
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        count_elem = root.find(".//Count")
        return int(count_elem.text) if count_elem is not None else 0

    def get_total_count(self, raw_query: str, days_back: int = 3650) -> int:
        """获取查询结果总数，使用 mindate/maxdate 参数"""
        qualified_query = self._qualify_query(raw_query)
        
        # 计算日期范围
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        mindate = start_date.strftime("%Y/%m/%d")
        maxdate = end_date.strftime("%Y/%m/%d")
        
        print(f"[BatchFetcher] Count query term: {qualified_query}")
        print(f"[BatchFetcher] Date range: {mindate} to {maxdate}")
        
        handle = Entrez.esearch(
            db="pubmed",
            term=qualified_query,
            retmax=0,
            mindate=mindate,
            maxdate=maxdate,
            datetype='pdat',   # publication date
            api_key=self.api_key
        )
        record = Entrez.read(handle)
        handle.close()
        return int(record["Count"])

    def search_all(self, raw_query: str, max_results: Optional[int] = None,
                   days_back: int = 3650, checkpoint_file: str = "pubmed_progress.json",
                   progress_callback: Optional[Callable] = None) -> List[Dict]:
        """全量检索所有文献（支持断点续传），使用 mindate/maxdate 参数"""
        qualified_query = self._qualify_query(raw_query)
        
        # 计算日期范围
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        mindate = start_date.strftime("%Y/%m/%d")
        maxdate = end_date.strftime("%Y/%m/%d")
        
        print(f"[BatchFetcher] 检索式: {qualified_query}")
        print(f"[BatchFetcher] 日期范围: {mindate} 至 {maxdate}")
        
        # 首次搜索，获取总数和WebEnv（使用日期参数）
        handle = Entrez.esearch(
            db="pubmed",
            term=qualified_query,
            retmax=0,
            usehistory="y",
            mindate=mindate,
            maxdate=maxdate,
            datetype='pdat',
            api_key=self.api_key
        )
        search_record = Entrez.read(handle)
        handle.close()
        
        total_count = int(search_record["Count"])
        webenv = search_record["WebEnv"]
        query_key = search_record["QueryKey"]
        
        print(f"[BatchFetcher] 共找到 {total_count} 篇文献")
        
        if max_results and max_results < total_count:
            total_count = max_results
            print(f"[BatchFetcher] 限制获取数量为 {total_count} 篇")
        
        # 加载断点进度（与之前相同）
        processed = 0
        articles = []
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            processed = checkpoint.get('processed', 0)
            articles = checkpoint.get('articles', [])
            print(f"[BatchFetcher] 从断点恢复，已完成 {processed}/{total_count} 篇")
        
        # 分批获取（与之前相同）
        for start in range(processed, total_count, self.batch_size):
            current_batch_size = min(self.batch_size, total_count - start)
            try:
                fetch_handle = Entrez.efetch(
                    db="pubmed",
                    rettype="xml",
                    retmode="text",
                    retstart=start,
                    retmax=current_batch_size,
                    webenv=webenv,
                    query_key=query_key,
                    api_key=self.api_key
                )
                batch_data = Entrez.read(fetch_handle)
                fetch_handle.close()
                
                batch_articles = self._parse_pubmed_batch(batch_data)
                articles.extend(batch_articles)
                processed = start + len(batch_articles)
                
                if progress_callback:
                    progress_callback(processed, total_count)
                
                print(f"[BatchFetcher] 进度: {processed}/{total_count} ({100*processed/total_count:.1f}%)")
                
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'processed': processed,
                        'articles': articles,
                        'query': qualified_query,
                        'last_update': datetime.now().isoformat()
                    }, f, indent=2)
                
                time.sleep(0.1 if self.api_key else 0.34)
                
            except Exception as e:
                print(f"[BatchFetcher] 错误发生在 start={start}: {e}")
                print(f"[BatchFetcher] 进度已保存至 {checkpoint_file}，可重新运行继续")
                raise
        
        if os.path.exists(checkpoint_file) and processed >= total_count:
            os.remove(checkpoint_file)
            print("[BatchFetcher] 所有文献获取完成，进度文件已清理")
        
        return articles

    def search_all_requests(self, raw_query: str, max_results: Optional[int] = None,
                        days_back: int = 3650, checkpoint_file: str = "pubmed_progress.json",
                        progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        全量检索所有文献（完全使用 requests，避免 Biopython 编码问题）
        支持断点续传、进度回调
        """
        # 1. 构建查询字符串（包含日期过滤）
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        date_filter = f' AND ("{start_date.strftime("%Y/%m/%d")}"[Date - Publication] : "{end_date.strftime("%Y/%m/%d")}"[Date - Publication])'
        full_query = f'({raw_query}){date_filter}' if days_back else raw_query
        
        print(f"[Requests] 检索式: {full_query}")
        
        # 2. 第一次 esearch 获取总数和 WebEnv/QueryKey
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": full_query,
            "retmax": 0,
            "usehistory": "y",
            "tool": self.tool,
            "email": Entrez.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        
        resp = requests.get(esearch_url, params=params)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        
        total_count = int(root.findtext("Count", "0"))
        webenv = root.findtext("WebEnv", "")
        query_key = root.findtext("QueryKey", "")
        
        print(f"[Requests] 共找到 {total_count} 篇文献")
        if max_results and max_results < total_count:
            total_count = max_results
            print(f"[Requests] 限制获取数量为 {total_count} 篇")
        
        # 3. 断点续传逻辑
        processed = 0
        articles = []
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            processed = checkpoint.get('processed', 0)
            articles = checkpoint.get('articles', [])
            print(f"[Requests] 从断点恢复，已完成 {processed}/{total_count} 篇")

        # 确保断点文件所在目录存在
        checkpoint_dir = os.path.dirname(checkpoint_file)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 4. 分批 efetch
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        batch_size = self.batch_size
        
        for start in range(processed, total_count, batch_size):
            current_batch_size = min(batch_size, total_count - start)
            efetch_params = {
                "db": "pubmed",
                "rettype": "xml",
                "retmode": "xml",
                "retstart": start,
                "retmax": current_batch_size,
                "query_key": query_key,
                "WebEnv": webenv,
                "tool": self.tool,
                "email": Entrez.email,
            }
            if self.api_key:
                efetch_params["api_key"] = self.api_key
            
            try:
                resp = requests.get(efetch_url, params=efetch_params)
                resp.raise_for_status()
                batch_articles = self._parse_efetch_xml(resp.content)
                articles.extend(batch_articles)
                processed = start + len(batch_articles)
                
                if progress_callback:
                    progress_callback(processed, total_count)
                print(f"[Requests] 进度: {processed}/{total_count} ({100*processed/total_count:.1f}%)")
                
                # 保存断点
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'processed': processed,
                        'articles': articles,
                        'query': full_query,
                        'last_update': datetime.now().isoformat()
                    }, f, indent=2)
                
                time.sleep(0.1 if self.api_key else 0.34)
            except Exception as e:
                print(f"[Requests] 错误发生在 start={start}: {e}")
                print(f"[Requests] 进度已保存至 {checkpoint_file}，可重新运行继续")
                raise
        
        # 清理断点文件
        if os.path.exists(checkpoint_file) and processed >= total_count:
            os.remove(checkpoint_file)
            print("[Requests] 所有文献获取完成，进度文件已清理")
        
        return articles

    def _parse_efetch_xml(self, xml_bytes: bytes) -> List[Dict]:
        """解析 efetch 返回的 XML 批量结果"""
        root = ET.fromstring(xml_bytes)
        articles = []
        for article_elem in root.findall(".//PubmedArticle"):
            try:
                pmid = article_elem.findtext(".//PMID", "")
                title = article_elem.findtext(".//ArticleTitle", "")
                # 摘要可能分多段
                abstract_parts = article_elem.findall(".//AbstractText")
                abstract = " ".join(part.text for part in abstract_parts if part.text)
                year_elem = article_elem.find(".//PubDate/Year")
                year = year_elem.text if year_elem is not None else ""
                journal_elem = article_elem.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else ""
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "year": year,
                })
            except Exception as e:
                print(f"解析单篇出错: {e}")
                continue
        return articles

    def _parse_pubmed_batch(self, batch_data) -> List[Dict]:
        """批量解析PubMed XML（来自efetch的批量结果）"""
        articles = []

        if 'PubmedArticle' in batch_data:
            pubmed_articles = batch_data['PubmedArticle']
        elif 'PubmedBookArticle' in batch_data:
            pubmed_articles = batch_data['PubmedBookArticle']
        else:
            return articles

        for article in pubmed_articles:
            try:
                medline = article.get('MedlineCitation', {})
                pmid = str(medline.get('PMID', ''))

                article_data = medline.get('Article', {})
                title = article_data.get('ArticleTitle') or ''

                # 摘要处理
                abstract = ""
                if 'Abstract' in article_data:
                    abstract_parts = article_data['Abstract'].get('AbstractText', [])
                    if isinstance(abstract_parts, list):
                        abstract = " ".join(str(part) for part in abstract_parts if part)
                    else:
                        abstract = str(abstract_parts) if abstract_parts else ""

                # 期刊
                journal = article_data.get('Journal', {})
                journal_title = journal.get('Title') or ''

                # 发表年份
                pub_date = journal.get('JournalIssue', {}).get('PubDate', {})
                year = pub_date.get('Year', '')
                if not year and 'MedlineDate' in pub_date:
                    medline_date = pub_date['MedlineDate']
                    import re
                    year_match = re.search(r'\b(19|20)\d{2}\b', str(medline_date))
                    if year_match:
                        year = year_match.group()

                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal_title,
                    "year": year,
                })
            except Exception as e:
                print(f"解析单篇文献时出错: {e}")
                continue

        return articles