"""
增强版文献检索器
整合：PubMed 摘要 + PMC 全文 + PDF 上传 + Unpaywall
"""
from typing import List, Dict, Optional, BinaryIO
from pathlib import Path

from core.retrieval.pubmed import PubMedFetcher
from core.retrieval.pmc_fulltext import PMCFulltextFetcher, UnpaywallIntegration
from core.retrieval.pdf_processor import PDFProcessor


class EnhancedLiteratureFetcher:
    """
    增强版文献检索器
    支持多源文献获取：PubMed、PMC 全文、PDF 上传、开放获取链接
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.pubmed_fetcher = PubMedFetcher(
            email=config["email"],
            api_key=config.get("pubmed_api_key")
        )
        self.pmc_fetcher = PMCFulltextFetcher(
            email=config["email"],
            api_key=config.get("pubmed_api_key"),
            output_dir=config.get("pmc_cache_dir", "./data/pmc_cache")
        )
        self.unpaywall = UnpaywallIntegration(email=config["email"])
        self.pdf_processor = PDFProcessor()
        
    def fetch_with_fulltext(self, query: str, max_papers: int = 50) -> List[Dict]:
        """
        检索文献并尽可能获取全文
        优先级：PMC 全文 > Unpaywall OA > PubMed 摘要
        """
        # 1. 基础 PubMed 检索
        pmids = self.pubmed_fetcher.search(query, max_results=max_papers)
        articles = self.pubmed_fetcher.fetch_details(pmids)
        
        # 2. 为每篇文献尝试获取全文
        enriched_articles = []
        
        for article in articles:
            enriched = article.copy()
            enriched["fulltext_available"] = False
            enriched["fulltext_content"] = None
            enriched["fulltext_source"] = None
            
            # 尝试获取 PMCID
            pmcid = article.get("pmcid")
            doi = article.get("doi")
            
            # 方法1：通过 PMC 获取全文
            if pmcid:
                try:
                    fulltext = self.pmc_fetcher.search_by_pmcid(pmcid)
                    if fulltext and fulltext.get("plaintext"):
                        enriched["fulltext_available"] = True
                        enriched["fulltext_content"] = fulltext["plaintext"]
                        enriched["fulltext_source"] = "pmc"
                        enriched["tables"] = fulltext.get("tables", [])
                        enriched["markdown"] = fulltext.get("markdown", "")
                except Exception as e:
                    print(f"PMC fulltext failed for {pmcid}: {e}")
            
            # 方法2：通过 Unpaywall 获取 OA 全文
            if not enriched["fulltext_available"] and doi:
                try:
                    oa_url = self.unpaywall.get_fulltext_url(doi)
                    if oa_url:
                        enriched["fulltext_available"] = True
                        enriched["fulltext_source"] = "unpaywall"
                        enriched["oa_url"] = oa_url
                        # 可选：下载并解析 OA PDF
                except Exception as e:
                    print(f"Unpaywall lookup failed for {doi}: {e}")
            
            enriched_articles.append(enriched)
        
        return enriched_articles
    
    def process_uploaded_paper(self, pdf_file: BinaryIO, filename: str) -> Dict:
        """
        处理用户上传的 PDF 文献
        提取全文并准备 RAG 索引
        """
        # 提取文本和分块
        chunks = self.pdf_processor.process_uploaded_pdf(pdf_file, filename)
        
        # 提取表格
        tables_md = self.pdf_processor.extract_tables_as_markdown(pdf_file)
        
        # 重新打开文件以重置指针（实际使用中需要重新获取文件）
        # 这里简化处理
        extracted = self.pdf_processor.extract_text_from_pdf(pdf_file)
        
        return {
            "source": filename,
            "type": "user_uploaded",
            "full_text": extracted["full_text"],
            "chunks": chunks,
            "tables_markdown": tables_md,
            "metadata": extracted["metadata"],
            "text_by_page": extracted["text_by_page"]
        }
    
    def enrich_with_unpaywall_metadata(self, articles: List[Dict]) -> List[Dict]:
        """
        使用 Unpaywall 丰富文献元数据
        包括：开放获取状态、最佳获取链接、许可证等
        """
        # 提取所有 DOI
        dois = [a.get("doi") for a in articles if a.get("doi")]
        
        if not dois:
            return articles
        
        # 批量查询 Unpaywall
        oa_results = self.unpaywall.batch_check(dois)
        
        # 丰富每篇文献
        for article in articles:
            doi = article.get("doi")
            if doi and doi in oa_results:
                oa_info = oa_results[doi]
                article["open_access"] = {
                    "is_oa": oa_info.get("is_oa", False),
                    "oa_status": oa_info.get("oa_status"),
                    "best_oa_url": oa_info.get("best_oa_location", {}).get("url"),
                    "license": oa_info.get("license")
                }
        
        return articles