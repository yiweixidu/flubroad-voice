"""
PMC 全文检索与下载模块
支持：PMC ID 全文下载、XML 解析、文本提取
"""
import os
from typing import List, Dict, Optional
from pathlib import Path

from pyeuropepmc import SearchClient, FullTextClient, FullTextXMLParser
from pyeuropepmc.ftp_downloader import FTPDownloader
from pmc_downloader import PMCDownloader


class PMCFulltextFetcher:
    """PMC 全文获取器 - 支持 XML/PDF 下载与解析"""
    
    def __init__(self, email: str, api_key: Optional[str] = None, 
                 output_dir: str = "./data/pmc_cache"):
        self.email = email
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def search_by_pmcid(self, pmcid: str) -> Dict:
        """
        通过 PMCID 搜索文献详细信息
        使用 pyeuropepmc 的全文客户端 [citation:1]
        """
        with FullTextClient() as client:
            # 下载 XML 全文
            xml_path = client.download_xml_by_pmcid(pmcid, output_dir=self.output_dir)
            
            # 解析 XML 提取结构化内容
            with open(xml_path, 'r') as f:
                parser = FullTextXMLParser(f.read())
            
            # 提取各类内容
            return {
                "metadata": parser.extract_metadata(),
                "plaintext": parser.to_plaintext(),
                "markdown": parser.to_markdown(),
                "tables": parser.extract_tables(),
                "references": parser.extract_references(),
                "xml_path": str(xml_path)
            }
    
    def search_by_query(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        通过查询检索 PMC 全文文献
        使用 Europe PMC 高级搜索 API [citation:1]
        """
        from pyeuropepmc import QueryBuilder
        
        # 构建查询（仅限开放获取全文）
        qb = QueryBuilder()
        full_query = (qb
            .keyword(query)
            .and_()
            .keyword("open access", field="ACC")
            .build())
        
        with SearchClient() as client:
            # 搜索并自动解析结果
            papers = client.search_and_parse(
                query=full_query,
                pageSize=max_results,
                sort="cited desc"
            )
            
            # 对每篇文献获取全文
            results = []
            for paper in papers:
                pmcid = paper.get('pmcid')
                if pmcid:
                    try:
                        full_content = self.search_by_pmcid(pmcid)
                        results.append({
                            **paper,
                            "fulltext": full_content
                        })
                    except Exception as e:
                        print(f"Failed to get fulltext for {pmcid}: {e}")
                        results.append(paper)
            
            return results
    
    def batch_download(self, pmcids: List[str]) -> Dict:
        """
        批量下载 PMC 文献
        使用 FTP 批量下载提高效率 [citation:1]
        """
        ftp_downloader = FTPDownloader()
        results = ftp_downloader.bulk_download_and_extract(
            pmcids=pmcids,
            output_dir=str(self.output_dir / "bulk")
        )
        return results


class UnpaywallIntegration:
    """Unpaywall 集成 - 自动查找合法全文链接 [citation:5]"""
    
    def __init__(self, email: str):
        self.email = email
        # 设置认证
        from unpywall.utils import UnpywallCredentials
        UnpywallCredentials(email)
        
    def get_fulltext_url(self, doi: str) -> Optional[str]:
        """通过 DOI 获取开放获取全文 URL"""
        from unpywall import Unpywall
        
        try:
            # 获取 PDF 链接
            pdf_url = Unpywall.get_pdf_link(doi=doi)
            if pdf_url:
                return pdf_url
            
            # 备选：获取文档链接
            doc_url = Unpywall.get_doc_link(doi=doi)
            return doc_url
        except Exception as e:
            print(f"Unpaywall lookup failed for {doi}: {e}")
            return None
    
    def batch_check(self, dois: List[str]) -> Dict[str, Optional[str]]:
        """批量检查多个 DOI 的开放获取状态"""
        from unpywall import Unpywall
        
        results = Unpywall.doi(dois=dois, progress=True)
        return results.to_dict() if results is not None else {}
    
    def get_best_oa_location(self, doi: str) -> Dict:
        """
        获取最佳开放获取位置
        返回 PDF URL、许可证、版本等信息
        """
        from unpywall import Unpywall
        
        json_data = Unpywall.get_json(doi=doi)
        if json_data:
            best_oa = json_data.get('best_oa_location', {})
            return {
                "url": best_oa.get('url_for_pdf') or best_oa.get('url'),
                "version": best_oa.get('version'),
                "license": best_oa.get('license'),
                "evidence": best_oa.get('evidence')
            }
        return {}