# core/retrieval/base.py
from abc import ABC, abstractmethod
from typing import List, Dict

class BaseFetcher(ABC):
    """所有文献检索器的基类"""
    
    @abstractmethod
    def search(self, query: str, max_results: int = 50, **kwargs) -> List[str]:
        """返回文献唯一标识符列表（如PMID, DOI）"""
        pass
    
    @abstractmethod
    def fetch_details(self, ids: List[str]) -> List[Dict]:
        """根据标识符获取文献详情（标题、摘要、元数据）"""
        pass
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """返回数据源名称，如'PubMed', 'bioRxiv'"""
        pass