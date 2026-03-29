# core/utils/query_builder.py
from datetime import datetime, timedelta

def build_pubmed_query(base_query: str, days_back: int = 3650) -> str:
    """
    构造 PubMed 查询字符串，自动添加 [Title/Abstract] 和日期过滤。
    base_query: 原始查询，不应包含字段限定符如 [Title/Abstract]
    """
    # 如果查询已经包含字段限定符，则不再添加
    if any(pat in base_query for pat in ['[Title/Abstract]', '[Title]', '[Abstract]', '[MeSH]']):
        query = base_query
    else:
        # 为整个查询添加 [Title/Abstract] 后缀，但保留括号和逻辑运算符
        # 方法：将整个查询字符串包裹在括号内，然后添加 [Title/Abstract]
        # 注意：PubMed 支持这种用法，例如 (term1 OR term2)[Title/Abstract]
        query = f'({base_query})[Title/Abstract]'

    # 添加日期过滤
    if days_back > 0:
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
        query = f'{query} AND ("{cutoff}"[Date - Publication] : "3000"[Date - Publication])'
    
    return query