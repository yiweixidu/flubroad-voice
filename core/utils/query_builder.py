# core/utils/query_builder.py
from typing import Optional

def build_pubmed_query(base_query: Optional[str] = None,
                       virus: str = "influenza",
                       include_antibody_genes: bool = True,
                       include_epitopes: bool = True,
                       include_platforms: bool = True) -> str:
    """
    根据多维度关键词构建PubMed查询字符串
    :param base_query: 用户提供的基础查询，若为空则自动生成
    :param virus: 病毒名称，如 influenza, SARS-CoV-2
    :param include_antibody_genes: 是否包含抗体基因/家族关键词
    :param include_epitopes: 是否包含结构域/表位关键词
    :param include_platforms: 是否包含技术平台关键词
    """
    # 核心主题
    core = '("broadly neutralizing antibodies"[Title/Abstract] OR "cross-reactive antibodies"[Title/Abstract])'
    
    # 病毒/抗原（若用户提供了base_query，则覆盖此部分）
    virus_part = f'("{virus}"[Title/Abstract])'
    
    # 可选维度
    parts = [virus_part, core]
    
    if include_antibody_genes:
        antibody_genes = '("IGHV1-69"[Title/Abstract] OR "IGHV3-*"[Title/Abstract] OR "IGHD3-3"[Title/Abstract] OR "VH1-69"[Title/Abstract])'
        parts.append(antibody_genes)
    
    if include_epitopes:
        epitopes = '("HA stem"[Title/Abstract] OR "fusion peptide"[Title/Abstract] OR "receptor binding domain"[Title/Abstract] OR RBD[Title/Abstract] OR "conserved epitope"[Title/Abstract])'
        parts.append(epitopes)
    
    if include_platforms:
        platforms = '("COBRA"[Title/Abstract] OR "deep mutational scanning"[Title/Abstract] OR mRNA[Title/Abstract] OR "structure-based design"[Title/Abstract] OR "germline targeting"[Title/Abstract])'
        parts.append(platforms)
    
    # 用 AND 连接
    query = " AND ".join(parts)
    
    # 如果提供了base_query，则将其作为额外条件（保留用户自定义）
    if base_query:
        query = f"({base_query}) AND ({query})"
    
    return query