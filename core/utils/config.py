# core/utils/config.py (修改)
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")
    EMAIL = os.getenv("EMAIL")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default")
    PERSIST_DIR = os.getenv("PERSIST_DIR", "./data/vector_db")
    PPT_TEMPLATE = os.getenv("PPT_TEMPLATE")
    
    # 新增
    ENABLE_BIORXIV = os.getenv("ENABLE_BIORXIV", "true").lower() == "true"
    # ENABLE_EUROPE_PMC = os.getenv("ENABLE_EUROPE_PMC", "false").lower() == "true"