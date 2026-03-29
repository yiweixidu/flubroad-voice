import json
from typing import List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    ChatOllama = None

from domain.virology.prompts.pmrc_templates import get_enhanced_template

# 可选 Pydantic 导入，若安装则用，否则跳过
try:
    from domain.virology.schemas.antibody_schema import AntibodyInfo, AntibodyList
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False


class NarrativeGenerator:
    def __init__(self, model: str = "llama3.2:3b", temperature: float = 0.1, llm_type: str = "ollama"):
        """
        llm_type: "openai" 或 "ollama"
        model: 对于 openai 是模型名（如 gpt-3.5-turbo）；对于 ollama 是模型名（如 llama3.2:3b）
        """
        self.llm_type = llm_type
        if llm_type == "openai":
            if ChatOpenAI is None:
                raise ImportError("Please install langchain-openai: pip install langchain-openai")
            self.llm = ChatOpenAI(model=model, temperature=temperature)
        elif llm_type == "ollama":
            if ChatOllama is None:
                raise ImportError("Please install langchain-community: pip install langchain-community")
            self.llm = ChatOllama(model=model, temperature=temperature)
        else:
            raise ValueError(f"Unknown llm_type: {llm_type}")

    def generate_review(self, context: str, topic: str = "flu_bnabs") -> str:
        """生成增强版综述"""
        template = get_enhanced_template(topic)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a senior virologist. Write a structured literature review."),
            ("human", f"Use the following framework:\n{template}\n\nLiterature context:\n{context}")
        ])
        chain = prompt | self.llm
        return chain.invoke({}).content

    def extract_antibodies(self, text: str, schema: Dict) -> List[Dict]:
        # Convert schema to JSON string and escape braces for LangChain
        schema_str = json.dumps(schema, indent=2)
        escaped_schema = schema_str.replace("{", "{{").replace("}", "}}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in virology and antibody research."),
            ("human", f"""Extract all broadly neutralizing antibodies mentioned in the text below.
    Return a JSON object with key "antibodies" containing a list of objects.
    Each object must follow this schema:
    {escaped_schema}

    Text:
    {text}

    Return only valid JSON.""")
        ])
        chain = prompt | self.llm
        response = chain.invoke({})
        content = response.content if hasattr(response, 'content') else str(response)
        try:
            data = json.loads(content)
            return data.get("antibodies", [])
        except Exception as e:
            print(f"Extraction failed: {e}")
            return []