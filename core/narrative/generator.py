"""
core/narrative/generator.py  —  fixed
Root cause of 404: if OPENAI_BASE_URL=https://api.anthropic.com/v1 is set in
the environment (from a previous test), langchain-openai silently uses it and
sends requests to Anthropic's API, which returns a 'not_found_error' for
'gpt-4o-mini'.  Fix: always pass openai_api_base explicitly so the env
variable cannot override it.
"""

import json
import os
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    ChatOllama = None

from domain.virology.prompts.pmrc_templates import get_enhanced_template

try:
    from domain.virology.schemas.antibody_schema import AntibodyInfo, AntibodyList
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

OPENAI_BASE_URL = "https://api.openai.com/v1"

# ── Per-section writing instructions ─────────────────────────────────────────
SECTION_INSTRUCTIONS = {
    "problem": (
        "Write the 'Problem' section of a virology literature review.\n"
        "- Describe the clinical challenge: antigenic drift, vaccine mismatch, pandemic risk.\n"
        "- Quantify the problem with specific statistics from the provided context.\n"
        "- Every factual claim MUST include a PMID in parentheses, e.g. (PMID: 34607819).\n"
        "- Length: 150-200 words. Do NOT include a section header in your output."
    ),
    "motivation": (
        "Write the 'Motivation' section of a virology literature review.\n"
        "- Explain WHY targeting conserved epitopes (HA stem, neuraminidase active site, "
        "fusion peptide) is a promising strategy.\n"
        "- Reference structural or immunological evidence from the context.\n"
        "- CRITICALLY compare this strategy to traditional strain-specific vaccination: "
        "what are the trade-offs?\n"
        "- Every factual claim MUST include a PMID.\n"
        "- Length: 150-200 words. Do NOT include a section header."
    ),
    "results": (
        "Write the 'Key Results' section of a virology literature review.\n"
        "- List and COMPARE at least 4-5 specific broadly neutralizing antibodies from "
        "the context.\n"
        "- For each antibody include: name, target epitope, IGHV gene usage, "
        "neutralization spectrum (which subtypes), IC50 if available, clinical stage.\n"
        "- CRITICALLY compare breadth and potency — which is broader? which is more potent?\n"
        "- Synthesize patterns across antibodies; do NOT just list them.\n"
        "- Every antibody claim MUST include a PMID.\n"
        "- Length: 300-400 words. Do NOT include a section header."
    ),
    "mechanisms": (
        "Write the 'Mechanisms of Action' section of a virology literature review.\n"
        "- Describe mechanisms BEYOND direct neutralization: Fc effector functions, ADCC, "
        "ADCP, complement.\n"
        "- Cite specific studies that tested these mechanisms (e.g. Fc-knockout experiments).\n"
        "- Note any controversy or inconsistency in the field.\n"
        "- Every factual claim MUST include a PMID.\n"
        "- Length: 150-200 words. Do NOT include a section header."
    ),
    "challenges": (
        "Write the 'Technical Challenges' section of a virology literature review.\n"
        "- Identify at least 3 specific challenges: immunodominance of head epitopes, "
        "low stem accessibility, germline gene requirements, escape mutations, manufacturing.\n"
        "- For each challenge, cite evidence AND describe current approaches to overcome it.\n"
        "- Be critical: which challenges remain unsolved?\n"
        "- Every factual claim MUST include a PMID.\n"
        "- Length: 200-250 words. Do NOT include a section header."
    ),
    "future": (
        "Write the 'Future Directions' section of a virology literature review.\n"
        "- Provide 3-4 SPECIFIC, forward-looking suggestions NOT directly quoted from papers.\n"
        "- Each suggestion must be justified with a brief mechanistic rationale.\n"
        "- Example: 'Combining deep mutational scanning of the HA stem with structure-guided "
        "mRNA-LNP immunogen design could pre-empt escape pathways before they emerge in nature.'\n"
        "- Do NOT use vague language like 'further research is needed'.\n"
        "- Length: 200-250 words. Do NOT include a section header."
    ),
}


class NarrativeGenerator:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        llm_type: str = "openai",
    ):
        self.llm_type = llm_type

        if llm_type == "openai":
            if ChatOpenAI is None:
                raise ImportError("Run: pip install langchain-openai")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set. "
                    "Add it to your .env file: OPENAI_API_KEY=sk-..."
                )
            # Force OpenAI's base URL — prevents OPENAI_BASE_URL env var from
            # accidentally routing requests to Anthropic or another provider.
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                openai_api_key=api_key,
                openai_api_base=OPENAI_BASE_URL,
            )

        elif llm_type == "ollama":
            if ChatOllama is None:
                raise ImportError("Run: pip install langchain-community")
            self.llm = ChatOllama(model=model, temperature=temperature)

        else:
            raise ValueError(f"Unknown llm_type: {llm_type!r}. Use 'openai' or 'ollama'.")

    # ── Section generation ────────────────────────────────────────────────────
    def generate_section(
        self,
        section_name: str,
        context: str,
        all_section_texts: Optional[Dict[str, str]] = None,
    ) -> str:
        instruction = SECTION_INSTRUCTIONS.get(
            section_name,
            "Write a concise academic paragraph. Cite PMIDs for every claim.",
        )

        prior_context = ""
        if all_section_texts:
            recent = list(all_section_texts.items())[-2:]
            prior_context = "\n\n".join(
                f"[Previously written — {k}]:\n{v[:400]}..." for k, v in recent
            )

        system = (
            "You are a senior virologist writing a peer-reviewed literature review. "
            "Be analytical, not merely descriptive. "
            "Compare studies, identify patterns, and note contradictions. "
            "Every factual claim must cite a PMID in parentheses."
        )
        human_parts = [f"TASK:\n{instruction}\n"]
        if prior_context:
            human_parts.append(f"PRIOR SECTIONS (for continuity):\n{prior_context}\n")
        human_parts.append(
            f"LITERATURE CONTEXT (cite these PMIDs):\n{context}\n\nWrite the section now:"
        )

        messages = [
            SystemMessage(content=system),
            HumanMessage(content="\n".join(human_parts)),
        ]
        response = self.llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    # ── Single-pass review (backward compat) ─────────────────────────────────
    def generate_review(self, context: str, template: str = "") -> str:
        system = (
            "You are a senior virologist. Write a structured, critical literature review. "
            "Compare studies, identify contradictions, and cite PMIDs for every claim."
        )
        human = (
            (f"Use this framework:\n{template}\n\n" if template else "")
            + f"Literature context:\n{context}"
        )
        messages = [SystemMessage(content=system), HumanMessage(content=human)]
        response = self.llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    # ── Antibody extraction ───────────────────────────────────────────────────
    def extract_antibodies(self, text: str, schema: Dict) -> List[Dict]:
        """
        Uses plain SystemMessage/HumanMessage (no ChatPromptTemplate) to avoid
        the KeyError caused by LangChain parsing {"antibodies"} as a template var.
        """
        schema_str = json.dumps(schema, indent=2)

        system_content = (
            "You are an expert in virology and antibody research. "
            "Return ONLY valid JSON — no markdown fences, no preamble."
        )
        human_content = "\n".join([
            "Extract all broadly neutralizing antibodies mentioned in the text below.",
            'Return a JSON object with key "antibodies" containing a list of objects.',
            "Each object must follow this schema:",
            schema_str,
            "",
            "Rules:",
            "- Only include antibodies explicitly named in the text.",
            "- Use null for missing optional fields.",
            "- ic50 must be a number or null.",
            "- pmid must be a string.",
            "",
            "Text:",
            text,
            "",
            "Return only the JSON object:",
        ])

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]
        try:
            response = self.llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            print(f"LLM call failed in extract_antibodies: {e}")
            return []

        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            content = "\n".join(lines[1:end])
        content = content.strip()

        try:
            data = json.loads(content)
            antibodies = data.get("antibodies", [])
            return [
                ab for ab in antibodies
                if isinstance(ab, dict) and ab.get("antibody_name")
            ]
        except json.JSONDecodeError as e:
            print(f"JSON parse failed in extract_antibodies: {e}")
            print(f"Raw output (first 600 chars):\n{content[:600]}")
            return []