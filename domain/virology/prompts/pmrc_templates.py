def get_enhanced_template(topic: str) -> str:
    templates = {
        "flu_bnabs": """
You are a senior virologist writing a literature review on broadly neutralizing antibodies against influenza.
Follow the PMRC structure strictly, and include the following elements:

**Problem**: 
- Describe the clinical and evolutionary challenge (antigenic drift/shift, vaccine mismatch).
- Cite specific PMIDs that quantify the problem.

**Motivation**:
- Explain why targeting conserved epitopes (e.g., HA stem, neuraminidase active site) is a promising strategy.
- Reference key structural or immunological studies that support this concept.

**Results**:
- List at least 3-5 broadly neutralizing antibodies, including their target epitope, gene usage, and neutralization spectrum.
- For each antibody, mention structural basis (if available) and any clinical development stage.
- Discuss mechanisms of action beyond neutralization (e.g., Fc effector functions, ADCC) if supported by literature.

**Technical Challenges**:
- Identify at least 2-3 key challenges in translating bnAbs to vaccines or therapeutics (e.g., immunodominance, epitope accessibility, germline targeting).
- Reference papers that address these obstacles.

**Future Directions**:
- Provide at least 2-3 original, forward-looking suggestions that are NOT directly quoted from the papers.
- Examples: “Combining structure-based immunogen design with mRNA-LNP delivery could accelerate clinical translation,” or “Deep mutational scanning of the HA stem may reveal escape pathways that need to be preemptively blocked.”
- Justify each suggestion with a brief rationale.

**Conclusion**:
- Summarize the current state and emphasize the potential impact of bnAbs on pandemic preparedness.

Use a professional, academic tone. Every factual statement must be accompanied by a PMID in parentheses.
""",
        "hiv_bnabs": """..."""   # 可添加其他主题模板
    }
    return templates.get(topic, "")

def get_template(topic: str) -> str:
    """Alias for get_enhanced_template to match orchestrator expectations."""
    return get_enhanced_template(topic)