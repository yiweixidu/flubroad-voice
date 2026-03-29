# Pydantic models for type validation
from pydantic import BaseModel, Field
from typing import Optional, Literal, List

class AntibodyInfo(BaseModel):
    antibody_name: str = Field(..., description="Antibody name or identifier")
    target_protein: Literal["HA", "NA", "S", "RBD", "FP", "other"] = Field(..., description="Target protein")
    epitope_region: str = Field(..., description="Epitope region description")
    epitope_residues: Optional[str] = Field(None, description="Residue range or specific positions")
    gene_usage: Optional[str] = Field(None, description="IGHV/IGHD/IGHJ gene usage")
    neutralization_spectrum: str = Field(..., description="Range of strains/subtypes neutralized")
    clinical_phase: Literal["preclinical", "phase1", "phase2", "phase3", "approved"] = Field("preclinical")
    ic50: Optional[float] = Field(None, description="Neutralization IC50 in µg/mL if available")
    pmid: str = Field(..., description="PubMed ID of the reference")

class AntibodyList(BaseModel):
    antibodies: List[AntibodyInfo] = Field(..., description="List of extracted antibodies")

# JSON Schema dictionary for LLM output guidance
antibody_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "antibody_name": {"type": "string"},
            "target_protein": {"type": "string", "enum": ["HA", "NA", "S", "RBD", "FP", "other"]},
            "epitope_region": {"type": "string"},
            "epitope_residues": {"type": ["string", "null"]},
            "gene_usage": {"type": ["string", "null"]},
            "neutralization_spectrum": {"type": "string"},
            "clinical_phase": {"type": "string", "enum": ["preclinical", "phase1", "phase2", "phase3", "approved"]},
            "ic50": {"type": ["number", "null"]},
            "pmid": {"type": "string"}
        },
        "required": ["antibody_name", "target_protein", "epitope_region", "neutralization_spectrum", "pmid"]
    }
}