from typing import Literal
from pydantic import BaseModel, Field, model_validator

ComponentType = Literal["System_Component", "Super_System", "Target_Object"]
InteractionQuality = Literal["Useful_Normal", "Insufficient", "Excessive", "Harmful"]
FunctionCategory = Literal["Productive", "Supporting", "Transport", "Corrective"]
ModificationType = Literal["Untouched", "Trimmed", "Replaced", "Modified", "Merged"]


class RawCaseRecord(BaseModel):
    """Unified output of all ingestion adapters."""
    case_id: str
    source: Literal["trizbench", "uspto"]
    domain: str
    before_text: str
    after_text: str


# --- Pass 1: Function Model Extraction ---

class TRIZNode(BaseModel):
    node_id: str = Field(..., pattern=r"^[a-z0-9_]+$", description="Slugified canonical name")
    label_name: str = Field(..., min_length=1)
    component_type: ComponentType
    is_standard_or_regulated: bool
    role_description: str = Field(..., min_length=5, description="1-sentence summary of component role")


class TRIZEdge(BaseModel):
    source: str = Field(..., pattern=r"^[a-z0-9_]+$")
    target: str = Field(..., pattern=r"^[a-z0-9_]+$")
    action_verb: str = Field(..., min_length=1)
    interaction_quality: InteractionQuality
    function_category: FunctionCategory


class TRIZGraph(BaseModel):
    nodes: list[TRIZNode]
    edges: list[TRIZEdge]

    @model_validator(mode="after")
    def validate_edge_references(self) -> "TRIZGraph":
        node_ids = {n.node_id for n in self.nodes}
        for edge in self.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(
                    f"Edge ({edge.source} -> {edge.target}) references unknown node_id"
                )
        return self


# --- Pass 2: Ground-Truth Rigidity Labeling ---

class NodeRigidityLabel(BaseModel):
    node_id: str = Field(..., pattern=r"^[a-z0-9_]+$")
    rigidity_label: Literal[0, 1]  # 0 = Flexible/Modifiable, 1 = Rigid/Untouchable
    modification_type: ModificationType
    evidence_quote: str = Field(..., min_length=3)
    confidence_score: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_label_consistency(self) -> "NodeRigidityLabel":
        if self.rigidity_label == 1 and self.modification_type != "Untouched":
            raise ValueError("Rigid (1) nodes must have modification_type='Untouched'")
        if self.rigidity_label == 0 and self.modification_type == "Untouched":
            raise ValueError("Flexible (0) nodes cannot have modification_type='Untouched'")
        return self


class RigidityLabelOutput(BaseModel):
    labels: list[NodeRigidityLabel]