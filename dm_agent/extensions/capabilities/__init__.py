"""Built-in optional Agent capabilities."""

from .evidence import EvidenceGraphCapability
from .lsp_impact import LspImpactCapability
from .repeat_call_redirect import RepeatCallRedirectCapability
from .semantic_workspace import SemanticWorkspaceCapability
from .verified_edits import VerifiedEditCapability

__all__ = [
    "EvidenceGraphCapability",
    "LspImpactCapability",
    "RepeatCallRedirectCapability",
    "SemanticWorkspaceCapability",
    "VerifiedEditCapability",
]
