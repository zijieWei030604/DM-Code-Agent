"""Built-in optional Agent capabilities."""

from .evidence import EvidenceGraphCapability
from .repeat_call_redirect import RepeatCallRedirectCapability
from .semantic_workspace import SemanticWorkspaceCapability
from .verified_edits import VerifiedEditCapability

__all__ = [
    "EvidenceGraphCapability",
    "RepeatCallRedirectCapability",
    "SemanticWorkspaceCapability",
    "VerifiedEditCapability",
]
