"""Built-in optional Agent capabilities."""

from .evidence import EvidenceGraphCapability
from .semantic_workspace import SemanticWorkspaceCapability
from .verified_edits import VerifiedEditCapability

__all__ = ["EvidenceGraphCapability", "SemanticWorkspaceCapability", "VerifiedEditCapability"]
