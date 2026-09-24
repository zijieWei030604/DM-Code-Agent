"""Memory and context management."""

from typing import Any


def __getattr__(name: str) -> Any:
    """Load legacy standalone helpers only for explicit legacy imports."""
    if name in __all__:
        from . import context_compressor

        return getattr(context_compressor, name)
    raise AttributeError(name)


__all__ = [
    "ContextCompressor",
    "Mem0StyleMemory",
    "MemoryHit",
    "MemoryItem",
]
