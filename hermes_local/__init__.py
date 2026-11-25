"""Hermes Local — an original, clean-room, offline-first agent framework.

Inspired by the architecture of Nous Research's Hermes Agent (agent loop +
tools + persistent memory + local models), implemented from scratch for
Aaron's RV command center. No Nous code is used or copied.

Built-in embedding layer: EmbeddingGemma 2 (with fallbacks). The generative
brain is auto-selected from a low-resource ladder of local models.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
