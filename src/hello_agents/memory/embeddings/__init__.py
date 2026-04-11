"""Embedding backends for the memory subsystem."""

from hello_agents.memory.base import Embedder
from hello_agents.memory.config import EmbedConfig
from hello_agents.memory.embeddings.dashscope import DashScopeEmbedder


def build_embedder(config: EmbedConfig) -> Embedder:
    """Build the configured embedding backend."""

    model_type = config.model_type.lower()
    if model_type in {"dashscope", "openai-compatible", "openai"}:
        return DashScopeEmbedder(config)
    raise ValueError(f"Unsupported embedding model type '{config.model_type}'.")


__all__ = ["DashScopeEmbedder", "build_embedder"]
