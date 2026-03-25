"""Core agent primitives for the hello_agents package."""

from __future__ import annotations

from abc import ABC, abstractmethod

from hello_agents.llm.client import LLMClient


class Agent(ABC):
    """Define the top-level abstract contract for all LLM-backed agents."""

    def __init__(self, name: str, llm: LLMClient) -> None:
        """Store the common agent identity and shared LLM dependency."""

        self.name = name
        self.llm = llm

    @abstractmethod
    def run(self, message: str) -> str:
        """Execute the agent's primary behavior."""
