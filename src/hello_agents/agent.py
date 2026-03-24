"""Core agent primitives for the hello_agents package."""

from dataclasses import dataclass


@dataclass(slots=True)
class Agent:
    """Represent a minimal agent instance."""

    name: str

    def run(self) -> str:
        """Return a simple status message for the agent."""

        return f"Agent {self.name} is ready."
