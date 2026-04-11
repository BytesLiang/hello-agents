"""Placeholder Neo4j graph-store adapter."""

from __future__ import annotations

from importlib import import_module

from hello_agents.memory.base import GraphStore
from hello_agents.memory.config import Neo4jStoreConfig


class Neo4jGraphStore(GraphStore):
    """Expose a health-checkable Neo4j adapter for future graph memory support."""

    def __init__(self, config: Neo4jStoreConfig) -> None:
        """Store the Neo4j configuration without enabling writes yet."""

        self._config = config
        self._driver = None
        if config.url and config.username and config.password:
            try:
                graph_database = import_module("neo4j").GraphDatabase
            except ModuleNotFoundError:
                self._driver = None
            else:
                self._driver = graph_database.driver(
                    config.url,
                    auth=(config.username, config.password),
                )

    def healthcheck(self) -> bool:
        """Return whether a Neo4j driver was successfully configured."""

        return self._driver is not None
