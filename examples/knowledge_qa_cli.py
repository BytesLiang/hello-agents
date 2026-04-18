"""Backward-compatible wrapper for the formal knowledge QA CLI."""

from hello_agents.apps.knowledge_qa.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
