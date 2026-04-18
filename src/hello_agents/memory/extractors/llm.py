"""LLM-assisted memory analyzer with rule-based fallback."""

from __future__ import annotations

import json
from collections.abc import Sequence
from json import JSONDecodeError
from typing import Any

from hello_agents.llm.client import LLMClient
from hello_agents.llm.types import LLMMessage
from hello_agents.memory.base import MemoryAnalyzer
from hello_agents.memory.models import (
    MemoryCandidate,
    MemoryKind,
    MemoryProposal,
    MemoryRecord,
    MemoryScope,
)
from hello_agents.tools.base import ToolResult

from .rule_based import RuleBasedMemoryAnalyzer, _normalize_key

ANALYZER_SYSTEM_PROMPT = """
Extract memory commands from a completed assistant turn.

Return JSON only.
Do not include Markdown or code fences.

Output schema:
{
  "working_records": [
    {
      "kind": "working_plan|working_context|working_message",
      "content": "string",
      "summary": "string",
      "pinned": false,
      "metadata": {}
    }
  ],
  "candidates": [
    {
      "kind": "semantic_preference|semantic_fact|episodic|procedural",
      "key": "optional stable key",
      "value": "optional value",
      "content": "string",
      "summary": "string",
      "confidence": 0.0,
      "confirmed": false,
      "requires_confirmation": false,
      "metadata": {}
    }
  ]
}

Rules:
- Only propose information grounded in the turn.
- Facts must be explicit or clearly confirmed.
- Procedures must be reusable; do not store generic successful answers.
- Working records may include plan/context/message items for immediate reuse.
""".strip()


class LLMMemoryAnalyzer(MemoryAnalyzer):
    """Use an LLM to propose memory commands with fallback behavior."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        fallback: MemoryAnalyzer | None = None,
    ) -> None:
        """Store the analyzer LLM and optional fallback analyzer."""

        self._llm = llm
        self._fallback = fallback or RuleBasedMemoryAnalyzer()

    def propose(
        self,
        message: str,
        response: str,
        *,
        scope: MemoryScope,
        tool_results: Sequence[ToolResult] = (),
        success: bool = True,
    ) -> MemoryProposal:
        """Return a memory proposal for a completed turn."""

        fallback_proposal = self._fallback.propose(
            message,
            response,
            scope=scope,
            tool_results=tool_results,
            success=success,
        )
        try:
            result = self._llm.chat(
                [
                    LLMMessage(role="system", content=ANALYZER_SYSTEM_PROMPT),
                    LLMMessage(
                        role="user",
                        content=_build_prompt(
                            message=message,
                            response=response,
                            tool_results=tool_results,
                            success=success,
                        ),
                    ),
                ]
            )
            payload = _load_json_object(result.content)
        except Exception:
            return fallback_proposal

        proposal = MemoryProposal(
            working_records=tuple(
                _build_working_records(scope, payload.get("working_records"))
            ),
            candidates=tuple(_build_candidates(payload.get("candidates"))),
        )
        return _merge_proposals(fallback_proposal, proposal)


def _build_prompt(
    *,
    message: str,
    response: str,
    tool_results: Sequence[ToolResult],
    success: bool,
) -> str:
    """Build the analyzer prompt."""

    tool_lines = ["None"]
    if tool_results:
        tool_lines = [
            f"- {tool_result.tool_name}: {tool_result.content}"
            for tool_result in tool_results
        ]
    return (
        f"Success: {success}\n\n"
        f"User message:\n{message}\n\n"
        f"Assistant response:\n{response}\n\n"
        "Tool results:\n" + "\n".join(tool_lines)
    )


def _load_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object from model output."""

    normalized = content.strip()
    if normalized.startswith("```"):
        normalized = normalized.removeprefix("```json").removeprefix("```")
        normalized = normalized.removesuffix("```").strip()
    try:
        parsed = json.loads(normalized)
    except JSONDecodeError:
        start = normalized.find("{")
        end = normalized.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise ValueError("Memory analyzer response must contain JSON.") from None
        parsed = json.loads(normalized[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Memory analyzer response must be a JSON object.")
    return parsed


def _build_working_records(
    scope: MemoryScope,
    payload: Any,
) -> list[MemoryRecord]:
    """Build working records from JSON output."""

    if not isinstance(payload, list):
        return []
    records: list[MemoryRecord] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        kind = _load_memory_kind(item.get("kind"))
        if kind not in {
            MemoryKind.WORKING_PLAN,
            MemoryKind.WORKING_CONTEXT,
            MemoryKind.WORKING_MESSAGE,
        }:
            continue
        content = _optional_string(item.get("content"))
        if content is None:
            continue
        records.append(
            MemoryRecord(
                kind=kind,
                user_id=scope.user_id,
                session_id=scope.session_id,
                agent_id=scope.agent_id,
                run_id=scope.run_id,
                content=content,
                summary=_optional_string(item.get("summary")) or content,
                pinned=bool(item.get("pinned", False)),
                metadata=_load_metadata(item.get("metadata")),
            )
        )
    return records


def _build_candidates(payload: Any) -> list[MemoryCandidate]:
    """Build long-term candidates from JSON output."""

    if not isinstance(payload, list):
        return []
    candidates: list[MemoryCandidate] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        kind = _load_memory_kind(item.get("kind"))
        if kind not in {
            MemoryKind.SEMANTIC_PREFERENCE,
            MemoryKind.SEMANTIC_FACT,
            MemoryKind.EPISODIC,
            MemoryKind.PROCEDURAL,
        }:
            continue
        content = _optional_string(item.get("content"))
        summary = _optional_string(item.get("summary"))
        if content is None or summary is None:
            continue
        key = _optional_string(item.get("key"))
        value = _optional_string(item.get("value"))
        if (
            kind
            in {
                MemoryKind.SEMANTIC_PREFERENCE,
                MemoryKind.SEMANTIC_FACT,
            }
            and value is not None
            and key is None
        ):
            key = _normalize_key(value)
        candidates.append(
            MemoryCandidate(
                kind=kind,
                key=key,
                value=value,
                content=content,
                summary=summary,
                confidence=_float_or_default(item.get("confidence"), default=0.0),
                confirmed=bool(item.get("confirmed", False)),
                requires_confirmation=bool(item.get("requires_confirmation", False)),
                metadata=_load_metadata(item.get("metadata")),
            )
        )
    return candidates


def _merge_proposals(
    fallback: MemoryProposal,
    proposed: MemoryProposal,
) -> MemoryProposal:
    """Merge fallback and proposed memory proposals."""

    return MemoryProposal(
        working_records=_dedupe_records(
            [*fallback.working_records, *proposed.working_records]
        ),
        candidates=_dedupe_candidates([*fallback.candidates, *proposed.candidates]),
    )


def _dedupe_records(records: list[MemoryRecord]) -> tuple[MemoryRecord, ...]:
    """Deduplicate generic memory records."""

    seen: set[tuple[MemoryKind, str, str]] = set()
    ordered: list[MemoryRecord] = []
    for record in records:
        fingerprint = (record.kind, record.summary, record.content)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        ordered.append(record)
    return tuple(ordered)


def _dedupe_candidates(
    candidates: list[MemoryCandidate],
) -> tuple[MemoryCandidate, ...]:
    """Deduplicate long-term candidates."""

    seen: set[tuple[MemoryKind, str, str]] = set()
    ordered: list[MemoryCandidate] = []
    for candidate in candidates:
        fingerprint = (candidate.kind, candidate.summary, candidate.content)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        ordered.append(candidate)
    return tuple(ordered)


def _load_memory_kind(value: Any) -> MemoryKind | None:
    """Load a memory kind from JSON."""

    if not isinstance(value, str):
        return None
    try:
        return MemoryKind(value)
    except ValueError:
        return None


def _load_metadata(value: Any) -> dict[str, object]:
    """Load metadata from JSON."""

    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _optional_string(value: Any) -> str | None:
    """Return a non-empty stripped string or None."""

    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _float_or_default(value: Any, *, default: float) -> float:
    """Return a numeric value or the supplied default."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default
