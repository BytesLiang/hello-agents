"""Rule-based memory analyzer for completed agent turns."""

from __future__ import annotations

import re
from collections.abc import Sequence

from hello_agents.memory.base import MemoryAnalyzer
from hello_agents.memory.models import (
    MemoryCandidate,
    MemoryKind,
    MemoryProposal,
    MemoryRecord,
    MemoryScope,
)
from hello_agents.tools.base import ToolResult


class RuleBasedMemoryAnalyzer(MemoryAnalyzer):
    """Analyze durable memory candidates with deterministic heuristics."""

    def propose(
        self,
        message: str,
        response: str,
        *,
        scope: MemoryScope,
        tool_results: Sequence[ToolResult] = (),
        success: bool = True,
    ) -> MemoryProposal:
        """Return working records and long-term candidates for a completed turn."""

        working_records = tuple(
            self._build_working_records(
                message,
                response,
                scope=scope,
                tool_results=tool_results,
            )
        )
        candidates = [
            *self._extract_preferences(message),
            *self._extract_facts(message),
            self._build_episode_candidate(
                message,
                response,
                tool_results=tool_results,
                success=success,
            ),
        ]
        if success:
            procedure = self._extract_procedure_candidate(
                message,
                response,
                tool_results=tool_results,
            )
            if procedure is not None:
                candidates.append(procedure)
        return MemoryProposal(
            working_records=working_records,
            candidates=tuple(
                candidate for candidate in candidates if candidate is not None
            ),
        )

    @staticmethod
    def _build_working_records(
        message: str,
        response: str,
        *,
        scope: MemoryScope,
        tool_results: Sequence[ToolResult],
    ) -> list[MemoryRecord]:
        """Build working-memory records for the current turn."""

        records = [
            MemoryRecord(
                kind=MemoryKind.WORKING_MESSAGE,
                user_id=scope.user_id,
                session_id=scope.session_id,
                agent_id=scope.agent_id,
                run_id=scope.run_id,
                content=message,
                summary=message,
                metadata={"message_role": "user"},
            ),
            MemoryRecord(
                kind=MemoryKind.WORKING_MESSAGE,
                user_id=scope.user_id,
                session_id=scope.session_id,
                agent_id=scope.agent_id,
                run_id=scope.run_id,
                content=response,
                summary=response,
                metadata={"message_role": "assistant"},
            ),
        ]
        plan = _extract_plan(message) or _extract_plan(response)
        if plan is not None:
            records.append(
                MemoryRecord(
                    kind=MemoryKind.WORKING_PLAN,
                    user_id=scope.user_id,
                    session_id=scope.session_id,
                    agent_id=scope.agent_id,
                    run_id=scope.run_id,
                    content=plan,
                    summary=plan,
                    pinned=True,
                )
            )
        records.extend(
            [
                MemoryRecord(
                    kind=MemoryKind.WORKING_CONTEXT,
                    user_id=scope.user_id,
                    session_id=scope.session_id,
                    agent_id=scope.agent_id,
                    run_id=scope.run_id,
                    content=f"Latest user request: {_truncate(message, 160)}",
                    summary=_truncate(message, 80),
                ),
                MemoryRecord(
                    kind=MemoryKind.WORKING_CONTEXT,
                    user_id=scope.user_id,
                    session_id=scope.session_id,
                    agent_id=scope.agent_id,
                    run_id=scope.run_id,
                    content=f"Latest assistant answer: {_truncate(response, 160)}",
                    summary=_truncate(response, 80),
                ),
            ]
        )
        for tool_result in tool_results:
            records.append(
                MemoryRecord(
                    kind=MemoryKind.WORKING_CONTEXT,
                    user_id=scope.user_id,
                    session_id=scope.session_id,
                    agent_id=scope.agent_id,
                    run_id=scope.run_id,
                    content=f"{tool_result.tool_name}: {tool_result.content}",
                    summary=_truncate(tool_result.content, 80),
                    metadata={"source": "tool"},
                )
            )
        return records

    @staticmethod
    def _extract_preferences(message: str) -> list[MemoryCandidate]:
        """Extract user preference candidates from the user message."""

        patterns = [
            re.compile(r"\bI prefer (?P<value>[^.!?\n]+)", re.IGNORECASE),
            re.compile(r"\bI like (?P<value>[^.!?\n]+)", re.IGNORECASE),
            re.compile(r"我(?:更喜欢|喜欢|偏好)(?P<value>[^。！？\n]+)"),
        ]
        candidates: list[MemoryCandidate] = []
        for pattern in patterns:
            match = pattern.search(message)
            if match is None:
                continue
            value = match.group("value").strip()
            candidates.append(
                MemoryCandidate(
                    kind=MemoryKind.SEMANTIC_PREFERENCE,
                    key=_normalize_key(value),
                    value=value,
                    content=value,
                    summary=f"User preference: {value}",
                    confidence=0.9,
                    confirmed=True,
                )
            )
        return candidates

    @staticmethod
    def _extract_facts(message: str) -> list[MemoryCandidate]:
        """Extract explicitly confirmed fact candidates from the user message."""

        patterns = [
            re.compile(r"\bremember that (?P<value>[^.!?\n]+)", re.IGNORECASE),
            re.compile(r"\bconfirmed:? (?P<value>[^.!?\n]+)", re.IGNORECASE),
            re.compile(r"请记住(?P<value>[^。！？\n]+)"),
            re.compile(r"已确认(?P<value>[^。！？\n]+)"),
        ]
        candidates: list[MemoryCandidate] = []
        for pattern in patterns:
            match = pattern.search(message)
            if match is None:
                continue
            value = match.group("value").strip()
            key = _normalize_key(value.split(" is ", maxsplit=1)[0])
            candidates.append(
                MemoryCandidate(
                    kind=MemoryKind.SEMANTIC_FACT,
                    key=key,
                    value=value,
                    content=value,
                    summary=f"Confirmed fact: {value}",
                    confidence=0.9,
                    confirmed=True,
                )
            )
        return candidates

    @staticmethod
    def _build_episode_candidate(
        message: str,
        response: str,
        *,
        tool_results: Sequence[ToolResult],
        success: bool,
    ) -> MemoryCandidate:
        """Build an episodic candidate for the completed task."""

        tool_names = tuple(result.tool_name for result in tool_results)
        summary = _truncate(f"Task: {message} | Result: {response}", 220)
        content_lines = [f"Task: {message}", f"Assistant: {response}"]
        if tool_results:
            content_lines.append("Tools:")
            content_lines.extend(
                f"- {result.tool_name}: {result.content}" for result in tool_results
            )
        return MemoryCandidate(
            kind=MemoryKind.EPISODIC,
            content="\n".join(content_lines),
            summary=summary,
            confidence=0.8 if success else 0.55,
            metadata={"task": message, "success": success, "tool_names": tool_names},
        )

    @staticmethod
    def _extract_procedure_candidate(
        message: str,
        response: str,
        *,
        tool_results: Sequence[ToolResult],
    ) -> MemoryCandidate | None:
        """Extract a reusable successful procedure candidate."""

        task_type = _task_type(message)
        if not tool_results and task_type == "general_task":
            return None
        tool_names = tuple(result.tool_name for result in tool_results)
        content = _truncate(f"Successful pattern for {task_type}: {response}", 260)
        return MemoryCandidate(
            kind=MemoryKind.PROCEDURAL,
            content=content,
            summary=f"Successful approach for {task_type}",
            confidence=0.75,
            metadata={
                "task_type": task_type,
                "tool_names": tool_names,
                "tool_count": len(tool_names),
            },
        )


def _extract_plan(message: str) -> str | None:
    """Extract an explicit plan from a message when present."""

    patterns = [
        re.compile(r"\bplan:\s*(?P<plan>.+)", re.IGNORECASE),
        re.compile(r"计划[:：]\s*(?P<plan>.+)"),
    ]
    for pattern in patterns:
        match = pattern.search(message)
        if match is not None:
            return match.group("plan").strip()
    return None


def _normalize_key(value: str) -> str:
    """Normalize free-form text into a stable semantic-memory key."""

    lowered = value.lower()
    collapsed = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", lowered).strip("_")
    return collapsed or "memory"


def _task_type(message: str) -> str:
    """Infer a coarse task type from the user's request."""

    lowered = message.lower()
    for keyword in ("summarize", "search", "plan", "write", "compare", "analyze"):
        if keyword in lowered:
            return keyword
    return "general_task"


def _truncate(value: str, limit: int) -> str:
    """Return a shortened string suitable for summaries."""

    if len(value) <= limit:
        return value
    return f"{value[: limit - 3].rstrip()}..."
