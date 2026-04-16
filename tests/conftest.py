from __future__ import annotations

import uuid
from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from orchestrator import AgentState, build_graph


def _state(**overrides: object) -> AgentState:
    base: AgentState = {
        "goal": "what is 2 + 2",
        "memory_context": "",
        "plan": [],
        "step_index": 0,
        "observations": [],
        "result": "",
        "critic_score": 0,
        "attempt_count": 0,
        "task_id": str(uuid.uuid4()),
        "failed_reason": None,
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


@pytest.fixture
def base_state() -> AgentState:
    return _state()


@pytest.fixture
def graph():
    return build_graph()


@pytest.fixture
def mock_memory(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mem = MagicMock()
    mem.recall = AsyncMock(return_value=[])
    mem.remember = AsyncMock()
    monkeypatch.setattr("agents.planner.get_memory", lambda: mem)
    monkeypatch.setattr("agents.critic.get_memory", lambda: mem)
    return mem


def patch_llm_sequence(
    monkeypatch: pytest.MonkeyPatch,
    responses: Sequence[AIMessage],
) -> None:
    seq = list(responses)

    async def _invoke(*_a: object, **_k: object) -> AIMessage:
        if not seq:
            raise AssertionError("invoke_with_fallback called more times than expected")
        return seq.pop(0)

    m = AsyncMock(side_effect=_invoke)
    monkeypatch.setattr("agents.planner.invoke_with_fallback", m)
    monkeypatch.setattr("agents.executor.invoke_with_fallback", m)
    monkeypatch.setattr("agents.critic.invoke_with_fallback", m)
