from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage

from tests.conftest import _state, patch_llm_sequence

MOCK_PLAN_ONE = '{"steps": ["answer succinctly"]}'
MOCK_PLAN_TWO = '{"steps": ["search for X", "summarize findings"]}'
MOCK_CRITIC_PASS = '{"score": 8, "reason": "complete"}'
MOCK_CRITIC_FAIL = '{"score": 4, "reason": "missing detail"}'


@pytest.mark.asyncio
async def test_max_retries_sets_failed_reason(
    graph,
    mock_memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    msgs: list[AIMessage] = []
    for _ in range(5):
        msgs.append(AIMessage(content=MOCK_PLAN_TWO))
        msgs.extend([AIMessage(content="exec-a"), AIMessage(content="exec-b")])
        msgs.append(AIMessage(content=MOCK_CRITIC_FAIL))
    patch_llm_sequence(monkeypatch, msgs)

    cfg = {"configurable": {"thread_id": "test-max-retries"}}
    final = await asyncio.wait_for(
        graph.ainvoke(_state(goal="stress"), cfg),
        timeout=60.0,
    )
    assert final["attempt_count"] == 5
    assert final["failed_reason"] == "max retries exceeded"
    mock_memory.remember.assert_not_called()


@pytest.mark.asyncio
async def test_happy_path_planner_executor_critic(
    graph,
    mock_memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_llm_sequence(
        monkeypatch,
        [
            AIMessage(content=MOCK_PLAN_ONE),
            AIMessage(content="The answer is 4."),
            AIMessage(content=MOCK_CRITIC_PASS),
        ],
    )
    cfg = {"configurable": {"thread_id": "test-happy"}}
    final = await graph.ainvoke(_state(), cfg)
    assert final["critic_score"] >= 7
    assert final["result"]
    mock_memory.remember.assert_awaited()


@pytest.mark.asyncio
async def test_critic_loop_replans_then_passes(
    graph,
    mock_memory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    msgs: list[AIMessage] = []
    msgs.extend(
        [
            AIMessage(content=MOCK_PLAN_TWO),
            AIMessage(content="obs-1"),
            AIMessage(content="obs-2"),
            AIMessage(content=MOCK_CRITIC_FAIL),
        ],
    )
    msgs.extend(
        [
            AIMessage(content=MOCK_PLAN_TWO),
            AIMessage(content="obs-1b"),
            AIMessage(content="obs-2b"),
            AIMessage(content=MOCK_CRITIC_FAIL),
        ],
    )
    msgs.extend(
        [
            AIMessage(content=MOCK_PLAN_TWO),
            AIMessage(content="obs-1c"),
            AIMessage(content="obs-2c"),
            AIMessage(content=MOCK_CRITIC_PASS),
        ],
    )
    patch_llm_sequence(monkeypatch, msgs)

    cfg = {"configurable": {"thread_id": "test-replan"}}
    final = await asyncio.wait_for(graph.ainvoke(_state(), cfg), timeout=60.0)
    assert final["critic_score"] >= 7
    assert final["attempt_count"] == 2
    mock_memory.remember.assert_awaited()
