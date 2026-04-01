"""LangGraph orchestrator: Planner → Executor (loop) → Critic."""

import logging
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.critic import critic_node
from agents.executor import executor_node
from agents.planner import planner_node

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    goal: str
    memory_context: str
    plan: list[str]
    step_index: int
    observations: list[str]
    result: str
    critic_score: int
    attempt_count: int
    task_id: str
    failed_reason: str | None


def route_after_executor(state: AgentState) -> Literal["executor", "critic"]:
    plan = state.get("plan") or []
    if not plan:
        return "critic"
    idx = int(state.get("step_index") or 0)
    if idx < len(plan):
        return "executor"
    return "critic"


def route_after_critic(state: AgentState) -> Literal["planner", "end"]:
    ac = int(state.get("attempt_count") or 0)
    if ac >= 5:
        return "end"
    score = int(state.get("critic_score") or 0)
    if score >= 7:
        return "end"
    if score < 7 and ac < 5:
        return "planner"
    return "end"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("critic", critic_node)
    g.add_edge(START, "planner")
    g.add_edge("planner", "executor")
    g.add_conditional_edges(
        "executor",
        route_after_executor,
        {"executor": "executor", "critic": "critic"},
    )
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {"planner": "planner", "end": END},
    )
    return g.compile(checkpointer=MemorySaver())
