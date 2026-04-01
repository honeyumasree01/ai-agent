"""Planner: decompose goal into JSON steps via LLM."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from memory.vector_store import get_memory
from utils.llm import invoke_with_fallback

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = """You are a planning agent. Use chain-of-thought internally, then output ONLY valid JSON \
with shape {"steps": ["atomic step 1", ...]} with 3–7 concise steps. No markdown, no prose outside JSON."""


def _strip_json_fence(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s.removeprefix("```")
        if "```" in s:
            s = s.split("```", 1)[0]
    return s.strip()


async def planner_node(state: dict) -> dict:
    goal = state["goal"]
    mem = get_memory()
    mc = (state.get("memory_context") or "").strip()
    past = await mem.recall(goal, k=5)
    merged = [mc] if mc else []
    merged.extend(p for p in past if p)
    ctx = "\n".join(merged[:8])
    mem_note = f"\nRelevant past memory:\n{ctx}\n" if ctx else ""
    retry_note = ""
    if state.get("failed_reason"):
        retry_note = f"\nPrevious attempt failed: {state['failed_reason']}\nImprove the plan.\n"
    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"Goal:{mem_note}{retry_note}\n{goal}"),
    ]
    msg = await invoke_with_fallback(messages, task_id=state["task_id"])
    raw = _strip_json_fence(msg.content or "")
    try:
        data = json.loads(raw)
        steps = list(data["steps"])
    except (json.JSONDecodeError, KeyError, TypeError):
        fix = await invoke_with_fallback(
            [
                SystemMessage(content=PLANNER_SYSTEM),
                HumanMessage(content=f"Return ONLY JSON. Previous output was invalid:\n{raw[:2000]}"),
            ],
            task_id=state["task_id"],
        )
        raw2 = _strip_json_fence(fix.content or "")
        try:
            data = json.loads(raw2)
            steps = list(data["steps"])
        except (json.JSONDecodeError, KeyError, TypeError):
            steps = [f"Research and answer: {goal}"]
    logger.info("planner steps=%s", len(steps))
    return {"plan": steps, "step_index": 0, "observations": [], "critic_score": 0}
