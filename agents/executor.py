"""Executor: one plan step — Claude tool-use via invoke_with_fallback."""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tools.api import call_external_api
from tools.code_exec import run_code
from tools.database import query_database
from tools.search import web_search
from utils.llm import invoke_with_fallback

logger = logging.getLogger(__name__)

TOOLS = [web_search, query_database, call_external_api, run_code]
TOOL_MAP = {t.name: t for t in TOOLS}

SYS = """You execute one step of a plan using tools when needed. \
Prefer a single tool call when the step requires external data. If you can answer from context, reply briefly."""


async def executor_node(state: dict) -> dict:
    plan: list[str] = state["plan"]
    i: int = state["step_index"]
    step = plan[i] if i < len(plan) else ""
    prior = list(state.get("observations") or [])
    obs_hist = "\n".join(prior[-12:])
    messages: list[Any] = [
        SystemMessage(content=SYS),
        HumanMessage(
            content=json.dumps(
                {"step": step, "goal": state["goal"], "prior_observations": obs_hist},
            ),
        ),
    ]
    msg = await invoke_with_fallback(messages, tools=TOOLS, task_id=state["task_id"])
    if not isinstance(msg, AIMessage):
        msg = AIMessage(content=str(msg))
    added: list[str] = []
    msgs = list(messages)
    if msg.tool_calls:
        msgs.append(msg)
        for tc in msg.tool_calls:
            name = tc["name"]
            args: dict[str, Any] = tc["args"] if isinstance(tc["args"], dict) else {}
            tool = TOOL_MAP.get(name)
            if not tool:
                added.append(f"unknown_tool:{name}")
                continue
            out = await tool.ainvoke(args)
            added.append(f"[{name}] {out!s}"[:20000])
            msgs.append(ToolMessage(content=str(out)[:15000], tool_call_id=tc["id"]))
        final = await invoke_with_fallback(msgs, tools=TOOLS, task_id=state["task_id"])
        tail = (final.content or "").strip()
        if tail:
            added.append(f"(summary){tail}"[:5000])
    else:
        added.append((msg.content or "").strip() or "(no output)")
    new_obs = prior + added
    new_idx = i + 1
    result = state.get("result") or ""
    if new_idx >= len(plan) and plan:
        result = "\n".join(new_obs)[-50000:]
    logger.info("executor step=%s idx=%s tools=%s", i, new_idx, bool(msg.tool_calls))
    return {"observations": new_obs, "step_index": new_idx, "result": result}
