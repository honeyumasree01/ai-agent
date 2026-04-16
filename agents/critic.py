import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from memory.vector_store import get_memory
from utils.llm import invoke_with_fallback

logger = logging.getLogger(__name__)

CRITIC_SYS = """You critique task completion. Return ONLY JSON: \
{"score": 1-10, "reason": "short justification"}.

Consider the complexity of the goal when scoring:
- Simple factual questions (recipes, definitions, how-to guides) should score 7-9 if the answer is complete and accurate
- Only score below 7 if the answer is genuinely incomplete, wrong, or missing key information
- Do not penalize simple answers to simple questions"""


async def critic_node(state: dict) -> dict:
    goal = state["goal"]
    obs = "\n".join(state.get("observations") or [])[-30000:]
    messages = [
        SystemMessage(content=CRITIC_SYS),
        HumanMessage(content=f"Goal:\n{goal}\n\nObservations:\n{obs}"),
    ]
    msg = await invoke_with_fallback(messages, task_id=state["task_id"])
    raw = (msg.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if "```" in raw:
            raw = raw.split("```", 1)[0]
    raw = raw.strip()
    try:
        data = json.loads(raw)
        score = int(data["score"])
        reason = str(data.get("reason", ""))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        score, reason = 5, "parse_error"
    score = max(1, min(10, score))
    out: dict = {"critic_score": score}
    prev_attempts = int(state.get("attempt_count") or 0)
    if score < 7:
        attempts = prev_attempts + 1
        out["attempt_count"] = attempts
        out["failed_reason"] = reason[:2000]
        if attempts >= 5:
            out["failed_reason"] = "max retries exceeded"
        logger.info("critic score=%s attempts=%s", score, attempts)
    else:
        out["failed_reason"] = None
        if state.get("result"):
            mem = get_memory()
            await mem.remember(
                state["result"],
                metadata={"task_id": state["task_id"], "timestamp": time.time()},
            )
        logger.info("critic score=%s attempts=%s", score, prev_attempts)
    return out
