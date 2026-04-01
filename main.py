"""FastAPI entry: /health and SSE /run for LangGraph."""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from memory.vector_store import get_memory
from orchestrator import build_graph
from utils.app_context import init_resources, shutdown_resources
from utils.auth import verify_run_token
from utils.health import run_health

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_resources()
    global _graph
    _graph = build_graph()
    yield
    await shutdown_resources()


app = FastAPI(title="Multi-Agent AI", lifespan=lifespan)


class GoalBody(BaseModel):
    goal: str = Field(..., min_length=1)


async def _graph_astream_producer(
    graph: Any,
    state: dict[str, Any],
    cfg: dict[str, Any],
    queue: asyncio.Queue,
) -> None:
    try:
        async for update in graph.astream(state, cfg, stream_mode="updates"):
            await queue.put(("update", update))
        await queue.put(("done", None))
    except asyncio.CancelledError:
        logger.info("graph astream producer cancelled")
        raise
    except Exception as exc:  # noqa: BLE001
        await queue.put(("error", exc))


async def _sse_stream(request: Request, goal: str, task_id: str) -> AsyncIterator[str]:
    mem = get_memory()
    ctx_bits = await mem.recall(goal, k=5)
    memory_context = "\n".join(ctx_bits)
    state = {
        "goal": goal,
        "memory_context": memory_context,
        "plan": [],
        "step_index": 0,
        "observations": [],
        "result": "",
        "critic_score": 0,
        "attempt_count": 0,
        "task_id": task_id,
        "failed_reason": None,
    }
    cfg = {"configurable": {"thread_id": task_id}}
    assert _graph is not None
    queue: asyncio.Queue = asyncio.Queue()
    worker = asyncio.create_task(_graph_astream_producer(_graph, state, cfg, queue))
    try:
        while True:
            if await request.is_disconnected():
                logger.info("sse client disconnected; cancelling graph task_id=%s", task_id)
                break
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                if worker.done():
                    async for chunk in _flush_queue(queue):
                        yield chunk
                    break
                continue
            if kind == "update":
                yield f"data: {json.dumps(payload, default=str)}\n\n"
            elif kind == "done":
                break
            elif kind == "error":
                err = json.dumps({"error": type(payload).__name__, "detail": str(payload)[:2000]})
                yield f"data: {err}\n\n"
                break
    finally:
        if not worker.done():
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                logger.info("graph task cancelled task_id=%s", task_id)


async def _flush_queue(queue: asyncio.Queue) -> AsyncIterator[str]:
    while True:
        try:
            kind, payload = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if kind == "update":
            yield f"data: {json.dumps(payload, default=str)}\n\n"
        elif kind == "error":
            err = json.dumps({"error": type(payload).__name__, "detail": str(payload)[:2000]})
            yield f"data: {err}\n\n"
            break


@app.post("/run")
async def run(
    request: Request,
    body: GoalBody,
    _auth: None = Depends(verify_run_token),
):
    task_id = str(uuid.uuid4())
    return StreamingResponse(
        _sse_stream(request, body.goal, task_id),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health():
    return await run_health()
