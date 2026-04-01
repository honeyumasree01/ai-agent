"""Single entry point for all LLM invocations — cascade with backoff + DLQ."""

import asyncio
import logging
from typing import Any

import redis.asyncio as redis
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import RateLimitError as AnthropicRateLimitError
from google.api_core import exceptions as google_exceptions
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from openai import APIStatusError as OpenAIAPIStatusError
from openai import RateLimitError as OpenAIRateLimitError

from utils.errors import AllModelsFailedError
from utils.llm_clients import claude, gemini, gpt4o
from utils.settings import get_settings

logger = logging.getLogger(__name__)

MODEL_BUILDERS = (
    ("claude-sonnet-4-20250514", claude),
    ("gpt-4o-2024-05-13", gpt4o),
    ("gemini-1.5-pro-001", gemini),
)


def _is_rate_or_server(exc: BaseException) -> bool:
    if isinstance(exc, AnthropicRateLimitError | OpenAIRateLimitError):
        return True
    if isinstance(exc, AnthropicAPIStatusError | OpenAIAPIStatusError):
        code = getattr(exc, "status_code", None)
        return code is not None and int(code) >= 500
    if isinstance(exc, (google_exceptions.ResourceExhausted, google_exceptions.InternalServerError)):
        return True
    return False


async def _push_dlq(r: redis.Redis, task_id: str) -> None:
    await r.rpush("dlq:failed_tasks", task_id)


async def invoke_with_fallback(
    messages: list[BaseMessage],
    tools: list[Any] | None = None,
    *,
    task_id: str = "",
) -> AIMessage:
    """Try each model with per-model retries + exponential backoff; DLQ on total failure."""
    settings = get_settings()
    r = redis.from_url(settings.redis_url, decode_responses=True)
    per = 3
    try:
        for name, builder in MODEL_BUILDERS:
            model: Runnable = builder(settings)
            if tools:
                model = model.bind_tools(tools)
            for attempt in range(per):
                try:
                    out = await model.ainvoke(messages)
                    logger.info("llm_success model=%s task_id=%s", name, task_id)
                    return out  # type: ignore[return-value]
                except BaseException as exc:  # noqa: BLE001
                    if _is_rate_or_server(exc) and attempt < per - 1:
                        wait = 2**attempt
                        logger.warning("llm_backoff model=%s attempt=%s wait_s=%s", name, attempt + 1, wait)
                        await asyncio.sleep(wait)
                        continue
                    logger.warning("llm_fail model=%s err=%s", name, type(exc).__name__)
                    break
        if task_id:
            await _push_dlq(r, task_id)
        raise AllModelsFailedError()
    finally:
        await r.aclose()
