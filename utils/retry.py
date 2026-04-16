import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import asyncpg
import httpx

T = TypeVar("T")

logger = logging.getLogger(__name__)

try:
    from tavily import TavilyError  # type: ignore[attr-defined]
except Exception:  # pragma: no cover

    class TavilyError(Exception):  # type: ignore[no-redef]
        pass


RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.TimeoutException,
    asyncpg.TooManyConnectionsError,
    TavilyError,
)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, RETRY_EXCEPTIONS):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500:
        return True
    return False


async def with_retry(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    max_attempts: int = 3,
    base_delay: float = 2.0,
) -> T:
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return await fn(*args)
        except BaseException as exc:  # noqa: BLE001
            if not _is_retryable(exc) or attempt == max_attempts - 1:
                raise
            last_exc = exc
            delay = base_delay**attempt
            logger.warning(
                "retry attempt=%s error_type=%s delay_s=%.2f",
                attempt + 1,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc
