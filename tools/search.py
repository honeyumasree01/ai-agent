"""Web search via Tavily (async-friendly via thread pool)."""

import asyncio
import logging
from typing import Any

from langchain_core.tools import tool
from tavily import TavilyClient

from utils.settings import get_settings
from utils.retry import with_retry

logger = logging.getLogger(__name__)


def _format(results: dict[str, Any]) -> str:
    lines: list[str] = []
    for r in results.get("results", [])[:5]:
        title = r.get("title", "")
        content = (r.get("content") or r.get("snippet") or "")[:500]
        lines.append(f"- {title}\n  {content}")
    return "\n".join(lines) if lines else "(no results)"


def _search_sync(query: str, api_key: str) -> dict[str, Any]:
    client = TavilyClient(api_key=api_key)
    return client.search(query=query, max_results=5, include_raw_content=True)  # type: ignore[no-any-return]


async def _web_search_impl(query: str) -> str:
    s = get_settings()
    raw = await asyncio.to_thread(_search_sync, query, s.tavily_api_key)
    return _format(raw if isinstance(raw, dict) else {"results": raw})  # type: ignore[arg-type]


@tool
async def web_search(query: str) -> str:
    """Search the web; returns titles and snippets (max 5)."""
    logger.debug("web_search query=%s", query[:80])
    return await with_retry(_web_search_impl, query)
