"""HTTP calls via httpx — bearer or none."""

import logging
from typing import Any

import httpx
from langchain_core.tools import tool

from utils.retry import with_retry
from utils.settings import get_settings

logger = logging.getLogger(__name__)

# TODO: add OAuth2 flow for third-party providers


async def _call_api_impl(
    url: str,
    method: str,
    body: dict[str, Any],
    auth_type: str,
) -> dict[str, Any]:
    s = get_settings()
    headers: dict[str, str] = {"Accept": "application/json"}
    if auth_type == "bearer":
        headers["Authorization"] = f"Bearer {s.external_api_token}"
    elif auth_type != "none":
        raise ValueError("auth_type must be bearer or none")
    async with httpx.AsyncClient() as client:
        resp = await client.request(method.upper(), url, json=body or None, headers=headers, timeout=10.0)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise httpx.HTTPStatusError(
            f"{exc.response.status_code} error from API",
            request=exc.request,
            response=exc.response,
        ) from exc
    if resp.content:
        return resp.json()
    return {}


@tool
async def call_external_api(
    url: str,
    method: str,
    body: dict[str, Any],
    auth_type: str,
) -> dict[str, Any]:
    """Call a REST API with JSON body. auth_type: bearer | none."""
    logger.debug("call_external_api %s %s", method, url[:60])
    return await with_retry(_call_api_impl, url, method, body, auth_type)
