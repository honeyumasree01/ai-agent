import hashlib
import json
import logging
from typing import Any

from langchain_core.tools import tool

from db.query_templates import MAX_ROWS, QUERY_TEMPLATES
from utils.app_context import db_pool, redis_client
from utils.retry import with_retry

logger = logging.getLogger(__name__)


def _cache_key(template_key: str, params: dict[str, Any]) -> str:
    payload = json.dumps({"k": template_key, "p": params}, sort_keys=True)
    h = hashlib.sha256(payload.encode()).hexdigest()[:32]
    return f"db:{template_key}:{h}"


async def _query_impl(template_key: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    if template_key not in QUERY_TEMPLATES:
        raise ValueError(f"Unknown template_key: {template_key}")
    r = redis_client()
    ck = _cache_key(template_key, params)
    cached = await r.get(ck)
    if cached:
        return json.loads(cached)
    sql = QUERY_TEMPLATES[template_key]
    pool = db_pool()
    vals = list(params.values())
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *vals) if vals else await conn.fetch(sql)
    out: list[dict[str, Any]] = [dict(x) for x in rows[:MAX_ROWS]]
    await r.setex(ck, 300, json.dumps(out, default=str))
    return out


@tool
async def query_database(template_key: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Allowlisted SQL only; params bound by asyncpg."""
    logger.debug("query_database key=%s", template_key)
    return await with_retry(_query_impl, template_key, params)
