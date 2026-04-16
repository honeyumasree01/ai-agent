from typing import Any

from openai import AsyncOpenAI
from pinecone import Pinecone

from utils.app_context import db_pool, redis_client
from utils.settings import get_settings


async def probe_postgres() -> str:
    pool = db_pool()
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    return "ok"


async def probe_redis() -> str:
    r = redis_client()
    pong = await r.ping()
    return "ok" if pong else "fail"


async def probe_pinecone() -> str:
    s = get_settings()
    if not s.pinecone_api_key:
        return "skipped"
    pc = Pinecone(api_key=s.pinecone_api_key)
    _ = pc.Index(s.pinecone_index)
    return "ok"


async def probe_models() -> list[str]:
    s = get_settings()
    ok: list[str] = []
    if s.anthropic_api_key:
        ok.append("anthropic_configured")
    if s.openai_api_key:
        try:
            client = AsyncOpenAI(api_key=s.openai_api_key)
            await client.models.list(timeout=5.0)
            ok.append("openai_ok")
        except Exception:
            ok.append("openai_fail")
    if s.google_api_key:
        ok.append("google_configured")
    return ok


async def run_health() -> dict[str, Any]:
    out: dict[str, Any] = {"status": "ok", "postgres": "", "redis": "", "pinecone": "", "models_available": []}
    try:
        out["postgres"] = await probe_postgres()
    except Exception as exc:  # noqa: BLE001
        out["postgres"] = f"fail:{type(exc).__name__}"
        out["status"] = "degraded"
    try:
        out["redis"] = await probe_redis()
    except Exception as exc:  # noqa: BLE001
        out["redis"] = f"fail:{type(exc).__name__}"
        out["status"] = "degraded"
    try:
        out["pinecone"] = await probe_pinecone()
    except Exception as exc:  # noqa: BLE001
        out["pinecone"] = f"fail:{type(exc).__name__}"
        out["status"] = "degraded"
    out["models_available"] = await probe_models()
    return out
