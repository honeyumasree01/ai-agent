import asyncio
import logging
import time
import uuid
from typing import Any

from openai import AsyncOpenAI
from pinecone import Pinecone

from utils.settings import get_settings

logger = logging.getLogger(__name__)


def _openai_configured() -> bool:
    return bool(get_settings().openai_api_key.strip())


def _sync_upsert(index: Any, id_: str, vec: list[float], meta: dict[str, Any]) -> None:
    index.upsert(vectors=[{"id": id_, "values": vec, "metadata": meta}])


def _sync_query(index: Any, vec: list[float], k: int) -> list[str]:
    q = index.query(vector=vec, top_k=k, include_metadata=True)
    matches = getattr(q, "matches", None) or (q.get("matches", []) if isinstance(q, dict) else [])
    out: list[str] = []
    for m in matches:
        md = getattr(m, "metadata", None) or (m.get("metadata") if isinstance(m, dict) else {}) or {}
        t = md.get("text") if isinstance(md, dict) else None
        if isinstance(t, str):
            out.append(t)
    return out


class VectorMemory:
    def __init__(self) -> None:
        s = get_settings()
        self._embed = AsyncOpenAI(api_key=s.openai_api_key)
        self._pc = Pinecone(api_key=s.pinecone_api_key)
        self._index_name = s.pinecone_index

    async def _embed_text(self, text: str) -> list[float] | None:
        if not _openai_configured():
            logger.warning("embeddings skipped: OPENAI_API_KEY empty")
            return None
        try:
            r = await self._embed.embeddings.create(model="text-embedding-3-small", input=text)
            return list(r.data[0].embedding)
        except Exception as exc:  # noqa: BLE001
            logger.warning("embeddings failed (%s): %s", type(exc).__name__, exc)
            return None

    async def remember(self, text: str, metadata: dict[str, Any]) -> None:
        s = get_settings()
        if not s.pinecone_api_key:
            logger.warning("skip remember: no PINECONE_API_KEY")
            return
        vec = await self._embed_text(text)
        if vec is None:
            return
        idx = self._pc.Index(self._index_name)
        vid = str(uuid.uuid4())
        meta = {**metadata, "text": text[:8000], "timestamp": metadata.get("timestamp", time.time())}
        await asyncio.to_thread(_sync_upsert, idx, vid, vec, meta)

    async def recall(self, query: str, k: int = 5) -> list[str]:
        s = get_settings()
        if not s.pinecone_api_key:
            return []
        vec = await self._embed_text(query)
        if vec is None:
            return []
        idx = self._pc.Index(self._index_name)
        return await asyncio.to_thread(_sync_query, idx, vec, k)


_memory: VectorMemory | None = None


def get_memory() -> VectorMemory:
    global _memory
    if _memory is None:
        _memory = VectorMemory()
    return _memory
