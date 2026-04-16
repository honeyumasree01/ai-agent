import asyncio
import logging
import sys

from langchain_core.tools import tool

from utils.retry import with_retry

logger = logging.getLogger(__name__)

MAX_OUT = 10 * 1024
EXEC_TIMEOUT = 30.0


async def _run_code_impl(code: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out_b, _err_b = await asyncio.wait_for(proc.communicate(), timeout=EXEC_TIMEOUT)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return "(timeout)"
    text = (out_b or b"").decode(errors="replace")
    return text[:MAX_OUT]


@tool
async def run_code(code: str) -> str:
    """Local python -c; stdout only, capped. Not a real sandbox."""
    logger.debug("run_code len=%s", len(code))
    return await with_retry(_run_code_impl, code)
