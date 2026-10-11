"""Production entrypoint: FastAPI host (health probes + background scanner) with Chainlit mounted at /chat.

    uvicorn main:app --host 0.0.0.0 --port 8000        # keep --workers 1 (see README)
"""
from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager

from chainlit.utils import mount_chainlit
from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse

from mmrag.config import get_settings
from mmrag.container import get_container
from mmrag.logging_setup import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(get_settings().log_level)
    container = get_container()
    await asyncio.to_thread(container.prepare)                     # load CLIP + index before serving
    scanner = asyncio.create_task(container.scanner_loop(), name="ingestion-scanner")
    try:
        yield
    finally:
        scanner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scanner


app = FastAPI(title="Multimodal PDF RAG", lifespan=lifespan)


@app.get("/healthz", include_in_schema=False)
async def healthz():
    return {"status": "ok"}                                         # liveness: the process is up


@app.get("/readyz", include_in_schema=False)
async def readyz():
    c = get_container()                                             # readiness: model + index loaded
    if not c.ready:
        return JSONResponse({"status": "starting"}, status_code=503)
    return {"status": "ready", **c.db.stats()}


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/chat")


mount_chainlit(app=app, target="ui/chainlit_app.py", path="/chat")
