"""Chainlit UI. All logic lives in mmrag/; this file only renders events."""
from __future__ import annotations

import asyncio
import hmac
import os
import re
import shutil
from pathlib import Path

import chainlit as cl

from mmrag.config import get_settings
from mmrag.container import get_container
from mmrag.models import Blocked, Final, Retrieved, Token

settings = get_settings()


# ------------------------------------------------------------------ optional auth
if settings.auth_enabled:
    @cl.password_auth_callback
    def auth(username: str, password: str):
        ok_user = hmac.compare_digest(username.encode(), settings.app_username.encode())
        ok_pass = hmac.compare_digest(password.encode(), settings.app_password.get_secret_value().encode())
        return cl.User(identifier=username, metadata={"role": "user"}) if (ok_user and ok_pass) else None


# ------------------------------------------------------------------ helpers
def _caller_key() -> str:
    user = cl.user_session.get("user")
    return user.identifier if user else cl.user_session.get("id")


def _safe_pdf_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(name).name).strip(" .") or "upload"
    return stem if stem.lower().endswith(".pdf") else stem + ".pdf"


async def _handle_uploads(files: list) -> None:
    c = get_container()
    if not settings.allow_uploads:
        await cl.Message(content="Uploads are disabled on this deployment.").send()
        return
    saved = []
    for f in files:
        src = Path(f.path)
        with open(src, "rb") as fh:
            header = fh.read(5)
        if header != b"%PDF-":
            await cl.Message(content=f"`{f.name}` doesn't look like a valid PDF - skipped.").send()
            continue
        if src.stat().st_size > settings.max_file_bytes:
            await cl.Message(content=f"`{f.name}` is larger than {settings.max_file_mb} MB - skipped.").send()
            continue
        dest = settings.inbox_dir / _safe_pdf_name(f.name)
        tmp = dest.with_name(f".upload-{os.getpid()}-{dest.name}")   # dot-prefixed: the scanner ignores it
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)                                         # atomic: scanner never sees a partial file
        saved.append(dest.name)
    if not saved:
        return
    async with cl.Step(name="Indexing documents", type="tool") as step:
        step.input = ", ".join(saved)
        await asyncio.sleep(max(settings.scan_settle_seconds, 0))     # let the file settle
        report = await asyncio.to_thread(c.pipeline.run)
        step.output = report.summary()
    problems = [f"`{d}`: {e}" for d, e in report.failed]
    await cl.Message(content=("Indexed: " + ", ".join(f"`{n}`" for n in saved)
                              + ("\n\nProblems:\n- " + "\n- ".join(problems) if problems else ""))).send()


# ------------------------------------------------------------------ chat lifecycle
@cl.set_starters
async def starters(user=None):
    return [
        cl.Starter(label="Summarise the documents", message="Summarise the main findings of the documents."),
        cl.Starter(label="What charts or figures are there?", message="What charts, figures or images appear in the documents, and what do they show?"),
        cl.Starter(label="Key numbers", message="List the key numbers and statistics mentioned, with their sources."),
    ]


@cl.on_chat_start
async def on_start():
    cl.user_session.set("history", [])
    c = get_container()
    if not c.ready:
        await cl.Message(content="The assistant is still starting up. Please try again in a moment.").send()
    elif c.db.stats()["documents_ready"] == 0:
        await cl.Message(content="No documents are indexed yet. **Attach a PDF** to this chat "
                                 "(or drop it in the server's inbox folder) and then ask your question.").send()


@cl.on_message
async def on_message(message: cl.Message):
    c = get_container()
    if not c.ready:
        await cl.Message(content="The assistant is still starting up. Please try again in a moment.").send()
        return

    pdfs = [e for e in (message.elements or [])
            if getattr(e, "mime", "") == "application/pdf" or str(getattr(e, "name", "")).lower().endswith(".pdf")]
    if pdfs:
        await _handle_uploads(pdfs)
    query = (message.content or "").strip()
    if not query:
        return

    history: list[tuple[str, str]] = cl.user_session.get("history")
    out = cl.Message(content="")
    final: Final | None = None

    async for ev in c.rag.ask(query, session_id=_caller_key(), history=history):
        if isinstance(ev, Retrieved):
            r = ev.result
            async with cl.Step(name="Retrieval", type="retrieval") as step:
                step.input = query
                step.output = ("\n".join(f"- {p.title} (score {p.score:.2f}, {p.modality})" for p in r.all)
                               or "No relevant passages found.")
        elif isinstance(ev, Token):
            await out.stream_token(ev.text)
        elif isinstance(ev, Blocked):
            out.content = ev.message
            await out.send()
            return
        elif isinstance(ev, Final):
            final = ev

    if final is None:
        return
    elements, source_names = [], []
    for p in final.sources:
        if p.modality == "image":
            elements.append(cl.Image(name=p.title, path=p.image_path, display="inline", size="medium"))
        else:
            elements.append(cl.Text(name=p.title, content=p.text, display="side"))
            source_names.append(p.title)
    out.content = final.answer + (("\n\n**Sources:** " + " · ".join(source_names)) if source_names else "")
    out.elements = elements
    await out.send()

    history.append((query, final.answer))
    cl.user_session.set("history", history[-settings.history_turns:] if settings.history_turns else [])
