# Multimodal PDF RAG

Ask questions about PDFs - text **and** the images/charts inside them. This app uses LangChain's chat,
prompt and retriever interfaces, with a CLIP vector space shared by text and images, a durable SQLite
ingestion store, guardrails, evaluation and a Chainlit UI.

```
 inbox/*.pdf ──► scanner ──► loader ──► chunker ──► injection scan ──► CLIP embed ──► SQLite
 (drop or upload) (sha256   (PyMuPDF)  (CLIP-token   (quarantine       (text+images)    │ source of truth
                  manifest)             + overlap)     bad pages)                        ▼
                                                                                   in-memory index
 question ─► guardrails ─► LangChain retriever (CLIP text + image search) ─► merge windows ─► │
              (len, moderation,                                                          ▼
                injection, rate)        stream ◄── LangChain chat model ◄── prompt ◄─ passages + images
                  │                       │
                  ▼                       ▼
            Chainlit UI ◄── output guard (leak canary, PII, markdown-image strip) ──► cited sources
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # set OPENAI_API_KEY, or configure Groq below
cp your.pdf data/inbox/         # or attach PDFs in the chat UI
uvicorn main:app --port 8000    # open http://localhost:8000
```
Docker: `docker compose up --build`. Probes: `/healthz` (liveness), `/readyz` (model + index loaded).

### Use Groq

Set `LLM_PROVIDER=groq` and `GROQ_API_KEY` in `.env`. With no `LLM_MODEL` override, the app uses
Groq's `qwen/qwen3.8-27b` vision model; image questions require a Groq model that supports image input.
The chat request uses LangChain's `ChatGroq` integration. OpenAI moderation and prompt-cache hints are
OpenAI-only; moderation is disabled when using Groq. CLIP embeddings remain local and do not use either API key.
The RAGAS evaluation script still uses OpenAI as its judge and needs `OPENAI_API_KEY`.

CLI (no server): `python scripts/ingest.py [--watch | --retry-failed | --status]`

## Layout

| Component | File |
|---|---|
| 1. Ingestion scanner | `mmrag/ingestion/scanner.py` - new / changed / deleted / unchanged via size+mtime+sha256 manifest |
| 1. Ingestion pipeline | `mmrag/ingestion/pipeline.py` - single-writer, idempotent, per-document failure isolation |
| 2. Chunks + overlap | `mmrag/ingestion/chunking.py` - sized in CLIP tokens, with overlap and exact offsets |
| 3. Embeddings + config | `mmrag/embeddings.py`, `mmrag/config.py` (every knob is an env var) |
| 4. Guardrails | `mmrag/guardrails.py` - length, moderation, rate limit, PII redaction, leak detection |
| 5. Document loader | `mmrag/loaders/pdf.py` - text clean-up, image filtering/dedupe/downscale |
| 6. Retrieval | `mmrag/retrieval.py`, `mmrag/store.py` |
| 7. Prompt caching | `mmrag/prompts.py`, `mmrag/llm.py` |
| 8. RAGAS evaluation | `scripts/evaluate.py` |
| 9. Prompt injection | `mmrag/injection.py` (+ `prompts.py`, `guardrails.py`) |
| UI / host | `ui/chainlit_app.py`, `main.py` |

## Decisions worth knowing

**Chunks are measured in CLIP tokens.** CLIP truncates text at 77 tokens; the notebook's 500-character chunks
were silently cut. Defaults are 60 tokens / 12 overlap. Small chunks embed well but read poorly, so retrieval is
*small-to-big*: embed the small chunk, give the LLM the chunk plus `CONTEXT_WINDOW_CHUNKS` neighbours (overlapping
windows on the same page are merged).

**Text and images are searched separately** (`TOP_K_TEXT` / `TOP_K_IMAGES`). CLIP text->image scores are lower than
text->text, so one shared top-k would almost never return an image. Tune `MIN_SCORE_*` on your own documents.

**SQLite is the source of truth; the vector index is rebuilt in memory** (exact cosine search, same results as FAISS
`IndexFlatIP`, no native dependency). Fine to a few hundred thousand chunks. If you change the CLIP model or chunk
settings, the app detects it and rebuilds from the inbox automatically.

**Prompt caching** (OpenAI, automatic, prompts >= 1,024 tokens, identical prefix): the prompt is ordered
`system -> chat history -> retrieved context + question`, so the static prefix is reused across users and turns, and every
request sends the same `prompt_cache_key`. Cache hits are logged per request (`cached_tokens`, `cache_hit_ratio`).
Short first-turn prompts fall under 1,024 tokens and won't hit; hits grow as conversations lengthen.

**Prompt injection, layered:** (1) heuristic scoring of user queries and document text (full-width/zero-width
obfuscation normalised); pages that trip it are quarantined at ingestion and passages re-checked at retrieval;
(2) retrieved text is HTML-escaped inside `<context>` tags the system prompt declares to be untrusted data;
(3) a per-deployment canary in the system prompt aborts the stream if it leaks, and markdown images (an exfiltration
channel) are stripped from answers. Heuristics are not a complete defence; keep the LLM unprivileged (this app has no tools).

**Guardrails fail open on moderation outages** (so an OpenAI blip doesn't take chat down). Change `InputGuard.moderate`
if you need fail-closed.

## Evaluation (RAGAS)

```bash
pip install -r requirements-eval.txt
# write real questions + reference answers in eval/testset.jsonl (the shipped file is a template)
python scripts/evaluate.py --min-faithfulness 0.8
```
Runs each question through the real pipeline, scores faithfulness, answer relevancy, context precision and context
recall, writes `eval/results.csv`, and exits 1 under threshold (a CI gate). RAGAS judges **text**: image passages are
passed as a placeholder, so answers that rely on a chart can look unfaithful. Treat image-heavy questions as manual review.

## LangChain integration

Chat generation uses a LangChain `ChatPromptTemplate` and streaming chat model (`ChatOpenAI` or `ChatGroq`);
hybrid retrieval is exposed as a LangChain retriever, and CLIP implements LangChain's embeddings interface.
The PDF-specific loader, transactional SQLite database and in-memory exact cosine index remain app-owned so
document ingestion, separate text/image quotas, context-window merging and existing citations are preserved.

## Operating notes

- **Run one worker.** The index is in memory and one process owns the ingestion lock. Scale up by moving the store
  to pgvector/Qdrant and the rate limiter to Redis.
- One shared corpus for all users (no per-user isolation). Set `ALLOW_UPLOADS=false` and manage the inbox yourself if
  users shouldn't add documents. Enable login with `APP_USERNAME`/`APP_PASSWORD`/`CHAINLIT_AUTH_SECRET`.
- Logs are JSON, one line per event; questions are logged as a hash, never raw.
- Text inside images is not scanned for injection (no OCR); the system prompt tells the model to treat it as data.
- Follow-up questions are embedded as written ("and page 2?" retrieves poorly). Add a query-rewrite step if that matters.
- Back up `data/` (the inbox is the real source; `rag.db` and `images/` can be rebuilt).

## Tests

`pytest -q` (or `python tests/run_all.py`). They use a fake embedder/loader, so they run without torch or PyMuPDF.
