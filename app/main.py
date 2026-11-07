"""HTTP API: POST /ask and GET /health.

Run (one worker only: limits and the index lock are per process):
    .venv/Scripts/uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

import logging
import math
import time
from contextlib import asynccontextmanager
from typing import Annotated, Callable

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StringConstraints, create_model

from app.config import Settings, settings as default_settings
from app.guards import AnswerCache, RateLimiter, SlotLimiter
from app.llm import LLM, LLMError, LLMTimeout, OllamaLLM
from app.qa import Answer, answer_question
from app.store import EmbeddingError, IndexMismatchError, OllamaEmbedder, VectorStore

log = logging.getLogger("uvicorn.error")  # shows up in uvicorn's console output


class CitationOut(BaseModel):
    file: str
    section: str


class AskResponse(BaseModel):
    answered: bool
    answer: str
    citations: list[CitationOut]
    reason: str
    top_similarity: float | None
    cached: bool


def ollama_problems(cfg: Settings) -> list[str]:
    """Empty list if Ollama is reachable and has both models, else what's wrong."""
    try:
        r = httpx.get(f"{cfg.ollama_url}/api/tags", timeout=2)
        r.raise_for_status()
        have = {m["name"] for m in r.json()["models"]}
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return [f"Ollama not reachable at {cfg.ollama_url}"]
    have |= {n.removesuffix(":latest") for n in have}
    return [f"Ollama model missing: {m}" for m in (cfg.embed_model, cfg.llm_model) if m not in have]


def _to_response(a: Answer, cached: bool) -> AskResponse:
    return AskResponse(
        answered=a.answered, answer=a.answer,
        citations=[CitationOut(file=c.file, section=c.section) for c in a.citations],
        reason=a.reason, top_similarity=a.top_similarity, cached=cached)


def create_app(cfg: Settings = default_settings, *, store: VectorStore | None = None,
               llm: LLM | None = None,
               check_ollama: Callable[[Settings], list[str]] = ollama_problems) -> FastAPI:
    """Build the app. Tests pass a fake store/LLM; uvicorn uses the defaults."""
    AskRequest = create_model("AskRequest", question=(Annotated[str, StringConstraints(
        strip_whitespace=True, min_length=1, max_length=cfg.max_question_chars)], ...))
    max_body_bytes = 4 * cfg.max_question_chars + 512  # worst-case UTF-8 + JSON overhead

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owns_store = store is None
        app.state.store = store or VectorStore(
            OllamaEmbedder(cfg.ollama_url, cfg.embed_model, cfg.embed_timeout_s), cfg.index_dir, cfg.collection)
        app.state.llm = llm or OllamaLLM(
            cfg.ollama_url, cfg.llm_model, cfg.llm_timeout_s, cfg.llm_max_tokens)
        try:
            yield
        finally:
            if owns_store:
                app.state.store.close()

    app = FastAPI(title="Policy Q&A", version="0.1.0", lifespan=lifespan)
    app.state.rate = RateLimiter(cfg.rate_limit_per_min)
    app.state.slots = SlotLimiter(cfg.max_concurrent_llm)
    app.state.cache = AnswerCache[Answer](cfg.cache_size)

    @app.middleware("http")
    async def guard_ask(request: Request, call_next):
        # Runs before the body is read, so oversized or spammy requests cost almost nothing.
        if request.method == "POST" and request.url.path == "/ask":
            size = request.headers.get("content-length", "")
            if size.isdigit() and int(size) > max_body_bytes:
                return JSONResponse(status_code=413, content={"detail": "Request body too large."})
            client = request.client.host if request.client else "unknown"
            retry = app.state.rate.check(client)
            if retry is not None:
                return JSONResponse(status_code=429,
                                    content={"detail": "Too many requests. Slow down."},
                                    headers={"Retry-After": str(max(1, math.ceil(retry)))})
        return await call_next(request)

    @app.post("/ask", response_model=AskResponse)
    async def ask(body: AskRequest) -> AskResponse:
        question = body.question
        hit = app.state.cache.get(question)
        if hit is not None:
            return _to_response(hit, cached=True)

        if not app.state.slots.try_acquire():
            raise HTTPException(503, "Busy answering other questions. Try again shortly.",
                                headers={"Retry-After": "5"})
        start = time.perf_counter()
        try:
            st = app.state.store
            if await run_in_threadpool(st.count) == 0:
                raise HTTPException(503, "Index is empty. Run: python scripts/ingest.py")
            a = await run_in_threadpool(answer_question, question, st, app.state.llm,
                                        top_k=cfg.top_k, min_similarity=cfg.min_similarity)
        except LLMTimeout as e:
            log.warning("ask: LLM timeout: %s", e)
            raise HTTPException(504, "The language model took too long to answer.")
        except IndexMismatchError as e:
            raise HTTPException(503, str(e))
        except (LLMError, EmbeddingError) as e:
            log.warning("ask: Ollama error: %s", e)
            raise HTTPException(503, "Ollama is unavailable. Try again later.")
        finally:
            app.state.slots.release()

        log.info("ask: answered=%s reason=%s top=%s %.1fs", a.answered, a.reason,
                 a.top_similarity, time.perf_counter() - start)
        app.state.cache.put(question, a)
        return _to_response(a, cached=False)

    @app.get("/health")
    async def health() -> JSONResponse:
        st = app.state.store
        problems = await run_in_threadpool(check_ollama, cfg)
        chunks = await run_in_threadpool(st.count)
        built = await run_in_threadpool(st.built_with)
        if chunks == 0:
            problems.append("Index is empty. Run: python scripts/ingest.py")
        elif built != st.embedder.model:
            problems.append(f"Index was built with '{built}' but EMBED_MODEL is "
                            f"'{st.embedder.model}'. Run: python scripts/ingest.py")
        body = {"status": "ok" if not problems else "unavailable", "problems": problems,
                "index_chunks": chunks, "embed_model": cfg.embed_model,
                "llm_model": cfg.llm_model}
        return JSONResponse(status_code=200 if not problems else 503, content=body)

    return app


app = create_app()
