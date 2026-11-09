# CLAUDE.md

Guidance for AI coding assistants working in this repository.

## What this is
A grounded RAG API: `POST /ask` answers questions only from the 6 policy docs in `mock-policies/`, cites file + section, and refuses when the docs don't cover the question. Ollama runs the models locally (`embeddinggemma-2` for embeddings, `gemma3` for answers). Qdrant runs in local mode (no server) and stores the index in `data/qdrant`.

`README.md` is the tester handbook. `plan.md` has the design, the build history and the eval results. Read both before larger changes.

## Commands
Windows (Git Bash) uses `.venv/Scripts/...`; Linux/macOS uses `.venv/bin/...`.

| Task | Command |
|---|---|
| Install | `python -m pip install -e ".[dev]"` |
| Fast tests (no Ollama) | `pytest -m "not integration"` |
| All tests | `pytest -v` (114; integration tests auto-skip without Ollama) |
| Build the index | `python scripts/ingest.py` |
| Ask from the terminal | `python scripts/ask.py "question"` |
| Run the API | `uvicorn app.main:app --host 127.0.0.1 --port 8000` |
| Quality eval | `python eval/run_eval.py` (all 34) or `--ids a01,t03` |

## Code map
- `app/config.py`: settings (pydantic-settings, reads `.env`)
- `app/chunking.py`: markdown → chunks by `##`/`###` headings (54 chunks)
- `app/store.py`: Ollama embeddings + Qdrant `VectorStore`; records which embed model built the index
- `app/llm.py`: Ollama `/api/chat` with a JSON schema
- `app/qa.py`: prompt, 3-layer refusal, citation check, label cleanup
- `app/guards.py`: rate limiter, slot limiter, answer cache
- `app/evaluation.py`: eval scoring and threshold sweep
- `app/main.py`: FastAPI app via `create_app()` (tests inject fakes)

## Rules and traps
- **Grounding is the product.** Never weaken the 3 refusal layers (similarity gate, model `answered=false`, citation check against the S-labels given). After changing the prompt, retrieval or the threshold, run the eval. **False answers must stay at 0.**
- **One process per index.** Qdrant local mode locks `data/qdrant`. Stop the API before ingest or the eval. Always close the store (`with VectorStore(...) as store:`).
- **Re-run ingest** after editing `mock-policies/` or changing `EMBED_MODEL`; a mismatched index raises a clear error.
- **No `from __future__ import annotations` in `app/main.py`.** It breaks FastAPI's dynamic `AskRequest` model: requests get a 422.
- **embeddinggemma needs task prefixes** (`task: search result | query: ` / `title: none | text: `); they're chosen by model name in `store.py`.
- **Unit and API tests use fakes** (`tests/conftest.py`: `FakeEmbedder`, `FakeLLM`). New logic gets a fast test; real-Ollama checks use the `integration` marker.
- **Don't edit `eval/questions.jsonl` to make a failing case pass.** Fix the code, or record it as a known limit in `plan.md` (like a18).
- `MIN_SIMILARITY=0.625` was tuned by the eval. Re-check it with the eval before changing it.
- Never commit `.env`, `data/` or `eval/results/` (they're in `.gitignore`).

## Known limit
Eval case a18: gemma3 gives the leave policy's date when asked about the expense policy's date (the two "Document Info" sections are nearly identical). Documented in `plan.md` §13 and the README.
