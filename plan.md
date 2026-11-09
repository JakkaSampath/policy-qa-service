# Plan: Policy Q&A Service (RAG API with Ollama + Qdrant)

## 1. Goal
An HTTP service that answers questions using only the 6 policy docs in `mock-policies/`. Every answer cites its file and section, and the service refuses when the docs don't cover the question. It ships with tests, a way to measure answer quality, and protection against abuse.

This is a **grounded RAG API**:
- **Retrieval:** it finds the relevant policy sections in Qdrant.
- **Augmented:** it puts those sections into the prompt.
- **Generation:** Ollama writes the answer from them.
- **API:** it's exposed over HTTP (`POST /ask`).

## 2. Where things run
| What | Where |
|---|---|
| Ollama (`gemma3`, `embeddinggemma-2`) | Developer's laptop, `http://localhost:11434` |
| Code is written | Remote dev server (`policy-qa-service/`) |
| Unit and API tests (fake LLM and fake embeddings) | Anywhere; no Ollama needed |
| Ingest, integration test, eval, running the API | Laptop (needs Ollama) |

## 3. Tech stack
| Part | Choice |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI + Uvicorn |
| Vector DB | Qdrant, local mode (`qdrant-client`: runs inside the app, saves to `./data/qdrant`, no server) |
| Embeddings | Ollama `embeddinggemma-2` |
| LLM | Ollama `gemma3` |
| Tests | pytest |
| Packaging | `pyproject.toml` |

## 4. Architecture

```
                 ┌──────────── INGEST (run once / on doc change) ────────────┐
mock-policies/*.md ─► split by ## section ─► embed (Ollama) ─► store in Qdrant  
                                                                   (text + file + section,
                                                                    + which embed model)

                 ┌──────────────────── QUERY (POST /ask) ────────────────────┐
question ─► validate ─► rate limit ─► cache? ─► embed ─► Qdrant top-k
         ─► best similarity < threshold? ──yes──► REFUSE (no LLM call)
                     │ no
                     ▼
         LLM (concurrency-limited, timeout) with "answer ONLY from these sections"
         ─► parse JSON {answer, citations}
         ─► every citation in retrieved set? ──no──► REFUSE
                     │ yes
                     ▼
         return answer + citations
```

## 5. Folder structure
```
policy-qa-service/
├── mock-policies/            # source docs
├── app/
│   ├── config.py             # settings from env / .env
│   ├── chunking.py           # md → sections
│   ├── store.py              # Qdrant + embeddings
│   ├── llm.py                # Ollama client, prompt, JSON parsing
│   ├── qa.py                 # retrieve → decide → answer → verify
│   ├── guards.py             # rate limit, concurrency, cache
│   ├── evaluation.py         # eval scoring, threshold sweep, report
│   └── main.py               # FastAPI routes
├── scripts/
│   ├── ingest.py             # build the vector index
│   └── ask.py                # ask one question from the terminal
├── eval/
│   ├── questions.jsonl       # 34 labeled questions
│   └── run_eval.py           # scores quality
├── tests/                    # pytest (fake LLM + fake embeddings)
├── .env.example
├── CLAUDE.md                 # guidance for AI coding assistants
├── plan.md                   # this file
└── README.md                 # tester handbook
```

## 6. API contract
**`POST /ask`**
```json
// request
{ "question": "How many days per week can I work remotely?" }

// answered
{ "answered": true,
  "answer": "Up to 3 days per week with manager approval.",
  "citations": [{"file": "remote-work-policy.md", "section": "Eligibility"}],
  "reason": "answered", "top_similarity": 0.82, "cached": false }

// refused
{ "answered": false,
  "answer": "I can't find this in the policy documents.",
  "citations": [],
  "reason": "no_relevant_docs", "top_similarity": 0.59, "cached": false }
```
`reason` is one of `answered`, `no_relevant_docs`, `model_refused`, `bad_citation`, `bad_llm_output`.

**`GET /health`**: reports the status of the app, Ollama and the index.

**Errors:** `413` request body too large · `422` invalid input or question too long · `429` rate limited · `503` LLM busy, Ollama down, index empty, or index built with another model · `504` LLM timeout

## 7. How refusal works (3 layers)
1. **Retrieval gate:** if the closest chunk's similarity is below the threshold, refuse without calling the LLM.
2. **Prompt rule:** the LLM must return `answered: false` if the sections don't contain the answer.
3. **Citation check:** if the LLM cites a section it wasn't given, or cites nothing, treat it as a refusal.

## 8. Abuse protection
Ollama is local and free, so abuse threatens the laptop's CPU/GPU rather than a bill.

| Risk | Protection |
|---|---|
| Huge input | Body over ~2.5 KB rejected with `413` before it's read; question max 500 characters (`422`) |
| Spam | Rate limit per IP (for example 10 requests per minute) |
| CPU overload | 1 answer at a time (`MAX_CONCURRENT_LLM`); extra requests get `503` + `Retry-After: 5`, not queued |
| Slow or hung Ollama | 90-second LLM timeout (`504`), 60-second embedding timeout; documents embedded in batches of 16 |
| Long output | `num_predict` cap of about 300 tokens |
| Repeated questions | Cache answers in memory, keyed by the normalized question |
| Prompt injection ("ignore instructions…") | System prompt plus the citation check; the eval includes injection cases |

## 9. Testing strategy
- **Unit tests:** chunking, the threshold logic, JSON parsing, the citation check and the guards
- **API tests:** all status codes and response shapes, using a **fake LLM and fake embeddings**, so they're fast and free, give the same result every run, and don't need Ollama
- **Integration tests (`integration` marker):** run against the real Ollama. They're auto-skipped, with the reason shown, when Ollama isn't reachable or a configured model is missing.

## 10. Measuring answer quality
`eval/questions.jsonl` holds 34 labeled cases:
- 20 answerable, each with the expected file and section, key facts the answer must contain, and sometimes facts it must not contain
- 7 not covered (for example "What's the cafeteria menu?")
- 7 tricky ones: prompt injection, a fake "SYSTEM:" message, near misses ("What is the dress code?", part-time vacation), a question spanning two policies (VPN), arithmetic (100 miles) and a "not allowed" answer (keystroke tracking)

`run_eval.py` reports:

| Metric | Meaning |
|---|---|
| Refusal accuracy | Refuses when it should and answers when it should |
| False answers | Answered a question the docs don't cover (worst error) |
| Wrongly refused | Refused a question the docs do cover |
| Citation accuracy | The cited file and section match the expected ones |
| Answer correctness | The answer contains the expected key facts and none of the forbidden ones |
| Latency | p50 and p95 |

It also sweeps the similarity threshold (0.50–0.85) and recommends `MIN_SIMILARITY`. Results are in the step 6 notes.

## 11. Laptop setup
```bash
# Ollama models (done)
ollama pull gemma3
ollama pull embeddinggemma-2

# Python environment
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"     # Windows: .venv\Scripts\pip

# Build the index, then run all tests (integration tests included)
.venv/bin/python scripts/ingest.py
.venv/bin/pytest -v
```
If `python3 -m venv` fails on Ubuntu without sudo, use `python3 -m venv --without-pip .venv`, then
`PYTHONPATH=/usr/lib/python3/dist-packages .venv/bin/python -m pip install --upgrade pip`.

## 12. Build steps
| # | Step | Status / done when |
|---|---|---|
| 0 | Ollama with `gemma3` + `embeddinggemma-2` on the laptop | ✅ Done by the user |
| 1 | Skeleton: pyproject, config, `.env.example` | ✅ Done |
| 2 | Chunking + tests | ✅ Done: 54 chunks, 10 tests |
| 3 | Vector store (Qdrant) + ingest script + tests | ✅ Done: laptop ingest indexed 54 chunks; **31/31 tests pass on the laptop** (incl. real-Ollama integration) |
| 4 | LLM, prompt, QA logic + tests | ✅ Done: 54/54 tests pass on the laptop (incl. 3 real-Ollama); `scripts/ask.py` answers and refuses correctly |
| 5 | FastAPI + guards + API tests | ✅ Done: **91/91 tests pass on the laptop**; real API checked (`/health` ok, answer in 27.7 s, repeat served from cache) |
| 6 | Eval set + runner | ✅ Done: **114/114 tests pass on the laptop**; eval 33/34, refusal accuracy 100%, **0 false answers**; `MIN_SIMILARITY=0.625` |
| 7 | README (tester handbook) | ✅ Done: followed from a fresh clone on the laptop (setup, ingest 54 chunks, `ask.py`, `/health` ok, answer + refusal via the API); **114/114 tests pass** |

### Step 6 notes
- `eval/questions.jsonl`: 34 labelled questions (20 answerable, 7 out of scope, 7 tricky). Run: `.venv/Scripts/python eval/run_eval.py` (`--ids a01,t03` for a few). The API must be stopped. The report is saved to `eval/results/` (not in git).
- Each question runs once with the gate off; the threshold sweep is applied afterwards (an estimate).
- **Baseline (laptop, 2026-11-07, MIN_SIMILARITY=0.5):** 31/34 passed, **0 false answers**, refusal accuracy 97%, citation accuracy 96%, answer correctness 91%, latency p50 19.3 s / p95 33.9 s (slowest 54.9 s).
  - a03: added "full-time" (not in Sick Leave) and left "according to." after label removal.
  - a18: cited the leave policy's "Document Info" for the expense policy date (every doc has a section with that name).
  - a20: refused the alcohol question, which is answered by an exclusion list.
- **Fixes:** prompt rules 6 (no added conditions) and 7 (exclusions count as answers); the source header now includes the document title; label cleanup removes "according to S1", "(see S1)" and "(S1, S2)" as whole phrases.
- **Threshold:** 0.50–0.725 all scored the same; above 0.75 good questions get refused. Set `MIN_SIMILARITY=0.625` (middle; skips the LLM for ~15% of questions). Lowest answerable top similarity 0.731, highest unanswerable 0.770 (part-time), so the gate can't separate them alone. The model (layer 2) does the rest.
- `LLM_TIMEOUT_S` 60 → 90: the slowest eval question took 55 s on the laptop CPU.
- **Run 2 (threshold 0.625):** 33/34, refusal accuracy 100%, 0 false answers, p50 15.9 s / p95 28.1 s. a03 and a20 fixed.
- a18 still cited the leave policy's date. A retrieval check showed the expense "Document Info" was ranked #1 (0.857), so the model mixed up two near-identical date blocks. Fix: the source header now starts with the policy name (`[S1] Expense and Reimbursement Policy > Document Info  (file: expense-policy.md)`), plus prompt rule 8 (a question about a specific policy uses only that policy's sources).
- **Final (laptop, commit 3800fba):** 114/114 tests pass; eval **33/34**, refusal accuracy 100%, 0 false answers, 0 wrongly refused. a18 still fails: a **known limit** of gemma3 (see section 13).

### Step 5 notes
- Run: `.venv/Scripts/uvicorn app.main:app --host 127.0.0.1 --port 8000` (Linux/macOS: `.venv/bin/uvicorn`). Try it at `http://127.0.0.1:8000/docs`.
- **1 server process only** (uvicorn's default): the rate limit and cache are in memory, and the index allows one process. Stop the API before running ingest or the tests.
- Order of checks on `POST /ask`: body size (`413`) → rate limit per IP (`429`) → input (`422`) → cache → answer slot (`503` busy) → index (`503`) → answer (`503` Ollama down, `504` timeout).
- Refusals are cached (asking the same out-of-scope question again is free); errors never are.
- Known limits: the rate limit is per IP (behind a proxy everyone shares one); a request without `Content-Length` skips the body-size check (the 500-character limit still applies).
- Seen in the laptop check: the answer "**Full-time** employees are entitled to 10 days of paid sick leave" cited only *Sick Leave*, but "full-time" comes from *Annual Leave*. Step 6 should catch facts that aren't in the cited section.

### Step 4 notes
- Sources are sent to gemma3 labelled S1, S2, …; it must reply in a fixed JSON shape (Ollama structured output) and cite labels. Citations are verified against the labels given; leftover labels in the answer text are removed.
- Every result has a `reason`: `answered`, `no_relevant_docs`, `model_refused`, `bad_citation`, `bad_llm_output`.
- `scripts/ask.py "question"` asks one question from the terminal (no API needed).

### Step 3 notes
- Each embedding model has its own task prefixes. For `embeddinggemma`: `task: search result | query: …` for questions, `title: none | text: …` for sections. They're picked automatically by model name in `app/store.py`.
- The index records which embedding model built it. If `EMBED_MODEL` changes, search raises a clear "run scripts/ingest.py" error instead of silently returning wrong results.
- **Only one process can open the index folder at a time** (Qdrant local mode locks it). Stop the API before running ingest; a second open gives a clear "already accessed" error. Code must close the store (`with VectorStore(...) as store:`).
- **Why Qdrant and not ChromaDB:** ChromaDB 1.x's native (Rust) core crashed with an access violation on the first `add` on a Windows laptop whose Visual C++ runtime (`msvcp140.dll` 14.13, from 2017) is too old (known issue: chroma-core/chroma #6052, #4399). Updating the runtime needed admin rights. ChromaDB's Python fallback needs `chroma-hnswlib`, which has no Windows build for Python 3.12. Qdrant local mode is plain Python, and its native dependencies (`numpy` ships its own runtime, `grpcio` has it built in) don't use the system's `msvcp140.dll`.

## 13. Known risks / open items
- **Known limit (eval a18): near-identical metadata blocks.** Asked when the expense policy was last updated, gemma3 answers with the leave policy's date ("September 2026" instead of "July 2026") and cites the leave policy, although the search ranks the expense policy's "Document Info" first (0.857 vs 0.767). The header and prompt fixes in step 6 didn't change it. It's a wrong-citation error inside the docs, not an outside-knowledge answer. The eval keeps checking it, so a stronger model or a later fix can be measured against it.
- **Threshold (first `embeddinggemma-2` numbers, laptop, 4 questions):** answerable 0.817–0.834; unanswerable 0.591 (cafeteria) and 0.680 (dress code). At the 0.5 placeholder, both unanswerable questions reached gemma3 and were refused by the model (`model_refused`, ~13 s each). A threshold around 0.70 would refuse them instantly. **Tuned in step 6: 0.625** (see the step 6 notes).
- **Near misses get past the threshold.** "What is the dress code?" scored close to answerable questions (it sounds like the Code of Conduct), so the LLM has to refuse it. That's covered by layers 2–3 and by the eval.
- **Speed (laptop):** 11–18 s per answered question, ~13 s per model refusal; the full test suite takes ~83 s. A first answer after a model load took 27.7 s through the API, which is why only 1 answer runs at a time. The eval's slowest question took 55 s, so the LLM timeout is 90 s (embedding: 60 s). A GPU machine can change these in `.env`.
- **A small model can be sloppy with JSON or citations.** The parser and citation check handle this, and the eval will show how often it happens.
- **Store calls are blocking.** The API runs them in a thread pool (`run_in_threadpool`), so one slow call doesn't freeze the API.
- **Rebuild isn't atomic.** It deletes the index, then re-adds everything (see the step 3 notes).
