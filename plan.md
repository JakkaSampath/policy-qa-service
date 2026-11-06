# Plan: Policy Q&A Service (RAG API with Ollama + ChromaDB)

## 1. Goal
An HTTP service that answers questions using only the 6 policy docs in `mock-policies/`. Every answer cites its file and section, and the service refuses when the docs don't cover the question. It ships with tests, a way to measure answer quality, and protection against abuse.

This is a **grounded RAG API**:
- **Retrieval:** it finds the relevant policy sections in ChromaDB.
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
| Vector DB | ChromaDB 1.x (runs inside the app, saves to `./data/chroma`) |
| Embeddings | Ollama `embeddinggemma-2` |
| LLM | Ollama `gemma3` |
| Tests | pytest |
| Packaging | `pyproject.toml`, Docker Compose |

## 4. Architecture

```
                 ┌──────────── INGEST (run once / on doc change) ────────────┐
mock-policies/*.md ─► split by ## section ─► embed (Ollama) ─► store in ChromaDB
                                                                   (text + file + section,
                                                                    + which embed model)

                 ┌──────────────────── QUERY (POST /ask) ────────────────────┐
question ─► validate ─► rate limit ─► cache? ─► embed ─► ChromaDB top-k
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
│   ├── store.py              # ChromaDB + embeddings
│   ├── llm.py                # Ollama client, prompt, JSON parsing
│   ├── qa.py                 # retrieve → decide → answer → verify
│   ├── guards.py             # rate limit, concurrency, cache
│   └── main.py               # FastAPI routes
├── scripts/ingest.py         # build the vector index
├── eval/
│   ├── questions.jsonl       # ~30 labeled questions
│   └── run_eval.py           # scores quality
├── tests/                    # pytest (fake LLM + fake embeddings)
├── .env.example
├── Dockerfile, docker-compose.yml
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
  "citations": [{"file": "remote-work-policy.md", "section": "Eligibility"}] }

// refused
{ "answered": false,
  "answer": "I can't find this in the policy documents.",
  "citations": [] }
```

**`GET /health`**: reports the status of the app, Ollama and the index.

**Errors:** `422` invalid input or question too long · `429` rate limited · `503` LLM busy, Ollama down, or index built with another model · `504` LLM timeout

## 7. How refusal works (3 layers)
1. **Retrieval gate:** if the closest chunk's similarity is below the threshold, refuse without calling the LLM.
2. **Prompt rule:** the LLM must return `answered: false` if the sections don't contain the answer.
3. **Citation check:** if the LLM cites a section it wasn't given, or cites nothing, treat it as a refusal.

## 8. Abuse protection
Ollama is local and free, so abuse threatens the laptop's CPU/GPU rather than a bill.

| Risk | Protection |
|---|---|
| Huge input | Max question length of 500 characters |
| Spam | Rate limit per IP (for example 10 requests per minute) |
| CPU overload | At most 2 concurrent LLM calls; reply `503` when full |
| Slow or hung LLM | 30-second timeout |
| Long output | `num_predict` cap of about 300 tokens |
| Repeated questions | Cache answers in memory, keyed by the normalized question |
| Prompt injection ("ignore instructions…") | System prompt plus the citation check; the eval includes injection cases |

## 9. Testing strategy
- **Unit tests:** chunking, the threshold logic, JSON parsing, the citation check and the guards
- **API tests:** all status codes and response shapes, using a **fake LLM and fake embeddings**, so they're fast and free, give the same result every run, and don't need Ollama
- **Integration tests (`integration` marker):** run against the real Ollama. They're auto-skipped, with the reason shown, when Ollama isn't reachable or a configured model is missing.

## 10. Measuring answer quality
`eval/questions.jsonl` holds about 30 labeled cases:
- about 18 answerable, each with the expected file and section, plus key facts the answer must contain
- about 8 not covered (for example "What's the cafeteria menu?")
- about 4 tricky ones: prompt injection, a question that mixes two policies, and near misses such as "What is the dress code?"

`run_eval.py` reports:

| Metric | Meaning |
|---|---|
| Refusal accuracy | Refuses when it should and answers when it should |
| Citation accuracy | The cited file and section match the expected ones |
| Answer correctness | The answer contains the expected key facts |
| Latency | p50 and p95 |

It also helps tune the similarity threshold.

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
| 3 | ChromaDB store + ingest script + tests | ✅ Done: code and tests pass (29 passed, 1 integration skipped on the dev server). **Laptop: run ingest + integration test.** |
| 4 | LLM, prompt, QA logic + tests | Fake-LLM tests pass for answer, refuse and bad-citation cases |
| 5 | FastAPI + guards + API tests | All status codes are tested |
| 6 | Eval set + runner | Eval report prints the metrics; threshold is tuned |
| 7 | README (tester handbook) + Docker Compose | A tester can run it from the README alone |

### Step 3 notes
- Each embedding model has its own task prefixes. For `embeddinggemma`: `task: search result | query: …` for questions, `title: none | text: …` for sections. They're picked automatically by model name in `app/store.py`.
- The index records which embedding model built it. If `EMBED_MODEL` changes, search raises a clear "run scripts/ingest.py" error instead of silently returning wrong results.
- After re-running ingest, restart the API so it picks up the new index. Don't run 2 ingests at once.

## 13. Known risks / open items
- **The threshold must be re-measured with `embeddinggemma-2`.** The first measurement used nomic: answerable questions scored 0.75–0.86 and unanswerable ones 0.49–0.70, which suggested about 0.65. Those numbers don't carry over to a different embedding model. The `min_similarity` default (0.5) is a placeholder until step 6.
- **Near misses get past the threshold.** "What is the dress code?" scored close to answerable questions (it sounds like the Code of Conduct), so the LLM has to refuse it. That's covered by layers 2–3 and by the eval.
- **Speed depends on the laptop.** `gemma3` speed hasn't been measured yet. The 30-second timeout and the concurrency limit of 2 may need tuning after step 4.
- **A small model can be sloppy with JSON or citations.** The parser and citation check handle this, and the eval will show how often it happens.
- **Store calls are blocking.** Step 5 must run them in a thread pool so one slow call doesn't freeze the API.
- **Rebuild isn't atomic.** It deletes the index, then re-adds everything (see the step 3 notes).
