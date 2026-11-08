# Policy Q&A Service

An HTTP service that answers employee questions **only from the company policy documents** in `mock-policies/`. Every answer cites the policy file and section it came from. If the documents don't cover the question, the service refuses.

This README is for testers. Follow it top to bottom on a Windows laptop with Git Bash. You don't need admin rights.

---

## 1. What you need

| Item | Notes |
|---|---|
| Windows 10/11 with **Git Bash** | All commands below are run in Git Bash |
| **Python 3.11 or newer** | We tested with 3.12.10 |
| **Ollama** | Runs the two local AI models. Free. https://ollama.com |
| ~5 GB free disk | For the models and the index |

The service runs entirely on your machine. No data leaves it, and no paid API is used.

---

## 2. Setup (once)

**a) Get the code**
```bash
git clone https://github.com/JakkaSampath/policy-qa-service.git
cd policy-qa-service
```

**b) Download the two Ollama models** (in any terminal, this takes a while):
```bash
ollama pull gemma3
ollama pull embeddinggemma-2
```
Check they're there:
```bash
ollama list
```
You should see both `gemma3` and `embeddinggemma-2`.

**c) Create the Python environment**
```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -e ".[dev]"
```
`.[dev]` installs the app plus pytest. After this, you can call the tools directly as `.venv/Scripts/python` and `.venv/Scripts/pytest`, with or without activating.

> **Windows note:** if `python -m venv` fails with an `ensurepip` error, the Python install is incomplete. Reinstall Python from python.org and tick "Add to PATH". You don't need admin rights for this.

**d) Create your settings file**
```bash
cp .env.example .env
```
The defaults work on a laptop. You only need to edit `.env` if you change the models or the Ollama address. See section 7.

---

## 3. Build the index (once, and after any policy change)

The index is a searchable copy of the policy documents. Build it with:
```bash
.venv/Scripts/python scripts/ingest.py
```
Expected output: `Loaded 54 chunks ...` and an indexed count. It takes about a minute.

Re-run it whenever a file in `mock-policies/` changes, or when you change `EMBED_MODEL`.

> **Important:** stop the API before running ingest. Only one process can open the index at a time. If you see "already accessed" or "locked", stop the API and try again.

---

## 4. Ask a question from the terminal (no API)

```bash
.venv/Scripts/python scripts/ask.py "How much is mileage reimbursed?"
```
This is the quickest way to check that everything works. It prints the answer, its citations, and the reason.

---

## 5. Run the API

```bash
.venv/Scripts/uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Leave this window open. Stop it with `Ctrl+C`.

Open a second Git Bash window to test it:

**Or use the browser:** open http://127.0.0.1:8000/docs, click `POST /ask`, then **Try it out**, type a question and click **Execute**.

**Health check** (should say `"status": "ok"`):
```bash
curl http://127.0.0.1:8000/health
```

**Ask a question:**
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many sick days do I get per year?"}'
```

An answered question looks like this:
```json
{"answered": true,
 "answer": "You get 10 days of paid sick leave per year.",
 "citations": [{"file": "leave-policy.md", "section": "Sick Leave"}],
 "reason": "answered", "top_similarity": 0.81, "cached": false}
```

A refused question (not covered by the docs):
```json
{"answered": false,
 "answer": "I can't find this in the policy documents.",
 "citations": [], "reason": "no_relevant_docs", "top_similarity": 0.52, "cached": false}
```

**Status codes you may see:**

| Code | Meaning |
|---|---|
| 200 | OK (answered or refused) |
| 413 | Request body too large |
| 422 | Question missing, empty, or longer than 500 characters |
| 429 | Too many requests from your address. Wait a minute. |
| 503 | Busy answering another question, or Ollama / the index is not ready. The response says which. Retry in a few seconds. |
| 504 | The model took too long (over 90 s) |

Repeat questions are answered from a cache and come back with `"cached": true`.

---

## 6. Run the tests

**Fast tests (no Ollama needed, about 1–2 minutes):**
```bash
.venv/Scripts/pytest -m "not integration"
```

**All tests, including the real-Ollama checks (Ollama must be running with both models):**
```bash
.venv/Scripts/pytest -v
```
Expected: **114 passed**. Integration tests are skipped with a reason if Ollama is not reachable.

The fast tests use fake models, so they give the same result every time.

---

## 7. Settings (`.env`)

| Setting | Default | What it does |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Where Ollama is running |
| `EMBED_MODEL` | `embeddinggemma-2` | Model that turns text into search vectors |
| `LLM_MODEL` | `gemma3` | Model that writes the answer |
| `TOP_K` | `4` | How many policy sections the model sees per question |
| `MIN_SIMILARITY` | `0.625` | Below this match score, the service refuses without calling the model. Tuned in the eval (section 8). |
| `MAX_QUESTION_CHARS` | `500` | Longest allowed question |
| `LLM_TIMEOUT_S` | `90` | Seconds to wait for the model before returning 504 |
| `EMBED_TIMEOUT_S` | `60` | Seconds to wait for the search-vector model |
| `LLM_MAX_TOKENS` | `300` | Longest answer the model may write |
| `MAX_CONCURRENT_LLM` | `1` | Questions answered at once. Extra requests get 503, not queued. |
| `RATE_LIMIT_PER_MIN` | `10` | Questions per minute per address |
| `CACHE_SIZE` | `256` | How many answers to remember |

After changing `.env`, restart the API. If you change `EMBED_MODEL`, re-run ingest.

---

## 8. Answer quality (the eval)

The eval asks 34 labelled questions and grades every answer automatically. Each question has the expected result: answer or refuse, the right citation, and the key facts.

**Run it** (stop the API first, takes about 10 minutes):
```bash
.venv/Scripts/python eval/run_eval.py                 # all 34 questions
.venv/Scripts/python eval/run_eval.py --ids a01,t03   # just these
```
The full report is saved to `eval/results/` (not in git).

**What the numbers mean:**

| Metric | Meaning | Target |
|---|---|---|
| **False answers** | Answered something the docs don't cover. The worst error. | 0 |
| **Wrongly refused** | Refused a question the docs do cover | As low as possible |
| **Refusal accuracy** | Answered or refused correctly, overall | High |
| **Citation accuracy** | Cited the right section when answering | High |
| **Answer correctness** | Key facts present and no forbidden facts | High |
| **Latency** | Seconds per question on the laptop | p50 about 15–20 s |

**Last result (commit `fb60026`):** 33 of 34 passed, **0 false answers**, refusal accuracy 100%, `MIN_SIMILARITY=0.625`.

---

## 9. Known limits

- **One question at a time.** A second question while one is being answered gets a 503. This protects the laptop's CPU and GPU.
- **Slow first answer.** After the models are loaded, expect 15–20 s per question on a laptop CPU. The very first question after starting the API can take about 30 s.
- **One wrong answer in the eval: question a18.** Asked when the expense policy was last updated, the service gives the leave policy's date (September 2026 instead of July 2026). The two policies have nearly identical "Document Info" sections, and the small model mixes them up. The citation shows the leave policy, so it can be spotted. This is a known limit of `gemma3`, not an invented answer.
- **Threshold tuned on 34 questions.** New or changed documents may need the eval re-run and the threshold checked.
- **No login.** Anyone who can reach the port can ask questions. The server listens only on `127.0.0.1`, so only your machine can reach it. Don't change this to `0.0.0.0` on a shared network.
- **Rate limits and cache are in memory.** They reset when the API restarts.
- **Ollama must be running** on the same machine. If it's down, `/health` says so.

---

## 10. Troubleshooting

| You see | Cause | Fix |
|---|---|---|
| `Ollama not reachable` | Ollama isn't running | Start the Ollama app, or run `ollama serve`, then retry |
| `Ollama model missing: gemma3` (or `embeddinggemma-2`) | Model not downloaded | `ollama pull gemma3` / `ollama pull embeddinggemma-2` |
| `Index is empty` | Ingest not run yet | `.venv/Scripts/python scripts/ingest.py` |
| `Index was built with '...' but EMBED_MODEL is '...'` | Index built with another embedding model | Re-run ingest |
| `already accessed` / `locked` during ingest or eval | The API is still running | Stop the API (`Ctrl+C`), then retry |
| `ensurepip` error when creating the venv | Incomplete Python install | Reinstall Python from python.org with "Add to PATH" |
| The program crashes (access violation) when using the index | Old Visual C++ runtime on the laptop | We use Qdrant (plain Python) instead of ChromaDB, which needs a newer runtime. Tell the developer. |
| `504` on questions | Model slow, or the laptop is busy | Close other heavy programs and retry. Raise `LLM_TIMEOUT_S` in `.env` if it keeps happening. |
| `422` on a valid question | Question is longer than 500 characters | Shorten it |

---

## 11. Project layout

```
app/
  config.py      settings from .env
  chunking.py    splits the policy docs into sections
  store.py       the search index (Qdrant) and the embedding calls
  llm.py         the answer model (Ollama)
  qa.py          retrieve -> decide -> answer -> check citations
  guards.py      rate limit, one-at-a-time, answer cache
  evaluation.py  the quality scorer
  main.py        the API routes
scripts/
  ingest.py      builds the index
  ask.py         asks one question from the terminal
eval/
  questions.jsonl   the 34 labelled questions
  run_eval.py       runs the eval and prints the report
tests/           pytest tests
mock-policies/   the 6 policy documents
plan.md          design notes and build history
```

---

## 12. How the refusals work

A question is refused at any of three points, and the reason is always reported:

1. **No close match in the docs** (`no_relevant_docs`). The search score is below `MIN_SIMILARITY`. The model is not called.
2. **The model says the sources don't answer it** (`model_refused`).
3. **The model cites a section that was not given to it, or its reply can't be read** (`bad_citation` / `bad_llm_output`). The answer is discarded and refused.
