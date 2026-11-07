import json

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.llm import LLMError, LLMTimeout
from app.main import create_app
from app.store import EmbeddingError
from conftest import FakeEmbedder, FakeLLM

GOOD = json.dumps({"answered": True, "answer": "It is $0.67 per mile.", "citations": ["S1"]})
Q = {"question": "mileage reimbursement rate per mile"}


def make_client(store, llm, problems=(), **overrides) -> TestClient:
    cfg = settings.model_copy(update={
        "min_similarity": 0.0, "rate_limit_per_min": 100, "max_concurrent_llm": 1,
        "cache_size": 16, "max_question_chars": 500, **overrides})
    app = create_app(cfg, store=store, llm=llm, check_ollama=lambda c: list(problems))
    return TestClient(app)


@pytest.fixture
def indexed(store, chunks):
    store.rebuild(chunks)
    return store


# ---- 200: answers and refusals ----

def test_answer_shape(indexed):
    with make_client(indexed, FakeLLM(GOOD)) as c:
        r = c.post("/ask", json=Q)
    assert r.status_code == 200
    body = r.json()
    assert body.pop("top_similarity") > 0
    assert body == {"answered": True, "answer": "It is $0.67 per mile.",
                    "citations": [{"file": "expense-policy.md", "section": "Mileage Reimbursement"}],
                    "reason": "answered", "cached": False}


def test_refusal_is_200_with_reason_and_no_llm_call(indexed):
    llm = FakeLLM(GOOD)
    with make_client(indexed, llm, min_similarity=0.99) as c:
        r = c.post("/ask", json=Q)
    assert r.status_code == 200
    assert (r.json()["answered"], r.json()["reason"], r.json()["citations"]) == (False, "no_relevant_docs", [])
    assert llm.calls == []


# ---- 422 / 413: bad input ----

@pytest.mark.parametrize("payload", [
    {"question": ""}, {"question": "   "}, {"question": "x" * 501},
    {}, {"question": 123}, {"q": "hi"},
], ids=["empty", "blank", "too-long", "missing", "not-string", "wrong-field"])
def test_invalid_input_is_422_and_never_reaches_llm(indexed, payload):
    llm = FakeLLM(GOOD)
    with make_client(indexed, llm) as c:
        r = c.post("/ask", json=payload)
    assert r.status_code == 422
    assert llm.calls == []


def test_question_at_max_length_is_accepted(indexed):
    with make_client(indexed, FakeLLM(GOOD)) as c:
        assert c.post("/ask", json={"question": "mileage " + "x" * 492}).status_code == 200


def test_oversized_body_is_413(indexed):
    with make_client(indexed, FakeLLM(GOOD)) as c:
        r = c.post("/ask", content=json.dumps({"question": "x" * 5000}),
                   headers={"content-type": "application/json"})
    assert r.status_code == 413


# ---- 429: rate limit ----

def test_rate_limit_is_429_with_retry_after(indexed):
    with make_client(indexed, FakeLLM(GOOD), rate_limit_per_min=2) as c:
        codes = [c.post("/ask", json=Q).status_code for _ in range(3)]
        r = c.post("/ask", json=Q)
    assert codes == [200, 200, 429]
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_invalid_requests_count_toward_rate_limit(indexed):
    with make_client(indexed, FakeLLM(GOOD), rate_limit_per_min=2) as c:
        c.post("/ask", json={"question": ""})
        c.post("/ask", json={"question": ""})
        assert c.post("/ask", json=Q).status_code == 429


# ---- 503 / 504: busy, Ollama problems, index problems ----

def test_busy_is_503_with_retry_after(indexed):
    llm = FakeLLM(GOOD)
    with make_client(indexed, llm) as c:
        slots = c.app.state.slots
        assert slots.try_acquire()          # another request is "in progress"
        r = c.post("/ask", json=Q)
        slots.release()
        ok = c.post("/ask", json=Q)
    assert r.status_code == 503 and r.headers["retry-after"] == "5"
    assert ok.status_code == 200
    assert len(llm.calls) == 1


@pytest.mark.parametrize("error, status", [
    (LLMTimeout("slow"), 504),
    (LLMError("down"), 503),
], ids=["llm-timeout", "llm-down"])
def test_llm_failures_map_to_status_and_free_the_slot(indexed, error, status):
    llm = FakeLLM(error)
    with make_client(indexed, llm) as c:
        assert c.post("/ask", json=Q).status_code == status
        llm.reply = GOOD
        r = c.post("/ask", json=Q)          # slot was released, error was not cached
    assert r.status_code == 200 and r.json()["cached"] is False
    assert len(llm.calls) == 2


def test_embedding_failure_is_503(indexed):
    class BrokenQuery(FakeEmbedder):
        def embed_query(self, text):
            raise EmbeddingError("Ollama down")

    indexed.embedder = BrokenQuery()
    with make_client(indexed, FakeLLM(GOOD)) as c:
        assert c.post("/ask", json=Q).status_code == 503


def test_index_built_with_other_model_is_503(indexed):
    class OtherModel(FakeEmbedder):
        model = "other-model"

    indexed.embedder = OtherModel()
    with make_client(indexed, FakeLLM(GOOD)) as c:
        r = c.post("/ask", json=Q)
    assert r.status_code == 503 and "ingest" in r.json()["detail"]


def test_empty_index_is_503(store):
    with make_client(store, FakeLLM(GOOD)) as c:
        r = c.post("/ask", json=Q)
    assert r.status_code == 503 and "empty" in r.json()["detail"]


# ---- cache ----

def test_repeat_question_is_served_from_cache(indexed):
    llm = FakeLLM(GOOD)
    with make_client(indexed, llm) as c:
        first = c.post("/ask", json={"question": "Mileage reimbursement rate per mile?"})
        second = c.post("/ask", json={"question": "  mileage   REIMBURSEMENT rate per mile "})
    assert first.json()["cached"] is False and second.json()["cached"] is True
    assert second.json()["answer"] == first.json()["answer"]
    assert len(llm.calls) == 1


def test_cached_answer_is_served_even_when_busy(indexed):
    with make_client(indexed, FakeLLM(GOOD)) as c:
        c.post("/ask", json=Q)
        assert c.app.state.slots.try_acquire()
        r = c.post("/ask", json=Q)
        c.app.state.slots.release()
    assert r.status_code == 200 and r.json()["cached"] is True


# ---- /health ----

def test_health_ok(indexed):
    with make_client(indexed, FakeLLM(GOOD)) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["index_chunks"] == 54


def test_health_reports_ollama_problem(indexed):
    with make_client(indexed, FakeLLM(GOOD), problems=["Ollama not reachable"]) as c:
        r = c.get("/health")
    assert r.status_code == 503
    assert r.json()["problems"] == ["Ollama not reachable"]


def test_health_reports_empty_index(store):
    with make_client(store, FakeLLM(GOOD)) as c:
        r = c.get("/health")
    assert r.status_code == 503 and "empty" in r.json()["problems"][0]
