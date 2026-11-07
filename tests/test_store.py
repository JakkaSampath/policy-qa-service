import json

import httpx
import pytest

from app.config import settings
from app.store import EmbeddingError, IndexMismatchError, OllamaEmbedder, VectorStore


# ---- index ----

def test_rebuild_indexes_all_chunks(store, chunks):
    assert store.rebuild(chunks) == 54


def test_rebuild_twice_does_not_duplicate(store, chunks):
    store.rebuild(chunks)
    assert store.rebuild(chunks) == 54


def test_index_persists_across_instances(tmp_path, fake_embedder, chunks):
    with VectorStore(fake_embedder, tmp_path, "test-policies") as s:
        s.rebuild(chunks)
    with VectorStore(fake_embedder, tmp_path, "test-policies") as s:
        assert s.count() == 54


def test_second_open_of_same_folder_is_refused(tmp_path, fake_embedder):
    with VectorStore(fake_embedder, tmp_path, "test-policies"):
        with pytest.raises(RuntimeError, match="already accessed"):
            VectorStore(fake_embedder, tmp_path, "test-policies")


def test_failed_rebuild_keeps_old_index(store, chunks):
    store.rebuild(chunks)

    class Broken:
        def embed_documents(self, texts):
            raise EmbeddingError("Ollama down")

    store.embedder = Broken()
    with pytest.raises(EmbeddingError):
        store.rebuild(chunks)
    assert store.count() == 54


# ---- search ----

def test_search_on_empty_store_returns_nothing(store):
    assert store.search("anything", k=4) == []


def test_search_returns_k_hits_best_first(store, chunks):
    store.rebuild(chunks)
    hits = store.search("sick leave days per year", k=4)
    sims = [h.similarity for h in hits]
    assert len(hits) == 4
    assert sims == sorted(sims, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in sims)


@pytest.mark.parametrize("question, file, section", [
    ("mileage reimbursement rate per mile", "expense-policy.md", "Mileage Reimbursement"),
    ("how many days of sick leave", "leave-policy.md", "Sick Leave"),
    ("who is eligible for remote work after 6 months with the company",
     "remote-work-policy.md", "Eligibility"),
])
def test_search_finds_the_right_section(store, chunks, question, file, section):
    store.rebuild(chunks)
    top = store.search(question, k=1)[0].chunk
    assert (top.file, top.section) == (file, section)


def test_hits_round_trip_chunk_data(store, chunks):
    store.rebuild(chunks)
    original = {c.id: c for c in chunks}
    for hit in store.search("expense approval", k=5):
        assert hit.chunk == original[hit.chunk.id]


# ---- index remembers its embedding model ----

def test_index_records_embedding_model(store, chunks):
    assert store.built_with() is None
    store.rebuild(chunks)
    assert store.built_with() == "fake"


def test_search_refuses_index_built_with_other_model(tmp_path, fake_embedder, chunks):
    with VectorStore(fake_embedder, tmp_path, "test-policies") as s:
        s.rebuild(chunks)

    class OtherModel(type(fake_embedder)):
        model = "embeddinggemma-2"

    with VectorStore(OtherModel(), tmp_path, "test-policies") as s:
        with pytest.raises(IndexMismatchError, match="ingest"):
            s.search("sick leave", k=1)


# ---- Ollama client (mocked HTTP) ----

def make_embedder(handler, model="embeddinggemma-2", batch_size=16) -> OllamaEmbedder:
    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return OllamaEmbedder("http://ollama", model, client=client, batch_size=batch_size)


def test_documents_are_embedded_in_batches_in_order():
    batches = []

    def handler(request):
        texts = json.loads(request.content)["input"]
        batches.append(len(texts))
        # vector = the text's number, so order can be checked
        return httpx.Response(200, json={"embeddings": [[float(t.split()[-1])] for t in texts]})

    vectors = make_embedder(handler).embed_documents([f"doc {i}" for i in range(40)])
    assert batches == [16, 16, 8]
    assert vectors == [[float(i)] for i in range(40)]


@pytest.mark.parametrize("model, query_input, doc_inputs", [
    ("embeddinggemma-2", "task: search result | query: hi",
     ["title: none | text: a", "title: none | text: b"]),
    ("nomic-embed-text", "search_query: hi", ["search_document: a", "search_document: b"]),
    ("some-other-model", "hi", ["a", "b"]),
])
def test_ollama_embedder_sends_model_and_prefixes(model, query_input, doc_inputs):
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]] * len(sent["input"])})

    emb = make_embedder(handler, model)
    assert emb.embed_query("hi") == [0.1, 0.2]
    assert sent == {"model": model, "input": [query_input]}
    emb.embed_documents(["a", "b"])
    assert sent["input"] == doc_inputs


def _refused(request):
    raise httpx.ConnectError("connection refused", request=request)


@pytest.mark.parametrize("handler", [
    lambda r: httpx.Response(500, text="boom"),
    lambda r: httpx.Response(200, json={"unexpected": 1}),
    lambda r: httpx.Response(200, json={"embeddings": []}),
    _refused,
], ids=["http-500", "bad-json", "wrong-count", "connection-refused"])
def test_ollama_failures_become_embedding_error(handler):
    with pytest.raises(EmbeddingError):
        make_embedder(handler).embed_query("hi")


# ---- integration (real Ollama; auto-skipped if unreachable) ----

@pytest.mark.integration
def test_real_ollama_finds_mileage_section(tmp_path, chunks):
    with VectorStore(OllamaEmbedder(settings.ollama_url, settings.embed_model,
                                    settings.embed_timeout_s),
                     tmp_path, "it-policies") as store:
        store.rebuild(chunks)
        top = store.search("How much do I get reimbursed per mile?", k=1)[0].chunk
    assert (top.file, top.section) == ("expense-policy.md", "Mileage Reimbursement")
