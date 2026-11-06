import hashlib
import math
import re

import httpx
import pytest

from app.chunking import load_documents
from app.config import settings
from app.store import VectorStore


class FakeEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors."""

    DIM = 512
    model = "fake"

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeLLM:
    """Stand-in for gemma3: returns a scripted reply and records every prompt."""

    def __init__(self, reply="{}"):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path, fake_embedder):
    with VectorStore(fake_embedder, tmp_path / "index", "test-policies") as s:
        yield s


@pytest.fixture(scope="session")
def chunks():
    return load_documents(settings.docs_dir)


def _ollama_problem() -> str | None:
    """None if Ollama is reachable and has the configured models, else the reason."""
    try:
        r = httpx.get(f"{settings.ollama_url}/api/tags", timeout=2)
        r.raise_for_status()
        have = {m["name"] for m in r.json()["models"]}
    except (httpx.HTTPError, KeyError, ValueError):
        return f"Ollama not reachable at {settings.ollama_url}"
    have |= {n.removesuffix(":latest") for n in have}
    missing = [m for m in (settings.embed_model, settings.llm_model) if m not in have]
    return f"Ollama is missing model(s): {', '.join(missing)}" if missing else None


def pytest_collection_modifyitems(config, items):
    problem = _ollama_problem()
    if problem is None:
        return
    skip = pytest.mark.skip(reason=problem)
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
