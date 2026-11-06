"""Vector store: embeds chunks with Ollama and searches them in Qdrant (local mode)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
from qdrant_client import QdrantClient, models

from app.chunking import Chunk


class EmbeddingError(RuntimeError):
    """Ollama could not produce embeddings (down, timed out, or bad response)."""


class IndexMismatchError(RuntimeError):
    """The index was built with a different embedding model than the one configured."""


class Embedder(Protocol):
    model: str
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


# Task prefixes each embedding model was trained with: (query, document).
# Matched by model-name prefix, so "embeddinggemma-2" uses the embeddinggemma entry.
_PREFIXES = {
    "embeddinggemma": ("task: search result | query: ", "title: none | text: "),
    "nomic-embed-text": ("search_query: ", "search_document: "),
}


def prefixes_for(model: str) -> tuple[str, str]:
    for name, prefixes in _PREFIXES.items():
        if model.startswith(name):
            return prefixes
    return ("", "")  # unknown model: no prefixes


class OllamaEmbedder:
    """Calls Ollama's /api/embed, adding the model's task prefixes."""

    def __init__(self, url: str, model: str, timeout_s: float = 10.0,
                 client: httpx.Client | None = None):
        self.model = model
        self._query_prefix, self._doc_prefix = prefixes_for(model)
        self._client = client or httpx.Client(base_url=url, timeout=timeout_s)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            r = self._client.post("/api/embed", json={"model": self.model, "input": texts})
            r.raise_for_status()
            vectors = r.json()["embeddings"]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise EmbeddingError(f"Ollama embedding failed: {e}") from e
        if len(vectors) != len(texts):
            raise EmbeddingError(f"Expected {len(texts)} embeddings, got {len(vectors)}")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self._doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self._query_prefix + text])[0]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    similarity: float  # cosine similarity: 1.0 = same meaning, 0 = unrelated


def _point_id(chunk_id: str) -> str:
    """Qdrant needs integer or UUID ids; derive a stable UUID from the chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class VectorStore:
    """Qdrant in local mode: plain Python, saves to a folder, no server.

    Only one VectorStore may have the folder open at a time; call close()
    (or use `with VectorStore(...) as store:`) when done.
    """

    def __init__(self, embedder: Embedder, path: Path, collection: str):
        self.embedder = embedder
        self._name = collection
        self._client = QdrantClient(path=str(path))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def rebuild(self, chunks: list[Chunk]) -> int:
        """Replace the whole index with these chunks. Safe to run repeatedly."""
        # Embed first: if Ollama fails here, the existing index is untouched.
        vectors = self.embedder.embed_documents([c.embed_text for c in chunks]) if chunks else []
        if self._client.collection_exists(self._name):
            self._client.delete_collection(self._name)
        if not chunks:
            return 0
        self._client.create_collection(
            self._name,
            vectors_config=models.VectorParams(size=len(vectors[0]),
                                               distance=models.Distance.COSINE),
        )
        self._client.upsert(self._name, points=[
            models.PointStruct(
                id=_point_id(c.id),
                vector=v,
                payload={"chunk_id": c.id, "file": c.file, "title": c.title,
                         "section": c.section, "text": c.text,
                         "embed_model": self.embedder.model},
            )
            for c, v in zip(chunks, vectors)
        ])
        return self.count()

    def count(self) -> int:
        if not self._client.collection_exists(self._name):
            return 0
        return self._client.count(self._name).count

    def built_with(self) -> str | None:
        """Embedding model the current index was built with (None if empty)."""
        if self.count() == 0:
            return None
        points, _ = self._client.scroll(self._name, limit=1, with_payload=["embed_model"])
        return points[0].payload.get("embed_model")

    def search(self, question: str, k: int) -> list[Hit]:
        """Top-k most similar chunks, best first. Empty index -> []."""
        if self.count() == 0:
            return []
        built = self.built_with()
        if built != self.embedder.model:
            raise IndexMismatchError(
                f"Index was built with '{built}' but EMBED_MODEL is "
                f"'{self.embedder.model}'. Run: python scripts/ingest.py")
        points = self._client.query_points(
            self._name, query=self.embedder.embed_query(question),
            limit=k, with_payload=True,
        ).points
        return [
            Hit(Chunk(p.payload["chunk_id"], p.payload["file"], p.payload["title"],
                      p.payload["section"], p.payload["text"]), round(p.score, 4))
            for p in points
        ]
