"""Vector store: embeds chunks with Ollama and searches them in ChromaDB."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chromadb
import httpx

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


class VectorStore:
    def __init__(self, embedder: Embedder, path: Path, collection: str):
        self.embedder = embedder
        self._client = chromadb.PersistentClient(path=str(path))
        self._name = collection
        self._col = self._open()

    def _open(self, metadata: dict | None = None):
        # embedding_function=None: we always pass vectors ourselves, so Chroma
        # must never download or use its own default model.
        return self._client.get_or_create_collection(
            self._name, embedding_function=None, metadata=metadata,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def rebuild(self, chunks: list[Chunk]) -> int:
        """Replace the whole index with these chunks. Safe to run repeatedly."""
        # Embed first: if Ollama fails here, the existing index is untouched.
        vectors = self.embedder.embed_documents([c.embed_text for c in chunks])
        self._client.delete_collection(self._name)
        self._col = self._open(metadata={"embed_model": self.embedder.model})
        if chunks:
            self._col.add(
                ids=[c.id for c in chunks],
                embeddings=vectors,
                documents=[c.text for c in chunks],
                metadatas=[{"file": c.file, "title": c.title, "section": c.section}
                           for c in chunks],
            )
        return self._col.count()

    def count(self) -> int:
        return self._col.count()

    def built_with(self) -> str | None:
        """Embedding model the current index was built with (None if never built)."""
        return (self._col.metadata or {}).get("embed_model")

    def search(self, question: str, k: int) -> list[Hit]:
        """Top-k most similar chunks, best first. Empty index -> []."""
        n = self.count()
        if n == 0:
            return []
        built = self.built_with()
        if built != self.embedder.model:
            raise IndexMismatchError(
                f"Index was built with '{built}' but EMBED_MODEL is "
                f"'{self.embedder.model}'. Run: python scripts/ingest.py")
        r = self._col.query(
            query_embeddings=[self.embedder.embed_query(question)],
            n_results=min(k, n),
            include=["documents", "metadatas", "distances"],
        )
        return [
            Hit(Chunk(cid, m["file"], m["title"], m["section"], doc), round(1.0 - dist, 4))
            for cid, doc, m, dist in zip(
                r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])
        ]
