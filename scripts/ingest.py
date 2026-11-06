"""Build (or rebuild) the vector index from the policy docs.

Usage:  .venv/bin/python scripts/ingest.py
"""
import sys
from collections import Counter

from app.chunking import load_documents
from app.config import settings
from app.store import EmbeddingError, OllamaEmbedder, VectorStore


def main() -> int:
    chunks = load_documents(settings.docs_dir)
    print(f"Loaded {len(chunks)} chunks from {settings.docs_dir}")

    embedder = OllamaEmbedder(settings.ollama_url, settings.embed_model)
    store = VectorStore(embedder, settings.chroma_dir, settings.collection)
    try:
        n = store.rebuild(chunks)
    except EmbeddingError as e:
        print(f"ERROR: {e}\nIs Ollama running at {settings.ollama_url} "
              f"with '{settings.embed_model}' pulled?", file=sys.stderr)
        return 1

    for file, count in sorted(Counter(c.file for c in chunks).items()):
        print(f"  {count:3d}  {file}")
    print(f"Indexed {n} chunks into {settings.chroma_dir} (collection '{settings.collection}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
