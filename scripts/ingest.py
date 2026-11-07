"""Build (or rebuild) the vector index from the policy docs.

Usage:  .venv/bin/python scripts/ingest.py      (Windows: .venv/Scripts/python)
Stop the API first: only one process can open the index folder at a time.
"""
import sys
from collections import Counter

from app.chunking import load_documents
from app.config import settings
from app.store import EmbeddingError, OllamaEmbedder, VectorStore


def fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    try:
        chunks = load_documents(settings.docs_dir)
    except FileNotFoundError as e:
        return fail(f"{e}\nCheck DOCS_DIR in .env.")
    print(f"Loaded {len(chunks)} chunks from {settings.docs_dir}")

    embedder = OllamaEmbedder(settings.ollama_url, settings.embed_model, settings.embed_timeout_s)
    try:
        store = VectorStore(embedder, settings.index_dir, settings.collection)
    except RuntimeError as e:  # folder locked by another process (e.g. the running API)
        return fail(f"{e}\nStop the API (or other ingest) and try again.")

    with store:
        try:
            n = store.rebuild(chunks)
        except EmbeddingError as e:
            return fail(f"{e}\nIs Ollama running at {settings.ollama_url} "
                        f"with '{settings.embed_model}' pulled?")

    for file, count in sorted(Counter(c.file for c in chunks).items()):
        print(f"  {count:3d}  {file}")
    print(f"Indexed {n} chunks into {settings.index_dir} (collection '{settings.collection}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
