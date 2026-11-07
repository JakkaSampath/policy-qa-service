"""Ask one question from the terminal (no API needed).

Usage:  .venv/Scripts/python scripts/ask.py "How much is mileage reimbursed?"
"""
import sys
import time

from app.config import settings
from app.llm import OllamaLLM
from app.qa import answer_question
from app.store import OllamaEmbedder, VectorStore


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "your question"', file=sys.stderr)
        return 2
    question = " ".join(sys.argv[1:])
    embedder = OllamaEmbedder(settings.ollama_url, settings.embed_model, settings.embed_timeout_s)
    llm = OllamaLLM(settings.ollama_url, settings.llm_model,
                    settings.llm_timeout_s, settings.llm_max_tokens)

    start = time.perf_counter()
    try:
        with VectorStore(embedder, settings.index_dir, settings.collection) as store:
            if store.count() == 0:
                print("ERROR: The index is empty. Run: python scripts/ingest.py", file=sys.stderr)
                return 1
            a = answer_question(question, store, llm, top_k=settings.top_k,
                                min_similarity=settings.min_similarity)
    except RuntimeError as e:  # Ollama down/timeout, index locked, wrong embed model
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    took = time.perf_counter() - start

    print(a.answer)
    for c in a.citations:
        print(f"  - {c.file} > {c.section}")
    top = f"{a.top_similarity:.3f}" if a.top_similarity is not None else "n/a"
    print(f"\n[answered={a.answered}  reason={a.reason}  top_similarity={top}  time={took:.1f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
