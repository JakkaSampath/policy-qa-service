"""Measure answer quality against the labelled questions, using the real Ollama.

Usage:  .venv/Scripts/python eval/run_eval.py              (all questions, ~10 min on a laptop CPU)
        .venv/Scripts/python eval/run_eval.py --ids a01,t03  (just these)
Stop the API first: it holds the index open.
"""
import argparse
import json
import sys
from datetime import datetime

from app.config import ROOT, settings
from app.evaluation import format_report, load_cases, report_json, run_cases, sweep
from app.llm import OllamaLLM
from app.store import OllamaEmbedder, VectorStore

QUESTIONS = ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
THRESHOLDS = [round(0.50 + 0.025 * i, 3) for i in range(15)]  # 0.500 .. 0.850


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ids", help="comma-separated case ids to run (default: all)")
    args = parser.parse_args()

    cases = load_cases(QUESTIONS)
    if args.ids:
        wanted = set(args.ids.split(","))
        cases = [c for c in cases if c.id in wanted]
    if not cases:
        print("ERROR: no matching questions", file=sys.stderr)
        return 2

    embedder = OllamaEmbedder(settings.ollama_url, settings.embed_model, settings.embed_timeout_s)
    llm = OllamaLLM(settings.ollama_url, settings.llm_model,
                    settings.llm_timeout_s, settings.llm_max_tokens)

    def progress(i, r):
        if r.answer is None:
            status = "ERROR"
        else:
            status = f"answered={str(r.answer.answered):5}  top={r.answer.top_similarity}"
        print(f"[{i:2d}/{len(cases)}] {r.case.id}  {status}  {r.seconds:5.1f}s  {r.case.question}")

    print(f"Running {len(cases)} questions with {settings.llm_model} / {settings.embed_model} ...")
    try:
        with VectorStore(embedder, settings.index_dir, settings.collection) as store:
            if store.count() == 0:
                print("ERROR: The index is empty. Run: python scripts/ingest.py", file=sys.stderr)
                return 1
            if store.built_with() != embedder.model:
                print(f"ERROR: Index was built with '{store.built_with()}' but EMBED_MODEL is "
                      f"'{embedder.model}'. Run: python scripts/ingest.py", file=sys.stderr)
                return 1
            results = run_cases(cases, store, llm, settings.top_k, progress)
    except RuntimeError as e:  # index locked by the API, wrong embed model
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    rows = sweep(results, THRESHOLDS)
    print(format_report(results, settings.min_similarity, rows))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report_json(results, settings.min_similarity, rows), indent=2),
                   encoding="utf-8")
    print(f"\nFull report saved to {out}")
    return 1 if any(r.answer is None for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
