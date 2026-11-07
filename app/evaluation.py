"""Answer-quality evaluation: run labelled questions, score answers, sweep the threshold.

Questions run once with the retrieval gate OFF (min_similarity=0), so the LLM's
decision is recorded for every question. Any threshold is then applied after the
fact: a question whose top similarity is below it counts as refused. (At a real
higher threshold, fewer low-scoring sources would reach the LLM, which can change
its answer slightly, so the sweep is a close estimate, not an exact replay.)
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.llm import LLM
from app.qa import REFUSAL, Answer, answer_question
from app.store import VectorStore

CATEGORIES = ("answerable", "out_of_scope", "tricky")


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    category: str
    expect_answered: bool
    citations: tuple[str, ...] = ()                   # "file > section"; citing any one is enough
    must_contain: tuple[tuple[str, ...], ...] = ()    # each group: at least one alternative appears
    must_not_contain: tuple[str, ...] = ()


@dataclass(frozen=True)
class Result:
    case: Case
    answer: Answer | None   # None if the question errored (timeout, Ollama down)
    seconds: float
    error: str | None = None


def load_cases(path: Path) -> list[Case]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        groups = tuple((g,) if isinstance(g, str) else tuple(g) for g in d.get("must_contain", []))
        cases.append(Case(d["id"], d["question"], d["category"], d["expect_answered"],
                          tuple(d.get("citations", [])), groups,
                          tuple(d.get("must_not_contain", []))))
    return cases


def _norm(s: str) -> str:
    return " ".join(s.casefold().split())


def apply_threshold(answer: Answer, threshold: float) -> Answer:
    """What the answer would be with the retrieval gate at `threshold`."""
    if answer.top_similarity is None or answer.top_similarity < threshold:
        return Answer(False, REFUSAL, (), "no_relevant_docs", answer.top_similarity)
    return answer


def score_case(case: Case, answer: Answer) -> dict:
    """Checks for one answer. citation_ok/content_ok are None when not applicable."""
    citation_ok = content_ok = None
    missing: list[str] = []
    forbidden: list[str] = []
    if case.expect_answered and answer.answered:
        cited = {f"{c.file} > {c.section}" for c in answer.citations}
        citation_ok = not case.citations or bool(cited & set(case.citations))
        text = _norm(answer.answer)
        missing = [g[0] for g in case.must_contain if not any(_norm(a) in text for a in g)]
        forbidden = [w for w in case.must_not_contain if _norm(w) in text]
        content_ok = not missing and not forbidden
    decision_ok = answer.answered == case.expect_answered
    return {"decision_ok": decision_ok, "citation_ok": citation_ok, "content_ok": content_ok,
            "missing": missing, "forbidden": forbidden,
            "passed": decision_ok and citation_ok is not False and content_ok is not False}


def _rate(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 3) if flags else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1)], 1)


def summarize(results: list[Result], threshold: float) -> dict:
    ok = [r for r in results if r.answer is not None]
    scored = [(r, score_case(r.case, apply_threshold(r.answer, threshold))) for r in ok]
    pos = [(r, s) for r, s in scored if r.case.expect_answered]
    neg = [(r, s) for r, s in scored if not r.case.expect_answered]
    answered_pos = [s for _, s in pos if s["citation_ok"] is not None]
    return {
        "threshold": threshold,
        "cases": len(results),
        "errors": [r.case.id for r in results if r.answer is None],
        "passed": sum(s["passed"] for _, s in scored),
        "refusal_accuracy": _rate([s["decision_ok"] for _, s in scored]),
        "answered_when_should": _rate([s["decision_ok"] for _, s in pos]),
        "refused_when_should": _rate([s["decision_ok"] for _, s in neg]),
        "false_answers": [r.case.id for r, s in neg if not s["decision_ok"]],
        "wrongly_refused": [r.case.id for r, s in pos if not s["decision_ok"]],
        "citation_accuracy": _rate([s["citation_ok"] for s in answered_pos]),
        "answer_correctness": _rate([s["content_ok"] for s in answered_pos]),
        "llm_skipped": _rate([r.answer.top_similarity is None or r.answer.top_similarity < threshold
                              for r in ok]),
        "latency_p50_s": percentile([r.seconds for r in ok], 50),
        "latency_p95_s": percentile([r.seconds for r in ok], 95),
    }


def sweep(results: list[Result], thresholds: list[float]) -> list[dict]:
    return [summarize(results, t) for t in thresholds]


def recommend(rows: list[dict]) -> float | None:
    """Best refusal accuracy, then fewest false answers; middle of any tie (safety margin).
    None if nothing was answered successfully (e.g. every question errored)."""
    if all(r["refusal_accuracy"] is None for r in rows):
        return None
    key = lambda r: (r["refusal_accuracy"] or 0, -len(r["false_answers"]))
    best = max(key(r) for r in rows)
    tied = sorted(r["threshold"] for r in rows if key(r) == best)
    return tied[len(tied) // 2]


def run_cases(cases: list[Case], store: VectorStore, llm: LLM, top_k: int,
              on_result: Callable[[int, Result], None] = lambda i, r: None) -> list[Result]:
    """Ask every question with the gate off; errors are recorded, not raised."""
    results = []
    for i, case in enumerate(cases, 1):
        start = time.perf_counter()
        try:
            a = answer_question(case.question, store, llm, top_k=top_k, min_similarity=0.0)
            r = Result(case, a, time.perf_counter() - start)
        except RuntimeError as e:  # Ollama down / timeout
            r = Result(case, None, time.perf_counter() - start, str(e))
        results.append(r)
        on_result(i, r)
    return results


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def format_report(results: list[Result], threshold: float, rows: list[dict]) -> str:
    s = summarize(results, threshold)
    lines = [f"\n=== Results at MIN_SIMILARITY={threshold} ({s['passed']}/{s['cases']} passed) ==="]
    for r in results:
        if r.answer is None:
            lines.append(f"  ERROR {r.case.id}: {r.case.question}\n        {r.error}")
            continue
        a = apply_threshold(r.answer, threshold)
        sc = score_case(r.case, a)
        if sc["passed"]:
            continue
        why = []
        if not sc["decision_ok"]:
            why.append("should have answered" if r.case.expect_answered else "should have REFUSED")
        if sc["citation_ok"] is False:
            got = ", ".join(f"{c.file} > {c.section}" for c in a.citations)
            why.append(f"wrong citation (got: {got})")
        if sc["missing"]:
            why.append(f"missing: {sc['missing']}")
        if sc["forbidden"]:
            why.append(f"should not say: {sc['forbidden']}")
        lines.append(f"  FAIL {r.case.id} [{r.case.category}] {r.case.question}\n"
                     f"       {'; '.join(why)}  (reason={a.reason}, top={a.top_similarity})\n"
                     f"       answer: {a.answer}")
    lines += [
        "",
        f"  Refusal accuracy      {_pct(s['refusal_accuracy'])}"
        f"   (answered when it should {_pct(s['answered_when_should'])},"
        f" refused when it should {_pct(s['refused_when_should'])})",
        f"  False answers         {s['false_answers'] or 'none'}   <- worst error: answered something not in the docs",
        f"  Wrongly refused       {s['wrongly_refused'] or 'none'}",
        f"  Citation accuracy     {_pct(s['citation_accuracy'])}   (of answered, answerable questions)",
        f"  Answer correctness    {_pct(s['answer_correctness'])}   (expected facts present, no forbidden ones)",
        f"  Latency               p50 {s['latency_p50_s']} s, p95 {s['latency_p95_s']} s (LLM called for every question)",
        f"  Errors                {s['errors'] or 'none'}",
        "",
        "=== Threshold sweep (estimate) ===",
        "  threshold  refusal-acc  false-answers  wrongly-refused  LLM-skipped",
    ]
    for row in rows:
        lines.append(f"  {row['threshold']:>9.3f}  {_pct(row['refusal_accuracy']):>11}"
                     f"  {len(row['false_answers']):>13}  {len(row['wrongly_refused']):>15}"
                     f"  {_pct(row['llm_skipped']):>11}")
    best = recommend(rows)
    lines.append(f"\n  Recommended MIN_SIMILARITY: {best}  (set it in .env or app/config.py)" if best is not None
                 else "\n  Recommended MIN_SIMILARITY: n/a (no successful answers; fix the errors above first)")
    return "\n".join(lines)


def report_json(results: list[Result], threshold: float, rows: list[dict]) -> dict:
    return {
        "summary": summarize(results, threshold),
        "recommended_threshold": recommend(rows),
        "sweep": rows,
        "cases": [{
            "id": r.case.id, "category": r.case.category, "question": r.case.question,
            "expect_answered": r.case.expect_answered, "seconds": round(r.seconds, 2),
            "error": r.error,
            **({} if r.answer is None else {
                "answered": r.answer.answered, "answer": r.answer.answer,
                "reason": r.answer.reason, "top_similarity": r.answer.top_similarity,
                "citations": [f"{c.file} > {c.section}" for c in r.answer.citations]}),
        } for r in results],
    }
