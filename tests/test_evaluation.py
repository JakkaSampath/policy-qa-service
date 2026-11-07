import json
from collections import Counter

import pytest

from app.config import ROOT
from app.evaluation import (CATEGORIES, Case, Result, apply_threshold, format_report,
                            load_cases, percentile, recommend, report_json, run_cases,
                            score_case, summarize, sweep)
from app.llm import LLMTimeout
from app.qa import Answer, Citation
from conftest import FakeLLM

QUESTIONS = ROOT / "eval" / "questions.jsonl"
MILEAGE = Citation("expense-policy.md", "Mileage Reimbursement")


def case(expect=True, **kw) -> Case:
    base = dict(id="x", question="q", category="answerable", expect_answered=expect,
                citations=("expense-policy.md > Mileage Reimbursement",) if expect else (),
                must_contain=(("0.67",), ("mile", "miles")) if expect else ())
    base.update(kw)
    return Case(**base)


def ans(answered=True, text="It is $0.67 per mile.", cites=(MILEAGE,), top=0.8) -> Answer:
    return Answer(answered, text if answered else "refused", cites if answered else (),
                  "answered" if answered else "model_refused", top)


# ---- the question set itself ----

def test_question_set_is_valid_and_points_at_real_sections(chunks):
    cases = load_cases(QUESTIONS)
    real = {f"{c.file} > {c.section}" for c in chunks}
    assert len(cases) >= 30
    assert len({c.id for c in cases}) == len(cases), "duplicate ids"
    counts = Counter(c.category for c in cases)
    assert set(counts) == set(CATEGORIES) and min(counts.values()) >= 5
    for c in cases:
        if c.expect_answered:
            assert c.citations and c.must_contain, c.id
            assert set(c.citations) <= real, f"{c.id} cites a section that doesn't exist"
        else:
            assert not c.citations and not c.must_contain, c.id


def test_load_cases_accepts_strings_and_alternatives(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(json.dumps({"id": "a", "question": "q", "category": "answerable",
                             "expect_answered": True, "citations": ["f > s"],
                             "must_contain": ["x", ["y", "z"]]}) + "\n\n")
    assert load_cases(p)[0].must_contain == (("x",), ("y", "z"))


# ---- scoring one answer ----

def test_correct_answer_passes():
    assert score_case(case(), ans())["passed"]


def test_alternatives_and_case_are_ignored():
    assert score_case(case(), ans(text="IT IS $0.67 PER MILE"))["passed"]


def test_missing_fact_fails():
    s = score_case(case(), ans(text="It is per mile."))
    assert (s["passed"], s["content_ok"], s["missing"]) == (False, False, ["0.67"])


def test_forbidden_phrase_fails():
    s = score_case(case(must_not_contain=("full-time",)),
                   ans(text="Full-time staff get $0.67 per mile."))
    assert (s["passed"], s["forbidden"]) == (False, ["full-time"])


def test_wrong_citation_fails():
    s = score_case(case(), ans(cites=(Citation("leave-policy.md", "Sick Leave"),)))
    assert (s["passed"], s["citation_ok"]) == (False, False)


def test_refusing_an_answerable_question_fails():
    s = score_case(case(), ans(answered=False))
    assert (s["passed"], s["decision_ok"], s["citation_ok"]) == (False, False, None)


@pytest.mark.parametrize("answered, passed", [(False, True), (True, False)])
def test_out_of_scope_must_be_refused(answered, passed):
    assert score_case(case(expect=False), ans(answered=answered))["passed"] is passed


# ---- threshold ----

def test_apply_threshold():
    assert apply_threshold(ans(top=0.8), 0.7).answered
    gated = apply_threshold(ans(top=0.6), 0.7)
    assert (gated.answered, gated.reason) == (False, "no_relevant_docs")
    assert not apply_threshold(ans(top=None), 0.0).answered


def sample_results() -> list[Result]:
    return [
        Result(case(id="p1"), ans(top=0.85), 10.0),
        Result(case(id="p2"), ans(top=0.75), 12.0),
        Result(case(expect=False, id="n1", category="out_of_scope"), ans(answered=False, top=0.55), 8.0),
        Result(case(expect=False, id="n2", category="tricky"), ans(top=0.68), 9.0),  # false answer
        Result(case(id="e1"), None, 60.0, "timeout"),
    ]


def test_summarize_counts_each_kind_of_error():
    s = summarize(sample_results(), 0.5)
    assert s["errors"] == ["e1"] and s["passed"] == 3
    assert s["false_answers"] == ["n2"] and s["wrongly_refused"] == []
    assert s["refusal_accuracy"] == 0.75
    assert (s["citation_accuracy"], s["answer_correctness"]) == (1.0, 1.0)
    assert s["llm_skipped"] == 0.0


def test_raising_threshold_fixes_false_answer_then_starts_refusing_good_ones():
    rows = {r["threshold"]: r for r in sweep(sample_results(), [0.5, 0.7, 0.8])}
    assert rows[0.7]["false_answers"] == [] and rows[0.7]["refusal_accuracy"] == 1.0
    assert rows[0.8]["wrongly_refused"] == ["p2"]
    assert recommend(list(rows.values())) == 0.7


def test_recommend_picks_middle_of_a_tie():
    rows = [{"threshold": t, "refusal_accuracy": 1.0, "false_answers": []} for t in (0.6, 0.65, 0.7)]
    rows.insert(0, {"threshold": 0.55, "refusal_accuracy": 0.9, "false_answers": ["x"]})
    assert recommend(rows) == 0.65


def test_recommend_is_none_when_nothing_succeeded():
    rows = [{"threshold": t, "refusal_accuracy": None, "false_answers": []} for t in (0.6, 0.7)]
    assert recommend(rows) is None


def test_percentile():
    assert percentile([5, 1, 3, 2, 4], 50) == 3 and percentile([5, 1, 3, 2, 4], 95) == 5
    assert percentile([], 50) is None


# ---- running (fake store + fake LLM) and the report ----

def test_run_cases_and_report(store, chunks):
    store.rebuild(chunks)
    good = json.dumps({"answered": True, "answer": "It is $0.67 per mile.", "citations": ["S1"]})
    cases = [case(id="p1", question="mileage reimbursement rate per mile")]
    seen = []
    results = run_cases(cases, store, FakeLLM(good), top_k=4, on_result=lambda i, r: seen.append(i))
    assert seen == [1] and results[0].answer.answered and results[0].error is None

    rows = sweep(results, [0.0, 0.99])
    text = format_report(results, 0.0, rows)
    assert "1/1 passed" in text and "Recommended MIN_SIMILARITY" in text
    data = report_json(results, 0.0, rows)
    assert data["cases"][0]["citations"] == ["expense-policy.md > Mileage Reimbursement"]
    json.dumps(data)  # must be serializable


def test_run_cases_records_errors_and_continues(store, chunks):
    store.rebuild(chunks)
    results = run_cases([case(id="a"), case(id="b")], store, FakeLLM(LLMTimeout("slow")), top_k=4)
    assert [r.error for r in results] == ["slow", "slow"]
    text = format_report(results, 0.5, sweep(results, [0.5]))
    assert "ERROR a" in text and "n/a (no successful answers" in text
