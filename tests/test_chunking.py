from pathlib import Path

import pytest

from app.chunking import DOC_INFO, chunk_markdown, load_documents
from app.config import settings

SAMPLE = """# Sample Policy

**Version:** 1.0

## Alpha

Alpha body.

## Beta

Beta intro.

### Sub One
- point one

### Sub Two
- point two

## Empty

## Gamma
Gamma body.
"""


def sections(chunks):
    return [c.section for c in chunks]


def test_splits_by_h2_and_h3_with_parent_names():
    chunks = chunk_markdown(SAMPLE, "sample.md")
    assert sections(chunks) == [
        DOC_INFO, "Alpha", "Beta", "Beta > Sub One", "Beta > Sub Two", "Gamma",
    ]


def test_metadata_block_becomes_document_info():
    info = chunk_markdown(SAMPLE, "sample.md")[0]
    assert info.section == DOC_INFO
    assert "Version" in info.text


def test_empty_sections_are_skipped():
    assert "Empty" not in sections(chunk_markdown(SAMPLE, "sample.md"))


def test_body_excludes_heading_and_title_comes_from_h1():
    alpha = chunk_markdown(SAMPLE, "sample.md")[1]
    assert alpha.text == "Alpha body."
    assert alpha.title == "Sample Policy"
    assert alpha.embed_text.startswith("Sample Policy — Alpha")


def test_ids_are_stable_and_unique_even_with_duplicate_headings():
    text = "# T\n## Notes\nfirst\n## Notes\nsecond\n"
    ids = [c.id for c in chunk_markdown(text, "t.md")]
    assert ids == ["t.md#notes", "t.md#notes-2"]


def test_title_falls_back_to_filename():
    chunk = chunk_markdown("## Only\nbody\n", "travel-rules.md")[0]
    assert chunk.title == "Travel Rules"


def test_headings_inside_code_fences_are_ignored():
    text = "# T\n## Real\n```\n## not a heading\n```\n"
    chunks = chunk_markdown(text, "t.md")
    assert sections(chunks) == ["Real"]
    assert "## not a heading" in chunks[0].text


def test_load_documents_errors_on_empty_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path)


# ---- against the real policy docs ----

def test_real_docs_chunk_counts():
    chunks = load_documents(settings.docs_dir)
    assert len({c.file for c in chunks}) == 6
    assert len(chunks) == 54
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.text.strip() for c in chunks)


def test_real_docs_known_sections():
    by_key = {(c.file, c.section): c for c in load_documents(settings.docs_dir)}
    assert "6 months" in by_key[("remote-work-policy.md", "Eligibility")].text
    assert "1.67 days" in by_key[("leave-policy.md", "Annual Leave > Accrual")].text
    assert "$0.67" in by_key[("expense-policy.md", "Mileage Reimbursement")].text
