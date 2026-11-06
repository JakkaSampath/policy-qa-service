"""Split policy markdown files into section-level chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DOC_INFO = "Document Info"
_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


@dataclass(frozen=True)
class Chunk:
    id: str        # "leave-policy.md#annual-leave-accrual"
    file: str      # "leave-policy.md"
    title: str     # "Leave and Time Off Policy"
    section: str   # "Annual Leave > Accrual"
    text: str      # section body, without the heading line

    @property
    def embed_text(self) -> str:
        """Text sent to the embedder: title + section give context to short bodies."""
        return f"{self.title} — {self.section}\n\n{self.text}"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def chunk_markdown(text: str, file: str) -> list[Chunk]:
    """Split one markdown document into chunks by ## and ### headings."""
    title = Path(file).stem.replace("-", " ").title()  # fallback if no "# " line
    h2: str | None = None
    section = DOC_INFO
    body: list[str] = []
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    in_fence = False

    def flush() -> None:
        content = "\n".join(body).strip()
        body.clear()
        if not content:
            return
        cid = f"{file}#{_slug(section)}"
        seen[cid] = seen.get(cid, 0) + 1
        if seen[cid] > 1:                      # duplicate heading names
            cid = f"{cid}-{seen[cid]}"
        chunks.append(Chunk(cid, file, title, section, content))

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if not m:
            body.append(line)
            continue
        level, name = len(m.group(1)), m.group(2)
        if level == 1:
            title = name
            continue
        flush()
        if level == 2:
            h2 = section = name
        else:
            section = f"{h2} > {name}" if h2 else name
    flush()
    return chunks


def load_documents(docs_dir: Path) -> list[Chunk]:
    """Chunk every .md file in docs_dir, in a stable (sorted) order."""
    files = sorted(Path(docs_dir).glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {docs_dir}")
    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name))
    return chunks
