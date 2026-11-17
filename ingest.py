"""Ingestion: upload -> extract -> clean -> split -> embed into the vector store."""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass

SUPPORTED = ("pdf", "docx", "txt", "md", "csv")


@dataclass
class IndexStats:
    files: int
    passages: int
    skipped: list[str]


def extract(name: str, data: bytes) -> list[tuple[str, int]]:
    """Return [(text, page)] for a file. page is 0 when the format has no pages."""
    ext = name.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return [(p.extract_text() or "", i) for i, p in enumerate(reader.pages, 1)]
    if ext == "docx":
        from docx import Document

        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        return [("\n\n".join(parts), 0)]
    if ext == "csv":
        text = data.decode("utf-8", errors="replace")
        rows = list(csv.reader(io.StringIO(text)))
        if not rows:
            return []
        header, body = rows[0], rows[1:]
        lines = ["; ".join(f"{h}: {v}" for h, v in zip(header, r)) for r in body]
        return [("\n".join(lines), 0)]
    if ext in ("txt", "md"):
        return [(data.decode("utf-8", errors="replace"), 0)]
    raise ValueError(f"Unsupported file type: .{ext}")


def clean(text: str, keep_lines: bool = False) -> str:
    text = text.replace("\x00", " ").replace("\u00ad", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not keep_lines:
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # soft line wraps -> spaces
    return text.strip()


def split_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Recursive split on paragraphs, lines, sentences, words; then pack with overlap."""

    def pieces(t: str, level: int) -> list[str]:
        if len(t) <= size:
            return [t]
        if level == 0:
            parts = t.split("\n\n")
        elif level == 1:
            parts = t.split("\n")
        elif level == 2:
            parts = re.split(r"(?<=[.!?])\s+", t)
        elif level == 3:
            parts = t.split(" ")
        else:
            return [t[i : i + size] for i in range(0, len(t), size)]
        out: list[str] = []
        for part in parts:
            if not part.strip():
                continue
            out.extend(pieces(part, level + 1) if len(part) > size else [part])
        return out

    chunks: list[str] = []
    current = ""
    for piece in pieces(text, 0):
        if current and len(current) + 1 + len(piece) > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            tail = tail.split(" ", 1)[1] if " " in tail else ""
            current = f"{tail} {piece}".strip()
        else:
            current = f"{current}\n{piece}" if current else piece
    if current.strip():
        chunks.append(current)
    return [c.strip() for c in chunks if len(c.strip()) > 20]


def build_index(collection, files: list[tuple[str, bytes]], size: int = 900, overlap: int = 150) -> IndexStats:
    ids, docs, metas = [], [], []
    skipped: list[str] = []
    used = 0
    for f_idx, (name, data) in enumerate(files):
        try:
            pages = extract(name, data)
        except Exception as exc:  # unreadable or unsupported file
            skipped.append(f"{name}: {exc}")
            continue
        keep_lines = name.lower().endswith(".csv")
        count_before = len(ids)
        for text, page in pages:
            for c_idx, chunk in enumerate(split_text(clean(text, keep_lines), size, overlap)):
                ids.append(f"{f_idx}-{page}-{c_idx}")
                docs.append(chunk)
                metas.append({"source": name, "page": page})
        if len(ids) == count_before:
            skipped.append(f"{name}: no readable text (scanned PDF?)")
        else:
            used += 1
    for i in range(0, len(ids), 100):
        collection.add(ids=ids[i : i + 100], documents=docs[i : i + 100], metadatas=metas[i : i + 100])
    return IndexStats(files=used, passages=len(ids), skipped=skipped)
