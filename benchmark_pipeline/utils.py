from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def approximate_token_count(text: str) -> int:
    if not text:
        return 0
    words = len(text.split())
    return max(1, int(words * 1.3))


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    tokens = text.split()
    if not tokens:
        return []

    chunks: List[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(tokens), step):
        chunk = tokens[start : start + chunk_size]
        if not chunk:
            continue
        chunks.append(" ".join(chunk))
        if start + chunk_size >= len(tokens):
            break
    return chunks


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            out.append(json.loads(line))
    return out
