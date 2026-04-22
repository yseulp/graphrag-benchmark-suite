from __future__ import annotations

import random
from typing import Dict, List

from .taxonomy import taxonomy_fields
from .utils import read_jsonl, write_jsonl


def build_ultradomain_split(
    output_path: str,
    hf_id: str,
    split: str,
    domains: List[str],
    samples_per_domain: int,
    seed: int,
) -> List[Dict]:
    from datasets import load_dataset
    from datasets.exceptions import DatasetGenerationError
    from datasets.table import CastError

    try:
        ds = load_dataset(hf_id, split=split, streaming=False)
    except (DatasetGenerationError, CastError):
        ds = load_dataset(hf_id, split=split, streaming=True)
    rng = random.Random(seed)

    by_domain: Dict[str, List[Dict]] = {d: [] for d in domains}
    for ex in ds:
        label = ex.get("label") or ex.get("domain")
        if label in by_domain:
            by_domain[label].append(ex)

    rows: List[Dict] = []
    idx = 0
    for domain in domains:
        data = by_domain.get(domain, [])
        if not data:
            continue
        rng.shuffle(data)
        subset = data[:samples_per_domain]
        for ex in subset:
            rows.append(
                {
                    "id": idx,
                    "domain": domain,
                    "question": ex.get("input", ""),
                    "context": ex.get("context", ""),
                    "gold_answer": (ex.get("answers", [""]) or [""])[0],
                    **taxonomy_fields(ex.get("input", "")),
                }
            )
            idx += 1

    write_jsonl(output_path, rows)
    return rows


def load_or_build_split(
    split_path: str,
    hf_id: str,
    split: str,
    domains: List[str],
    samples_per_domain: int,
    seed: int,
) -> List[Dict]:
    try:
        return read_jsonl(split_path)
    except FileNotFoundError:
        return build_ultradomain_split(
            output_path=split_path,
            hf_id=hf_id,
            split=split,
            domains=domains,
            samples_per_domain=samples_per_domain,
            seed=seed,
        )
