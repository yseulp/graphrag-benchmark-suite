from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

from .taxonomy import taxonomy_fields
from .utils import read_jsonl, write_jsonl


def _collect_by_domain(examples, domains: List[str]) -> Dict[str, List[Dict]]:
    by_domain: Dict[str, List[Dict]] = {d: [] for d in domains}
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        label = ex.get("label") or ex.get("domain")
        if label in by_domain:
            by_domain[label].append(ex)
    return by_domain


def _iter_ultradomain_rows_from_snapshot(hf_id: str, split: str):
    from huggingface_hub import snapshot_download

    snapshot_dir = Path(
        snapshot_download(
            repo_id=hf_id,
            repo_type="dataset",
            allow_patterns=["**/*.json", "**/*.jsonl"],
        )
    )

    files = sorted(snapshot_dir.glob(f"**/*{split}*.jsonl"))
    files += sorted(snapshot_dir.glob(f"**/*{split}*.json"))
    if not files:
        files = sorted(snapshot_dir.glob("**/*.jsonl"))
        files += sorted(snapshot_dir.glob("**/*.json"))

    for path in files:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
            continue

        with path.open("r", encoding="utf-8") as handle:
            obj = json.load(handle)

        if isinstance(obj, list):
            for row in obj:
                if isinstance(row, dict):
                    yield row
        elif isinstance(obj, dict):
            split_rows = obj.get(split)
            if isinstance(split_rows, list):
                for row in split_rows:
                    if isinstance(row, dict):
                        yield row


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
        by_domain = _collect_by_domain(ds, domains)
    except (DatasetGenerationError, CastError):
        by_domain = _collect_by_domain(_iter_ultradomain_rows_from_snapshot(hf_id, split), domains)
    rng = random.Random(seed)

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
