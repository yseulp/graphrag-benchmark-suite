# benchmark_datasets/ultradomain_loader.py
import json
from typing import Optional, List
from datasets import load_dataset
from pathlib import Path

ULTRADOMAIN_HF_ID = "TommyChien/UltraDomain"
def load_ultradomain(
    domain: Optional[str] = None,
    limit: Optional[int] = None,
    streaming: bool = True
):
    """
    Load UltraDomain from HuggingFace.

    Args:
        domain: optional domain filter, e.g. "agriculture", "cs", "legal", "mix".
        limit: optional max number of samples.

    Returns:
        A HuggingFace Dataset object.
    """
    ds = load_dataset(ULTRADOMAIN_HF_ID, split="train", streaming=streaming)

    if domain is not None:
        ds = ds.filter(lambda ex: ex.get("label") == domain)

    if limit is not None:
        if streaming:
            ds = ds.take(limit)
        else:
            ds = ds.select(range(min(limit, len(ds))))

    return ds

def export_ultradomain_by_domain(output_dir: str = "data/ultradomain"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ds = load_dataset(ULTRADOMAIN_HF_ID, split="train")

    domains = sorted(set(ds["domain"]))
    print("Found domains:", domains)

    for d in domains:
        sub = ds.filter(lambda ex, d=d: ex["domain"] == d)
        out_path = Path(output_dir) / f"{d}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for ex in sub:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(sub)} samples to {out_path}")


if __name__ == "__main__":
    ds = load_ultradomain(limit=3)
    print("Columns:", ds.column_names)
    for ex in ds:
        print("----")
        print("domain:", ex.get("domain"))
        print("meta:", ex.get("metadata"))
        print("text snippet:", str(ex.get("context", ""))[:200], "...")