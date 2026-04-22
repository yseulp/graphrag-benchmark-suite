import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

def normalize(text: str) -> List[str]:
    """Simple tokenization/normalization for F1 (works okay as baseline)."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    # keep alphanumerics; split on non-word
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens

def f1_score(pred: str, gold: str) -> Tuple[float, float, float]:
    pred_toks = normalize(pred)
    gold_toks = normalize(gold)

    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0, 0.0, 0.0

    pred_cnt = Counter(pred_toks)
    gold_cnt = Counter(gold_toks)
    common = pred_cnt & gold_cnt
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def main(run_file: str):
    path = Path(run_file)
    assert path.exists(), f"Not found: {path}"

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    f1s = []
    for r in rows:
        pred = r.get("pred_answer", "") or ""
        gold = r.get("gold_answer", "") or ""
        f1, p, rc = f1_score(pred, gold)
        f1s.append(f1)
        r["f1"] = f1
        r["precision"] = p
        r["recall"] = rc

    avg_f1 = sum(f1s) / max(1, len(f1s))
    print(f"Loaded: {len(rows)} samples")
    print(f"Average token-F1: {avg_f1:.4f}")

    # show worst 3
    worst = sorted(rows, key=lambda x: x["f1"])[:3]
    print("\nWorst 3 examples:")
    for w in worst:
        print("-" * 60)
        print("id:", w.get("id"), "domain:", w.get("domain"))
        print("Q:", w.get("question"))
        print("GOLD:", (w.get("gold_answer") or "")[:200], "...")
        print("PRED:", (w.get("pred_answer") or "")[:200], "...")
        print("F1:", w["f1"])

    # save annotated file
    out_path = path.with_suffix(".scored.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote scored file: {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.eval_run_f1 runs/ultradomain_lightrag/<file>.jsonl")
        raise SystemExit(1)
    main(sys.argv[1])