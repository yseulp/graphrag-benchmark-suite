from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from .utils import tokenize


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if " ".join(tokenize(pred)) == " ".join(tokenize(gold)) else 0.0


def f1_score(pred: str, gold: str) -> Tuple[float, float, float]:
    pred_toks = tokenize(pred)
    gold_toks = tokenize(gold)

    if not pred_toks and not gold_toks:
        return 1.0, 1.0, 1.0
    if not pred_toks or not gold_toks:
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


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def aggregate_reference_scores(rows: List[Dict]) -> Dict[str, float]:
    return {
        "exact_match": mean([r.get("exact_match", 0.0) for r in rows]),
        "token_f1": mean([r.get("token_f1", 0.0) for r in rows]),
        "precision": mean([r.get("precision", 0.0) for r in rows]),
        "recall": mean([r.get("recall", 0.0) for r in rows]),
    }
