from __future__ import annotations

from typing import Dict


def infer_query_category(question: str) -> str:
    q = question.lower()

    if any(x in q for x in ["summarize", "overall", "main themes", "across", "in general"]):
        return "global_synthesis"

    if any(x in q for x in ["relationship", "how does", "connect", "impact", "therefore", "because"]):
        return "multi_hop_relation"

    if any(x in q for x in ["roles", "between", "among", "interaction", "combination"]):
        return "n_ary_relational"

    if any(x in q for x in ["according to", "what is", "who is", "when", "where"]):
        return "local_factoid"

    return "cross_document_reasoning"


def task_group_for_category(category: str) -> str:
    if category == "global_synthesis":
        return "global_synthesis"
    return "multi_hop_knowledge_reasoning"


def taxonomy_fields(question: str) -> Dict[str, str]:
    category = infer_query_category(question)
    return {
        "query_category": category,
        "task_group": task_group_for_category(category),
    }
