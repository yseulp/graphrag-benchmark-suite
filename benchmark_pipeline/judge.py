from __future__ import annotations

import json
from typing import Dict

from .ollama_client import OllamaClient


JUDGE_PROMPT_TEMPLATE = """You are an evaluator for RAG answers.
Return JSON only with integer scores 1-5 and a short justification.

Question:
{question}

Reference answer:
{gold_answer}

Predicted answer:
{pred_answer}

Retrieved context:
{retrieved_context}

JSON schema:
{{
  "correctness": 1-5,
  "relevance": 1-5,
  "comprehensiveness": 1-5,
  "diversity": 1-5,
  "logical_consistency": 1-5,
  "coherence": 1-5,
  "groundedness": 1-5,
  "justification": "one short sentence"
}}
"""


def _default_scores() -> Dict[str, int | str]:
    return {
        "correctness": 1,
        "relevance": 1,
        "comprehensiveness": 1,
        "diversity": 1,
        "logical_consistency": 1,
        "coherence": 1,
        "groundedness": 1,
        "justification": "Judge parse failed.",
    }


def _try_parse_json(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def judge_answer(
    client: OllamaClient,
    judge_model: str,
    question: str,
    gold_answer: str,
    pred_answer: str,
    retrieved_context: str,
) -> Dict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        pred_answer=pred_answer,
        retrieved_context=retrieved_context[:4000],
    )

    output = client.generate(
        model=judge_model,
        prompt=prompt,
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
    )
    try:
        parsed = _try_parse_json(output)
    except Exception:
        return _default_scores()

    base = _default_scores()
    base.update(parsed)
    return base
