from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_run_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _to_ragas_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        contexts = row.get("retrieved_texts") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        if not isinstance(contexts, list):
            contexts = []

        clean_contexts = [c for c in contexts if isinstance(c, str) and c.strip()]
        out.append(
            {
                "question": str(row.get("question") or ""),
                "answer": str(row.get("pred_answer") or ""),
                "contexts": clean_contexts,
                "ground_truth": str(row.get("gold_answer") or ""),
            }
        )
    return out


def _build_ollama_models(base_url: str, llm_model: str, embedding_model: str):
    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies for Ollama-backed RAGAS. "
            "Install with: pip install ragas langchain-ollama"
        ) from exc

    llm = ChatOllama(model=llm_model, base_url=base_url, temperature=0)
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
    return llm, embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate unified benchmark outputs with RAGAS")
    parser.add_argument("--run-file", required=True, help="Path to run jsonl (e.g. runs/.../lightrag.jsonl)")
    parser.add_argument(
        "--out-summary",
        default=None,
        help="Path to summary json output (default: <run-file>.ragas.summary.json)",
    )
    parser.add_argument(
        "--out-scored",
        default=None,
        help="Path to per-sample scored jsonl output (default: <run-file>.ragas.scored.jsonl)",
    )
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--llm-model", default="qwen2.5:14b")
    parser.add_argument("--embedding-model", default="mxbai-embed-large")
    args = parser.parse_args()

    run_path = Path(args.run_file)
    if not run_path.exists():
        raise FileNotFoundError(f"Run file not found: {run_path}")

    rows = _load_run_rows(run_path)
    if not rows:
        raise ValueError(f"Run file has no rows: {run_path}")

    ragas_rows = _to_ragas_rows(rows)
    empty_contexts = sum(1 for r in ragas_rows if not r["contexts"])

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise ImportError(
            "Missing RAGAS dependencies. Install with: pip install ragas datasets"
        ) from exc

    llm, embeddings = _build_ollama_models(
        base_url=args.ollama_base_url,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
    )

    ds = Dataset.from_list(ragas_rows)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    try:
        scored_df = result.to_pandas()
        scored_rows = scored_df.to_dict(orient="records")
    except Exception:
        scored_rows = ragas_rows

    summary = {}
    if hasattr(result, "items"):
        summary = {k: float(v) for k, v in dict(result).items()}
    elif hasattr(result, "to_dict"):
        raw = result.to_dict()
        summary = {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}

    summary_payload = {
        "run_file": str(run_path),
        "samples": len(rows),
        "samples_without_contexts": empty_contexts,
        "metrics": summary,
    }

    out_summary = Path(args.out_summary) if args.out_summary else run_path.with_suffix(".ragas.summary.json")
    out_scored = Path(args.out_scored) if args.out_scored else run_path.with_suffix(".ragas.scored.jsonl")

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_scored.parent.mkdir(parents=True, exist_ok=True)

    out_summary.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_scored.open("w", encoding="utf-8") as handle:
        for row in scored_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Loaded {len(rows)} rows from: {run_path}")
    print(f"Rows with empty contexts: {empty_contexts}")
    print("RAGAS metrics:")
    for key, value in summary.items():
        print(f"- {key}: {value:.4f}")
    print(f"Wrote summary: {out_summary}")
    print(f"Wrote per-sample scores: {out_scored}")


if __name__ == "__main__":
    main()
