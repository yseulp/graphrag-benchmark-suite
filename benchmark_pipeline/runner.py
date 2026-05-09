from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .config import ExperimentConfig, load_experiment_config
from .dataset import load_or_build_split
from .methods import build_method
from .metrics import mean
from .ollama_client import OllamaClient
from .utils import write_jsonl


def _method_output_path(base_dir: str, run_id: str, method_name: str) -> Path:
    return Path(base_dir) / run_id / f"{method_name}.jsonl"


def _summary_path(base_dir: str, run_id: str) -> Path:
    return Path(base_dir) / run_id / "summary.json"


def run_benchmark(cfg: ExperimentConfig) -> Dict:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    client = OllamaClient(base_url=cfg.models.ollama_base_url, timeout_s=cfg.runtime.timeout_s)

    eval_rows = load_or_build_split(
        split_path=cfg.dataset.split_file,
        hf_id=cfg.dataset.hf_id,
        split=cfg.dataset.split,
        domains=cfg.dataset.domains,
        samples_per_domain=cfg.dataset.samples_per_domain,
        seed=cfg.dataset.seed,
    )

    contexts = list(dict.fromkeys([r["context"] for r in eval_rows if r.get("context")]))
    all_method_summaries: List[Dict] = []

    print(
        f"Run {run_id}: loaded {len(eval_rows)} eval rows and {len(contexts)} unique contexts",
        flush=True,
    )

    for method_name in cfg.methods:
        print(f"[{method_name}] building method", flush=True)
        method = build_method(
            method_name=method_name,
            client=client,
            generation_model=cfg.models.generation,
            embedding_model=cfg.models.embedding,
            retrieval_token_budget=cfg.runtime.retrieval_token_budget,
            answer_max_tokens=cfg.runtime.max_answer_tokens,
            temperature=cfg.runtime.temperature,
            top_p=cfg.runtime.top_p,
        )

        indexing_start = time.perf_counter()
        print(f"[{method_name}] indexing started", flush=True)
        method.build_index(
            contexts=contexts,
            chunk_size=cfg.chunking.chunk_size,
            chunk_overlap=cfg.chunking.chunk_overlap,
        )
        indexing_time_s = time.perf_counter() - indexing_start
        print(f"[{method_name}] indexing finished in {indexing_time_s:.2f}s", flush=True)

        method_rows: List[Dict] = []
        total = len(eval_rows)
        for i, row in enumerate(eval_rows, start=1):
            q = row["question"]
            start = time.perf_counter()
            pred = method.answer(q)
            latency_ms = (time.perf_counter() - start) * 1000

            method_rows.append(
                {
                    **row,
                    "method": method_name,
                    "pred_answer": pred.answer,
                    "retrieved_texts": pred.retrieved_texts,
                    "retrieval_token_cost": pred.retrieval_token_cost,
                    "retrieval_time_ms": pred.retrieval_time_ms,
                    "latency_ms": latency_ms,
                    "raw": pred.raw,
                }
            )

            if i % 10 == 0 or i == total:
                print(
                    f"[{method_name}] progress {i}/{total} samples, last latency={latency_ms:.1f}ms",
                    flush=True,
                )

        output_path = _method_output_path(cfg.output_dir, run_id, method_name)
        write_jsonl(output_path, method_rows)
        print(f"[{method_name}] wrote rows to {output_path}", flush=True)

        retrieval_tokens = mean([float(r.get("retrieval_token_cost", 0)) for r in method_rows])
        retrieval_ms = mean([float(r.get("retrieval_time_ms", 0)) for r in method_rows])
        latency_ms = mean([float(r.get("latency_ms", 0)) for r in method_rows])

        domain_summary: Dict[str, Dict[str, float]] = {}
        for row in method_rows:
            domain = str(row.get("domain", "unknown"))
            bucket = domain_summary.setdefault(
                domain,
                {
                    "samples": 0,
                    "avg_retrieval_token_cost": 0.0,
                    "avg_retrieval_time_ms": 0.0,
                    "avg_latency_ms": 0.0,
                },
            )
            bucket["samples"] += 1
            bucket["avg_retrieval_token_cost"] += float(row.get("retrieval_token_cost", 0) or 0)
            bucket["avg_retrieval_time_ms"] += float(row.get("retrieval_time_ms", 0) or 0)
            bucket["avg_latency_ms"] += float(row.get("latency_ms", 0) or 0)

        for bucket in domain_summary.values():
            samples = max(1, int(bucket["samples"]))
            bucket["avg_retrieval_token_cost"] /= samples
            bucket["avg_retrieval_time_ms"] /= samples
            bucket["avg_latency_ms"] /= samples

        all_method_summaries.append(
            {
                "method": method_name,
                "samples": len(method_rows),
                "indexing_time_s": indexing_time_s,
                "avg_retrieval_token_cost": retrieval_tokens,
                "avg_retrieval_time_ms": retrieval_ms,
                "avg_latency_ms": latency_ms,
                "by_domain": domain_summary,
            }
        )
        print(
            f"[{method_name}] done: avg_latency={latency_ms:.1f}ms, avg_retrieval={retrieval_ms:.1f}ms, "
            f"avg_tokens={retrieval_tokens:.1f}",
            flush=True,
        )

    summary = {
        "run_id": run_id,
        "track": cfg.track,
        "config_name": cfg.name,
        "models": {
            "generation": cfg.models.generation,
            "embedding": cfg.models.embedding,
            "judge": cfg.models.judge,
        },
        "runtime": {
            "retrieval_token_budget": cfg.runtime.retrieval_token_budget,
            "max_answer_tokens": cfg.runtime.max_answer_tokens,
        },
        "dataset": {
            "split_file": cfg.dataset.split_file,
            "samples": len(eval_rows),
        },
        "methods": all_method_summaries,
    }

    out = _summary_path(cfg.output_dir, run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local unified UltraDomain benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    summary = run_benchmark(cfg)
    print("Run complete:")
    print(summary["run_id"])
    for method in summary["methods"]:
        print(
            f"- {method['method']}: avg_latency={method['avg_latency_ms']:.1f}ms, "
            f"avg_retrieval={method['avg_retrieval_time_ms']:.1f}ms"
        )


if __name__ == "__main__":
    main()
