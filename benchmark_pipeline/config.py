from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class ModelConfig:
    generation: str
    embedding: str
    judge: str
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64


@dataclass
class RuntimeConfig:
    retrieval_token_budget: int = 2000
    max_answer_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    timeout_s: int = 180


@dataclass
class DatasetConfig:
    hf_id: str = "TommyChien/UltraDomain"
    split: str = "train"
    domains: List[str] = field(default_factory=lambda: ["agriculture", "cs", "legal", "mix"])
    samples_per_domain: int = 100
    seed: int = 42
    split_file: str = "data/splits/ultradomain_v1/unified_eval.jsonl"


@dataclass
class ExperimentConfig:
    name: str
    track: str
    methods: List[str]
    seeds: List[int]
    models: ModelConfig
    chunking: ChunkingConfig
    runtime: RuntimeConfig
    dataset: DatasetConfig
    output_dir: str


def _as_dataclass(data: Dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        name=data["name"],
        track=data.get("track", "unified"),
        methods=data["methods"],
        seeds=data.get("seeds", [42]),
        models=ModelConfig(**data["models"]),
        chunking=ChunkingConfig(**data.get("chunking", {})),
        runtime=RuntimeConfig(**data.get("runtime", {})),
        dataset=DatasetConfig(**data.get("dataset", {})),
        output_dir=data.get("output_dir", "runs/unified_benchmark"),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return _as_dataclass(raw)
