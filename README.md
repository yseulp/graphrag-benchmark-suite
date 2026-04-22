# RAG Benchmarks (UltraDomain, Local-Only)

Reusable benchmark framework for comparing graph-based and graph-inspired RAG systems on UltraDomain under fair, shared controls.

## Key Policies

- No proprietary APIs.
- Local open models only (Ollama runtime).
- Unified fairness controls across methods.
- Reproduce relative method behavior patterns, not exact paper absolute scores.

## Main Methods

- `naive_rag`
- `vector_rag`
- `graphrag`
- `lightrag`
- `pathrag`
- `hypergraphrag`

Optional:
- `memorag`
- `cdf_rag` (auxiliary causal track)

## Models

Default in `config/benchmark_default.yaml`:
- Generation: `qwen2.5:14b`
- Embedding: `mxbai-embed-large`
- Judge: `qwen2.5:14b`

Alternative generation model: `llama3.1:8b`.

## Benchmark Artifacts

- Design spec: `docs/benchmark_design_spec.md`
- Report template: `docs/result_reporting_template.md`
- Unified config: `config/benchmark_default.yaml`
- Native template config: `config/benchmark_native_template.yaml`
- Pipeline package: `benchmark_pipeline/`

## Quick Start

1. Pull local models in Ollama:

```bash
ollama pull qwen2.5:14b
ollama pull llama3.1:8b
ollama pull mxbai-embed-large
```

2. Build a frozen UltraDomain split:

```bash
python -m scripts.build_ultradomain_split \
  --output data/splits/ultradomain_v1/unified_eval.jsonl \
  --samples-per-domain 100 \
  --seed 42
```

3. Run unified benchmark:

```bash
python -m scripts.run_unified_benchmark --config config/benchmark_default.yaml
```

Outputs are written under `runs/unified_benchmark/<run_id>/`.

## Notes on Method Implementations

- `naive_rag`, `vector_rag`, and `lightrag` are executable through the shared pipeline.
- `graphrag`, `pathrag`, `hypergraphrag`, `memorag`, `cdf_rag` are wired into the method registry and currently default to shared placeholder behavior unless you swap in dedicated adapters.
- The framework is structured so each method can preserve native internals while still producing a unified result schema.
