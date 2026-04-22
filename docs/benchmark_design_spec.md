# UltraDomain Graph-RAG Benchmark Design

## 1. Benchmark Design Specification

### Goal
Create a reusable, fair, local-only benchmark for graph-based or graph-inspired RAG systems on UltraDomain.

### Core principles
- No proprietary APIs.
- Local open-source models via Ollama.
- Compare methods under common controls in a unified track.
- Optionally run native reproduction settings for method-specific behavior checks.

### Main methods
- Naive RAG
- Standard Vector RAG
- GraphRAG
- LightRAG
- PathRAG
- HyperGraphRAG

### Optional methods
- MemoRAG (extended baseline)
- CDF-RAG (auxiliary causal-only track)

### Shared controls
1. Same corpus input
2. Same preprocessing
3. Same chunking (unless method-required override)
4. Same embedding model
5. Same generation model
6. Same context budget
7. Same answer prompt
8. Same evaluation sample set
9. Same random seed policy
10. Same runtime environment

### Fairness normalization
Normalize retrieval by token budget rather than only top-k.

## 2. Method Normalization Table

| Method | Core retrieval unit | Method-specific component | Unified normalization |
|---|---|---|---|
| Naive RAG | fixed chunks | none | common prompt/model/budget |
| Vector RAG | vector chunks | retriever backend | common prompt/model/budget |
| GraphRAG | graph summaries/communities | graph build and graph traversal | common prompt/model/budget |
| LightRAG | dual-level graph retrieval | LightRAG internals | common prompt/model/budget |
| PathRAG | relational path retrieval | path extraction and path prompting | common prompt/model/budget |
| HyperGraphRAG | hyperedge retrieval | hypergraph construction and reasoning | common prompt/model/budget |
| MemoRAG (opt) | long-context memory units | memory mechanism | common prompt/model/budget |
| CDF-RAG (aux) | causal graph units | causal discovery modules | separate auxiliary track |

## 3. Default Experimental Configuration

- Generation: `qwen2.5:14b`
- Embedding: `mxbai-embed-large`
- Judge: `qwen2.5:14b`
- Chunk size/overlap: `512/64`
- Retrieval token budget: `2000`
- Max answer tokens: `256`
- Temperature/top_p: `0.0/1.0`
- Seeds: `[13, 42, 123]`

See `config/benchmark_default.yaml`.

## 4. Query Taxonomy

- local_factoid
- multi_hop_relation
- global_synthesis
- cross_document_reasoning
- n_ary_relational

Task groups:
- Group 1: global synthesis
- Group 2: multi-hop knowledge reasoning

## 5. Evaluation Metric Definitions

### Layer 1: reference-based
- Exact Match
- Token F1

### Layer 2: LLM-as-judge (local open model)
- correctness
- relevance
- comprehensiveness
- diversity
- logical_consistency
- coherence
- groundedness

### Efficiency metrics
- retrieval token cost
- latency
- indexing time
- retrieval time

## 6. Experiment Matrix

1. Unified main run:
   - methods: naive, vector, graphrag, lightrag, pathrag, hypergraphrag
   - shared models and prompts
2. Unified budget sweep:
   - token budgets: 1000, 2000, 4000
3. Unified generator swap:
   - `qwen2.5:14b` and `llama3.1:8b`
4. Optional native run:
   - per-method paper-inspired settings
5. Optional tracks:
   - MemoRAG
   - CDF-RAG auxiliary causal subset

## 7. Reproduction Checklist

For each method:
- Adapter returns unified schema (`answer`, `retrieved_texts`, timing, token cost).
- Retrieval respects token budget cap.
- Uses unified prompt/model in unified track.
- Expected behavior appears versus Naive RAG:
  - GraphRAG: stronger global synthesis
  - LightRAG: broader contextual coverage / efficiency
  - PathRAG: improved logic structure and reduced redundancy
  - HyperGraphRAG: better n-ary reasoning under budget constraints
  - MemoRAG: stronger long-context behavior

## 8. Result Reporting Template

See `docs/result_reporting_template.md`.
