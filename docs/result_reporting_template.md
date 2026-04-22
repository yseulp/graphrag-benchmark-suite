# UltraDomain Graph-RAG Benchmark Report

## Run Metadata
- Track: unified | native
- Commit: <git-hash>
- Config: <config-name>
- Dataset split file: <path>
- Generation model: <ollama tag>
- Embedding model: <model tag>
- Judge model: <ollama tag>
- Seeds: <list>
- Retrieval token budget: <int>

## Unified Results Table
| Method | EM | TokenF1 | Judge Correctness | Judge Groundedness | Retrieval Token Cost | Avg Latency (ms) | Index Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive_rag |  |  |  |  |  |  |  |
| vector_rag |  |  |  |  |  |  |  |
| graphrag |  |  |  |  |  |  |  |
| lightrag |  |  |  |  |  |  |  |
| pathrag |  |  |  |  |  |  |  |
| hypergraphrag |  |  |  |  |  |  |  |

## Slice Analysis

### Global Synthesis
| Method | TokenF1 | Correctness | Coherence | Groundedness |
|---|---:|---:|---:|---:|

### Multi-hop / N-ary
| Method | TokenF1 | Logical Consistency | Groundedness |
|---|---:|---:|---:|

## Reproduction Behavior Check
- GraphRAG expected pattern: pass/fail + note
- LightRAG expected pattern: pass/fail + note
- PathRAG expected pattern: pass/fail + note
- HyperGraphRAG expected pattern: pass/fail + note
- MemoRAG expected pattern (optional): pass/fail + note

## Fairness Compliance
- Same prompt across methods: yes/no
- Same generation model: yes/no
- Same judge model: yes/no
- Same retrieval token budget: yes/no
- Same evaluation split: yes/no
- Deviations: <list>

## Error Analysis
- recurring failure mode 1
- recurring failure mode 2
- recurring failure mode 3
