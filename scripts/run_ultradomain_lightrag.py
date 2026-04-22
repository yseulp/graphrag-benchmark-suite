#scripts/run_ultradomain_lightrag.py
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from lightrag import LightRAG

from benchmark_datasets.ultradomain_loader import load_ultradomain
from rag_systems.lightrag_adapter import LightRAGAdapter

async def ollama_embedding(texts: list[str]) -> list[list[float]]:
    r = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "mxbai-embed-large:latest", "prompt": texts[0]},
        timeout=60,
    )
    r.raise_for_status()

    first = r.json()["embedding"]
    out = [first]

    for t in texts[1:]:
        r = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "mxbai-embed-large:latest", "prompt": t},
            timeout=60,
        )
        r.raise_for_status()
        out.append(r.json()["embedding"])

    return out


_tmp = requests.post(
    "http://localhost:11434/api/embeddings",
    json={"model": "mxbai-embed-large:latest", "prompt": "dim_check"},
    timeout=60,
)
_tmp.raise_for_status()
ollama_embedding.embedding_dim = len(_tmp.json()["embedding"])


async def ollama_llm(prompt: str, **kwargs) -> str:
    model = "llama3.1:8b" 
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["response"]


def dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def unique_preserve_order(texts: List[str]) -> List[str]:
    # avoid indexing duplicate contexts
    return list(dict.fromkeys(texts))

def main(
    domain: Optional[str] = "agriculture",
    limit: int = 10,
    mode: str = "naive",
    top_k: int = 5,
    save_raw: bool = False,
) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


    dom_tag = domain or "all"
    storage_dir = Path(f"storage/lightrag_ultradomain/{dom_tag}_{limit}_{mode}_{run_id}")
    out_path = Path(f"runs/ultradomain_lightrag/{dom_tag}_{limit}_{mode}_{run_id}.jsonl")

    # load ultradomain
    ds = load_ultradomain(domain=domain, limit=limit, streaming=True)
    ds = list(ds)
    print(f"Loaded UltraDomain: domain={domain}, size={len(ds)}")


    # build context
    contexts = [ex["context"] for ex in ds]
    contexts = unique_preserve_order(contexts)
    print(f"Contexts to index: {len(contexts)} (after dedup)")

    #initialize lightrag engine 
    engine = LightRAG(working_dir=str(storage_dir), embedding_func=ollama_embedding, llm_model_func=ollama_llm,)
    asyncio.run(engine.initialize_storages())

    rag = LightRAGAdapter(engine= engine, mode=mode)

    #index all contexts 
    rag.build_index(contexts)
    print("Index built!")

    #batch questions
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds): 
        q = ex["input"]
        gold_list = ex.get("answers", []) or []
        gold = gold_list[0] if len(gold_list) > 0 else ""
        dom = ex.get("label", domain)

        pred = rag.answer(q, top_k=top_k)

        row = {
            "id": i, 
            "domain": dom,
            "question": q, 
            "gold_answer": gold, 
            "pred_answer": pred.get("answer", ""), 
            "retrieved_texts": pred.get("retrieved_texts", []), 
        }
        if save_raw:
            row["raw"] = pred.get("raw", {})

        rows.append(row)

        if (i+1) % 10 == 0 or (i+1) == len(ds):
            print(f"[{i+1}/{len(ds)}] done")

    dump_jsonl(out_path, rows)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main(domain="agriculture", limit=10, mode="naive", top_k=5, save_raw=False)