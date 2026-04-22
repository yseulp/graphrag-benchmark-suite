import json
import asyncio
from lightrag import LightRAG
from rag_systems.lightrag_adapter import LightRAGAdapter
import numpy as np
import requests

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


# Automatically determine embedding dimension
# by making a single embedding call
_tmp = requests.post(
    "http://localhost:11434/api/embeddings",
    json={"model": "mxbai-embed-large:latest", "prompt": "dim_check"},
    timeout=60,
)
_tmp.raise_for_status()
ollama_embedding.embedding_dim = len(_tmp.json()["embedding"])


async def ollama_llm(prompt: str, **kwargs) -> str:
    """
    Simple Ollama LLM wrapper for LightRAG.

    Returns a single generated response (non-streaming).
    """
    model = "llama3.1:8b"  # small model for smoke testing
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


def main():
    # Initialize LightRAG engine with Ollama-based embedding and LLM
    engine = LightRAG(
        working_dir="./tmp_lightrag",
        embedding_func=ollama_embedding,
        llm_model_func=ollama_llm,
    )

    # Required initialization step for LightRAG storages
    asyncio.run(engine.initialize_storages())

    # Wrap LightRAG with the BaseRAG-compatible adapter
    rag = LightRAGAdapter(engine, mode="naive")

    # Minimal corpus for smoke testing
    corpus = [
        "UltraDomain is a benchmark dataset for evaluating retrieval-augmented generation (RAG) systems.",
        "PTW is an institute at TU Darmstadt.",
        "Ollama runs LLMs locally."
    ]

    rag.build_index(corpus)

    out = rag.answer("What is UltraDomain?")
    print("ANSWER:\n", out["answer"])
    print("\nRAW KEYS:", list(out["raw"].keys()))
    print("\nRAW (first 1500 chars):")
    print(json.dumps(out["raw"], ensure_ascii=False, indent=2)[:1500])
    print("RETRIEVED_TEXTS:", out["retrieved_texts"])


if __name__ == "__main__":
    main()
