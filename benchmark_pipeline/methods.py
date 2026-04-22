from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from .ollama_client import OllamaClient
from .utils import approximate_token_count, chunk_text


ANSWER_PROMPT_TEMPLATE = """You are a careful question-answering assistant.
Use only the provided context. If context is insufficient, say so briefly.
Be concise and factual.

Question:
{question}

Context:
{context}

Answer:
"""


@dataclass
class MethodResult:
    answer: str
    retrieved_texts: List[str]
    retrieval_token_cost: int
    retrieval_time_ms: float
    raw: Dict[str, Any]


class BenchmarkMethod:
    def __init__(
        self,
        name: str,
        client: OllamaClient,
        generation_model: str,
        embedding_model: str,
        retrieval_token_budget: int,
        answer_max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        self.name = name
        self.client = client
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.retrieval_token_budget = retrieval_token_budget
        self.answer_max_tokens = answer_max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def build_index(self, contexts: List[str], chunk_size: int, chunk_overlap: int) -> None:
        raise NotImplementedError

    def answer(self, question: str) -> MethodResult:
        raise NotImplementedError

    def _generate_answer(self, question: str, retrieved_texts: List[str]) -> str:
        context = "\n\n".join(retrieved_texts)
        prompt = ANSWER_PROMPT_TEMPLATE.format(question=question, context=context)
        return self.client.generate(
            model=self.generation_model,
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.answer_max_tokens,
        )


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class NaiveRAGMethod(BenchmarkMethod):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("naive_rag", *args, **kwargs)
        self.chunks: List[str] = []

    def build_index(self, contexts: List[str], chunk_size: int, chunk_overlap: int) -> None:
        chunks: List[str] = []
        for text in contexts:
            chunks.extend(chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
        self.chunks = chunks

    def answer(self, question: str) -> MethodResult:
        start = time.perf_counter()
        out: List[str] = []
        budget = 0
        for chunk in self.chunks:
            c = approximate_token_count(chunk)
            if budget + c > self.retrieval_token_budget:
                break
            out.append(chunk)
            budget += c
        retrieval_ms = (time.perf_counter() - start) * 1000
        answer = self._generate_answer(question, out)
        return MethodResult(answer=answer, retrieved_texts=out, retrieval_token_cost=budget, retrieval_time_ms=retrieval_ms, raw={})


class VectorRAGMethod(BenchmarkMethod):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("vector_rag", *args, **kwargs)
        self.chunks: List[str] = []
        self.embeddings: List[List[float]] = []

    def build_index(self, contexts: List[str], chunk_size: int, chunk_overlap: int) -> None:
        chunks: List[str] = []
        for text in contexts:
            chunks.extend(chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

        self.chunks = chunks
        self.embeddings = [self.client.embed(self.embedding_model, ch) for ch in chunks]

    def answer(self, question: str) -> MethodResult:
        start = time.perf_counter()
        q_emb = self.client.embed(self.embedding_model, question)
        scored = [(_cosine_sim(q_emb, emb), idx) for idx, emb in enumerate(self.embeddings)]
        scored.sort(reverse=True)

        selected: List[str] = []
        budget = 0
        for _, idx in scored:
            ch = self.chunks[idx]
            c = approximate_token_count(ch)
            if budget + c > self.retrieval_token_budget:
                continue
            selected.append(ch)
            budget += c
            if budget >= self.retrieval_token_budget:
                break

        retrieval_ms = (time.perf_counter() - start) * 1000
        answer = self._generate_answer(question, selected)
        return MethodResult(answer=answer, retrieved_texts=selected, retrieval_token_cost=budget, retrieval_time_ms=retrieval_ms, raw={})


class LightRAGMethod(BenchmarkMethod):
    def __init__(self, mode: str = "naive", *args, **kwargs) -> None:
        super().__init__("lightrag", *args, **kwargs)
        self.mode = mode
        self.adapter = None

    def build_index(self, contexts: List[str], chunk_size: int, chunk_overlap: int) -> None:
        del chunk_size, chunk_overlap
        try:
            import asyncio
            from lightrag import LightRAG

            from rag_systems.lightrag_adapter import LightRAGAdapter
        except Exception as exc:
            raise RuntimeError("lightrag package unavailable; cannot run method=lightrag") from exc

        async def _embedding(texts: Any) -> Any:
            import numpy as np

            if isinstance(texts, str):
                batch = [texts]
            else:
                batch = list(texts)
            if not batch:
                return np.asarray([], dtype=np.float32)

            rows = [self.client.embed(self.embedding_model, t) for t in batch]
            return np.asarray(rows, dtype=np.float32)

        first = self.client.embed(self.embedding_model, "dim_check")
        embedding_dim = len(first)

        embedding_func = _embedding
        try:
            try:
                from lightrag.utils import EmbeddingFunc, wrap_embedding_func_with_attrs

                embedding_func = wrap_embedding_func_with_attrs(
                    embedding_dim=embedding_dim,
                    max_token_size=8192,
                )(_embedding)
            except Exception:
                from lightrag import EmbeddingFunc  # type: ignore[attr-defined]

                embedding_func = EmbeddingFunc(
                    embedding_dim=embedding_dim,
                    max_token_size=8192,
                    func=_embedding,
                )
        except Exception:
            _embedding.embedding_dim = embedding_dim
            _embedding.max_token_size = 8192
            embedding_func = _embedding

        async def _llm(prompt: str, **kwargs) -> str:
            del kwargs
            return self.client.generate(
                model=self.generation_model,
                prompt=prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.answer_max_tokens,
            )

        engine = LightRAG(
            working_dir="./tmp_lightrag",
            embedding_func=embedding_func,
            llm_model_func=_llm,
        )
        asyncio.run(engine.initialize_storages())
        self.adapter = LightRAGAdapter(engine=engine, mode=self.mode)
        self.adapter.build_index(contexts)

    def answer(self, question: str) -> MethodResult:
        if self.adapter is None:
            raise RuntimeError("LightRAG method not initialized")
        start = time.perf_counter()
        out = self.adapter.answer(question)
        retrieval_ms = (time.perf_counter() - start) * 1000
        retrieved = out.get("retrieved_texts", [])
        budget = sum(approximate_token_count(t) for t in retrieved)
        return MethodResult(
            answer=out.get("answer", ""),
            retrieved_texts=retrieved,
            retrieval_token_cost=budget,
            retrieval_time_ms=retrieval_ms,
            raw=out.get("raw", {}),
        )


class PlaceholderGraphMethod(VectorRAGMethod):
    def __init__(self, method_name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = method_name


def build_method(
    method_name: str,
    client: OllamaClient,
    generation_model: str,
    embedding_model: str,
    retrieval_token_budget: int,
    answer_max_tokens: int,
    temperature: float,
    top_p: float,
) -> BenchmarkMethod:
    common = dict(
        client=client,
        generation_model=generation_model,
        embedding_model=embedding_model,
        retrieval_token_budget=retrieval_token_budget,
        answer_max_tokens=answer_max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    if method_name == "naive_rag":
        return NaiveRAGMethod(**common)
    if method_name == "vector_rag":
        return VectorRAGMethod(**common)
    if method_name == "lightrag":
        return LightRAGMethod(**common)
    if method_name in {"graphrag", "pathrag", "hypergraphrag", "memorag", "cdf_rag"}:
        return PlaceholderGraphMethod(method_name=method_name, **common)

    raise ValueError(f"Unknown method: {method_name}")
