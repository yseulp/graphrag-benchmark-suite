# rag_systems/lightrag_adapter.py

import asyncio
from typing import Any, Dict, List, Optional

from .base import BaseRAG
from lightrag import LightRAG, QueryParam


def _approximate_token_count(text: str) -> int:
    if not text:
        return 0
    words = len(text.split())
    return max(1, int(words * 1.3))

class LightRAGAdapter(BaseRAG):
    def __init__(
        self,
        engine: LightRAG,
        mode: str = "global",
    ):
        """
        Adapter wrapper for LightRAG to fit the unified BaseRAG interface.

        Args:
            engine: An already initialized LightRAG instance.
            mode: Query mode used by LightRAG
                  (e.g., 'local', 'global', 'hybrid', 'naive').
        """
        self.engine = engine
        self.mode = mode

    def build_index(self, corpus: List[str]) -> None:
        """
        Build the LightRAG index by inserting all documents.

        Args:
            corpus: List of raw document texts used as the knowledge base.
        """
        for doc in corpus:
            self.engine.insert(doc)

    def _run_coroutine(self, coro):
        """
        Helper function to safely execute an async coroutine.

        LightRAG internally uses asyncio. This wrapper ensures that
        coroutines can be executed even if no event loop is currently running.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    def answer(
        self,
        question: str,
        top_k: int = 5,
        retrieval_token_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Answer a single UltraDomain question using LightRAG.

        This method:
        1. Executes LightRAG's aquery_llm API.
        2. Extracts the generated answer text.
        3. Extracts retrieved context chunks for evaluation.

        Args:
            question: The input question.
            top_k: Optional retrieval parameter (currently unused).

        Returns:
            A dictionary with:
            - "answer": generated answer string
            - "retrieved_texts": list of retrieved context texts
            - "raw": full raw output from LightRAG (for debugging / analysis)
        """
        param = QueryParam()
        param.mode = self.mode
        param.enable_rerank = False

        if hasattr(param, "top_k"):
            setattr(param, "top_k", top_k)

        if retrieval_token_budget is not None:
            for attr in (
                "max_token_for_text_unit",
                "max_token_for_global_context",
                "max_token_for_local_context",
                "max_total_tokens",
            ):
                if hasattr(param, attr):
                    setattr(param, attr, retrieval_token_budget)

        raw_data = self._run_coroutine(self.engine.aquery_llm(question, param=param))

        llm_resp = raw_data.get("llm_response", {}) or {}
        answer_text: str = llm_resp.get("content") or ""

        data = raw_data.get("data", {}) or {}

        retrieved_texts: List[str] = self._extract_retrieved_texts(data)
        retrieved_texts = self._trim_to_budget(retrieved_texts, retrieval_token_budget)

        return {
            "answer": answer_text,
            "retrieved_texts": retrieved_texts,
            "raw": raw_data,  
        }

    def _trim_to_budget(self, texts: List[str], budget: Optional[int]) -> List[str]:
        if budget is None or budget <= 0:
            return texts

        out: List[str] = []
        used = 0
        for text in texts:
            tok = _approximate_token_count(text)
            if used + tok > budget:
                continue
            out.append(text)
            used += tok
            if used >= budget:
                break
        return out


    def _extract_retrieved_texts(self, data: Dict[str, Any]) -> List[str]:
        """
        Extract retrieved text chunks from LightRAG's structured output.

        LightRAG returns retrieved contexts under data["chunks"].
        Each chunk contains a "content" field holding the raw text.
        """
        chunks = data.get("chunks",[])
        texts: List[str] = []
        for ch in chunks:
            if isinstance(ch, dict) and ch.get("content"):
                texts.append(ch["content"])
        return texts
