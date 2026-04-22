from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseRAG(ABC):
    @abstractmethod
    def build_index(self, corpus: List[str]) -> None:
        ...

    @abstractmethod
    def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Returns:
          {
            "answer": str,
            "retrieved_texts": List[str],
          }
        """
        ...
