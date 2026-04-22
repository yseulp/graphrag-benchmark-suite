from __future__ import annotations

from typing import Any, Dict, List


class OllamaClient:
    def __init__(self, base_url: str, timeout_s: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def embed(self, model: str, text: str) -> List[float]:
        import requests

        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 256,
    ) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                },
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload: Dict[str, Any] = response.json()
        return payload.get("response", "")
