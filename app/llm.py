"""LLM client: asks Ollama's chat API for a JSON answer."""
from __future__ import annotations

from typing import Protocol

import httpx


class LLMError(RuntimeError):
    """Ollama could not produce an answer (down, model missing, bad response)."""


class LLMTimeout(LLMError):
    """Ollama took longer than the configured timeout."""


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


# JSON shape the model must return. Ollama enforces it ("structured outputs").
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answered": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answered", "answer", "citations"],
}


class OllamaLLM:
    def __init__(self, url: str, model: str, timeout_s: float, max_tokens: int,
                 client: httpx.Client | None = None):
        self.model = model
        self.max_tokens = max_tokens
        self._client = client or httpx.Client(base_url=url, timeout=timeout_s)

    def complete(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "format": ANSWER_SCHEMA,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {"temperature": 0, "num_predict": self.max_tokens},
        }
        try:
            r = self._client.post("/api/chat", json=body)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except httpx.TimeoutException as e:
            raise LLMTimeout(f"Ollama did not answer in time: {e}") from e
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as e:
            raise LLMError(f"Ollama chat failed: {e}") from e
