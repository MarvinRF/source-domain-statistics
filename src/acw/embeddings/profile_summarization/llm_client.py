"""Backend-agnostic LLM client (local Ollama or an OpenAI-compatible vLLM server).

The backend is selected with an environment variable:
    LLM_BACKEND=ollama   (default; local Ollama server)
    LLM_BACKEND=vllm     (vLLM server via the OpenAI-compatible API)

Optional variables (defaults in acw.settings):
    OLLAMA_URL           (e.g. http://localhost:11434/api/generate)
    VLLM_BASE_URL        (e.g. http://localhost:8000/v1)
    VLLM_MODEL_PATH      (model path served by vLLM)

Usage:
    from .llm_client import build_client

    client = build_client(model="llama3.2:3b")   # Ollama
    client = build_client()                       # vLLM (read from env)

    text = client.generate(prompt, temperature=0.7)
    ok   = client.health_check()
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx
import requests

from acw.settings import OLLAMA_URL as _DEFAULT_OLLAMA_URL
from acw.settings import VLLM_BASE_URL as _DEFAULT_VLLM_BASE_URL
from acw.settings import VLLM_MODEL_PATH as _DEFAULT_VLLM_MODEL_PATH

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Common interface
# ─────────────────────────────────────────────────────────────────────────────


class LLMClient:
    """Minimal interface that every backend implements."""

    model: str  # model name/path, set by each subclass

    def generate(self, prompt: str, temperature: float) -> str:
        raise NotImplementedError

    async def agenerate(self, prompt: str, temperature: float) -> str:
        """Async variant. Default falls back to running the sync `generate`
        in a worker thread — correct but doesn't add real concurrency.
        Backends that talk to a server worth saturating (vLLM) should
        override this with a native async HTTP call."""
        return await asyncio.to_thread(self.generate, prompt, temperature)

    def health_check(self) -> bool:
        try:
            out = self.generate("hello", 0.1)
            return bool(out)
        except Exception as exc:
            log.error(f"Health check failed: {exc}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Backend Ollama (local)
# ─────────────────────────────────────────────────────────────────────────────


class OllamaClient(LLMClient):
    """Client for a local Ollama server (POST /api/generate)."""

    def __init__(
        self,
        model: str,
        timeout: int = 120,
        url: str | None = None,
    ):
        self.model = model
        self.timeout = timeout
        self.url = url or os.getenv("OLLAMA_URL", _DEFAULT_OLLAMA_URL)

    def generate(self, prompt: str, temperature: float) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 220,
            },
        }
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["response"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# vLLM backend (OpenAI-compatible server)
# ─────────────────────────────────────────────────────────────────────────────


class VLLMClient(LLMClient):
    """Client for a vLLM server exposing the OpenAI-compatible API, e.g.
        python -m vllm.entrypoints.openai.api_server --port 8000

    Environment variables:
        VLLM_BASE_URL   — server base URL (e.g. http://localhost:8000/v1)
        VLLM_MODEL_PATH — model path on the server
    """

    def __init__(
        self,
        model: str | None = None,
        timeout: int = 120,
        base_url: str | None = None,
    ):
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL", _DEFAULT_VLLM_BASE_URL)).rstrip("/")
        self.model = model or os.getenv("VLLM_MODEL_PATH", _DEFAULT_VLLM_MODEL_PATH)
        self.timeout = timeout
        # The model is instruction-tuned, so use /chat/completions rather than
        # raw /completions. Otherwise the model treats the prompt as text to
        # continue and tends to extend the constraint list at the end of the
        # prompt (e.g. inventing an extra rule such as "No explanations.")
        # before answering; this leaked into most H4 profiles.
        self._url = f"{self.base_url}/chat/completions"
        self._async_client: httpx.AsyncClient | None = None

    def _get_async_client(self) -> httpx.AsyncClient:
        # Lazily created and reused across calls (one connection pool for
        # the whole run) instead of opening/closing a client per request.
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client

    def generate(self, prompt: str, temperature: float) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 220,
            "temperature": temperature,
        }
        response = requests.post(self._url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    async def agenerate(self, prompt: str, temperature: float) -> str:
        """Native async call — lets many requests be in flight at once so
        vLLM's own request batching (--max-num-seqs) actually gets used,
        instead of one request at a time leaving the GPU idle in between."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 220,
            "temperature": temperature,
        }
        client = self._get_async_client()
        response = await client.post(self._url, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def health_check(self) -> bool:
        try:
            health_url = f"{self.base_url.rsplit('/v1', 1)[0]}/health"
            resp = requests.get(health_url, timeout=5)
            return resp.status_code == 200
        except Exception as exc:
            log.error(f"vLLM health check failed: {exc}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def build_client(model: str | None = None) -> LLMClient:
    """Instantiate the client selected by LLM_BACKEND.

    Examples:
        # Ollama (default)
        client = build_client("llama3.2:3b")

        # vLLM; set before running:
        #   export LLM_BACKEND=vllm
        #   export VLLM_MODEL_PATH=/path/to/llama-3.1-70b-awq
        client = build_client()
    """
    backend = os.getenv("LLM_BACKEND", "ollama").lower()

    if backend == "vllm":
        client = VLLMClient(model=model)
        log.info(f"LLM backend: vLLM | url={client.base_url} | model={client.model}")
    elif backend == "ollama":
        if model is None:
            raise ValueError(
                "model is required for the Ollama backend. "
                "Pass --model <name> or set DEFAULT_LLM_MODEL in the config."
            )
        client = OllamaClient(model=model)
        log.info(f"LLM backend: Ollama | url={client.url} | model={client.model}")
    else:
        raise ValueError(f"Invalid LLM_BACKEND: '{backend}'. Use 'ollama' or 'vllm'.")

    return client
