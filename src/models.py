"""OpenRouter model adapter."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is listed but this keeps errors friendly.
    load_dotenv = None


@dataclass
class ModelResponse:
    text: str
    latency_seconds: float
    usage: dict[str, Any] | None = None


class OpenRouterClient:
    """Tiny OpenAI-compatible chat-completions client for OpenRouter."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: int = 60,
    ) -> None:
        if load_dotenv is not None:
            load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> ModelResponse:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "The 'requests' package is required for OpenRouter calls. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. Add it to .env or use --dry-run."
            )

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "SLM Agent Cascades",
        }

        start = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

        latency = time.perf_counter() - start
        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenRouter request failed with HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenRouter response shape: {data}") from exc

        return ModelResponse(text=str(text), latency_seconds=latency, usage=data.get("usage"))
