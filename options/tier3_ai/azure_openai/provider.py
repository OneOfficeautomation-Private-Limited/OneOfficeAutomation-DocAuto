"""Tier 3 option: Azure OpenAI (3.4.0): a model deployed in the customer's own Azure resource.

The OpenAI-compatible connector with Azure's address and key: {endpoint}/openai/deployments/{deployment}/chat/completions
?api-version=..., header api-key. The prompt, JSON shape and hallucination guard are the shared ones.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import httpx

from docauto.errors import AuthFailed, ProviderError, ProviderTimeout, RateLimited
from docauto.provider import HealthStatus


def _load_shared():
    path = Path(__file__).resolve().parent.parent / "llm_openai_compatible" / "provider.py"
    spec = importlib.util.spec_from_file_location("docauto_option_llm_shared_azure", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_llm = _load_shared()
DEFAULT_API_VERSION = "2024-10-21"


class AzureOpenAIProvider(_llm.LlmExtractionProvider):
    def init(self, ctx):
        super().init(ctx)
        cfg = ctx.config
        self.endpoint = (cfg.get("endpoint") or "").rstrip("/")
        self.deployment = (cfg.get("deployment") or "").strip()
        self.api_version = (cfg.get("apiVersion") or DEFAULT_API_VERSION).strip()
        self.model = self.deployment
        self.send_images = str(cfg.get("sendImages", True)).lower() not in ("false", "0", "no")

    def health_check(self) -> HealthStatus:
        if not self.endpoint or not self.deployment:
            return HealthStatus(False, "endpoint/deployment not configured")
        if not self.api_key:
            return HealthStatus(False, "key not configured")
        return HealthStatus(True, f"deployment={self.deployment}")

    def _chat(self, user_content: Any) -> str:
        body = {"temperature": self.temperature, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": _llm.SYSTEM_PROMPT}, {"role": "user", "content": user_content}]}
        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
        try:
            r = httpx.post(url, json=body, headers={"Content-Type": "application/json", "api-key": self.api_key}, timeout=self.timeout)
        except httpx.TimeoutException as e:
            raise ProviderTimeout(str(e), self.info.id)
        except httpx.HTTPError as e:
            raise ProviderError(str(e), self.info.id)
        if r.status_code in (401, 403):
            raise AuthFailed(r.text[:300], self.info.id)
        if r.status_code == 429:
            raise RateLimited(r.text[:300], self.info.id)
        if r.status_code >= 400:
            raise ProviderError(f"HTTP {r.status_code}: {r.text[:300]}", self.info.id)
        return r.json()["choices"][0]["message"]["content"]
