"""Tier 3 option: Google Gemini (3.4.0), through Google AI Studio or Vertex AI with an API key.

The prompt, the JSON shape and the hallucination guard are the OpenAI-compatible connector's (every value must cite OCR
line ids); only the call differs: Gemini's generateContent with a JSON reply, page images sent inline. AI Studio:
https://generativelanguage.googleapis.com; Vertex AI (express mode, API key): https://aiplatform.googleapis.com.
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
    spec = importlib.util.spec_from_file_location("docauto_option_llm_shared_gemini", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_llm = _load_shared()
SERVICES = {"ai-studio": "https://generativelanguage.googleapis.com/v1beta/models",
            "vertex": "https://aiplatform.googleapis.com/v1/publishers/google/models"}


def gemini_parts(user_content: Any) -> list[dict]:
    """The OpenAI-style message content (text, or text and data-URL images) as Gemini parts."""
    if isinstance(user_content, str):
        return [{"text": user_content}]
    parts = []
    for p in user_content:
        if p.get("type") == "text":
            parts.append({"text": p["text"]})
        elif p.get("type") == "image_url":
            head, _, data = p["image_url"]["url"].partition(",")
            mime = head[len("data:"):].split(";")[0] or "image/jpeg"
            parts.append({"inline_data": {"mime_type": mime, "data": data}})
    return parts


class GeminiProvider(_llm.LlmExtractionProvider):
    def init(self, ctx):
        super().init(ctx)
        cfg = ctx.config
        self.service = (cfg.get("service") or "ai-studio").strip()
        self.base_url = (cfg.get("baseUrl") or SERVICES.get(self.service, SERVICES["ai-studio"])).rstrip("/")
        self.model = cfg.get("model") or "gemini-2.5-flash"
        self.send_images = str(cfg.get("sendImages", True)).lower() not in ("false", "0", "no")

    def health_check(self) -> HealthStatus:
        if not self.api_key:
            return HealthStatus(False, "API key not configured")
        return HealthStatus(True, f"model={self.model} ({self.service})")

    def _chat(self, user_content: Any) -> str:
        body = {"systemInstruction": {"parts": [{"text": _llm.SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": gemini_parts(user_content)}],
                "generationConfig": {"temperature": self.temperature, "responseMimeType": "application/json"}}
        url = f"{self.base_url}/{self.model}:generateContent"
        try:
            r = httpx.post(url, json=body, headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
                           timeout=self.timeout)
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
        cands = r.json().get("candidates") or []
        if not cands:
            raise ProviderError("Gemini returned no answer (blocked or empty)", self.info.id)
        return "".join(p.get("text", "") for p in (cands[0].get("content") or {}).get("parts", []))
