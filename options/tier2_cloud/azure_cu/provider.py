"""Tier 2 option: Azure Content Understanding (3.4.0), recognising and splitting a whole file.

Content Understanding classifies and splits from category names and descriptions, with no training: the process's
document types become an analyzer's `contentCategories` (with "other" for none of them) and `enableSegment` splits a
batch. The analyzer is created once per set of types (its id is a hash of them), then the file is sent with
`:analyzeBinary` and the result polled. REST API version 2025-11-01 (GA).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Optional

import httpx

from docauto.errors import AuthFailed, InvalidInput, ProviderError, ProviderTimeout, ProviderUnavailable, RateLimited
from docauto.provider import HealthStatus, Segment, SplitProvider

OTHER = "other"


def category_keys(labels: list[str]) -> dict[str, str]:
    """A category name for each label, as the service accepts them (letters, digits, underscores), told apart."""
    out: dict[str, str] = {}
    used = {OTHER}
    for lab in labels:
        key = re.sub(r"[^A-Za-z0-9_]", "_", lab).strip("_") or "type"
        if not key[0].isalpha():
            key = "t_" + key
        base, n = key, 2
        while key.lower() in used:
            key, n = f"{base}_{n}", n + 1
        used.add(key.lower())
        out[lab] = key
    return out


def analyzer_body(labels: list[str], descriptions: Optional[dict[str, str]], segment: bool, model: str) -> dict:
    keys = category_keys(labels)
    cats = {keys[lab]: {"description": ((descriptions or {}).get(lab) or lab).strip()} for lab in labels}
    cats[OTHER] = {"description": "Any document that is none of the other kinds."}
    body: dict[str, Any] = {"description": "OneOffice Document Automation: recognise and split",
                            "baseAnalyzerId": "prebuilt-document",
                            "config": {"enableSegment": segment, "contentCategories": cats}}
    if model:
        body["models"] = {"completion": model}
    return body


def analyzer_id(body: dict) -> str:
    return "ooa_split_" + hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def segments_from_result(result: dict, keys: dict[str, str], page_count: int) -> list[Segment]:
    """The analyze result's parts: each content's segments when it was split, else the content itself."""
    back = {v: k for k, v in keys.items()}
    out: list[Segment] = []
    for c in result.get("contents") or []:
        parts = c.get("segments") or [c]
        for s in parts:
            cat = s.get("category")
            start = int(s.get("startPageNumber") or c.get("startPageNumber") or 1)
            end = int(s.get("endPageNumber") or c.get("endPageNumber") or page_count or start)
            out.append(Segment(label=None if not cat or cat == OTHER else back.get(cat, cat),
                               pages=list(range(start, end + 1)), confidence=s.get("confidence")))
    return out


class AzureContentUnderstandingProvider(SplitProvider):
    def init(self, ctx):
        super().init(ctx)
        cfg = ctx.config
        self.endpoint = (cfg.get("endpoint") or "").rstrip("/")
        self.api_version = cfg.get("apiVersion") or "2025-11-01"
        self.model = (cfg.get("completionModel") or "").strip()
        self.location = (cfg.get("processingLocation") or "geography").strip()
        self.key = ctx.credentials.get("apiKey", "")
        self.timeout_s = float(cfg.get("timeoutSeconds") or 180)
        if cfg.get("region"):
            self.info.regions = [cfg["region"]]
        self._ready: set[str] = set()

    def health_check(self) -> HealthStatus:
        if not self.endpoint or not self.key:
            return HealthStatus(False, "endpoint/apiKey not configured")
        return HealthStatus(True, "configured")

    def split(self, data, mime, page_count, labels, descriptions=None, segment=True) -> list[Segment]:
        if not data:
            raise InvalidInput("azure-content-understanding needs the file", self.info.id)
        body = analyzer_body(labels, descriptions, segment, self.model)
        aid = analyzer_id(body)
        try:
            with httpx.Client(timeout=60) as client:
                self._ensure(client, aid, body)
                r = client.post(f"{self.endpoint}/contentunderstanding/analyzers/{aid}:analyzeBinary",
                                params={"api-version": self.api_version, "processingLocation": self.location},
                                headers={**self._auth(), "Content-Type": mime or "application/octet-stream"}, content=data)
                self._raise(r)
                op = r.headers.get("operation-location")
                if not op:
                    raise ProviderError("missing operation-location header", self.info.id)
                done = self._wait(client, op)
        except httpx.TimeoutException as e:
            raise ProviderTimeout(f"azure timeout: {e}", self.info.id)
        except httpx.HTTPError as e:
            raise ProviderUnavailable(f"cannot reach Azure endpoint: {e}", self.info.id)
        return segments_from_result(done.get("result") or {}, category_keys(labels), page_count)

    def _auth(self) -> dict:
        return {"Ocp-Apim-Subscription-Key": self.key}

    def _ensure(self, client: httpx.Client, aid: str, body: dict) -> None:
        """The analyzer for these types exists (made once; the same types always give the same id)."""
        if aid in self._ready:
            return
        url = f"{self.endpoint}/contentunderstanding/analyzers/{aid}"
        r = client.get(url, params={"api-version": self.api_version}, headers=self._auth())
        if r.status_code == 404:
            r = client.put(url, params={"api-version": self.api_version}, headers={**self._auth(), "Content-Type": "application/json"},
                           json=body)
            self._raise(r)
            if r.headers.get("operation-location"):
                self._wait(client, r.headers["operation-location"])
        else:
            self._raise(r)
        self._ready.add(aid)

    def _wait(self, client: httpx.Client, op: str) -> dict:
        deadline = time.time() + self.timeout_s
        while True:
            r = client.get(op, headers=self._auth())
            self._raise(r)
            body = r.json()
            status = str(body.get("status", "")).lower()
            if status == "succeeded":
                return body
            if status in ("failed", "canceled"):
                raise ProviderError(f"analysis {status}: {body.get('error')}", self.info.id)
            if time.time() >= deadline:
                raise ProviderTimeout("azure analysis timed out", self.info.id)
            time.sleep(float(r.headers.get("retry-after", 1)))

    def _raise(self, r: httpx.Response) -> None:
        if r.status_code < 400:
            return
        msg = r.text[:300]
        if r.status_code in (401, 403):
            raise AuthFailed(msg, self.info.id)
        if r.status_code == 429:
            raise RateLimited(msg, self.info.id)
        if r.status_code in (400, 415):
            raise InvalidInput(msg, self.info.id)
        raise ProviderError(f"HTTP {r.status_code}: {msg}", self.info.id)
