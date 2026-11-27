"""Keys from the Command Center's Model connections (3.4.0, plan DA-2).

An administrator chooses, in the Command Center, the Model connection each AI engine takes its key from. When DocFlow
loads its engines it asks once (POST {DOCAUTO_PLATFORM_URL}/api/docflow-callback/credentials, this service's token,
the tenant) and gets, for each engine that has one, the connection's provider, model, address and key. They are put
into that engine's settings in memory only; nothing is written to disk. An engine without a connection keeps the key
typed on the Engines page. When the Command Center cannot be reached the typed keys are used (and it is logged).
"""
from __future__ import annotations

import copy
import logging
from typing import Any

import httpx

from . import events

log = logging.getLogger("docauto.connections")

#: the engines that can take a Model connection (the Command Center checks the provider fits)
ENGINES = ("claude", "gemini", "bedrock", "azure-openai", "llm-cloud", "llm-local")


def _post(url: str, payload: dict, headers: dict, timeout: float) -> httpx.Response:
    """One request on its own client (not the shared httpx.post other callbacks use)."""
    with httpx.Client(timeout=timeout) as client:
        return client.post(url, json=payload, headers=headers)


def fetch(tenant: str, engines: list[str], timeout: float = 5) -> dict[str, dict]:
    """The connections the Command Center has for these engines ({} when none, or it is not set up or not reached)."""
    asked = [e for e in engines if e in ENGINES]
    base = events.platform_base(tenant)
    if not asked or not base:
        return {}
    try:
        r = _post(base + "/api/docflow-callback/credentials", {"engines": asked},
                  {"X-OOA-Tenant": tenant, **events.platform_auth()}, timeout)
    except httpx.HTTPError as e:
        log.warning("Model connections not read (the Command Center was not reached: %s); typed keys are used", e)
        return {}
    if r.status_code >= 300:
        log.warning("Model connections not read (the Command Center answered %s); typed keys are used", r.status_code)
        return {}
    try:
        body = r.json() if r.content else {}
    except ValueError:
        log.warning("Model connections not read (the Command Center's answer is not JSON); typed keys are used")
        return {}
    for engine, why in (body.get("refused") or {}).items():
        log.warning("%s: its Model connection was refused (%s); its typed key is used", engine, why)
    return {k: v for k, v in (body.get("engines") or {}).items() if k in asked and isinstance(v, dict)}


def apply(engine: str, settings: dict, conn: dict) -> dict:
    """The engine's settings ({config, credentials}) with the connection's key, model and address in place."""
    out = copy.deepcopy(settings or {})
    cfg, creds = out.setdefault("config", {}), out.setdefault("credentials", {})
    model, base = (conn.get("model") or "").strip(), (conn.get("baseUrl") or "").strip()
    if engine == "bedrock":
        creds["accessKeyId"], creds["secretAccessKey"] = conn.get("accessKeyId", ""), conn.get("secretAccessKey", "")
        creds.pop("sessionToken", None)
        if conn.get("region"):
            cfg["region"] = conn["region"]
    else:
        creds["apiKey"] = conn.get("apiKey", "")
    if engine == "azure-openai":
        if base:
            cfg["endpoint"] = base
        if model:
            cfg["deployment"] = model
        return out
    if model:
        cfg["model"] = model
    if engine in ("llm-cloud", "llm-local") and base:
        cfg["baseUrl"] = base
    if engine == "gemini":
        cfg["service"] = "ai-studio" if not base or "generativelanguage" in base else "vertex"
    return out


def describe(conn: dict[str, Any]) -> str:
    return f"Model connection {conn.get('connection', '')}".strip()
