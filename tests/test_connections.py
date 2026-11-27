"""3.4.0: AI engines take their keys from the Command Center's Model connections. DocFlow asks once when it loads its
engines; the key is put into the engine's settings in memory only, and a typed key is used when there is none."""
import httpx
from fastapi.testclient import TestClient

from conftest import make_config
from docauto import connections
from docauto.registry import ProviderRegistry


def test_each_engine_gets_the_connection_in_its_own_settings():
    typed = {"config": {"model": "typed-model", "timeoutSeconds": 60}, "credentials": {"apiKey": "typed"}}
    claude = connections.apply("claude", typed, {"apiKey": "vault", "model": "claude-haiku-5-5"})
    assert claude == {"config": {"model": "claude-haiku-5-5", "timeoutSeconds": 60}, "credentials": {"apiKey": "vault"}}
    assert typed["credentials"]["apiKey"] == "typed"                        # the typed settings are not changed
    az = connections.apply("azure-openai", {}, {"apiKey": "k", "model": "gpt-vision", "baseUrl": "https://acme.openai.azure.com"})
    assert az["config"] == {"endpoint": "https://acme.openai.azure.com", "deployment": "gpt-vision"}
    br = connections.apply("bedrock", {"config": {"region": "us-east-1"}}, {"accessKeyId": "AK", "secretAccessKey": "S", "region": "eu-central-1", "model": "m"})
    assert br["credentials"] == {"accessKeyId": "AK", "secretAccessKey": "S"} and br["config"] == {"region": "eu-central-1", "model": "m"}
    assert connections.apply("gemini", {}, {"apiKey": "g", "baseUrl": ""})["config"]["service"] == "ai-studio"
    assert connections.apply("gemini", {}, {"apiKey": "g", "baseUrl": "https://aiplatform.googleapis.com/v1"})["config"]["service"] == "vertex"
    cloud = connections.apply("llm-cloud", {}, {"apiKey": "o", "model": "gpt-x", "baseUrl": "https://api.example.com/v1"})
    assert cloud["config"] == {"model": "gpt-x", "baseUrl": "https://api.example.com/v1"}


def _cfg():
    cfg = make_config()
    cfg.setdefault("option_packs", {})["tier3_ai"] = True
    return cfg


def test_the_registry_asks_once_and_the_engine_is_ready_with_the_vault_key(monkeypatch):
    calls = []

    def post(url, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers})
        return httpx.Response(200, json={"engines": {"llm-cloud": {"connection": "openai-prod", "provider": "openai", "apiKey": "sk-vault",
                                                                   "model": "gpt-x", "baseUrl": "https://api.example.com/v1"}},
                                         "refused": {"gemini": "not a fitting connection"}}, request=httpx.Request("POST", url))

    monkeypatch.setenv("DOCAUTO_PLATFORM_URL", "http://cc.local")
    monkeypatch.setenv("DOCAUTO_API_TOKEN", "svc-token")
    monkeypatch.setattr(connections, "_post", post)
    r = ProviderRegistry(_cfg())
    r.discover()
    assert len(calls) == 1 and calls[0]["url"] == "http://cc.local/api/docflow-callback/credentials"
    assert calls[0]["headers"]["Authorization"] == "Bearer svc-token" and calls[0]["headers"]["X-OOA-Tenant"] == "default"
    assert {"claude", "gemini", "llm-cloud", "llm-local"} <= set(calls[0]["json"]["engines"])
    p = r.providers["llm-cloud"]
    assert p.available and p.instance.api_key == "sk-vault" and p.instance.base_url == "https://api.example.com/v1"
    assert "key from Model connection openai-prod" in p.reason
    assert p.instance.ctx.tenant_id == "default"


def test_typed_keys_are_used_when_the_command_center_is_not_set_up_or_not_reached(monkeypatch):
    monkeypatch.delenv("DOCAUTO_PLATFORM_URL", raising=False)
    assert connections.fetch("default", ["claude"]) == {}
    monkeypatch.setenv("DOCAUTO_PLATFORM_URL", "http://cc.local")

    def down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(connections, "_post", down)
    r = ProviderRegistry(_cfg())
    r.discover()
    assert r.connections == {} and not r.providers["llm-cloud"].available    # nothing typed in this test either
    assert connections.fetch("default", ["tesseract"]) == {}                  # not an AI engine: not asked


def test_the_command_center_can_make_docflow_reload_its_engines(monkeypatch):
    monkeypatch.setenv("DOCAUTO_FOLDER_WATCHER", "false")
    import docauto.api.app as appmod
    from docauto.pipeline import DocumentService
    appmod.set_service(DocumentService(make_config()))
    appmod._df = None
    r = TestClient(appmod.app).post("/api/docs/v1/admin/reload")
    assert r.status_code == 200 and any(p["id"] == "tesseract" for p in r.json()["providers"])
