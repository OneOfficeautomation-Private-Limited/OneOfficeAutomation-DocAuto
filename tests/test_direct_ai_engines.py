"""3.4.0: direct connections to Google Gemini, Amazon Bedrock and Azure OpenAI. Each is called as its vendor expects
(address, key header, request shape: checked on vendor-shaped replies, no network), and reads fields with the shared
prompt and hallucination guard, and recognises types from their descriptions."""
import json

import httpx
import pytest

from conftest import load_option
from docauto.errors import AuthFailed, RateLimited
from docauto.models import BBox, DocumentResult, FieldDef, FieldSchema, Line, Page
from docauto.provider import Capability, ExtractRequest, ProviderContext, ProviderInfo

gemini = load_option("tier3_ai/gemini/provider.py")
azure = load_option("tier3_ai/azure_openai/provider.py")
bedrock = load_option("tier3_ai/bedrock/provider.py")

SCHEMA = FieldSchema(name="Invoice", fields=[FieldDef(name="invoice_number"), FieldDef(name="total", type="currency")])
FIELDS = {"fields": {"invoice_number": {"value": "INV-7", "lines": ["p1.l0"], "confidence": 0.95},
                     "total": {"value": "1150.00", "lines": ["p1.l1"], "confidence": 0.9}}}


def doc():
    p = Page(pageNumber=1, width=1, height=1, lines=[
        Line(text="Invoice Number: INV-7", bbox=BBox(x=0.1, y=0.1, w=0.4, h=0.02), confidence=0.96),
        Line(text="Total: 1,150.00", bbox=BBox(x=0.5, y=0.3, w=0.4, h=0.02), confidence=0.9)])
    d = DocumentResult(documentId="d", pages=[p]); d.rebuild_text()
    return d


def make(mod, cls, eid, config, credentials):
    p = getattr(mod, cls)()
    p.info = ProviderInfo(id=eid, displayName="", version="1", vendor="", tier=3, kind="extraction",
                          capabilities=[Capability.EXTRACT_SCHEMA, Capability.CLASSIFY], languages=["*"], runsLocally=False)
    p.init(ProviderContext(config=config, credentials=credentials))
    return p


def fake_post(monkeypatch, mod, reply_json, status=200):
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers})
        return httpx.Response(status, json=reply_json, request=httpx.Request("POST", url))

    monkeypatch.setattr(mod.httpx, "post", post)
    return calls


# ------------------------------------------------------------------ Gemini
def gemini_reply(obj):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(obj)}]}}]}


def test_gemini_reads_fields_through_ai_studio_with_its_key_header(monkeypatch):
    calls = fake_post(monkeypatch, gemini, gemini_reply(FIELDS))
    p = make(gemini, "GeminiProvider", "gemini", {"model": "gemini-2.5-flash"}, {"apiKey": "g-key"})
    out = {f.name: f for f in p.extract(ExtractRequest(document=doc(), schema=SCHEMA))}
    assert calls[0]["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert calls[0]["headers"]["x-goog-api-key"] == "g-key"
    body = calls[0]["json"]
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert "extract data" in body["systemInstruction"]["parts"][0]["text"]
    assert "[p1.l0] Invoice Number: INV-7" in body["contents"][0]["parts"][0]["text"]
    assert out["invoice_number"].value == "INV-7" and out["total"].value == 1150.0


def test_gemini_on_vertex_and_with_page_images():
    from PIL import Image
    p = make(gemini, "GeminiProvider", "gemini", {"service": "vertex", "model": "gemini-2.5-pro", "sendImages": True}, {"apiKey": "k"})
    assert p.base_url == "https://aiplatform.googleapis.com/v1/publishers/google/models"
    parts = gemini.gemini_parts(p._content("read this", [Image.new("RGB", (20, 20), "white")]))
    assert parts[0] == {"text": "read this"} and parts[1]["inline_data"]["mime_type"] == "image/jpeg" and parts[1]["inline_data"]["data"]


def test_gemini_recognises_a_type_from_the_descriptions(monkeypatch):
    calls = fake_post(monkeypatch, gemini, gemini_reply({"label": "Invoice", "confidence": 0.93}))
    p = make(gemini, "GeminiProvider", "gemini", {}, {"apiKey": "k"})
    c = p.classify(doc(), ["Invoice", "Receipt"], {"Invoice": "A bill from a supplier."})
    assert c.label == "Invoice" and c.confidence == pytest.approx(0.93)
    assert "- Invoice: A bill from a supplier." in calls[0]["json"]["contents"][0]["parts"][0]["text"]


def test_gemini_errors_are_mapped_and_a_missing_key_is_not_ready(monkeypatch):
    fake_post(monkeypatch, gemini, {"error": {"message": "API key not valid"}}, status=403)
    p = make(gemini, "GeminiProvider", "gemini", {}, {"apiKey": "bad"})
    with pytest.raises(AuthFailed):
        p.extract(ExtractRequest(document=doc(), schema=SCHEMA))
    assert not make(gemini, "GeminiProvider", "gemini", {}, {}).health_check().ok


# ------------------------------------------------------------------ Azure OpenAI
def test_azure_openai_calls_the_deployment_with_its_api_key(monkeypatch):
    calls = fake_post(monkeypatch, azure, {"choices": [{"message": {"content": json.dumps(FIELDS)}}]})
    p = make(azure, "AzureOpenAIProvider", "azure-openai", {"endpoint": "https://acme.openai.azure.com/", "deployment": "gpt-vision"},
             {"apiKey": "az-key"})
    out = {f.name: f for f in p.extract(ExtractRequest(document=doc(), schema=SCHEMA))}
    assert calls[0]["url"] == "https://acme.openai.azure.com/openai/deployments/gpt-vision/chat/completions?api-version=2024-10-21"
    assert calls[0]["headers"]["api-key"] == "az-key" and "Authorization" not in calls[0]["headers"]
    assert "model" not in calls[0]["json"] and calls[0]["json"]["response_format"] == {"type": "json_object"}
    assert out["invoice_number"].value == "INV-7"


def test_azure_openai_throttling_and_readiness(monkeypatch):
    fake_post(monkeypatch, azure, {"error": {"code": "429"}}, status=429)
    p = make(azure, "AzureOpenAIProvider", "azure-openai", {"endpoint": "https://a", "deployment": "d"}, {"apiKey": "k"})
    with pytest.raises(RateLimited):
        p.classify(doc(), ["Invoice"], None)
    assert not make(azure, "AzureOpenAIProvider", "azure-openai", {"endpoint": "https://a"}, {"apiKey": "k"}).health_check().ok


# ------------------------------------------------------------------ Amazon Bedrock
class FakeBedrock:
    def __init__(self, reply=None, error=None):
        self.calls, self.reply, self.error = [], reply, error

    def converse(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return {"output": {"message": {"role": "assistant", "content": [{"text": json.dumps(self.reply)}]}}, "stopReason": "end_turn"}


class ClientError(Exception):
    """Shaped as botocore's ClientError: the code is in response['Error']['Code']."""
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code, "Message": code}}


def bedrock_with(monkeypatch, client, config=None):
    p = make(bedrock, "BedrockProvider", "bedrock", config or {"region": "eu-central-1", "model": "anthropic.claude-test"}, {})
    monkeypatch.setattr(p, "_client", lambda: client)
    return p


def test_bedrock_reads_fields_with_converse_and_page_images(monkeypatch):
    from PIL import Image
    client = FakeBedrock(FIELDS)
    p = bedrock_with(monkeypatch, client, {"region": "eu-central-1", "model": "anthropic.claude-test", "sendImages": True})
    out = {f.name: f for f in p.extract(ExtractRequest(document=doc(), schema=SCHEMA, images=[Image.new("RGB", (20, 20), "white")]))}
    kw = client.calls[0]
    assert kw["modelId"] == "anthropic.claude-test" and "extract data" in kw["system"][0]["text"]
    content = kw["messages"][0]["content"]
    assert "[p1.l0]" in content[0]["text"] and content[1]["image"]["format"] == "jpeg" and content[1]["image"]["source"]["bytes"][:2] == b"\xff\xd8"
    assert out["total"].value == 1150.0 and p.info.regions == ["eu-central-1"]


def test_bedrock_recognises_types_and_maps_its_errors(monkeypatch):
    p = bedrock_with(monkeypatch, FakeBedrock({"label": "Receipt", "confidence": 0.8}))
    assert p.classify(doc(), ["Invoice", "Receipt"], {"Receipt": "Proof of a payment."}).label == "Receipt"
    p = bedrock_with(monkeypatch, FakeBedrock(error=ClientError("AccessDeniedException")))
    with pytest.raises(AuthFailed):
        p.extract(ExtractRequest(document=doc(), schema=SCHEMA))
    p = bedrock_with(monkeypatch, FakeBedrock(error=ClientError("ThrottlingException")))
    with pytest.raises(RateLimited):
        p.extract(ExtractRequest(document=doc(), schema=SCHEMA))


def test_the_engines_are_in_the_catalogue_with_their_settings():
    from docauto import settings
    for eid, keys in (("gemini", {"GEMINI_API_KEY", "GEMINI_MODEL"}), ("bedrock", {"BEDROCK_REGION", "BEDROCK_MODEL"}),
                      ("azure-openai", {"AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_KEY"})):
        assert keys <= {f["key"] for f in settings.ENGINES[eid]["fields"]}
    assert {"GEMINI_API_KEY", "BEDROCK_SECRET_ACCESS_KEY", "AZURE_OPENAI_KEY"} <= settings.SECRET_KEYS
