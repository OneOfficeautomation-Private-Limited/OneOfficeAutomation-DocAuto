"""3.4.0: cloud services that recognise and split a whole file (Azure Content Understanding, Google Document AI
splitter or classifier, Amazon Bedrock Data Automation). Each is called as its vendor expects, on vendor-shaped
replies (no network), and a process splits a batch with one of them as its recognise engine."""
import json
import shutil

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import load_option, make_config
from docauto.docflow import parts_from_segments
from docauto.errors import AuthFailed, RateLimited
from docauto.provider import Capability, ProviderContext, ProviderInfo, Segment, SplitProvider
from test_docflow import DF, INV, PO, pdf, up

azure = load_option("tier2_cloud/azure_cu/provider.py")
google = load_option("tier2_cloud/google_split/provider.py")
aws = load_option("tier2_cloud/aws_bda/provider.py")

NAMES = {"Invoice": "invoice", "Purchase order": "purchase_order"}


def make(mod, cls, eid, config, credentials):
    p = getattr(mod, cls)()
    p.info = ProviderInfo(id=eid, displayName="", version="1", vendor="", tier=2, kind="classify",
                          capabilities=[Capability.CLASSIFY, Capability.SPLIT], languages=["*"], runsLocally=False)
    p.init(ProviderContext(config=config, credentials=credentials))
    return p


# ------------------------------------------------------------------ parts of a file
def test_parts_match_types_by_name_and_keep_every_page():
    segs = [Segment("invoice", [1, 2], 0.95), Segment("Purchase Order", [3], 0.9), Segment("Invoice", [5], 0.3)]
    parts = parts_from_segments(segs, 6, NAMES, 0.5, True)
    assert [(p["doctype"], p["pages"]) for p in parts] == [
        ("invoice", [1, 2]), ("purchase_order", [3]), (None, [4]), (None, [5]), (None, [6])]   # 4, 6 left out; 5 too unsure


def test_parts_without_a_confidence_count_as_sure_and_one_document_keeps_the_best_type():
    parts = parts_from_segments([Segment(None, [1]), Segment("Invoice", [2, 3])], 3, NAMES, 0.5, False)
    assert parts == [{"doctype": "invoice", "pages": [1, 2, 3], "score": 1.0}]
    assert parts_from_segments([Segment("Invoice", [1, 2]), Segment("Invoice", [2, 3])], 3, NAMES, 0.5, True)[1]["pages"] == [3]


# ------------------------------------------------------------------ Azure Content Understanding
class FakeAzure:
    """httpx.Client as Content Understanding answers: no analyzer yet, made on PUT, the file analyzed and polled."""
    calls: list = []
    analyzers: set = set()

    def __init__(self, timeout=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def _r(self, status, body=None, headers=None, url="https://x"):
        return httpx.Response(status, json=body or {}, headers=headers or {}, request=httpx.Request("GET", url))

    def get(self, url, params=None, headers=None):
        FakeAzure.calls.append(("GET", url, params, headers, None))
        if "/analyzerResults/" in url:
            return self._r(200, {"status": "Succeeded", "result": {"contents": [{
                "category": "Invoice", "startPageNumber": 1, "endPageNumber": 3, "segments": [
                    {"category": "Invoice", "startPageNumber": 1, "endPageNumber": 2},
                    {"category": "Purchase_order", "startPageNumber": 3, "endPageNumber": 3},
                    {"category": "other", "startPageNumber": 4, "endPageNumber": 4}]}]}})
        if "/operations/" in url:
            return self._r(200, {"status": "Succeeded"})
        aid = url.rsplit("/", 1)[-1]
        return self._r(200 if aid in FakeAzure.analyzers else 404, {})

    def put(self, url, params=None, headers=None, json=None):
        FakeAzure.calls.append(("PUT", url, params, headers, json))
        FakeAzure.analyzers.add(url.rsplit("/", 1)[-1])
        return self._r(201, json, {"operation-location": url + "/operations/1"})

    def post(self, url, params=None, headers=None, content=None):
        FakeAzure.calls.append(("POST", url, params, headers, content))
        return self._r(202, {"status": "NotStarted"}, {"operation-location": "https://acme/contentunderstanding/analyzerResults/9"})


def test_azure_makes_an_analyzer_from_the_descriptions_once_and_splits_the_file(monkeypatch):
    FakeAzure.calls, FakeAzure.analyzers = [], set()
    monkeypatch.setattr(azure.httpx, "Client", FakeAzure)
    p = make(azure, "AzureContentUnderstandingProvider", "azure-content-understanding",
             {"endpoint": "https://acme.services.ai.azure.com/", "completionModel": "gpt-4.1"}, {"apiKey": "cu-key"})
    labels, desc = ["Invoice", "Purchase order"], {"Invoice": "A bill from a supplier.", "Purchase order": ""}
    segs = p.split(b"%PDF", "application/pdf", 4, labels, desc)
    assert [(s.label, s.pages) for s in segs] == [("Invoice", [1, 2]), ("Purchase order", [3]), (None, [4])]
    put = next(c for c in FakeAzure.calls if c[0] == "PUT")
    assert put[1].startswith("https://acme.services.ai.azure.com/contentunderstanding/analyzers/ooa_split_")
    assert put[2] == {"api-version": "2025-11-01"} and put[3]["Ocp-Apim-Subscription-Key"] == "cu-key"
    body = put[4]
    assert body["baseAnalyzerId"] == "prebuilt-document" and body["models"] == {"completion": "gpt-4.1"}
    cats = body["config"]["contentCategories"]
    assert body["config"]["enableSegment"] is True and cats["Invoice"]["description"] == "A bill from a supplier."
    assert cats["Purchase_order"]["description"] == "Purchase order" and "other" in cats   # no description: the name
    post = next(c for c in FakeAzure.calls if c[0] == "POST")
    assert post[1].endswith(":analyzeBinary") and post[2]["processingLocation"] == "geography"
    assert post[3]["Content-Type"] == "application/pdf" and post[4] == b"%PDF"
    FakeAzure.calls = []
    p.split(b"%PDF", "application/pdf", 4, labels, desc)
    assert not [c for c in FakeAzure.calls if c[0] == "PUT" or (c[0] == "GET" and "/analyzers/" in c[1])]   # made once
    assert azure.category_keys(["A b", "A-b", "1st"]) == {"A b": "A_b", "A-b": "A_b_2", "1st": "t_1st"}


def test_azure_one_document_and_a_refused_key(monkeypatch):
    FakeAzure.calls, FakeAzure.analyzers = [], set()
    monkeypatch.setattr(azure.httpx, "Client", FakeAzure)
    p = make(azure, "AzureContentUnderstandingProvider", "azure-content-understanding", {"endpoint": "https://a"}, {"apiKey": "k"})
    p.split(b"x", "image/png", 1, ["Invoice"], None, segment=False)
    body = next(c for c in FakeAzure.calls if c[0] == "PUT")[4]
    assert body["config"]["enableSegment"] is False and "models" not in body
    with pytest.raises(AuthFailed):
        p._raise(httpx.Response(401, text="no", request=httpx.Request("GET", "https://a")))
    assert not make(azure, "AzureContentUnderstandingProvider", "azure-content-understanding", {}, {}).health_check().ok


# ------------------------------------------------------------------ Google Document AI
def test_google_splitter_parts_and_classifier_answers():
    splitter = {"entities": [{"type": "invoice", "confidence": 0.97, "pageAnchor": {"pageRefs": [{}, {"page": "1"}]}},
                             {"type": "purchase_order", "confidence": 0.9, "pageAnchor": {"pageRefs": [{"page": "2"}]}}]}
    assert [(s.label, s.pages, s.confidence) for s in google.segments_from_document(splitter, 3)] == [
        ("invoice", [1, 2], 0.97), ("purchase_order", [3], 0.9)]
    classifier = {"entities": [{"type": "invoice", "confidence": 0.2}, {"type": "purchase_order", "confidence": 0.8}]}
    assert [(s.label, s.pages) for s in google.segments_from_document(classifier, 2)] == [("purchase_order", [1, 2])]
    assert google.segments_from_document({}, 2) == []


def test_google_sends_the_file_to_its_processor(monkeypatch):
    p = make(google, "GoogleSplitProvider", "google-docai-split", {"projectId": "p", "location": "eu", "processorId": "abc"}, {})
    seen = []
    monkeypatch.setattr(p, "_process", lambda data, mime: seen.append((data, mime)) or {"entities": [
        {"type": "invoice", "confidence": 0.9, "pageAnchor": {"pageRefs": [{}]}}]})
    assert p.split(b"img", "", 1, ["Invoice"])[0].label == "invoice" and seen == [(b"img", "application/pdf")]
    assert p.info.regions == ["eu"]


# ------------------------------------------------------------------ Amazon Bedrock Data Automation
class FakeBda:
    def __init__(self, reply=None, error=None):
        self.calls, self.reply, self.error = [], reply, error

    def invoke_data_automation(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return self.reply


class ClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code, "Message": code}}


def bda(monkeypatch, client):
    p = make(aws, "AwsBdaProvider", "aws-bda", {"region": "eu-central-1", "projectArn": "arn:proj",
                                                 "profileArn": "arn:profile"}, {})
    monkeypatch.setattr(p, "_client", lambda: client)
    return p


def test_bda_sends_the_file_inline_and_reads_each_segment(monkeypatch):
    reply = {"outputSegments": [
        {"customOutputStatus": "MATCH", "customOutput": json.dumps({
            "matched_blueprint": {"arn": "arn:bp1", "name": "Invoice", "confidence": 0.96}, "split_document": {"page_indices": [0, 1]}})},
        {"customOutputStatus": "NO_MATCH", "standardOutput": json.dumps({"metadata": {"start_page_index": 2, "end_page_index": 2}})}]}
    client = FakeBda(reply)
    segs = bda(monkeypatch, client).split(b"%PDF", "application/pdf", 3, ["Invoice"])
    assert [(s.label, s.pages, s.confidence) for s in segs] == [("Invoice", [1, 2], 0.96), (None, [3], None)]
    kw = client.calls[0]
    assert kw["inputConfiguration"] == {"bytes": b"%PDF"} and kw["dataAutomationProfileArn"] == "arn:profile"
    assert kw["dataAutomationConfiguration"] == {"dataAutomationProjectArn": "arn:proj", "stage": "LIVE"}
    assert aws.default_profile("eu-central-1", "123") == "arn:aws:bedrock:eu-central-1:123:data-automation-profile/eu.data-automation-v1"


def test_bda_errors_are_mapped(monkeypatch):
    with pytest.raises(AuthFailed):
        bda(monkeypatch, FakeBda(error=ClientError("AccessDeniedException"))).split(b"x", "", 1, [])
    with pytest.raises(RateLimited):
        bda(monkeypatch, FakeBda(error=ClientError("ThrottlingException"))).split(b"x", "", 1, [])


def test_the_services_are_in_the_catalogue_with_their_settings():
    from docauto import settings
    for eid, keys in (("azure-content-understanding", {"AZURE_CU_ENDPOINT", "AZURE_CU_KEY"}),
                      ("google-docai-split", {"DOCAI_SPLIT_PROCESSOR"}), ("aws-bda", {"BDA_REGION", "BDA_PROJECT_ARN"})):
        assert keys <= {f["key"] for f in settings.ENGINES[eid]["fields"]}
    assert {"AZURE_CU_KEY", "BDA_SECRET_ACCESS_KEY"} <= settings.SECRET_KEYS


# ------------------------------------------------------------------ a process splits with a service
class FakeSplitter(SplitProvider):
    """Says page 1 is an invoice and page 2 a purchase order, by the names it is given."""
    calls: list = []

    def split(self, data, mime, page_count, labels, descriptions=None, segment=True):
        FakeSplitter.calls.append((len(data), mime, page_count, list(labels), segment))
        return [Segment("invoice", [1], 0.93), Segment("PURCHASE ORDER", [2], 0.88)]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DOCAUTO_FOLDER_WATCHER", "false")
    import docauto.api.app as appmod
    from docauto.pipeline import DocumentService
    svc = DocumentService(make_config())
    svc.registry.register(FakeSplitter(), ProviderInfo(
        id="fake-split", displayName="Fake splitter", version="1.0.0", vendor="test", tier=2, kind="classify",
        capabilities=[Capability.CLASSIFY, Capability.SPLIT], languages=["*"], runsLocally=True,
        creditCostPerPage={"classify": 5}, optionPack="tier2_cloud"))
    appmod.set_service(svc)
    appmod._df = None
    FakeSplitter.calls = []
    return TestClient(appmod.app)


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_a_process_splits_a_batch_with_the_service(client, monkeypatch):
    monkeypatch.setenv("DOCAUTO_DEFAULT_CLASSIFY", "fake-split")
    d = up(client, "smart-inbox", pdf([PO, INV]), "scan_batch.pdf")      # the words would say PO first: the service decides
    parent = client.get(f"{DF}/documents/{d['documentId']}").json()
    assert parent["status"] == "split"
    assert [ch["doctype_id"] for ch in parent["children"]] == ["invoice", "purchase_order"]
    size, mime, count, labels, segment = FakeSplitter.calls[0]
    assert size > 0 and mime == "application/pdf" and count == 2 and "Invoice" in labels and segment is True
    steps = client.get(f"/api/docs/v1/documents/{d['documentId']}/result").json()["provenance"]["steps"]
    assert any(s["stage"] == "classify" and s["providerId"] == "fake-split" and s["pages"] == 2 for s in steps)
