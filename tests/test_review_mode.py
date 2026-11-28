"""3.4.1: a process says how much goes to a person: values the engine is unsure of (as before), every document (for a
new process's first weeks), or nothing for confidence alone (failed checks and missing or invalid values still go)."""
import shutil

import pytest

from docauto.docflow import EVERY_DOCUMENT, ReviewSettings, review_mode
from docauto.models import ReviewReason
from test_docflow import DF, INV, client, pdf, up  # noqa: F401  (client is the shared fixture)

LOW = ReviewReason(field="total", reason="low_confidence", threshold=0.85, confidence=0.6)
DOC = ReviewReason(reason="document_confidence", threshold=0.8, confidence=0.7)
MISSING = ReviewReason(field="total", reason="required_missing")
CHECK = ReviewReason(reason="check_failed", message="Check failed: total > 50000")


def test_the_modes_keep_or_drop_the_confidence_reasons():
    assert ReviewSettings().mode == "unsure"
    assert review_mode("unsure", [LOW, DOC, CHECK]) == [LOW, DOC, CHECK]
    assert review_mode("none", [LOW, DOC, MISSING, CHECK]) == [MISSING, CHECK]
    assert review_mode("all", []) == [EVERY_DOCUMENT]
    assert review_mode("all", [LOW]) == [LOW]


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_every_document_goes_to_a_person_and_none_lets_it_through(client):  # noqa: F811
    every = client.post(f"{DF}/flows", json={"name": "First weeks", "docTypes": ["invoice"],
                                             "review": {"mode": "all", "fieldConfidence": 0, "documentConfidence": 0}}).json()
    assert every["review"]["mode"] == "all"
    d = up(client, every["id"], pdf([INV]))
    doc = client.get(f"{DF}/documents/{d['documentId']}").json()
    assert doc["status"] == "validation"
    none = client.post(f"{DF}/flows", json={"name": "Trusted", "docTypes": ["invoice"],
                                            "review": {"mode": "none", "fieldConfidence": 0.999, "documentConfidence": 0.999}}).json()
    d = up(client, none["id"], pdf([INV]))
    assert client.get(f"{DF}/documents/{d['documentId']}").json()["status"] == "success"   # unsure values alone do not stop it
    assert client.post(f"{DF}/flows", json={"name": "Bad", "docTypes": ["invoice"], "review": {"mode": "maybe"}}).status_code == 422


def test_a_webhook_header_can_come_from_the_vault(monkeypatch):
    """3.4.1: a header value "vault:folder/credential/attribute" is read from the Command Center at delivery time, as
    the person who set the delivery up, and never kept in the process; one that cannot be read sends nothing."""
    import httpx
    import docauto.docflow as dfm
    from conftest import make_config
    from docauto.docflow import Delivery, DocFlowService
    from docauto.pipeline import DocumentService
    monkeypatch.setenv("DOCAUTO_FOLDER_WATCHER", "false")
    svc = DocumentService(make_config())
    df = DocFlowService(svc)
    asked, sent = [], []
    monkeypatch.setattr(df, "_platform", lambda path, payload, timeout=30: asked.append((path, payload)) or {"value": "s3cret"})
    monkeypatch.setattr(dfm.httpx, "post", lambda url, json=None, headers=None, timeout=None:
                        sent.append(headers) or httpx.Response(200, request=httpx.Request("POST", url)))
    dv = Delivery(type="webhook", name="ERP", params={"url": "https://erp.example.com/in", "requestedBy": "ann",
                                                       "headers": {"X-Api-Key": "vault:keys/erp/api", "X-Tenant": "acme"}})
    ok, detail, _ = df._send_delivery(dv, "doc1", "success", {"documentId": "doc1"})
    assert ok and asked == [("secret", {"requestedBy": "ann", "ref": "keys/erp/api"})]
    assert sent[0]["X-Api-Key"] == "s3cret" and sent[0]["X-Tenant"] == "acme"

    def refused(path, payload, timeout=30):
        raise RuntimeError("the Command Center refused (403)")
    monkeypatch.setattr(df, "_platform", refused)
    sent.clear()
    ok, detail, retry = df._send_delivery(dv, "doc1", "success", {})
    assert not ok and not retry and "X-Api-Key" in detail and not sent
    svc.close()


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_a_dry_run_reads_as_the_process_and_keeps_nothing(client):  # noqa: F811
    """3.4.1: Try in this process reads a document with the process's checks and review settings, says what would be
    delivered, and forgets it: no document, no review, no delivery."""
    import docauto.api.app as appmod
    fl = client.post(f"{DF}/flows", json={"name": "Try me", "docTypes": ["invoice"], "status": "live",
                                          "review": {"mode": "all"},
                                          "deliveries": [{"type": "webhook", "name": "ERP", "on": ["success", "validation"],
                                                          "params": {"url": "http://127.0.0.1:9/never"}}]}).json()
    before = len(appmod.docflow().store.instance_documents(fl["id"], ["success", "validated", "validation", "invalid", "failed"]))
    r = client.post(f"{DF}/flows/{fl['id']}/try", files={"file": ("inv.pdf", pdf([INV]), "application/pdf")})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["status"] == "validation" and out["doctypeName"] == "Invoice" and out["fields"]
    assert out["reviewReasons"] and out["deliveries"] == ["ERP"]       # why it would go to a person, and what would be sent
    after = appmod.docflow().store.instance_documents(fl["id"], ["success", "validated", "validation", "invalid", "failed"])
    assert len(after) == before                                              # nothing kept
    assert not appmod.docflow().store.delivery_list(fl["id"])               # nothing sent or queued
    assert not [d for d in client.get(f"{DF}/review").json() if d.get("flowId") == fl["id"]]


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_a_draft_holds_its_deliveries(client):  # noqa: F811
    import docauto.api.app as appmod
    from docauto.docflow import HELD
    fl = client.post(f"{DF}/flows", json={"name": "Still a draft", "docTypes": ["invoice"],
                                          "deliveries": [{"type": "webhook", "name": "ERP", "on": ["success", "validation"],
                                                          "params": {"url": "http://127.0.0.1:9/never"}}]}).json()
    assert fl["status"] == "draft"
    up(client, fl["id"], pdf([INV]))
    log = appmod.docflow().store.delivery_list(fl["id"])
    assert log and log[0]["detail"] == HELD and not log[0]["ok"]
    assert not [o for o in appmod.docflow().store.outbox_pending() if o.get("flow_id") == fl["id"]] \
        if hasattr(appmod.docflow().store, "outbox_pending") else True


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_the_documents_list_says_why_a_document_failed(client, monkeypatch):  # noqa: F811
    """3.4.2: GET df/flows/{id}/documents?fields=true gives a failed document's error, for the Inbox's Needs you list."""
    import docauto.api.app as appmod
    fl = client.post(f"{DF}/flows", json={"name": "Fails", "docTypes": ["invoice"]}).json()
    d = up(client, fl["id"], pdf([INV]))
    store = appmod.service().store
    store.set_doc_status(d["documentId"], "failed")
    monkeypatch.setattr(store, "last_job_error", lambda doc_id: {"code": "OCR.FAILED", "message": "The engine could not read the page"})
    rows = client.get(f"{DF}/flows/{fl['id']}/documents?fields=true").json()
    assert rows[0]["status"] == "failed" and rows[0]["error"]["message"] == "The engine could not read the page"
    plain = client.get(f"{DF}/flows/{fl['id']}/documents").json()
    assert "error" not in plain[0]
