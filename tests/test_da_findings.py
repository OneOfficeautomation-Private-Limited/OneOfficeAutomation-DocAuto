"""3.4.3: what the Document Automation test of 28 Nov 2026 found - plain names for built-in checks, no checks of a type
the document is not, one document answered as itself (not as a batch), a type change that is not a rejection, the
receiver's reason in the delivery log, lookups with a condition, and whether DocFlow can call the Command Center."""
import httpx

from docauto.scoring import _rule_check, check_name
from test_docflow import DF, INV, client, pdf, up  # noqa: F401  (client is the shared fixture)
import docauto.docflow as dfm

LETTER = ["Dear neighbour,", "We are planting trees on Saturday", "in the park. Bring gloves and water."]


def test_the_invoice_check_has_a_plain_name_and_skips_a_missing_total():
    assert check_name(dfm.INVOICE_TOTAL_CHECK).startswith("Amounts add up")
    assert check_name(dfm.INVOICE_TOTAL_CHECK + " (TypeError)").startswith("Amounts add up")
    assert check_name("total > 5") == "total > 5"           # an own check is shown as written
    vals = {"subtotal": 100.0, "vat_amount": 15.0, "discount": None, "other_charges": None}
    assert _rule_check(dfm.INVOICE_TOTAL_CHECK, {**vals, "total": None}) == ("skip", None)
    assert _rule_check(dfm.INVOICE_TOTAL_CHECK, {**vals, "total": 115.0}) == ("pass", None)
    assert _rule_check(dfm.INVOICE_TOTAL_CHECK, {**vals, "total": 999.0})[0] == "fail"


def test_a_document_of_no_type_gets_no_checks_of_that_type(client):  # noqa: F811
    review = client.post(f"{DF}/flows", json={"name": "Invoices, check type", "docTypes": ["invoice"],
                                              "classify": {"mode": "single", "checkType": True, "whenNoType": "review"}}).json()
    d = client.get(f"{DF}/documents/{up(client, review['id'], pdf([LETTER]))['documentId']}").json()
    assert d["status"] == "validation" and d["doctype_id"] is None
    assert [r["reason"] for r in d["reviewReasons"]] == ["unclassified"]
    strict = client.post(f"{DF}/flows", json={"name": "Invoices only", "docTypes": ["invoice"],
                                              "classify": {"mode": "single", "checkType": True, "whenNoType": "reject"}}).json()
    d = client.get(f"{DF}/documents/{up(client, strict['id'], pdf([LETTER]))['documentId']}").json()
    assert d["status"] == "invalid" and d["reviewReasons"] == []
    assert d["flags"][0]["message"] == "Document type not recognised"


def test_one_document_is_answered_as_itself_and_a_type_change_is_not_a_rejection(client):  # noqa: F811
    fl = client.post(f"{DF}/flows", json={"name": "Post", "docTypes": ["invoice", "purchase_order"],
                                          "classify": {"mode": "auto", "split": True}}).json()
    d = up(client, fl["id"], pdf([INV]))
    assert d.get("batchId") and d["documentId"] != d["batchId"]          # the part, not the batch
    part = client.get(f"{DF}/documents/{d['documentId']}").json()
    assert part["doctype_id"] == "invoice" and part["status"] in ("success", "validation")
    assert client.get(f"{DF}/documents/{d['batchId']}").json()["onlyPart"] == d["documentId"]
    new = client.post(f"{DF}/documents/{d['documentId']}/reclassify", json={"docType": "purchase_order"}).json()
    old = client.get(f"{DF}/documents/{d['documentId']}").json()
    assert old["status"] == "replaced" and old["flags"][0]["replacedBy"] == new["id"]
    stats = client.get(f"{DF}/flows/{fl['id']}/stats").json()
    assert stats["byStatus"]["invalid"] == 0 and stats["total"] == 1
    listed = [r["id"] for r in client.get(f"{DF}/flows/{fl['id']}/documents").json()]
    assert d["documentId"] not in listed and new["id"] in listed
    assert client.get(f"{DF}/documents/{d['batchId']}").json()["onlyPart"] == new["id"]


def test_the_delivery_log_keeps_the_receivers_reason(monkeypatch):
    from conftest import make_config
    from docauto.docflow import Delivery, DocFlowService
    from docauto.pipeline import DocumentService
    monkeypatch.setenv("DOCAUTO_FOLDER_WATCHER", "false")
    df = DocFlowService(DocumentService(make_config()))
    reply = {"error": {"code": "CR.NO_DEFAULT_RUNNER", "message": "admin has no default runner",
                       "suggestedFix": "Set one in Users (Default runner), or choose a runner."}}
    monkeypatch.setattr(dfm.httpx, "post", lambda url, json=None, headers=None, timeout=None:
                        httpx.Response(400, json=reply, request=httpx.Request("POST", url)))
    dv = Delivery(type="webhook", name="ERP", params={"url": "https://erp.example.com/in"})
    ok, detail, retry = df._send_delivery(dv, "doc1", "success", {"documentId": "doc1"})
    assert not ok and not retry
    assert detail == "HTTP 400: admin has no default runner Set one in Users (Default runner), or choose a runner."


def test_a_lookup_can_require_a_value_in_another_column(client):  # noqa: F811
    csv = "Supplier name,VAT number,Status\r\nACME,300123456700003,blocked\r\n".encode()
    t = client.post(f"{DF}/reftables", files={"file": ("s.csv", csv, "text/csv")}, data={"name": "Suppliers"}).json()
    rule = {"type": "lookup", "name": "Approved", "params": {"field": "supplier_vat", "table": t["id"], "column": "VAT number",
                                                             "whereColumn": "Status", "whereValue": "active"}}
    fl = client.post(f"{DF}/flows", json={"name": "AP", "docTypes": ["invoice"], "rules": [rule]}).json()
    d = client.get(f"{DF}/documents/{up(client, fl['id'], pdf([INV]))['documentId']}").json()
    assert d["status"] == "validation"
    assert "its Status is 'blocked', not 'active'" in d["flags"][0]["message"]
    bad = {**rule, "params": {**rule["params"], "whereColumn": "State"}}
    r = client.post(f"{DF}/flows", json={"name": "AP 2", "docTypes": ["invoice"], "rules": [bad]})
    assert r.status_code == 400 and "no column 'State'" in r.text


def test_the_version_says_whether_docflow_can_call_back(client, monkeypatch):  # noqa: F811
    monkeypatch.delenv("DOCAUTO_PLATFORM_URL", raising=False)
    assert client.get("/api/docs/v1/admin/version").json()["platformCallback"] is False
    monkeypatch.setenv("DOCAUTO_PLATFORM_URL", "http://127.0.0.1:8720")
    assert client.get("/api/docs/v1/admin/version").json()["platformCallback"] is True
