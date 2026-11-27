"""3.4.0: processes recognise each page's document type with an engine of their choice (DOCAUTO_DEFAULT_CLASSIFY), from
every type's name and plain-words description; the words on the page stay the offline way and the fallback."""
import shutil

import pytest
from fastapi.testclient import TestClient

from conftest import make_config
from docauto import recognise
from docauto.docflow import DocType, split_pages
from docauto.models import Classification
from docauto.provider import Capability, ExtractionProvider, ProviderInfo
from test_docflow import DF, INV, PO, pdf, up


def test_the_question_names_every_type_with_its_description():
    q = recognise.question(["Invoice", "Purchase order"], {"Invoice": "A bill from a supplier.", "Purchase order": ""}, "Tax Invoice 42")
    assert "- Invoice: A bill from a supplier." in q
    assert "- Purchase order\n" in q + "\n"            # no description: the name alone
    assert '"other"' in q and q.endswith("Tax Invoice 42")
    assert len(recognise.question(["A"], None, "x" * 20000)) < 8300   # the page text is cut to MAX_CHARS


def test_split_pages_takes_each_page_type_as_the_engine_gave_it():
    types = [DocType(id="invoice", name="Invoice"), DocType(id="purchase_order", name="Purchase order")]
    segs = split_pages(["a", "b", "c"], types, 2.0, True, labels=[("invoice", 0.9), ("purchase_order", 0.8), ("invoice", 0.7)])
    assert [(s["doctype"], s["pages"]) for s in segs] == [("invoice", [1]), ("purchase_order", [2]), ("invoice", [3])]
    joined = split_pages(["a", "b"], types, 2.0, False, labels=[("invoice", 0.9), (None, 0.0)])
    assert [(s["doctype"], s["pages"]) for s in joined] == [("invoice", [1, 2])]


class FakeRecogniser(ExtractionProvider):
    """Recognises by the descriptions it is given: answers the type whose description's first word is on the page."""
    calls: list = []

    def extract(self, req):
        return []

    def classify(self, doc, labels, descriptions=None):
        FakeRecogniser.calls.append((list(labels), dict(descriptions or {})))
        low = doc.text.lower()
        for label in labels:
            first = ((descriptions or {}).get(label) or "").split(" ")[0].lower()
            if first and first in low:
                return Classification(label=label, confidence=0.9)
        return Classification(label="other", confidence=0.6)


def _info():
    return ProviderInfo(id="fake-recognise", displayName="Fake recogniser", version="1.0.0", vendor="test", tier=3,
                        kind="extraction", capabilities=[Capability.CLASSIFY], languages=["*"], runsLocally=True,
                        creditCostPerPage={"classify": 2}, optionPack="tier3_ai")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DOCAUTO_FOLDER_WATCHER", "false")
    import docauto.api.app as appmod
    from docauto.pipeline import DocumentService
    svc = DocumentService(make_config())
    svc.registry.register(FakeRecogniser(), _info())
    appmod.set_service(svc)
    appmod._df = None
    FakeRecogniser.calls = []
    return TestClient(appmod.app)


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_a_process_recognises_pages_with_the_chosen_engine_from_the_descriptions(client, monkeypatch):
    # the descriptions decide: "Tax" for invoices, "Purchase" for purchase orders
    for dt, words in (("invoice", "Tax invoices from suppliers."), ("purchase_order", "Purchase orders we send.")):
        t = client.get(f"{DF}/doctypes/{dt}").json()
        assert client.put(f"{DF}/doctypes/{dt}", json={**t, "description": words}).status_code == 200
    monkeypatch.setenv("DOCAUTO_DEFAULT_CLASSIFY", "fake-recognise")
    d = up(client, "smart-inbox", pdf([INV, PO]), "scan_batch.pdf")
    parent = client.get(f"{DF}/documents/{d['documentId']}").json()
    assert parent["status"] == "split"
    assert [ch["doctype_id"] for ch in parent["children"]] == ["invoice", "purchase_order"]
    labels, descriptions = FakeRecogniser.calls[0]
    assert "Invoice" in labels and descriptions["Invoice"] == "Tax invoices from suppliers."
    assert descriptions["Purchase order"] == "Purchase orders we send."
    steps = client.get(f"/api/docs/v1/documents/{d['documentId']}/result").json()["provenance"]["steps"]
    assert any(s["stage"] == "classify" and s["providerId"] == "fake-recognise" and s["pages"] == 2 for s in steps)


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_an_engine_that_cannot_run_falls_back_to_the_words_on_the_page(client, monkeypatch):
    monkeypatch.setenv("DOCAUTO_DEFAULT_CLASSIFY", "no-such-engine")
    d = up(client, "smart-inbox", pdf([INV, PO]), "scan_batch.pdf")
    parent = client.get(f"{DF}/documents/{d['documentId']}").json()
    assert [ch["doctype_id"] for ch in parent["children"]] == ["invoice", "purchase_order"]   # the words still work
    res = client.get(f"/api/docs/v1/documents/{d['documentId']}/result").json()
    assert any("Recognised by the words on the page" in w for w in res["warnings"])
    assert not FakeRecogniser.calls


def test_the_setting_is_editable_and_empty_means_the_words_on_the_page():
    from docauto import settings
    f = next(x for x in settings.DEFAULTS if x["key"] == "DOCAUTO_DEFAULT_CLASSIFY")
    assert f["default"] == "" and f.get("allowEmpty") and "DOCAUTO_DEFAULT_CLASSIFY" in settings.EDITABLE_KEYS
