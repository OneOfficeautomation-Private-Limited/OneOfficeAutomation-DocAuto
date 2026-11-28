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
