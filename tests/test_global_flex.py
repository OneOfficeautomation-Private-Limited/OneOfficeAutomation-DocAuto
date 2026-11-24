"""Global and flexible: field types, how fields are read, choices, check strictness, "write as", names in any
script and reference-table ids (audit of 25 Nov 2026)."""
from docauto.api.app import ref_table_id
from docauto.docflow import _field_name_ok, _slug, _snake
from docauto.extraction import coerce
from docauto.models import DocumentResult, ExtractedField, FieldDef, FieldSchema
from docauto.normalize import write_as
from docauto.scoring import score


def test_new_field_types_are_read():
    assert coerce("15 %", "percentage") == (15.0, True)
    assert coerce("2026-11-20 14:30", "datetime") == ("2026-11-20T14:30", True)
    assert coerce("20/11/2026", "datetime") == ("2026-11-20T00:00", True)
    assert coerce("Mail: ap@example.com today", "email") == ("ap@example.com", True)
    assert coerce("+49 30 1234 5678", "phone")[1]
    assert coerce("12", "phone")[1] is False
    assert coerce("☑", "checkbox") == (True, True)
    assert coerce("[ ]", "checkbox") == (False, True)
    assert coerce("Ja", "boolean") == (True, True)
    assert coerce("nein", "boolean") == (False, True)
    assert coerce("J. Smith", "signature") == (True, True)
    assert coerce("Travel, Meals; Hotel", "multienum") == (["Travel", "Meals", "Hotel"], True)


def test_write_as():
    assert write_as("2026-11-05", "date", "DD/MM/YYYY") == "05/11/2026"
    assert write_as("2026-11-05", "date", "MM/DD/YYYY") == "11/05/2026"
    assert write_as("2026-11-05T09:30", "datetime", "YYYY-MM-DD HH:mm") == "2026-11-05 09:30"
    assert write_as(1234.5, "currency", "2 decimals") == "1234.50"
    assert write_as(1234.567, "currency", "0 decimals") == "1235"
    assert write_as("acme trading", "string", "UPPER") == "ACME TRADING"
    assert write_as("  a   b ", "string", "Trimmed") == "a b"
    assert write_as("+49 (30) 1234", "phone", "Digits only") == "49301234"
    assert write_as("0049 30 1234", "phone", "International (+)") == "+4930 1234".replace(" ", "")
    assert write_as("x", "string", "something unknown") is None
    assert write_as(None, "date", "DD/MM/YYYY") is None
    f = ExtractedField(name="d", type="date", value="2026-11-05", writeAs="DD.MM.YYYY")
    assert f.out() == "05.11.2026"
    f.value = "2026-12-01"   # a reviewer's correction is written the same way
    assert f.out() == "01.12.2026"


def _doc(**vals):
    return DocumentResult(documentId="d", fileName="f.pdf", mimeType="application/pdf", pageCount=1, pages=[],
                          fields=[ExtractedField(name=k, type="string", value=v, confidence=0.9) for k, v in vals.items()])


def test_other_values_are_kept_when_allowed():
    schema = FieldSchema(name="t", fields=[FieldDef(name="currency", type="enum", enumValues=["EUR", "USD"], allowOther=True),
                                           FieldDef(name="kind", type="enum", enumValues=["A", "B"])])
    doc = score(_doc(currency="JPY", kind="C"), schema, 0.8, 0.8)
    cur, kind = doc.fields
    assert cur.validation.passed and cur.notes == ["not one of the choices: JPY"] and cur.confidence == 0.9
    assert not kind.validation.passed


def test_check_strictness():
    fields = [FieldDef(name="a", regex=r"^\d+$"), FieldDef(name="b", regex=r"^\d+$", patternSeverity="warn"),
              FieldDef(name="c", regex=r"^\d+$", patternSeverity="off")]
    doc = score(_doc(a="x1", b="x1", c="x1"), FieldSchema(name="t", fields=fields), 0.5, 0.5)
    a, b, c = doc.fields
    assert not a.validation.passed and a.confidence == 0.45
    assert b.validation.passed and b.notes == ["does not match the format"] and b.confidence == 0.9
    assert c.validation.passed and not c.notes


def test_names_in_any_script():
    assert _field_name_ok("رقم_الفاتورة") and _field_name_ok("Rechnungsnummer") and _field_name_ok("発行日")
    assert not _field_name_ok("1st") and not _field_name_ok("a b") and not _field_name_ok("")
    assert _snake("Numéro de facture") == "numéro_de_facture"
    a, b = _slug("فواتير الموردين", "flow"), _slug("فواتير العملاء", "flow")
    assert a.startswith("flow-") and b.startswith("flow-") and a != b
    assert _slug("AP invoices") == "ap-invoices"


def test_reference_tables_never_overwrite_another():
    existing = [{"id": "suppliers", "name": "Suppliers"}, {"id": "table", "name": "الموردون"}]
    assert ref_table_id("suppliers", existing) == "suppliers"          # same name: replaced, as before
    assert ref_table_id("Suppliers!", existing) == "suppliers-2"       # another name, same slug: a new table
    assert ref_table_id("العملاء", existing) == "table-2"             # another non-Latin name: a new table
    assert ref_table_id("الموردون", existing) == "table"              # its own name: replaced
    assert ref_table_id("Renamed", existing, "suppliers") == "suppliers"   # replace by id, renamed


def test_ai_reading_gets_method_and_choice_meanings():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "options" / "tier3_ai" / "llm_openai_compatible" / "provider.py"
    spec = importlib.util.spec_from_file_location("llm_shared_t", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    schema = FieldSchema(name="Expense", fields=[
        FieldDef(name="kind", type="enum", method="classify", enumValues=["Travel", "Meals"],
                 enumDescriptions={"Travel": "flights, trains, taxis"}),
        FieldDef(name="overdue", type="boolean", method="infer", description="Is the due date before today?")])
    doc = DocumentResult(documentId="d", fileName="f.pdf", mimeType="application/pdf", pageCount=0, pages=[])
    prompt, _ = mod.build_prompt(doc, schema)
    assert '"method": "classify"' in prompt and '"choiceMeanings"' in prompt and '"method": "infer"' in prompt
    assert "English and Arabic" not in mod.SYSTEM_PROMPT
    out = mod.parse_llm_fields('{"fields": {"kind": {"value": "Travel", "lines": [], "confidence": 0.9}, '
                               '"overdue": {"value": true, "lines": [], "confidence": 0.8}}}', schema, {}, doc, "llm")
    assert all(f.validation.passed for f in out)   # answers about the document are not "not on the page"


# ------------------------------------------------------------------ the built-in library is global
def test_invoice_total_check_allows_missing_tax_discounts_and_charges():
    from docauto.docflow import INVOICE_TOTAL_CHECK
    from docauto.extraction import safe_eval
    none = dict(subtotal=None, discount=None, other_charges=None, vat_amount=None)
    assert safe_eval(INVOICE_TOTAL_CHECK, {**none, "total": 99.0})                       # a US bill with no tax line
    assert safe_eval(INVOICE_TOTAL_CHECK, {**none, "subtotal": 100.0, "discount": 10.0, "other_charges": 5.0,
                                           "vat_amount": 19.0, "total": 114.0})
    assert safe_eval(INVOICE_TOTAL_CHECK, {**none, "subtotal": 1000, "total": 1000.4})   # within 0.1 %
    assert not safe_eval(INVOICE_TOTAL_CHECK, {**none, "subtotal": 100.0, "vat_amount": 19.0, "total": 129.0})


def test_library_has_no_country_rules_and_an_other_type():
    from docauto.docflow import LIBRARY
    fields = [f for t in LIBRARY.values() for f in t["fields"]]
    assert not [f["name"] for f in fields if f.get("regex")]        # no Saudi VAT / IBAN / ID patterns
    cur = next(f for f in LIBRARY["invoice"]["fields"] if f["name"] == "currency")
    assert cur["allowOther"] and "JPY" in cur["enumValues"] and "INR" in cur["enumValues"]
    assert {"other", "credit_note", "delivery_note", "utility_bill", "contract"} <= set(LIBRARY)
    assert "Rechnungsnummer" in next(f for f in LIBRARY["invoice"]["fields"] if f["name"] == "invoice_number")["aliases"]


def test_amount_after_a_printed_tax_rate():
    from docauto.extraction import RulesExtractor
    from docauto.models import BBox, Line, Page
    from docauto.provider import ExtractRequest
    lines = ["VAT 15%: 150.00", "GST (18%) 90.00"]
    doc = DocumentResult(documentId="d", fileName="f.pdf", mimeType="application/pdf", pageCount=1,
                         pages=[Page(pageNumber=1, width=1, height=1, lines=[Line(text=t, bbox=BBox(x=0, y=i / 10, w=1, h=0.05)) for i, t in enumerate(lines)])])
    doc.rebuild_text()
    ex = RulesExtractor.__new__(RulesExtractor)
    ex.info = type("I", (), {"id": "builtin-rules"})()
    got = ex.extract(ExtractRequest(document=doc, schema=FieldSchema(name="t", fields=[
        FieldDef(name="vat", type="currency", label="VAT"), FieldDef(name="gst", type="currency", label="GST")])))
    assert [f.value for f in got] == [150.0, 90.0]


def test_installs_with_the_old_saudi_defaults_are_upgraded_once(monkeypatch):
    from tests.conftest import make_config
    from docauto.docflow import DocFlowService, OLD_INVOICE_CHECK, INVOICE_TOTAL_CHECK
    from docauto.pipeline import DocumentService
    monkeypatch.setenv("DOCAUTO_FOLDER_WATCHER", "false")
    svc = DocumentService(make_config())
    df = DocFlowService(svc)
    dt = df.doctype("invoice")          # make it look like a 3.2 seed
    for f in dt.fields:
        if f.name == "supplier_vat":
            f.regex = r"3\d{14}"
        if f.name == "currency":
            f.enumValues, f.allowOther = ["SAR", "AED", "USD", "EUR", "GBP"], False
    dt.fields = [f for f in dt.fields if f.name not in ("discount", "other_charges")]
    dt.checks = [OLD_INVOICE_CHECK]
    df.save_doctype(dt)
    for t in ("credit_note", "other"):
        svc.store.df_delete("doctype", t) if hasattr(svc.store, "df_delete") else None
    svc.store.meta_set("df.library.global", "")
    df._upgrade_global()
    dt = df.doctype("invoice")
    by = {f.name: f for f in dt.fields}
    assert by["supplier_vat"].regex is None and by["currency"].allowOther and "JPY" in by["currency"].enumValues
    assert dt.checks == [INVOICE_TOTAL_CHECK] and "discount" in by and "other_charges" in by
    assert df.doctype("other").name == "Other document"
    svc.close()
