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


# ------------------------------------------------------------------ e-invoice standards
CII_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rsm:CrossIndustryInvoice xmlns:rsm="urn:un:unece:uncefact:data:standard:CrossIndustryInvoice:100"
 xmlns:ram="urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100"
 xmlns:udt="urn:un:unece:uncefact:data:standard:UnqualifiedDataType:100">
 <rsm:ExchangedDocumentContext><ram:GuidelineSpecifiedDocumentContextParameter><ram:ID>urn:cen.eu:en16931:2017</ram:ID></ram:GuidelineSpecifiedDocumentContextParameter></rsm:ExchangedDocumentContext>
 <rsm:ExchangedDocument><ram:ID>RE-2026-118</ram:ID><ram:TypeCode>380</ram:TypeCode>
  <ram:IssueDateTime><udt:DateTimeString format="102">20261103</udt:DateTimeString></ram:IssueDateTime></rsm:ExchangedDocument>
 <rsm:SupplyChainTradeTransaction>
  <ram:IncludedSupplyChainTradeLineItem>
   <ram:SpecifiedTradeProduct><ram:Name>Beratung</ram:Name></ram:SpecifiedTradeProduct>
   <ram:SpecifiedLineTradeAgreement><ram:NetPriceProductTradePrice><ram:ChargeAmount>100.00</ram:ChargeAmount></ram:NetPriceProductTradePrice></ram:SpecifiedLineTradeAgreement>
   <ram:SpecifiedLineTradeDelivery><ram:BilledQuantity unitCode="HUR">10</ram:BilledQuantity></ram:SpecifiedLineTradeDelivery>
   <ram:SpecifiedLineTradeSettlement><ram:SpecifiedTradeSettlementLineMonetarySummation><ram:LineTotalAmount>1000.00</ram:LineTotalAmount></ram:SpecifiedTradeSettlementLineMonetarySummation></ram:SpecifiedLineTradeSettlement>
  </ram:IncludedSupplyChainTradeLineItem>
  <ram:ApplicableHeaderTradeAgreement>
   <ram:SellerTradeParty><ram:Name>Muster GmbH</ram:Name><ram:SpecifiedTaxRegistration><ram:ID schemeID="VA">DE123456789</ram:ID></ram:SpecifiedTaxRegistration></ram:SellerTradeParty>
   <ram:BuyerTradeParty><ram:Name>Kunde AG</ram:Name></ram:BuyerTradeParty>
   <ram:BuyerOrderReferencedDocument><ram:IssuerAssignedID>PO-77</ram:IssuerAssignedID></ram:BuyerOrderReferencedDocument>
  </ram:ApplicableHeaderTradeAgreement>
  <ram:ApplicableHeaderTradeSettlement><ram:InvoiceCurrencyCode>EUR</ram:InvoiceCurrencyCode>
   <ram:SpecifiedTradePaymentTerms><ram:DueDateDateTime><udt:DateTimeString format="102">20261203</udt:DateTimeString></ram:DueDateDateTime></ram:SpecifiedTradePaymentTerms>
   <ram:SpecifiedTradeSettlementHeaderMonetarySummation><ram:TaxBasisTotalAmount>1000.00</ram:TaxBasisTotalAmount>
    <ram:TaxTotalAmount currencyID="EUR">190.00</ram:TaxTotalAmount><ram:GrandTotalAmount>1190.00</ram:GrandTotalAmount>
   </ram:SpecifiedTradeSettlementHeaderMonetarySummation></ram:ApplicableHeaderTradeSettlement>
 </rsm:SupplyChainTradeTransaction>
</rsm:CrossIndustryInvoice>"""


def test_cii_e_invoices_are_read():
    from docauto import einvoice
    inv = einvoice.parse_xml(CII_XML)
    v = inv["values"]
    assert inv["raw"]["standard"] == "cii" and "Factur-X" in inv["standard"]
    assert (v["invoice_number"], v["invoice_date"], v["due_date"]) == ("RE-2026-118", "2026-11-03", "2026-12-03")
    assert (v["supplier_name"], v["supplier_vat"], v["customer_name"], v["po_number"]) == ("Muster GmbH", "DE123456789", "Kunde AG", "PO-77")
    assert (v["currency"], v["subtotal"], v["vat_amount"], v["total"]) == ("EUR", 1000.0, 190.0, 1190.0)
    assert v["line_items"] == [{"description": "Beratung", "quantity": 10.0, "unit_price": 100.0, "amount": 1000.0}]
    assert einvoice.parse_xml(CII_XML, ["ubl"]) is None          # a process reading UBL only leaves it alone


def test_payment_qr_codes_and_chosen_standards():
    from docauto import einvoice
    swiss = "\n".join(["SPC", "0200", "1", "CH4431999123000889012", "S", "Robert Schneider AG", "Rue du Lac", "1268",
                       "2501", "Biel", "CH", "", "", "", "", "", "", "", "1949.75", "CHF", "S", "Pia-Maria Rutschmann",
                       "Grosse Marktgasse", "28", "9400", "Rorschach", "CH", "QRR", "210000000003139471430009017",
                       "Order of 15 June", "EPD"])
    q = einvoice.parse_qr(swiss)
    assert q["source"] == "swiss-qr" and q["values"]["total"] == 1949.75 and q["values"]["currency"] == "CHF"
    assert q["values"]["supplier_name"] == "Robert Schneider AG" and q["values"]["iban"] == "CH4431999123000889012"
    epc = "\n".join(["BCD", "002", "1", "SCT", "BPOTBEB1", "Red Cross Belgium", "BE72000000001616", "EUR12.30", "CHAR", "",
                     "Urgency fund"])
    q = einvoice.parse_qr(epc)
    assert q["source"] == "epc-qr" and q["values"]["total"] == 12.3 and q["values"]["currency"] == "EUR"
    assert einvoice.parse_qr(epc, ["zatca-qr", "gst-qr"]) is None   # not a standard the process reads
    ids = {s["id"] for s in einvoice.standards()}
    assert {"ubl", "cii", "zatca-qr", "gst-qr", "swiss-qr", "epc-qr"} <= ids


# ------------------------------------------------------------------ checks for every country
def test_tax_check_presets_and_own_rates():
    from docauto.docflow import tax_problems
    ok = {"supplier_vat": "300123456700003", "subtotal": 1000.0, "vat_amount": 150.0, "total": 1150.0}
    assert tax_problems({"preset": "ksa"}, ok) == []
    assert [f for f, _ in tax_problems({"preset": "ksa"}, {**ok, "supplier_vat": "123"})] == ["supplier_vat"]
    de = {"supplier_vat": "DE123456789", "subtotal": 100.0, "vat_amount": 7.0, "total": 107.0}
    assert tax_problems({"preset": "germany"}, de) == []                       # the reduced rate
    assert [f for f, _ in tax_problems({"preset": "germany"}, {**de, "vat_amount": 10.0, "total": 110.0})] == ["vat_amount"]
    assert tax_problems({"preset": "eu"}, {**de, "vat_amount": 21.0, "total": 121.0}) == []   # no rate list: any rate
    gst = {"supplier_vat": "29ABCDE1234F1Z5", "subtotal": 1000.0, "vat_amount": 180.0, "total": 1180.0}
    assert tax_problems({"preset": "india-gst"}, gst) == []
    assert tax_problems({"preset": "custom", "rates": "8.1, 2.6", "idPattern": r"CHE-\d{3}\.\d{3}\.\d{3}"},
                        {"supplier_vat": "CHE-123.456.789", "subtotal": 100.0, "vat_amount": 8.1, "total": 108.1}) == []
    assert tax_problems({"preset": "ksa"}, {"subtotal": 1000.0, "total": 1150.0}) == []   # nothing to check


def test_date_and_pattern_checks():
    from datetime import date, timedelta
    from docauto.docflow import date_problem
    today = date.today()
    assert date_problem((today + timedelta(days=3)).isoformat(), {"notFuture": True}) == "is in the future"
    assert date_problem((today - timedelta(days=400)).isoformat(), {"notOlderThanDays": 365}) == "is more than 365 days ago"
    assert date_problem(today.isoformat(), {"notFuture": True, "notOlderThanDays": 30}) is None
    assert date_problem("not a date", {"notFuture": True}) == "is not a date"


# ------------------------------------------------------------------ personal data of every country
def test_personal_data_kinds_and_own_patterns(monkeypatch):
    from docauto import pii
    monkeypatch.delenv("DOCAUTO_PII_PATTERNS", raising=False)
    monkeypatch.setenv("DOCAUTO_PII_EXTRA", "EMPLOYEE=EMP-\d{6}")
    m = pii.PiiMasker()
    text = ("Card 4111 1111 1111 1111, SSN 123-45-6789, NINO AB 12 34 56 C, Emirates ID 784-1990-1234567-1, "
            "Aadhaar 2345 6789 0123, staff EMP-004512, ref 4111111111111112")
    out = m.mask(text)
    for secret in ("4111 1111 1111 1111", "123-45-6789", "AB 12 34 56 C", "784-1990-1234567-1", "2345 6789 0123", "EMP-004512"):
        assert secret not in out, secret
    assert "4111111111111112" in out             # fails the card checksum: not a card, kept
    assert m.unmask(out) == text
    monkeypatch.setenv("DOCAUTO_PII_PATTERNS", "email")
    monkeypatch.setenv("DOCAUTO_PII_EXTRA", "")
    assert pii.PiiMasker().mask("SSN 123-45-6789 a@b.co") == "SSN 123-45-6789 [EMAIL_1]"


def test_residency_region_groups():
    from docauto.policy import region_matches
    assert region_matches("westeurope", "eu") and region_matches("europe-west3", "eu") and region_matches("eu-central-1", "eu")
    assert region_matches("me-central2", "ksa") and region_matches("uaenorth", "uae") and region_matches("centralindia", "india")
    assert region_matches("eastus2", "us") and not region_matches("westeurope", "us") and not region_matches("eastus", "eu")
    assert region_matches("westeurope", "westeurope") and region_matches("westeurope", "west*")


# ------------------------------------------------------------------ languages, numbers and dates
def test_numbers_and_dates_of_many_countries(monkeypatch):
    from docauto.extraction import LOCALE, coerce
    monkeypatch.delenv("DOCAUTO_DATE_ORDER", raising=False)
    monkeypatch.delenv("DOCAUTO_DECIMAL_MARK", raising=False)
    for text, want in [("1,234.50", 1234.5), ("1.234,50", 1234.5), ("1'234.50", 1234.5), ("1 234,5", 1234.5),
                       ("(1,234.00)", -1234.0), ("1.234,00-", -1234.0), ("١٢٣٫٤٥", 123.45), ("१२३४", 1234.0)]:
        assert coerce(text, "currency") == (want, True), text
    for text, want in [("15. März 2026", "2026-03-15"), ("3 de noviembre de 2026", "2026-11-03"), ("2026年10月15日", "2026-10-15"),
                       ("2026년 10월 15일", "2026-10-15"), ("15 مارس 2026", "2026-03-15"), ("March 3, 2026", "2026-03-03"),
                       ("03/04/26", "2026-04-03")]:
        assert coerce(text, "date") == (want, True), text
    monkeypatch.setenv("DOCAUTO_DATE_ORDER", "MDY")
    assert coerce("03/04/2026", "date") == ("2026-03-04", True)
    monkeypatch.setenv("DOCAUTO_DECIMAL_MARK", ",")
    assert coerce("12.345", "currency") == (12345.0, True)
    token = LOCALE.set({"dateOrder": "DMY", "decimalMark": "."})   # a document type's own wins over the server's
    try:
        assert coerce("03/04/2026", "date") == ("2026-04-03", True) and coerce("12.345", "currency") == (12.345, True)
    finally:
        LOCALE.reset(token)


def test_scripts_and_directions():
    from docauto.normalize import detect_direction, detect_language
    assert (detect_language("שלום עולם"), detect_direction("שלום עולם")) == ("he", "rtl")
    assert detect_language("Привет мир") == "ru" and detect_language("東京タワー") == "ja" and detect_language("नमस्ते") == "hi"
    assert detect_language("مرحبا") == "ar" and detect_direction("Hello") == "ltr"


def test_company_headings_and_engine_languages():
    from docauto.grounding import as_number
    from docauto.provider import ProviderInfo
    assert as_number("¥12,000") == 12000.0 and as_number("CHF 1'250.00") == 1250.0 and as_number("INV-2026") is None
    info = ProviderInfo.__new__(ProviderInfo)
    info.languages = ["en", "ar"]
    assert info.supports_languages(["en", "fr"]) and info.coverage(["en", "fr"]) == 0.5
    assert not info.supports_languages(["ja"])


def test_reference_tables_in_more_formats():
    from docauto.docflow import parse_table_file
    head, rows = parse_table_file(b"code\tname\nS1\tAcme\nS2\tGlobex\n", "suppliers.tsv")
    assert head == ["code", "name"] and rows[1] == {"code": "S2", "name": "Globex"}
    head, rows = parse_table_file(b'[{"code": "S1", "vat": "DE1"}, {"code": "S2", "iban": "GB1"}]', "s.json")
    assert head == ["code", "vat", "iban"] and rows[1] == {"code": "S2", "vat": "", "iban": "GB1"}
    head, rows = parse_table_file(b"Supplier list 2026\ncode;name\nS1;Acme\n", "s.csv", header_row=2)
    assert head == ["code", "name"] and rows == [{"code": "S1", "name": "Acme"}]
    head, rows = parse_table_file(b"S1,Acme\nS2,Globex\n", "s.csv", header_row=0)
    assert head == ["col1", "col2"] and len(rows) == 2


def test_accuracy_from_reviewed_documents(monkeypatch):
    from fastapi.testclient import TestClient
    from tests.conftest import make_config
    from tests.test_einvoice import invoice_pdf
    import docauto.api.app as appmod
    from docauto.pipeline import DocumentService
    monkeypatch.setenv("DOCAUTO_FOLDER_WATCHER", "false")
    monkeypatch.delenv("DOCAUTO_PLATFORM_URL", raising=False)
    monkeypatch.delenv("DOCAUTO_API_TOKEN", raising=False)
    appmod.set_service(DocumentService(make_config()))
    c = TestClient(appmod.app)
    DF = "/api/docs/v1/df"
    fid = c.post(f"{DF}/flows", json={"name": "AP", "docTypes": ["invoice"], "intake": {"einvoiceFirst": False}}).json()["id"]
    lines = ["ACME Trading Co.", "Invoice", "Invoice Number: INV-77", "Invoice Date: 2026-11-03", "Subtotal: 100.00", "Total: 100.00 EUR"]
    r = c.post(f"{DF}/flows/{fid}/documents", params={"wait": "true"}, files=[("files", ("a.pdf", invoice_pdf(lines), "application/pdf"))])
    doc = r.json()["documents"][0]["documentId"]
    c.post(f"{DF}/documents/{doc}/review", json={"action": "submit", "fields": {"invoice_number": "INV-0077"}, "note": "number fixed"})
    a = c.get(f"{DF}/flows/{fid}/accuracy").json()
    by = {f["name"]: f for f in a["fields"]}
    assert a["reviewed"] == 1 and by["invoice_number"]["corrected"] == 1 and by["total"]["corrected"] == 0
    assert a["measured"] > 0 and len(a["levels"]) == 9 and all(0 <= (x["wrongPassed"] or 0) <= 1 for x in a["levels"])
    res = c.get(f"/api/docs/v1/documents/{doc}/result").json()
    assert "Reviewer note: number fixed" in res["warnings"]


def test_monthly_budget_switches_to_local_engines(monkeypatch):
    from tests.conftest import make_config
    from docauto.docflow import DocFlowService, Flow
    from docauto.pipeline import DocumentService
    monkeypatch.setenv("DOCAUTO_FOLDER_WATCHER", "false")
    monkeypatch.delenv("DOCAUTO_PRICES", raising=False)
    monkeypatch.delenv("DOCAUTO_PRICE_RATE", raising=False)
    svc = DocumentService(make_config())
    df = DocFlowService(svc)
    fl = df.save_flow(Flow(name="Budgeted", docTypes=["invoice"]), new=True)
    ids = svc.submit(b"%PDF-1.4 x", "a.pdf", __import__("docauto.pipeline", fromlist=["x"]).ProcessOptions(), wait=False)
    svc.store.set_doc_meta(ids["documentId"], fl.id, 1, "upload")
    svc.store.record_usage(tenant_id="t", job_id="j", document_id=ids["documentId"], provider_id="azure-di", option_pack="tier2_cloud",
                           capability="document_ocr", pages=2000, credits=0, runs_locally=False, regions=[])
    assert df.month_spend(fl.id) == 3.0            # 2,000 pages at 1.50 per 1,000
    assert not df.over_budget(fl)                  # no budget: no limit
    fl.engine.monthlyBudget = 2.5
    assert df.over_budget(fl) and df._options(fl, df.doctype("invoice"), "upload").localOnly
    fl.engine.extractionByType = {"invoice": "claude"}
    assert df._options(fl, df.doctype("invoice"), "upload").extractionProvider == "claude"
    svc.close()


def test_improve_instructions_from_a_test_run(monkeypatch):
    import json as _json
    from types import SimpleNamespace
    from tests.conftest import make_config
    from tests.test_einvoice import invoice_pdf
    from docauto.docflow import DocFlowService
    from docauto.pipeline import DocumentService
    monkeypatch.setenv("DOCAUTO_FOLDER_WATCHER", "false")
    svc = DocumentService(make_config())
    df = DocFlowService(svc)
    s = df.testsets.create("Invoices", "invoice")
    pdf = invoice_pdf(["ACME Trading Co.", "Invoice", "Invoice Number: INV-1", "Total: 100.00 EUR"])
    df.testsets.add_file(s["id"], pdf, "a.pdf", {"invoice_number": "INV-0001", "total": "100.00"})
    df.testsets.start_run(s["id"], [{"ocr": "auto", "extraction": "builtin-rules"}], wait=True)
    seen = {}

    def chat(prompt):
        seen["prompt"] = prompt
        return _json.dumps({"fields": {"invoice_number": "The number after 'Invoice Number', with its leading zeros"}})
    llm = SimpleNamespace(info=SimpleNamespace(id="fake-ai", runsLocally=True), instance=SimpleNamespace(_chat=chat))
    out = df.testsets.improve(s["id"], llm=llm)
    p = {x["name"]: x for x in out["proposals"]}
    assert "invoice_number" in p and "leading zeros" in p["invoice_number"]["proposed"]
    assert p["invoice_number"]["examples"][0]["right"] == "INV-0001" and "INV-0001" in seen["prompt"]
    assert "total" not in p                                    # read right: no new instruction
    assert df.doctype("invoice").fields[0].description in (None, "")   # nothing saved
    svc.close()
