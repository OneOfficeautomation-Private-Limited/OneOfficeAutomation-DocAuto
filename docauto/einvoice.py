"""E-invoices: use the invoice's own data instead of (or to check) what was read from the page.

Every standard is in STANDARDS (one parser each); a process chooses which it reads (all by default).

* UBL 2.1 XML (Peppol BIS, EN 16931, XRechnung UBL, ZATCA...) and UN/CEFACT CII XML (Factur-X, ZUGFeRD,
  XRechnung CII), uploaded on their own or embedded in a
  PDF/A-3: the invoice fields come from the XML with confidence 1.0 (source "einvoice-xml") and no OCR
  runs. A PDF still has its pages rendered (and its text layer read, free) for the review screen.
* QR codes on the page (docauto.barcodes): the ZATCA QR (base64 TLV: 1 seller name, 2 VAT number,
  3 time stamp, 4 total with VAT, 5 VAT; 6-9 are the signature parts, kept raw) and the India GST
  e-invoice QR (a signed JWT; the payload is read WITHOUT checking the signature: SellerGstin, BuyerGstin,
  DocNo, DocDt, TotInvVal, Irn...). Their values cross-check the extracted fields: a value that agrees
  is confirmed (high confidence, grounded by the QR), a different value sends the document to review
  with the reason, and a value the extraction missed is filled in from the QR.

Field names follow the built-in Invoice document type (invoice_number, invoice_date, supplier_name,
supplier_vat, customer_name, customer_vat, currency, subtotal, vat_amount, total, line_items...). A GSTIN
fills supplier_gstin / customer_gstin when the document type has them, else supplier_vat / customer_vat.
"""
from __future__ import annotations

import base64
import binascii
import json
import re
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from typing import Any, Optional

from .extraction import coerce
from .grounding import HALLUCINATION, as_number, key
from .models import Barcode, DocumentResult, ExtractedField, FieldSchema, FieldValidation

XML_SOURCE = "einvoice-xml"
ZATCA_SOURCE = "zatca-qr"
GST_SOURCE = "gst-qr"
QR_CONFIDENCE = 0.99
QR_PAGES = 3          # e-invoice QR codes are looked for on the first pages only

NS = {"cbc": "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2",
      "cac": "urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2"}
# value name -> document type fields it may fill, in order of preference
TARGETS = {"supplier_gstin": ["supplier_gstin", "supplier_vat"], "customer_gstin": ["customer_gstin", "customer_vat"]}
LABELS = {"total": "total", "vat_amount": "VAT", "supplier_vat": "VAT number", "supplier_gstin": "seller GSTIN",
          "customer_gstin": "buyer GSTIN", "invoice_date": "date", "invoice_number": "invoice number",
          "supplier_name": "seller", "irn": "IRN"}
_NUMBERS = {"subtotal", "vat_amount", "total", "quantity", "unit_price", "amount"}
_SOFT = {"supplier_name"}   # names are often printed in another language than the QR holds: warn, don't block


# ------------------------------------------------------------------ UBL XML
def _num(text: Optional[str]) -> Optional[float]:
    try:
        return float(text) if text not in (None, "") else None
    except ValueError:
        return None


def parse_ubl(data: bytes) -> Optional[dict]:
    """{"values": {...}, "raw": {...}} from a UBL 2.1 Invoice / CreditNote / DebitNote, or None."""
    head = data[:4096].lstrip(b"\xef\xbb\xbf \t\r\n")
    if not head.startswith(b"<") or b"<!DOCTYPE" in head.upper() or b"<!ENTITY" in data.upper():
        return None   # no DTDs or entities: e-invoices never need them
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None
    kind = root.tag.rsplit("}", 1)[-1]
    if kind not in ("Invoice", "CreditNote", "DebitNote") or not root.tag.startswith("{urn:oasis:names:specification:ubl"):
        return None

    def t(path: str, node=root) -> Optional[str]:
        v = node.findtext(path, namespaces=NS)
        return v.strip() if v and v.strip() else None

    def party(path: str) -> tuple[Optional[str], Optional[str]]:
        p = root.find(path + "/cac:Party", NS)
        if p is None:
            return None, None
        name = t("cac:PartyLegalEntity/cbc:RegistrationName", p) or t("cac:PartyName/cbc:Name", p)
        return name, t("cac:PartyTaxScheme/cbc:CompanyID", p)

    currency = t("cbc:DocumentCurrencyCode")
    sup_name, sup_vat = party("cac:AccountingSupplierParty")
    cus_name, cus_vat = party("cac:AccountingCustomerParty")
    subtotal = _num(t("cac:LegalMonetaryTotal/cbc:TaxExclusiveAmount"))
    total = _num(t("cac:LegalMonetaryTotal/cbc:TaxInclusiveAmount")) or _num(t("cac:LegalMonetaryTotal/cbc:PayableAmount"))
    vat = None
    # ZATCA has two TaxTotal blocks: one in the invoice currency (with the breakdown), one in the tax currency
    for tt in sorted(root.findall("cac:TaxTotal", NS), key=lambda e: e.find("cac:TaxSubtotal", NS) is None):
        amt = tt.find("cbc:TaxAmount", NS)
        if amt is not None and (currency is None or amt.get("currencyID") in (None, currency)):
            vat = _num(amt.text)
            break
    if vat is None and subtotal is not None and total is not None:
        vat = round(total - subtotal, 2)
    rows = []
    qty = {"Invoice": "cbc:InvoicedQuantity", "CreditNote": "cbc:CreditedQuantity", "DebitNote": "cbc:DebitedQuantity"}[kind]
    for ln in root.findall(f"cac:{kind}Line", NS):
        rows.append({k: v for k, v in {
            "description": t("cac:Item/cbc:Name", ln),
            "quantity": _num(t(qty, ln)),
            "unit_price": _num(t("cac:Price/cbc:PriceAmount", ln)),
            "amount": _num(t("cbc:LineExtensionAmount", ln)),
            "vat_amount": _num(t("cac:TaxTotal/cbc:TaxAmount", ln))}.items()})
    values = {"invoice_number": t("cbc:ID"), "invoice_date": t("cbc:IssueDate"),
              "due_date": t("cbc:DueDate") or t("cac:PaymentMeans/cbc:PaymentDueDate"),
              "supplier_name": sup_name, "supplier_vat": sup_vat, "customer_name": cus_name, "customer_vat": cus_vat,
              "po_number": t("cac:OrderReference/cbc:ID"), "currency": currency,
              "subtotal": subtotal, "vat_amount": vat, "total": total, "line_items": rows or None}
    raw = {"documentType": kind, "uuid": t("cbc:UUID"), "issueTime": t("cbc:IssueTime"),
           "typeCode": t("cbc:InvoiceTypeCode"), "profile": t("cbc:ProfileID")}
    for ref in root.findall("cac:AdditionalDocumentReference", NS):
        if t("cbc:ID", ref) == "QR":   # the ZATCA QR text is also inside the XML
            qr = parse_zatca_qr(t("cac:Attachment/cbc:EmbeddedDocumentBinaryObject", ref) or "")
            if qr:
                raw["qr"] = qr["values"]
    return {"values": {k: v for k, v in values.items() if v is not None}, "raw": {k: v for k, v in raw.items() if v}}


def pdf_attachments(data: bytes) -> list[tuple[str, bytes]]:
    """Files embedded in a PDF (PDF/A-3 e-invoices carry their XML this way)."""
    import pypdfium2 as pdfium

    from .preprocess import PDFIUM_LOCK
    with PDFIUM_LOCK:
        try:
            pdf = pdfium.PdfDocument(data)
        except Exception:  # damaged or encrypted PDF: no attachments, the normal path reports problems
            return []
        try:
            out = []
            for i in range(pdf.count_attachments()):
                a = pdf.get_attachment(i)
                out.append((a.get_name() or "", bytes(a.get_data())))
            return out
        except Exception:
            return []
        finally:
            pdf.close()


def read_file(data: bytes, mime: str, wanted: Optional[list[str]] = None) -> Optional[dict]:
    """The e-invoice XML of an uploaded file (standalone XML, or embedded in a PDF), by the standards wanted
    (None or empty: every one), or None."""
    if mime in ("application/xml", "text/xml"):
        inv = parse_xml(data, wanted)
        if inv:
            inv["raw"]["container"] = "xml"
        return inv
    if mime == "application/pdf":
        for name, blob in pdf_attachments(data):
            if name.lower().endswith(".xml") or blob.lstrip(b"\xef\xbb\xbf \r\n\t").startswith(b"<"):
                inv = parse_xml(blob, wanted)
                if inv:
                    inv["raw"].update(container="pdf", attachment=name)
                    return inv
    return None


def _target(name: str, names: Optional[set[str]]) -> Optional[str]:
    for cand in TARGETS.get(name, [name]):
        if names is None or cand in names:
            return cand
    return None


def xml_fields(inv: dict, schema: Optional[FieldSchema]) -> list[ExtractedField]:
    """The document type's fields filled from the e-invoice XML (confidence 1.0; missing ones empty)."""
    values = inv["values"]
    if schema is None:
        defs = [(n, "table" if n == "line_items" else "currency" if n in _NUMBERS else "date" if n.endswith("_date")
                 else "string", []) for n in values]
    else:
        defs = [(f.name, f.type, [c.name for c in f.columns]) for f in schema.fields]
    by_target = {}
    for n, v in values.items():
        tgt = _target(n, {d[0] for d in defs})
        if tgt and tgt not in by_target:
            by_target[tgt] = v
    out = []
    for name, ftype, cols in defs:
        v = by_target.get(name)
        if v is not None and ftype == "table":
            v = [{c: r.get(c) for c in cols} for r in v] if cols else v
        elif v is not None and ftype == "date":
            v, _ = coerce(str(v), "date")
        if v in (None, "", []):
            out.append(ExtractedField(name=name, type=ftype, value=None, confidence=0.0, providerId=XML_SOURCE))
            continue
        out.append(ExtractedField(name=name, type=ftype, value=v, rawText=None if ftype == "table" else str(v),
                                  confidence=1.0, providerId=XML_SOURCE, grounded=True, groundedBy=XML_SOURCE))
    return out


# ------------------------------------------------------------------ QR codes
def _tlv(raw: bytes, ber: bool) -> Optional[dict[int, bytes]]:
    tags, i = {}, 0
    while i < len(raw):
        if i + 2 > len(raw):
            return None
        tag, ln = raw[i], raw[i + 1]
        i += 2
        if ber and ln in (0x81, 0x82):   # some encoders write long values with BER lengths
            n = ln - 0x80
            ln, i = int.from_bytes(raw[i:i + n], "big"), i + n
        if i + ln > len(raw):
            return None
        tags[tag], i = raw[i:i + ln], i + ln
    return tags


def parse_zatca_qr(text: str) -> Optional[dict]:
    """ZATCA (Saudi) e-invoice QR: base64 of TLV tags 1-5 (6-9 kept raw, base64). None when it is not one."""
    try:
        raw = base64.b64decode(text.strip(), validate=True)
    except (binascii.Error, ValueError):
        return None
    for ber in (False, True):
        tags = _tlv(raw, ber)
        if tags and all(t in tags for t in (1, 2, 3, 4, 5)):
            break
    else:
        return None
    try:
        name, vat, stamp, total, tax = (tags[t].decode("utf-8").strip() for t in (1, 2, 3, 4, 5))
    except UnicodeDecodeError:
        return None
    if as_number(total) is None or as_number(tax) is None:
        return None
    date = re.match(r"\d{4}-\d{2}-\d{2}", stamp)
    values = {"supplier_name": name, "supplier_vat": vat, "total": as_number(total), "vat_amount": as_number(tax)}
    if date:
        values["invoice_date"] = date.group(0)
    raw_tags = {str(t): (v.decode("utf-8", "replace") if t <= 5 else base64.b64encode(v).decode())
                for t, v in sorted(tags.items())}
    return {"source": ZATCA_SOURCE, "values": values, "raw": {"timestamp": stamp, "tags": raw_tags}}


def _b64url(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def parse_gst_qr(text: str) -> Optional[dict]:
    """India GST e-invoice QR: a JWT signed by the IRP. The payload is decoded WITHOUT verifying the signature
    (it is only used to cross-check the page). None when it is not one."""
    parts = text.strip().split(".")
    if len(parts) != 3:
        return None
    try:
        payload = json.loads(_b64url(parts[1]))
        data = payload.get("data", payload) if isinstance(payload, dict) else None
        if isinstance(data, str):
            data = json.loads(data)
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or not ({"SellerGstin", "Irn", "DocNo"} & set(data)):
        return None
    values: dict[str, Any] = {"supplier_gstin": data.get("SellerGstin"), "customer_gstin": data.get("BuyerGstin"),
                              "invoice_number": data.get("DocNo"), "irn": data.get("Irn")}
    if data.get("DocDt"):
        d, ok = coerce(str(data["DocDt"]), "date", "%d/%m/%Y")
        if ok:
            values["invoice_date"] = d
    if data.get("TotInvVal") not in (None, ""):
        values["total"] = as_number(str(data["TotInvVal"]))
    return {"source": GST_SOURCE, "values": {k: v for k, v in values.items() if v not in (None, "")}, "raw": data}


def parse_qr(text: str, wanted: Optional[list[str]] = None) -> Optional[dict]:
    """The QR code's values by the first enabled QR standard that reads it, or None."""
    for sid, parse in _enabled("qr", wanted):
        qr = parse(text)
        if qr:
            qr["standard"] = STANDARDS[sid]["name"]
            return qr
    return None


def find_qr(pages_codes: list[tuple[int, list[Barcode]]], wanted: Optional[list[str]] = None) -> Optional[dict]:
    """The first e-invoice QR among the codes found on the pages [(page number, codes)], with where it is."""
    for pno, codes in pages_codes:
        for c in codes:
            qr = parse_qr(c.text, wanted)
            if qr:
                qr.update(page=pno, bbox=c.bbox)
                return qr
    return None


# ------------------------------------------------------------------ UN/CEFACT CII (Factur-X, ZUGFeRD, XRechnung CII)
CII = {"rsm": "urn:un:unece:uncefact:data:standard:CrossIndustryInvoice:100",
       "ram": "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100",
       "udt": "urn:un:unece:uncefact:data:standard:UnqualifiedDataType:100"}


def _cii_date(s: Optional[str]) -> Optional[str]:
    """CII dates are format 102 (YYYYMMDD); other formats are kept as written."""
    if s and re.fullmatch(r"\d{8}", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def parse_cii(data: bytes) -> Optional[dict]:
    """{"values", "raw"} from a UN/CEFACT Cross Industry Invoice (the XML inside Factur-X / ZUGFeRD PDFs and
    XRechnung CII), or None."""
    head = data[:4096].lstrip(b"\xef\xbb\xbf \t\r\n")
    if not head.startswith(b"<") or b"<!DOCTYPE" in head.upper() or b"<!ENTITY" in data.upper():
        return None
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None
    if root.tag != "{%s}CrossIndustryInvoice" % CII["rsm"]:
        return None

    def t(path: str, node=root) -> Optional[str]:
        v = node.findtext(path, namespaces=CII)
        return v.strip() if v and v.strip() else None

    tx = root.find("rsm:SupplyChainTradeTransaction", CII)
    agr = tx.find("ram:ApplicableHeaderTradeAgreement", CII) if tx is not None else None
    stl = tx.find("ram:ApplicableHeaderTradeSettlement", CII) if tx is not None else None

    def party(tag: str) -> tuple[Optional[str], Optional[str]]:
        p = agr.find(f"ram:{tag}", CII) if agr is not None else None
        if p is None:
            return None, None
        tax = None
        for reg in p.findall("ram:SpecifiedTaxRegistration/ram:ID", CII):
            if reg.text and (tax is None or reg.get("schemeID") == "VA"):
                tax = reg.text.strip()
        return t("ram:Name", p), tax

    sup_name, sup_vat = party("SellerTradeParty")
    cus_name, cus_vat = party("BuyerTradeParty")
    sums = stl.find("ram:SpecifiedTradeSettlementHeaderMonetarySummation", CII) if stl is not None else None
    s = (lambda name: _num(t(f"ram:{name}", sums)) if sums is not None else None)
    rows = []
    for ln in (tx.findall("ram:IncludedSupplyChainTradeLineItem", CII) if tx is not None else []):
        rows.append({k: v for k, v in {
            "description": t("ram:SpecifiedTradeProduct/ram:Name", ln),
            "quantity": _num(t("ram:SpecifiedLineTradeDelivery/ram:BilledQuantity", ln)),
            "unit_price": _num(t("ram:SpecifiedLineTradeAgreement/ram:NetPriceProductTradePrice/ram:ChargeAmount", ln)),
            "amount": _num(t("ram:SpecifiedLineTradeSettlement/ram:SpecifiedTradeSettlementLineMonetarySummation/"
                             "ram:LineTotalAmount", ln))}.items() if v is not None})
    type_code = t("rsm:ExchangedDocument/ram:TypeCode")
    values = {"invoice_number": t("rsm:ExchangedDocument/ram:ID"),
              "invoice_date": _cii_date(t("rsm:ExchangedDocument/ram:IssueDateTime/udt:DateTimeString")),
              "due_date": _cii_date(t("ram:SpecifiedTradePaymentTerms/ram:DueDateDateTime/udt:DateTimeString", stl))
              if stl is not None else None,
              "supplier_name": sup_name, "supplier_vat": sup_vat, "customer_name": cus_name, "customer_vat": cus_vat,
              "po_number": t("ram:BuyerOrderReferencedDocument/ram:IssuerAssignedID", agr) if agr is not None else None,
              "currency": t("ram:InvoiceCurrencyCode", stl) if stl is not None else None,
              "subtotal": s("TaxBasisTotalAmount"), "vat_amount": s("TaxTotalAmount"),
              "total": s("GrandTotalAmount") or s("DuePayableAmount"), "line_items": rows or None}
    raw = {"documentType": "CreditNote" if type_code == "381" else "Invoice", "typeCode": type_code,
           "profile": t("rsm:ExchangedDocumentContext/ram:GuidelineSpecifiedDocumentContextParameter/ram:ID")}
    return {"values": {k: v for k, v in values.items() if v is not None}, "raw": {k: v for k, v in raw.items() if v}}


# ------------------------------------------------------------------ payment QR codes (Swiss QR-bill, SEPA / EPC)
SWISS_SOURCE = "swiss-qr"
EPC_SOURCE = "epc-qr"


def parse_swiss_qr(text: str) -> Optional[dict]:
    """Swiss QR-bill (SPC, version 02xx): the creditor, IBAN, amount, currency, debtor and reference."""
    lines = text.replace("\r\n", "\n").split("\n")
    if len(lines) < 31 or lines[0] != "SPC" or not lines[1].startswith("02"):
        return None
    amount = as_number(lines[18]) if lines[18].strip() else None
    values = {"supplier_name": lines[5].strip() or None, "iban": lines[3].strip() or None, "total": amount,
              "currency": lines[19].strip() or None, "customer_name": lines[21].strip() or None,
              "payment_reference": lines[28].strip() or None}
    return {"source": SWISS_SOURCE, "values": {k: v for k, v in values.items() if v not in (None, "")},
            "raw": {"referenceType": lines[27], "message": lines[29] if len(lines) > 29 else ""}}


def parse_epc_qr(text: str) -> Optional[dict]:
    """SEPA credit transfer QR (EPC069-12, "GiroCode"): beneficiary, IBAN, amount in EUR, reference."""
    lines = text.replace("\r\n", "\n").split("\n")
    if len(lines) < 7 or lines[0] != "BCD" or lines[3] != "SCT":
        return None
    amt = lines[7] if len(lines) > 7 else ""
    m = re.fullmatch(r"([A-Z]{3})([\d.]+)", amt.strip())
    values = {"supplier_name": lines[5].strip() or None, "iban": lines[6].strip() or None,
              "total": float(m.group(2)) if m else None, "currency": m.group(1) if m else None,
              "payment_reference": (lines[9].strip() if len(lines) > 9 else "") or (lines[10].strip() if len(lines) > 10 else "") or None}
    return {"source": EPC_SOURCE, "values": {k: v for k, v in values.items() if v not in (None, "")},
            "raw": {"bic": lines[4].strip()}}


# ------------------------------------------------------------------ the standards a process may read
# id -> what it is; kind "xml" replaces reading the page, kind "qr" checks what was read. A process lists the ones it
# reads (Intake.einvoiceStandards); an empty list reads every one. New standards are added here, one parser each.
STANDARDS: dict[str, dict] = {
    "ubl": {"name": "UBL 2.1 XML (Peppol, EN 16931, XRechnung UBL, ZATCA, and others)", "kind": "xml", "parse": parse_ubl},
    "cii": {"name": "CII XML (Factur-X, ZUGFeRD, XRechnung CII)", "kind": "xml", "parse": parse_cii},
    "zatca-qr": {"name": "Saudi Arabia ZATCA QR code", "kind": "qr", "parse": parse_zatca_qr},
    "gst-qr": {"name": "India GST e-invoice QR code", "kind": "qr", "parse": parse_gst_qr},
    "swiss-qr": {"name": "Swiss QR-bill", "kind": "qr", "parse": parse_swiss_qr},
    "epc-qr": {"name": "SEPA payment QR code (EPC, GiroCode)", "kind": "qr", "parse": parse_epc_qr},
}


def standards() -> list[dict]:
    """The standards DocFlow reads, for the screens: id, name, kind."""
    return [{"id": k, "name": v["name"], "kind": v["kind"]} for k, v in STANDARDS.items()]


def _enabled(kind: str, wanted: Optional[list[str]]) -> list[tuple[str, Any]]:
    return [(k, v["parse"]) for k, v in STANDARDS.items() if v["kind"] == kind and (not wanted or k in wanted)]


def parse_xml(data: bytes, wanted: Optional[list[str]] = None) -> Optional[dict]:
    """The e-invoice in an XML file, by the first enabled XML standard that reads it, or None."""
    for sid, parse in _enabled("xml", wanted):
        inv = parse(data)
        if inv:
            inv["raw"]["standard"] = sid
            inv["standard"] = STANDARDS[sid]["name"]
            return inv
    return None


def _fmt(v: Any) -> str:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{v:,.2f}"
    n = as_number(str(v)) if isinstance(v, str) and re.fullmatch(r"[\d.,\s]+", v) else None
    return f"{n:,.2f}" if n is not None else str(v)


def agree(name: str, read: Any, qr: Any) -> bool:
    if isinstance(qr, (int, float)):
        r = read if isinstance(read, (int, float)) and not isinstance(read, bool) else as_number(str(read))
        return r is not None and abs(float(r) - float(qr)) < 0.01
    if name.endswith("_date"):
        v, _ = coerce(str(read), "date")
        return v == qr
    a, b = key(read), key(qr)
    if name in _SOFT:
        return a == b or SequenceMatcher(None, a, b).ratio() >= 0.85
    return a == b


def cross_check(doc: DocumentResult, qr: dict, schema: Optional[FieldSchema]) -> list[dict]:
    """Compare the QR values with the extracted fields, in place (see the module notes). Returns the checks,
    also kept in doc.einvoice: [{"field", "qr", "read", "result": match | mismatch | differs | filled}]."""
    defs = {f.name: f for f in schema.fields} if schema else {}
    names = set(defs) if schema else None
    by = {f.name: f for f in doc.fields}
    src, checks = qr["source"], []
    for name, qv in qr["values"].items():
        target = _target(name, names)
        if target is None or qv in (None, ""):
            continue
        fld = by.get(target)
        if fld is None or fld.value in (None, "", []):
            if fld is None:
                fd = defs.get(target)
                fld = ExtractedField(name=target, type=fd.type if fd else "currency" if isinstance(qv, float) else "string")
                doc.fields.append(fld)
                by[target] = fld
            fld.value, fld.rawText, fld.confidence, fld.providerId = qv, str(qv), QR_CONFIDENCE, src
            fld.page, fld.bbox, fld.grounded, fld.groundedBy = qr.get("page"), qr.get("bbox"), True, "qr"
            fld.validation = FieldValidation()
            checks.append({"field": target, "qr": qv, "read": None, "result": "filled"})
        elif agree(name, fld.value, qv):
            fld.confidence = max(fld.confidence or 0.0, QR_CONFIDENCE)
            if not fld.grounded:
                fld.page, fld.bbox, fld.grounded, fld.groundedBy = qr.get("page"), qr.get("bbox"), True, "qr"
            if HALLUCINATION in fld.validation.errors:   # the QR confirms what the AI read
                fld.validation.errors.remove(HALLUCINATION)
                fld.validation.passed = not fld.validation.errors
            checks.append({"field": target, "qr": qv, "read": fld.value, "result": "match"})
        else:
            msg = f"QR code says {LABELS.get(name, name.replace('_', ' '))} {_fmt(qv)}; page reads {_fmt(fld.value)}"
            if name in _SOFT:
                if msg not in doc.warnings:
                    doc.warnings.append(msg)
                checks.append({"field": target, "qr": qv, "read": fld.value, "result": "differs"})
                continue
            if msg not in fld.validation.errors:
                fld.validation.errors.append(msg)
            fld.validation.passed = False
            fld.confidence = min(fld.confidence if fld.confidence is not None else 0.5, 0.5)
            checks.append({"field": target, "qr": qv, "read": fld.value, "result": "mismatch"})
    doc.einvoice = {"source": src, "standard": qr.get("standard"), "page": qr.get("page"),
                    "bbox": qr["bbox"].model_dump() if qr.get("bbox") else None,
                    "values": qr["values"], "raw": qr.get("raw"), "checks": checks}
    return checks
