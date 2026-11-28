"""Oneoffice DocFlow: the Document Automation component of the Oneoffice RPA platform.

Concepts
--------
* Doc Type   - a reusable definition of one kind of document: fields (form + table), labels in
               English/Arabic, keywords used to recognise it, consistency checks, learned labels.
* Doc Flow   - a business process that documents travel through:
               Intake -> Classify & split -> Extract -> Rules & lookups -> Review -> Deliver.
               A flow can accept one or many Doc Types; it chooses the engine option (1/2/3),
               review thresholds and SLA, rules, and where results are delivered.
* Smart Inbox - a built-in flow that accepts ANY document: every page is classified against all
               Doc Types, mixed batches are split, and each part is extracted with its own type.
* Review Desk - human review of documents flagged by confidence, checks or rules (with SLA).
* Reference Data - lookup tables (supplier master, PO list...) used by lookup rules.

Document statuses: queued, processing, success, validation (in Review Desk), validated, invalid,
failed, split (a batch that was split into child documents).
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, Field

from . import einvoice, events, folders, outbox, pii, scoring
from .provider import Capability
from .errors import DocAutoError, InvalidInput, PolicyDenied
from .extraction import coerce, safe_eval
from .grounding import locations
from .models import DocumentResult, ExtractedField, FieldDef, FieldSchema, ReviewReason, ReviewStatus
from .pipeline import DocumentService, ProcessOptions
from .scoring import review_reasons

log = logging.getLogger("docauto.docflow")
READABLE_MIME = {"application/pdf", "image/png", "image/jpeg", "image/tiff", "image/bmp", "image/webp", "application/xml",
                 "text/xml"}
SUPPORTED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".xml"}
INBOX_ID = "smart-inbox"


def teach_max_samples() -> int:
    """How many example files Teach from samples reads (DOCAUTO_TEACH_MAX_SAMPLES, 5 by default, 1 to 20)."""
    try:
        return max(1, min(20, int(os.environ.get("DOCAUTO_TEACH_MAX_SAMPLES") or 5)))
    except ValueError:
        return 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(name: str, fallback: str = "item") -> str:
    """An id from a name: Latin letters and digits joined by -. A name with none (another script, such as
    "فواتير الموردين") gets the fallback and a short random part, so two such names never share an id."""
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return (s or f"{fallback}-{uuid.uuid4().hex[:8]}")[:40]


def _field_name_ok(name: str) -> bool:
    """A field name: letters of any script, digits and _, not starting with a digit (a Python identifier, so
    checks such as 'total >= subtotal' can use it)."""
    return bool(name) and name.isidentifier()


def _snake(label: str) -> str:
    s = re.sub(r"[^\w]+", "_", label).strip("_").lower()
    return s if s and (s[0].isalpha() or s[0] == "_") else ""


# ================================================================== tax and date checks
# tax presets: rates a country uses and the format of its tax id. A "tax" check takes a preset and/or its own values;
# empty rates = no rate check (mixed-rate invoices), empty idPattern = no format check.
TAX_PRESETS: dict[str, dict[str, Any]] = {
    "ksa": {"name": "Saudi Arabia VAT", "rates": [0.15], "idPattern": r"3\d{13}3"},
    "uae": {"name": "UAE VAT", "rates": [0.05], "idPattern": r"\d{15}"},
    "bahrain": {"name": "Bahrain VAT", "rates": [0.10], "idPattern": r"\d{15}"},
    "oman": {"name": "Oman VAT", "rates": [0.05], "idPattern": r"OM\d{10}"},
    "india-gst": {"name": "India GST", "rates": [0.05, 0.12, 0.18, 0.28, 0.0],
                  "idPattern": r"\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]"},
    "uk": {"name": "United Kingdom VAT", "rates": [0.20, 0.05, 0.0], "idPattern": r"GB(\d{9}|\d{12}|GD\d{3}|HA\d{3})"},
    "eu": {"name": "EU VAT (any member state, standard and reduced rates)", "rates": [],
           "idPattern": r"(AT|BE|BG|CY|CZ|DE|DK|EE|EL|ES|FI|FR|HR|HU|IE|IT|LT|LU|LV|MT|NL|PL|PT|RO|SE|SI|SK|XI)[0-9A-Z+*]{2,12}"},
    "germany": {"name": "Germany VAT", "rates": [0.19, 0.07, 0.0], "idPattern": r"DE\d{9}"},
    "france": {"name": "France VAT", "rates": [0.20, 0.10, 0.055, 0.021, 0.0], "idPattern": r"FR[0-9A-Z]{2}\d{9}"},
    "australia": {"name": "Australia GST", "rates": [0.10, 0.0], "idPattern": r"\d{11}"},
    "new-zealand": {"name": "New Zealand GST", "rates": [0.15, 0.0], "idPattern": r"\d{8,9}"},
    "singapore": {"name": "Singapore GST", "rates": [0.09, 0.0], "idPattern": r"([MT]\d{8}[A-Z]|\d{9}[A-Z]|\d{8}[A-Z])"},
    "canada": {"name": "Canada GST/HST", "rates": [0.05, 0.13, 0.15, 0.0], "idPattern": r"\d{9}(RT\d{4})?"},
    "japan": {"name": "Japan consumption tax", "rates": [0.10, 0.08], "idPattern": r"T\d{13}"},
    "custom": {"name": "Own rates and format", "rates": [], "idPattern": ""},
}


def _rates(v) -> list[float]:
    """Rates as fractions from a list or text ("15, 5" or "0.15"): numbers over 1 are percentages."""
    items = v if isinstance(v, list) else re.split(r"[,;\s]+", str(v or ""))
    out = []
    for x in items:
        try:
            f = float(str(x).strip().rstrip("%"))
        except ValueError:
            continue
        out.append(f / 100 if f > 1 else f)
    return out


def tax_problems(p: dict, vals: dict) -> list[tuple[str, str]]:
    """What a tax check finds: [(field, message)]. p: preset, rates, idPattern, idField, subtotalField, taxField,
    totalField, tolerance (absolute, default 0.05), tolerancePct (default 0.1 %)."""
    pre = TAX_PRESETS.get(str(p.get("preset") or "custom"), TAX_PRESETS["custom"])
    rates = _rates(p["rates"]) if p.get("rates") not in (None, "", []) else list(pre["rates"])
    if "rate" in p and not p.get("rates"):          # ksa_vat processes saved before 3.3 had a single rate
        rates = _rates([p["rate"]])
    id_pattern = p.get("idPattern") if p.get("idPattern") is not None else pre["idPattern"]
    idf = p.get("idField") or p.get("vatNumberField") or "supplier_vat"
    sub = p.get("subtotalField", "subtotal")
    tax = p.get("taxField") or p.get("vatField") or "vat_amount"
    tot = p.get("totalField", "total")
    tol_abs = float(p.get("tolerance", 0.05) or 0.05)
    tol_pct = float(p.get("tolerancePct", 0.001) or 0.001)
    out = []
    num = lambda k: vals.get(k) if isinstance(vals.get(k), (int, float)) and not isinstance(vals.get(k), bool) else None
    vn = re.sub(r"\s", "", str(vals.get(idf) or ""))
    if vn and id_pattern and not re.fullmatch(id_pattern, vn):
        out.append((idf, f"The tax id is not in the {pre['name']} format"))
    s, t, total = num(sub), num(tax), num(tot)
    if s is not None and t is not None and rates and s:
        if not any(abs(round(s * r, 2) - t) <= max(tol_abs, s * tol_pct) for r in rates):
            words = " or ".join(f"{r * 100:g}%" for r in rates)
            out.append((tax, f"Tax {t:,.2f} is not {words} of subtotal {s:,.2f}"))
    if s is not None and t is not None and total is not None:
        if abs(s + t - total) > max(tol_abs, abs(total) * tol_pct):
            out.append((tot, "Total does not equal subtotal + tax"))
    return out


def reply_reason(r) -> str:
    """3.4.3: why a receiver refused a delivery, from its reply (an error message, or the start of its text)."""
    try:
        b = r.json()
        e = b.get("error") if isinstance(b, dict) else None
        msg = (e.get("message") if isinstance(e, dict) else e) or (b.get("message") or b.get("detail") if isinstance(b, dict) else None)
        fix = e.get("suggestedFix") if isinstance(e, dict) else None
        text = " ".join(str(x).strip() for x in (msg, fix) if x)
    except Exception:
        try:
            text = str(getattr(r, "text", "") or "").strip()
        except Exception:
            text = ""
    text = re.sub(r"\s+", " ", text)[:300]
    return f": {text}" if text else ""


def date_problem(v, p: dict) -> Optional[str]:
    """A date check: notFuture, and notOlderThanDays / notLaterThanDays (from today). None when it passes."""
    if v in (None, ""):
        return None
    try:
        d = datetime.fromisoformat(str(v)[:10]).date()
    except ValueError:
        return "is not a date"
    today = datetime.now(timezone.utc).date()
    # a day of grace: a document dated today east of UTC is still tomorrow in UTC
    if p.get("notFuture") and d > today + timedelta(days=1):
        return "is in the future"
    older = p.get("notOlderThanDays")
    if older not in (None, "") and (today - d).days > int(older):
        return f"is more than {int(older)} days ago"
    later = p.get("notLaterThanDays")
    if later not in (None, "") and (d - today).days > int(later):
        return f"is more than {int(later)} days ahead"
    return None


# ================================================================== models
class LearnedLabels(BaseModel):
    enabled: bool = True
    aliases: dict[str, list[str]] = Field(default_factory=dict)


class DocType(BaseModel):
    id: str = ""
    name: str
    description: str = ""
    category: str = "custom"           # invoice | purchase_order | receipt | bank_statement | id_document | custom
    languages: list[str] = Field(default_factory=list)   # empty: the languages offered on this server
    dateOrder: Optional[str] = None     # DMY | MDY | YMD: how numeric dates of this type are read (None: server setting)
    decimalMark: Optional[str] = None   # "." | ",": the decimal mark of this type's amounts (None: server setting)
    keywords: list[str] = Field(default_factory=list)   # words that identify this type on a page
    fields: list[FieldDef] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)     # cross-field expressions
    learning: LearnedLabels = Field(default_factory=LearnedLabels)
    # teaching examples: approved documents whose right values AI reading sees as examples (like labelled samples)
    examples: list[str] = Field(default_factory=list)
    version: int = 1
    createdAt: str = ""
    updatedAt: str = ""

    def schema(self) -> FieldSchema:
        learned = self.learning.aliases if self.learning.enabled else {}
        fields = []
        for f in self.fields:
            f2 = f.model_copy(deep=True)
            f2.aliases += [a for a in learned.get(f.name, []) if a not in f2.aliases]
            fields.append(f2)
        return FieldSchema(id=f"doctype:{self.id}", name=self.name, description=self.description,
                           documentLanguages=self.languages, fields=fields, validation=self.checks)

    def recognition_terms(self) -> list[tuple[str, float]]:
        """Weighted terms used to recognise this type on a page."""
        terms: list[tuple[str, float]] = [(k, 3.0) for k in self.keywords]
        terms += [(w, 1.5) for w in re.split(r"[\s/_-]+", self.name) if len(w) > 3]
        for f in self.fields:
            for part in (f.label or "").split("/"):
                if part.strip():
                    terms.append((part.strip(), 2.0 if f.required else 1.0))
            terms += [(a, 0.8) for a in f.aliases]
        return terms


class EngineChoice(BaseModel):
    option: Literal["tier1_opensource", "tier2_cloud", "tier3_ai"] = "tier1_opensource"
    ocrProvider: str = "auto"
    extractionProvider: str = "auto"
    capabilities: list[str] = Field(default_factory=lambda: ["ocr_printed"])
    # cloud AI models: mask IDs, IBANs, phones and e-mails in the text (None = the server/tenant setting, on).
    # Page images cannot be masked: with masking on they are only sent when allowImages is true.
    maskPii: Optional[bool] = None
    allowImages: bool = False
    # most to spend on cloud and AI engines in a calendar month (in the price currency, docauto.costs); once reached,
    # the process reads with the engines on this server until the month ends. None = no limit
    monthlyBudget: Optional[float] = None
    # an engine per document type of the process (doc type id -> extraction engine id); others use extractionProvider
    extractionByType: dict[str, str] = Field(default_factory=dict)


class ClassifySettings(BaseModel):
    mode: Literal["single", "auto"] = "single"   # single: every document is the flow's (first) doc type
    split: bool = False                          # split mixed batches into one document per part
    minScore: float = 2.0                        # below this a page is 'unclassified' -> Review Desk
    # a document that fits none of the types: review (a person chooses the type), other (kept as an Other document,
    # its text and labelled values, and delivered), reject (marked invalid, with the reason)
    whenNoType: Literal["review", "other", "reject"] = "review"
    # single mode: check the document looks like the process's type (its recognition words score minScore or more);
    # one that does not is handled like a document that fits no type
    checkType: bool = False


class ReviewSettings(BaseModel):
    # 3.4.1: how much goes to a person. unsure: values under the confidence thresholds (as before); all: every
    # document, for a new process's first weeks; none: nothing for confidence alone (failed checks and missing or
    # invalid values still go)
    mode: Literal["unsure", "all", "none"] = "unsure"
    fieldConfidence: float = 0.85
    documentConfidence: float = 0.80
    fieldThresholds: dict[str, float] = Field(default_factory=dict)   # {field name: 0-1}, over fieldConfidence
    slaHours: float = 24
    slaWebhook: str = ""


class Rule(BaseModel):
    id: str = ""
    # ksa_vat is the "tax" check with the Saudi preset (kept for processes saved before 3.3)
    type: Literal["duplicate", "lookup", "ksa_vat", "tax", "threshold", "required", "expression", "pattern", "date_range"]
    name: str = ""
    enabled: bool = True
    # route: the document goes to another process (params routeTo) instead; notify: people hear of it (params notify:
    # addresses, user names, role:, team:) and the document goes on as with warn
    action: Literal["review", "reject", "warn", "route", "notify"] = "review"
    docTypes: list[str] = Field(default_factory=list)   # empty = all doc types of the flow
    params: dict[str, Any] = Field(default_factory=dict)


class Delivery(BaseModel):
    id: str = ""
    # email: sent by the Command Center (its mail settings); cloud_storage: files PUT to S3 / Azure Blob / a web address
    type: Literal["webhook", "folder", "excel", "start_bot", "email", "cloud_storage"]
    name: str = ""
    enabled: bool = True
    on: list[str] = Field(default_factory=lambda: ["success", "validated"])
    params: dict[str, Any] = Field(default_factory=dict)


class Intake(BaseModel):
    inputFolder: str = ""
    pollSeconds: int = 30
    # email inbox, read by the OOA Command Center (IMAP or Microsoft 365; secrets are its vault references)
    email: dict[str, Any] = Field(default_factory=dict)
    # e-invoices first: a UBL XML (alone or inside a PDF/A-3) is read instead of OCR, and a ZATCA / India GST
    # QR code on the page cross-checks the extracted fields. Only acts when such data is found.
    einvoiceFirst: bool = True
    einvoiceStandards: list[str] = Field(default_factory=list)   # e-invoice standards read (empty = every one)
    upload: bool = True   # people may add documents on the screens (agents, e-mail and folders are not affected)


class Flow(BaseModel):
    id: str = ""
    name: str
    description: str = ""
    folder: str = ""                                    # where it sits in the platform, e.g. "Finance / Accounts payable"
    docTypes: list[str] = Field(default_factory=list)   # Smart Inbox: empty = all doc types
    classify: ClassifySettings = Field(default_factory=ClassifySettings)
    engine: EngineChoice = Field(default_factory=EngineChoice)
    review: ReviewSettings = Field(default_factory=ReviewSettings)
    rules: list[Rule] = Field(default_factory=list)
    deliveries: list[Delivery] = Field(default_factory=list)
    intake: Intake = Field(default_factory=Intake)
    status: Literal["draft", "live"] = "draft"
    isInbox: bool = False
    version: int = 1
    createdAt: str = ""
    updatedAt: str = ""


# ================================================================== doc type library
def _f(name, type_="string", label=None, required=False, aliases=None, regex=None, fmt=None, enum=None, columns=None,
       fallback=None):
    d = {"name": name, "type": type_, "label": label, "required": required, "aliases": aliases or []}
    if fallback:
        d["fallback"] = fallback
    if regex:
        d["regex"] = regex
    if fmt:
        d["format"] = fmt
    if enum:
        d["enumValues"] = enum
    if columns:
        d["columns"] = columns
    return d


LINE_ITEMS = _f("line_items", "table", "Line items", columns=[
    _f("description", label="Description", aliases=["Item", "Items", "Details", "Particulars", "Article", "Artikel", "Désignation",
                                                     "Descripción", "Descrição", "Descrizione", "Omschrijving", "البيان", "الصنف",
                                                     "品名", "摘要", "विवरण"]),
    _f("quantity", "number", "Quantity", aliases=["Qty", "Qty.", "Menge", "Quantité", "Cantidad", "Quantidade", "Quantità",
                                                   "Aantal", "الكمية", "数量", "मात्रा"]),
    _f("unit_price", "currency", "Unit price", aliases=["Price", "Rate", "Unit Cost", "Einzelpreis", "Prix unitaire",
                                                         "Precio unitario", "Preço unitário", "Prezzo unitario", "السعر", "单价", "単価", "दर"]),
    _f("amount", "currency", "Amount", aliases=["Line Total", "Net Amount", "Total", "Betrag", "Montant", "Importe", "Valor",
                                                 "Importo", "Bedrag", "المبلغ", "الإجمالي", "金额", "金額", "राशि"])])

from .config import env_number  # noqa: E402
from .words import CURRENCIES, ISO_CURRENCIES, TAX_ID_WORDS, TAX_WORDS  # noqa: E402  (word lists shared with instances)

def _cur(name="currency"):
    d = _f(name, "enum", "Currency", aliases=["Currency Code", "Währung", "Devise", "Moneda", "Moeda", "Valuta", "العملة",
                                               "币种", "通貨", "मुद्रा"], enum=CURRENCIES)
    d["allowOther"] = True
    return d


# the invoice's amounts add up: subtotal - discount + other charges + tax = total, to a cent or 0.1% (a missing
# subtotal, discount, charge or tax counts as nothing, so invoices without them pass)
INVOICE_TOTAL_CHECK = ("abs(value_or(subtotal, total - value_or(vat_amount, 0)) - value_or(discount, 0) + value_or(other_charges, 0)"
                       " + value_or(vat_amount, 0) - total) <= max(0.05, abs(total) * 0.001)")
OLD_INVOICE_CHECK = "abs(subtotal + vat_amount - total) < 0.05"
scoring.CHECK_NAMES[INVOICE_TOTAL_CHECK] = "Amounts add up (subtotal - discount + charges + tax = total)"
scoring.CHECK_NAMES[OLD_INVOICE_CHECK] = "Amounts add up (subtotal + tax = total)"
scoring.CHECK_NAMES["abs(value_or(subtotal, total - value_or(vat_amount, 0)) + value_or(vat_amount, 0) - total) <= max(0.05, abs(total) * 0.001)"] =     "Amounts add up (subtotal + tax = total)"

LIBRARY: dict[str, dict[str, Any]] = {
    "invoice": {"name": "Invoice", "description": "Bills from suppliers asking for payment: tax invoices, commercial invoices.",
                "keywords": ["invoice", "tax invoice", "Rechnung", "facture", "factura", "fatura", "fattura", "factuur",
                             "faktura", "فاتورة", "发票", "請求書", "चालान", "VAT", "GST"],
                "fields": [
                    _f("invoice_number", label="Invoice number", required=True,
                       aliases=["Invoice No", "Invoice #", "Bill No", "Rechnungsnummer", "Rechnung Nr", "N° de facture",
                                "Numéro de facture", "Número de factura", "Nº Fatura", "Numero fattura", "Factuurnummer",
                                "رقم الفاتورة", "发票号码", "請求書番号", "चालान संख्या"]),
                    _f("invoice_date", "date", "Invoice date", True, ["Date", "Issue Date", "Rechnungsdatum", "Date de facture",
                                                                       "Fecha", "Data", "Factuurdatum", "تاريخ الفاتورة", "开票日期", "発行日"]),
                    _f("due_date", "date", "Due date", aliases=["Payment Due", "Fällig", "Échéance", "Vencimiento", "Scadenza",
                                                                 "Vervaldatum", "تاريخ الاستحقاق", "支払期限"]),
                    _f("supplier_name", label="Supplier", aliases=["Vendor", "Seller", "From", "Lieferant", "Fournisseur",
                                                                    "Proveedor", "Fornecedor", "Fornitore", "Leverancier", "المورد",
                                                                    "销售方", "供应商"], fallback="header_company"),
                    _f("supplier_vat", label="Supplier tax ID", aliases=TAX_ID_WORDS),
                    _f("customer_name", label="Customer", aliases=["Bill To", "Buyer", "Sold To", "Kunde", "Client", "Cliente",
                                                                    "Klant", "العميل", "购买方"]),
                    _f("customer_vat", label="Customer tax ID", aliases=["Buyer VAT", "Customer VAT No", "Buyer Tax ID",
                                                                          "الرقم الضريبي للعميل"]),
                    _f("po_number", label="PO number", aliases=["Purchase Order", "PO No", "Order No", "Bestellnummer",
                                                                 "Bon de commande", "Orden de compra", "رقم أمر الشراء"]),
                    _f("subtotal", "currency", "Subtotal", False, ["Sub Total", "Net Amount", "Net", "Nettobetrag", "Total HT",
                                                                   "Base imponible", "Imponibile", "المجموع الفرعي", "小计"]),
                    _f("discount", "currency", "Discount", aliases=["Rabatt", "Remise", "Descuento", "Desconto", "Sconto",
                                                                     "Korting", "خصم", "折扣"]),
                    _f("other_charges", "currency", "Other charges", aliases=["Shipping", "Freight", "Delivery", "Handling",
                                                                               "Versand", "Frais de port", "Envío", "Frete",
                                                                               "رسوم الشحن", "运费"]),
                    _f("vat_amount", "currency", "Tax amount", False, TAX_WORDS),
                    _f("total", "currency", "Total", True, ["Total Amount", "Grand Total", "Amount Due", "Balance Due",
                                                             "Gesamtbetrag", "Total TTC", "Importe total", "Valor total",
                                                             "Totale", "Totaal", "الإجمالي", "合计", "合計", "कुल"]),
                    _cur(), LINE_ITEMS],
                "checks": [INVOICE_TOTAL_CHECK]},
    "credit_note": {"name": "Credit note", "description": "Credit notes from suppliers that reduce what we owe.",
                    "keywords": ["credit note", "credit memo", "Gutschrift", "avoir", "nota de crédito", "nota di credito",
                                 "creditnota", "إشعار دائن"],
                    "fields": [_f("credit_note_number", label="Credit note number", required=True, aliases=["Credit Note No", "Credit Memo No"]),
                               _f("credit_note_date", "date", "Date", True, ["Issue Date"]),
                               _f("original_invoice", label="Original invoice", aliases=["Invoice No", "Reference"]),
                               _f("supplier_name", label="Supplier", aliases=["Vendor", "Seller"], fallback="header_company"),
                               _f("vat_amount", "currency", "Tax amount", False, TAX_WORDS),
                               _f("total", "currency", "Total", True, ["Total Credit", "Amount", "Grand Total"]), _cur()],
                    "checks": []},
    "purchase_order": {"name": "Purchase order", "description": "Purchase orders issued to vendors.",
                       "keywords": ["purchase order", "P.O.", "Bestellung", "bon de commande", "orden de compra",
                                    "pedido de compra", "ordine di acquisto", "inkooporder", "أمر شراء", "采购订单", "注文書"],
                       "fields": [
                           _f("po_number", label="PO number", required=True, aliases=["Purchase Order No", "PO No", "Order No",
                                                                                     "Bestellnummer", "رقم أمر الشراء"]),
                           _f("po_date", "date", "PO date", True, ["Date", "Order Date", "Bestelldatum", "التاريخ"]),
                           _f("vendor_name", label="Vendor", aliases=["Supplier", "Lieferant", "Fournisseur", "Proveedor", "المورد"]),
                           _f("delivery_date", "date", "Delivery date", aliases=["Liefertermin", "Date de livraison", "تاريخ التسليم"]),
                           _f("total", "currency", "Total", True, ["Total Amount", "Grand Total", "Gesamt", "الإجمالي"]),
                           _cur(), LINE_ITEMS],
                       "checks": []},
    "delivery_note": {"name": "Delivery note", "description": "Delivery notes / packing slips that come with goods.",
                      "keywords": ["delivery note", "packing slip", "goods received", "Lieferschein", "bon de livraison",
                                   "albarán", "guia de remessa", "bolla", "pakbon", "إشعار تسليم", "送货单", "納品書"],
                      "fields": [_f("delivery_number", label="Delivery note number", required=True, aliases=["Delivery No", "DN No", "Lieferschein-Nr"]),
                                 _f("delivery_date", "date", "Delivery date", True, ["Date", "Ship Date"]),
                                 _f("po_number", label="PO number", aliases=["Order No", "Purchase Order"]),
                                 _f("supplier_name", label="Supplier", aliases=["Shipper", "Vendor"], fallback="header_company"),
                                 _f("line_items", "table", "Items", columns=[
                                     _f("description", label="Description", aliases=["Item", "Article"]),
                                     _f("quantity", "number", "Quantity", aliases=["Qty", "Delivered"])])],
                      "checks": []},
    "receipt": {"name": "Receipt", "description": "Retail / expense receipts.",
                "keywords": ["receipt", "Quittung", "Beleg", "reçu", "recibo", "ricevuta", "bon", "إيصال", "收据", "領収書",
                             "cashier", "thank you"],
                "fields": [_f("merchant_name", label="Merchant", aliases=["Store", "Shop", "التاجر"], fallback="header_company"),
                           _f("receipt_date", "date", "Date", True, ["Datum", "Fecha", "التاريخ"]),
                           _f("vat_amount", "currency", "Tax amount", aliases=TAX_WORDS),
                           _f("total", "currency", "Total", True, ["Amount", "Total Paid", "Summe", "Total à payer", "الإجمالي"]),
                           _cur(),
                           {**_f("payment_method", "enum", "Payment method",
                                 enum=["Cash", "Card", "Bank transfer", "Cheque", "Mobile wallet", "UPI", "Pix", "iDEAL",
                                       "Apple Pay", "Google Pay", "Mada", "Other"]), "allowOther": True}],
                "checks": []},
    "bank_statement": {"name": "Bank statement", "description": "Account statements with balances and transactions.",
                       "keywords": ["statement", "account statement", "IBAN", "opening balance", "Kontoauszug",
                                    "relevé de compte", "extracto", "extrato", "estratto conto", "rekeningafschrift",
                                    "كشف حساب", "对账单"],
                       "fields": [_f("bank_name", label="Bank", aliases=["البنك"]),
                                  _f("account_number", label="Account number", required=True,
                                     aliases=["Account No", "IBAN", "A/C No", "Kontonummer", "Numéro de compte", "Número de cuenta",
                                              "رقم الحساب", "رقم الآيبان", "账号"]),
                                  _f("period_from", "date", "From", aliases=["Period from", "Von", "Du", "Desde", "من"]),
                                  _f("period_to", "date", "To", aliases=["Period to", "Bis", "Au", "Hasta", "إلى"]),
                                  _f("opening_balance", "currency", "Opening balance", True, ["Previous Balance", "Anfangssaldo",
                                                                                               "Solde initial", "Saldo inicial",
                                                                                               "الرصيد الافتتاحي"]),
                                  _f("closing_balance", "currency", "Closing balance", True, ["New Balance", "Endsaldo", "Solde final",
                                                                                               "Saldo final", "الرصيد الختامي"]),
                                  _cur(),
                                  _f("transactions", "table", "Transactions", columns=[
                                      _f("date", "date", "Date"), _f("description", label="Description"),
                                      _f("debit", "currency", "Debit"), _f("credit", "currency", "Credit"),
                                      _f("balance", "currency", "Balance")])],
                       "checks": []},
    "id_document": {"name": "ID document", "description": "National ID cards, residence permits and passport data pages.",
                    "keywords": ["identity", "identity card", "passport", "residence permit", "Personalausweis", "Reisepass",
                                 "carte d'identité", "passeport", "documento de identidad", "pasaporte", "هوية", "جواز",
                                 "iqama", "Aadhaar", "nationality", "身份证", "護照"],
                    "fields": [_f("full_name", label="Name", required=True, aliases=["Surname", "Given names", "Name", "Nom", "Nombre", "الاسم"]),
                               _f("id_number", label="ID number", required=True,
                                  aliases=["ID No", "Document No", "Passport No", "Card No", "Ausweisnummer", "N° du document",
                                           "Número de documento", "Iqama No", "رقم الهوية"]),
                               _f("nationality", label="Nationality", aliases=["Staatsangehörigkeit", "Nationalité", "Nacionalidad", "الجنسية"]),
                               _f("date_of_birth", "date", "Date of birth", aliases=["Birth date", "Geburtsdatum", "Date de naissance",
                                                                                      "Fecha de nacimiento", "تاريخ الميلاد"]),
                               _f("expiry_date", "date", "Expiry date", True, ["Valid until", "Date of expiry", "Gültig bis",
                                                                               "Date d'expiration", "Fecha de caducidad", "تاريخ الانتهاء"])],
                    "checks": []},
    "utility_bill": {"name": "Utility bill", "description": "Electricity, water, gas, phone and internet bills.",
                     "keywords": ["electricity", "water", "gas", "utility", "meter", "kWh", "Stromrechnung", "facture d'électricité",
                                  "factura de luz", "فاتورة كهرباء"],
                     "fields": [_f("provider_name", label="Provider", aliases=["Supplier"], fallback="header_company"),
                                _f("account_number", label="Account number", required=True, aliases=["Customer No", "Account No", "Kundennummer"]),
                                _f("bill_date", "date", "Bill date", True, ["Date", "Invoice Date"]),
                                _f("due_date", "date", "Due date", aliases=["Pay by", "Payment Due"]),
                                _f("total", "currency", "Amount due", True, ["Total", "Total Due", "Amount Payable"]), _cur()],
                     "checks": []},
    "contract": {"name": "Contract", "description": "Agreements between two or more parties.",
                 "keywords": ["agreement", "contract", "hereinafter", "parties", "Vertrag", "contrat", "contrato", "contratto",
                              "overeenkomst", "عقد", "合同", "契約"],
                 "fields": [_f("title", label="Title", fallback="header_company"),
                            {**_f("parties", label="Parties", required=True), "method": "infer",
                             "description": "The names of the parties to the agreement, separated by semicolons"},
                            _f("effective_date", "date", "Effective date", aliases=["Commencement Date", "Start Date"]),
                            _f("end_date", "date", "End date", aliases=["Expiry Date", "Termination Date"]),
                            {**_f("summary", label="Summary"), "method": "infer",
                             "description": "What the contract is about, in one sentence"}],
                 "checks": []},
    "other": {"name": "Other document", "description": "Any document that is none of the other types: its text, labelled values "
                                                       "and tables are kept, with no fixed fields.",
              "keywords": [],
              "fields": [{**_f("title", label="Title"), "method": "infer",
                          "description": "The document's title or what kind of document it is, in a few words"},
                         _f("document_date", "date", "Date", aliases=["Datum", "Fecha", "التاريخ"]),
                         {**_f("summary", label="Summary"), "method": "infer",
                          "description": "What the document is about, in one sentence"}],
              "checks": []},
}
# built-in types seeded by DocFlow before 3.3 carried Saudi-only defaults; _upgrade_global() replaces exactly these
OLD_SAUDI_REGEX = {r"3\d{14}", r"SA\d{2}[A-Z0-9]{18}", r"[12]\d{9}|[A-Z]\d{7,8}"}
OLD_CURRENCIES = ["SAR", "AED", "USD", "EUR", "GBP"]
OLD_PAYMENT = ["Cash", "Card", "Mada", "Apple Pay", "Other"]


def library() -> list[dict]:
    return [{"category": k, **v} for k, v in LIBRARY.items()]


# ================================================================== rules
def run_rules(store, flow: Flow, doctype: Optional[DocType], doc_id: str, result: DocumentResult,
              refs: "RefData") -> list[dict]:
    """Return flags: {rule, type, action, message, field?}. May enrich result fields (lookups)."""
    flags: list[dict] = []
    vals = {f.name: f.value for f in result.fields}
    by = {f.name: f for f in result.fields}

    def flag(rule: Rule, message: str, field: Optional[str] = None):
        flags.append({"rule": rule.name or rule.type, "type": rule.type, "action": rule.action,
                      "message": message, "field": field,
                      **({"routeTo": rule.params.get("routeTo")} if rule.action == "route" else {}),
                      **({"notify": rule.params.get("notify"), "requestedBy": rule.params.get("requestedBy")}
                         if rule.action == "notify" else {})})
        if field and field in by and rule.action not in ("warn", "notify"):
            by[field].validation.passed = False
            if message not in by[field].validation.errors:
                by[field].validation.errors.append(message)

    for r in flow.rules:
        if not r.enabled or (r.docTypes and (not doctype or doctype.id not in r.docTypes)):
            continue
        p = r.params
        try:
            if r.type == "duplicate":
                keys = [k for k in p.get("fields", []) if k]
                if keys and all(vals.get(k) not in (None, "") for k in keys):
                    key = "|".join(str(vals[k]).strip().lower() for k in keys)
                    other = store.dedupe_find(flow.id, key, doc_id)
                    if other:
                        flag(r, f"Possible duplicate of document {other[:8]} (same {', '.join(keys)})", keys[0])
                    store.dedupe_add(flow.id, key, doc_id)
            elif r.type == "lookup":
                field, table, col = p.get("field"), p.get("table"), p.get("column")
                v = vals.get(field)
                if field and table and col and v not in (None, ""):
                    row = refs.find(table, col, v)
                    wc, wv = str(p.get("whereColumn") or "").strip(), str(p.get("whereValue") or "").strip()
                    if row is None:
                        flag(r, p.get("message") or f"{field} '{v}' not found in {refs.name(table)}", field)
                    elif wc and str(row.get(wc, "")).strip().lower() != wv.lower():
                        # 3.4.3: the row must also have this value in a column (e.g. Status = active)
                        flag(r, f"{field} '{v}' is in {refs.name(table)}, but its {wc} is '{row.get(wc, '')}', not '{wv}'", field)
                    else:
                        for target, src in (p.get("enrich") or {}).items():
                            if src in row and row[src] not in (None, ""):
                                if target in by:
                                    if by[target].value in (None, ""):
                                        by[target].value, by[target].confidence = row[src], 1.0
                                        by[target].providerId = f"lookup:{table}"
                                    elif _same_text(by[target].value, row[src]) and by[target].validation.passed:
                                        by[target].confidence = 1.0   # the reference data confirms what was read
                                else:
                                    nf = ExtractedField(name=target, value=row[src], confidence=1.0,
                                                        providerId=f"lookup:{table}")
                                    result.fields.append(nf)
                                    by[target] = nf
            elif r.type in ("ksa_vat", "tax"):
                for field, message in tax_problems(p if r.type == "tax" else {"preset": "ksa", **p}, vals):
                    flag(r, message, field)
            elif r.type == "pattern":
                field, pattern = p.get("field"), str(p.get("pattern") or "")
                v = vals.get(field)
                if pattern and v not in (None, "") and not re.fullmatch(pattern, str(v).strip()):
                    flag(r, p.get("message") or f"{field} '{v}' is not in the expected format", field)
            elif r.type == "date_range":
                field = p.get("field")
                problem = date_problem(vals.get(field), p)
                if problem:
                    flag(r, p.get("message") or f"{field} {problem}", field)
            elif r.type == "threshold":
                field, op, limit = p.get("field"), p.get("op", ">"), float(p.get("value", 0))
                v = vals.get(field)
                if isinstance(v, (int, float)):
                    hi = float(p.get("value2", limit) or limit)
                    hit = {">": v > limit, ">=": v >= limit, "<": v < limit, "<=": v <= limit, "==": v == limit,
                           "!=": v != limit, "between": min(limit, hi) <= v <= max(limit, hi),
                           "outside": not (min(limit, hi) <= v <= max(limit, hi))}.get(op, False)
                    if hit:
                        flag(r, p.get("message") or f"{field} {v:,.2f} {op} {limit:,.2f}", field)
            elif r.type == "required":
                for field in p.get("fields", []):
                    if vals.get(field) in (None, "", []):
                        flag(r, f"{field} is required", field)
            elif r.type == "expression":
                expr = p.get("expression", "")
                if expr and not safe_eval(expr, vals):
                    flag(r, p.get("message") or f"Check failed: {expr}")
        except Exception as e:  # a broken rule must not break processing
            flag(r, f"Rule could not run: {type(e).__name__}: {e}")
            flags[-1]["action"] = "review"
    return flags


def _same_text(a: Any, b: Any) -> bool:
    return re.sub(r"\s+", " ", str(a)).strip().casefold() == re.sub(r"\s+", " ", str(b)).strip().casefold()


UNCLASSIFIED = ReviewReason(reason="unclassified", message="The document type was not recognised: choose it")


def _flag_reasons(flags: list[dict]) -> list[ReviewReason]:
    """The process's rules that sent the document to review (or rejected it), as review reasons."""
    return [ReviewReason(field=f.get("field"), reason="rule", rule=f.get("rule"), message=f.get("message", ""))
            for f in flags if f.get("action") not in ("warn", "notify", "route")]


def _merge_reasons(*groups: list[ReviewReason]) -> list[ReviewReason]:
    """Reasons in order, each once (a rule's message is also the field's validation error)."""
    out, seen = [], set()
    for r in (r for g in groups for r in g):
        k = (r.field, r.message.casefold())
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


class RefData:
    def __init__(self, store):
        self.store = store
        self._cache: dict[str, dict] = {}

    def table(self, ref_id: str) -> Optional[dict]:
        if ref_id not in self._cache:
            self._cache[ref_id] = self.store.ref_get(ref_id)
        return self._cache[ref_id]

    def name(self, ref_id: str) -> str:
        t = self.table(ref_id)
        return t["name"] if t else ref_id

    def find(self, ref_id: str, column: str, value: Any) -> Optional[dict]:
        t = self.table(ref_id)
        if not t:
            return None
        want = re.sub(r"\s+", " ", str(value)).strip().lower()
        for row in t["rows"]:
            if re.sub(r"\s+", " ", str(row.get(column, ""))).strip().lower() == want:
                return row
        return None


def document_markdown(res: DocumentResult, body: Optional[dict] = None) -> str:
    """A read document as Markdown: its name and type, the values read, then each page's text with its tables as
    Markdown tables. A Knowledge base's indexing agent (Knowledge Q&A, Build index) reads these files."""
    body = body or {}
    out = [f"# {body.get('fileName') or getattr(res, 'fileName', None) or res.documentId}", ""]
    if body.get("docTypeName"):
        out += [f"Document type: {body['docTypeName']}", ""]
    vals = [(f.name, f.out()) for f in res.fields if f.value not in (None, "", []) and f.type != "table"]
    if vals:
        out += ["## Values read", ""] + [f"- **{n}**: {v}" for n, v in vals] + [""]
    for p in res.pages:
        out += [f"## Page {p.pageNumber}", ""] + [l.text for l in p.lines] + [""]
        for t in p.tables:
            rows = [[c.text.replace("|", "/").replace("\n", " ") for c in sorted(r, key=lambda c: c.col)]
                    for r in _table_rows(t)]
            if rows:
                out += ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] + \
                       ["| " + " | ".join(r) + " |" for r in rows[1:]] + [""]
    return "\n".join(out)


def _table_rows(t) -> list[list]:
    rows: dict[int, list] = {}
    for c in getattr(t, "cells", []):
        rows.setdefault(c.row, []).append(c)
    return [rows[k] for k in sorted(rows)]


def parse_table_file(data: bytes, file_name: str, header_row: int = 1) -> tuple[list[str], list[dict]]:
    """A reference table from CSV / TSV / text (any of , ; tab | as separator), Excel (.xlsx, .xlsm), OpenDocument
    (.ods) or JSON (a list of objects, or {"rows": [...]}). header_row: the row with the column names (1 = the first;
    0 = none: the columns are named col1, col2...)."""
    low = file_name.lower()
    if low.endswith(".json"):
        try:
            obj = json.loads(data.decode("utf-8-sig"))
        except ValueError as e:
            raise InvalidInput(f"the JSON file cannot be read: {e}")
        items = obj.get("rows", obj.get("data")) if isinstance(obj, dict) else obj
        if not isinstance(items, list) or not all(isinstance(x, dict) for x in items) or not items:
            raise InvalidInput("a JSON table is a list of objects, one per row")
        head = list(dict.fromkeys(k for x in items for k in x))
        return head, [{h: ("" if x.get(h) is None else str(x.get(h))) for h in head} for x in items]
    if low.endswith((".xlsx", ".xlsm")):
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        ws = wb.active
        rows = [[("" if c is None else str(c)).strip() for c in r] for r in ws.iter_rows(values_only=True)]
    elif low.endswith(".ods"):
        rows = _ods_rows(data)
    else:
        text = data.decode("utf-8-sig", errors="replace")
        try:
            # the separator from the rows from the heading row on (a title line above it has none)
            sample = "\n".join(text.splitlines()[max(0, header_row - 1):][:30])[:4000]
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|") if sample.strip() else csv.excel
        except csv.Error:
            dialect = csv.excel_tab if "\t" in text[:2000] else csv.excel
        rows = [[c.strip() for c in r] for r in csv.reader(io.StringIO(text), dialect)]
    rows = [r for r in rows if any(r)]
    if not rows:
        raise InvalidInput("the file is empty")
    if header_row <= 0:
        width = max(len(r) for r in rows)
        head = [f"col{i + 1}" for i in range(width)]
        return head, [dict(zip(head, r)) for r in rows]
    if header_row > len(rows):
        raise InvalidInput(f"the file has {len(rows)} rows, so row {header_row} cannot hold the column names")
    head = [h or f"col{i + 1}" for i, h in enumerate(rows[header_row - 1])]
    return head, [dict(zip(head, r)) for r in rows[header_row:]]


def _ods_rows(data: bytes) -> list[list[str]]:
    """The first sheet of an OpenDocument spreadsheet, read with the standard library (zip + XML)."""
    import xml.etree.ElementTree as ETree
    import zipfile
    ns = {"table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0", "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0"}
    try:
        root = ETree.fromstring(zipfile.ZipFile(io.BytesIO(data)).read("content.xml"))
    except (zipfile.BadZipFile, KeyError, ETree.ParseError):
        raise InvalidInput("the .ods file cannot be read")
    sheet = root.find(".//table:table", ns)
    rows = []
    for tr in (sheet.findall(".//table:table-row", ns) if sheet is not None else []):
        row = []
        for tc in tr.findall("table:table-cell", ns):
            n = int(tc.get("{%s}number-columns-repeated" % ns["table"], "1"))
            text = " ".join("".join(p.itertext()) for p in tc.findall("text:p", ns)).strip()
            row.extend([text] * min(n, 200))
        while row and not row[-1]:
            row.pop()
        rows.append(row)
    return rows


# ================================================================== classification & split
class _PageText:
    """A page's text as an engine's classify() reads it (it uses .text only)."""
    def __init__(self, text: str):
        self.text = text


def classify_page(text: str, doctypes: list[DocType]) -> tuple[Optional[str], float, dict[str, float]]:
    low = text.lower()
    scores = {}
    for dt in doctypes:
        s = 0.0
        for term, w in dt.recognition_terms():
            t = term.lower().strip()
            if len(t) >= 2 and re.search(r"(?<![\w])" + re.escape(t) + r"(?![\w])", low):
                s += w
        scores[dt.id] = round(s, 2)
    if not scores:
        return None, 0.0, {}
    best = max(scores, key=scores.get)
    return best, scores[best], scores


def start_anchor(dt: DocType) -> Optional[str]:
    req = next((f for f in dt.fields if f.required and f.type != "table"), None)
    if not req or not req.label:
        return None
    return req.label.split("/")[0].strip().lower()


#: a dry run of a process (3.4.1): read, checked and forgotten (no events, deliveries, routing or messages)
TRY_SOURCE = "try"
#: what the delivery log says for a delivery a draft process holds back (3.4.1)
HELD = "Held: this process is a draft"
#: a delivery header value read from the Command Center's vault at delivery time (3.4.1): "vault:folder/credential/attribute"
VAULT_PREFIX = "vault:"

#: the reasons that are only about how sure an engine was (left out when a process reviews nothing for confidence)
CONFIDENCE_ONLY = ("low_confidence", "document_confidence")
EVERY_DOCUMENT = ReviewReason(reason="review_all", message="Every document is reviewed in this process")


def review_mode(mode: str, reasons: list) -> list:
    """The reasons for review as the process's review mode has them (3.4.1): all adds one when there is none;
    none drops the confidence-only ones."""
    if mode == "none":
        return [r for r in reasons if r.reason not in CONFIDENCE_ONLY]
    if mode == "all" and not reasons:
        return [EVERY_DOCUMENT]
    return reasons


def split_pages(pages_text: list[str], doctypes: list[DocType], min_score: float, split: bool,
                labels: Optional[list[tuple[Optional[str], float]]] = None) -> list[dict]:
    """Group pages into documents: [{doctype, pages:[1..], score}]. doctype None = unclassified. `labels`: each page's
    type and score as an engine recognised them (3.4.0); without them, the words on the page."""
    by_id = {d.id: d for d in doctypes}
    segs: list[dict] = []
    for i, txt in enumerate(pages_text, 1):
        if labels is not None:
            dt_id, score = labels[i - 1]
        else:
            dt_id, score, _ = classify_page(txt, doctypes)
            if score < min_score:
                dt_id = None
        anchor = start_anchor(by_id[dt_id]) if dt_id else None
        starts_new = bool(anchor and anchor in txt.lower())
        cur = segs[-1] if segs else None
        if not split:
            if cur is None:
                segs.append({"doctype": dt_id, "pages": [i], "score": score})
            else:
                cur["pages"].append(i)
                if score > cur["score"]:
                    cur["doctype"], cur["score"] = dt_id, score
            continue
        if cur and (dt_id == cur["doctype"] or (dt_id is None and cur["doctype"] is not None and not starts_new)) \
                and not (starts_new and cur.get("anchored")):
            cur["pages"].append(i)
            cur["anchored"] = cur.get("anchored") or starts_new
        else:
            segs.append({"doctype": dt_id, "pages": [i], "score": score, "anchored": starts_new})
    for s in segs:
        s.pop("anchored", None)
    return segs


def _plain(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def parts_from_segments(segments: list, count: int, names: dict[str, str], min_conf: float,
                        split: bool) -> list[dict]:
    """A recognise-and-split service's parts as the flow's parts [{doctype, pages, score}] (3.4.0). A part's type is
    matched to the process's types by name (case, spaces and signs ignored); below `min_conf` it fits none. Pages
    the service left out become parts of their own with no type; a page in two parts stays in the first. `split`
    False: one part for the whole file, as the most confident type."""
    plain = {_plain(n): i for n, i in names.items()}
    taken: set[int] = set()
    out: list[dict] = []
    for s in segments:
        pages = [p for p in sorted(set(int(x) for x in s.pages)) if 1 <= p <= count and p not in taken]
        if not pages:
            continue
        taken.update(pages)
        conf = 1.0 if s.confidence is None else float(s.confidence)
        dt = names.get(s.label or "") or plain.get(_plain(s.label or "")) if s.label else None
        out.append({"doctype": dt if dt and conf >= min_conf else None, "pages": pages, "score": round(conf, 3)})
    rest = [p for p in range(1, count + 1) if p not in taken]
    for p in rest:
        if out and out[-1]["doctype"] is None and out[-1]["pages"][-1] == p - 1 and out[-1].get("rest"):
            out[-1]["pages"].append(p)
        else:
            out.append({"doctype": None, "pages": [p], "score": 0.0, "rest": True})
    out.sort(key=lambda s: s["pages"][0])
    for s in out:
        s.pop("rest", None)
    if not split and out:
        best = max(out, key=lambda s: (s["doctype"] is not None, s["score"]))
        return [{"doctype": best["doctype"], "pages": list(range(1, count + 1)), "score": best["score"]}]
    return out


def extract_pages(data: bytes, mime: str, pages: list[int]) -> tuple[bytes, str]:
    """Return a new file containing only `pages` (1-based) of the original."""
    if mime == "application/pdf":
        import pypdfium2 as pdfium

        from .preprocess import PDFIUM_LOCK
        with PDFIUM_LOCK:   # PDFium is not thread-safe
            src = pdfium.PdfDocument(data)
            out = pdfium.PdfDocument.new()
            try:
                out.import_pages(src, [p - 1 for p in pages])
                buf = io.BytesIO()
                out.save(buf)
            finally:
                out.close()
                src.close()
        return buf.getvalue(), ".pdf"
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    frames = []
    for p in pages:
        img.seek(p - 1)
        frames.append(img.copy().convert("RGB"))
    buf = io.BytesIO()
    if len(frames) == 1:
        frames[0].save(buf, format="PNG")
        return buf.getvalue(), ".png"
    frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:])
    return buf.getvalue(), ".tif"


# ================================================================== teach from samples
# "Label: value" in any script: a label starts with a letter and has letters, digits and a few signs
_LABEL_VALUE = re.compile(r"^\s*([^\W\d_][\w .%#/()'&-]{1,38}?)\s*[:：]\s*(.+?)\s*$")


def infer_type(values: list[str]) -> str:
    vals = [v for v in values if v]
    if not vals:
        return "string"
    if all(coerce(v, "date")[1] and re.search(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", v) for v in vals):
        return "date"
    cur = r"(?:\b(?:" + "|".join(ISO_CURRENCIES) + r")\b|R\$|Rs\.?|[$€£¥₹﷼₩₺₽₪₫₦₱฿]|ر\.س|kr|zł|Kč|CHF)"
    if all(re.fullmatch(cur + r"?\s*-?\d[\d,]*(\.\d+)?\s*" + cur + "?", v.strip()) for v in vals):
        has_dec = any(re.search(r"[.,]\d{2}\b", v) for v in vals)
        return "currency" if has_dec or any(re.search(cur, v) for v in vals) else "number"
    return "string"


def suggest_from_results(results: list[DocumentResult]) -> dict:
    """Heuristic proposal of a Doc Type from OCR results of a few samples (no AI needed)."""
    found: dict[str, dict] = {}
    texts = []
    for r in results:
        texts.append(r.text)
        for p in r.pages:
            for line in p.lines:
                m = _LABEL_VALUE.match(line.text)
                if not m:
                    continue
                label, value = m.group(1).strip(), m.group(2).strip()
                if re.fullmatch(r"[\d\s.,/-]+", label):
                    continue
                key = label.lower()
                e = found.setdefault(key, {"label": label, "values": [], "count": 0})
                e["values"].append(value)
                e["count"] += 1
    # category guess
    best_cat, best_score = "custom", 0.0
    joined = "\n".join(texts)
    for cat, lib in LIBRARY.items():
        dt = DocType(name=lib["name"], keywords=lib["keywords"], fields=[FieldDef.model_validate(f) for f in lib["fields"]])
        _, s, _ = classify_page(joined, [dt])
        if s > best_score:
            best_cat, best_score = cat, s
    fields, used = [], set()
    for e in sorted(found.values(), key=lambda x: -x["count"]):
        name = _snake(e["label"]) or f"field_{len(fields) + 1}"
        if name in used:
            continue
        used.add(name)
        fields.append({"name": name, "type": infer_type(e["values"]), "label": e["label"], "aliases": [],
                       "required": e["count"] >= len(results), "samples": e["values"][:3], "source": "found on samples"})
    # merge with library template when the category is recognised
    if best_cat != "custom" and best_score >= 3:
        for f in LIBRARY[best_cat]["fields"]:
            labels = [p.strip().lower() for p in (f.get("label") or "").split("/")] + [a.lower() for a in f.get("aliases", [])]
            match = next((x for x in fields if x["label"].lower() in labels), None)
            if match:
                match.update({"name": f["name"], "type": f["type"], "label": f["label"], "aliases": f.get("aliases", []),
                              **({"regex": f["regex"]} if f.get("regex") else {}),
                              **({"columns": f["columns"]} if f.get("columns") else {})})
                match["source"] = "found on samples · matched template"
            elif f["name"] not in used:
                fields.append({**f, "samples": [], "source": f"{LIBRARY[best_cat]['name']} template (not seen on samples)",
                               "suggestOff": True})
                used.add(f["name"])
    langs = sorted({l.language for r in results for p in r.pages for l in p.lines if l.language})
    # title lines (short, no 'label: value') near the top identify the document type
    titles = []
    for r in results[:1]:
        for line in [l for p in r.pages[:1] for l in p.lines][:4]:
            t = line.text.strip()
            if 3 <= len(t) <= 40 and ":" not in t and not re.search(r"\d", t):
                titles.append(t)
    return {"category": best_cat, "categoryScore": best_score,
            "name": LIBRARY[best_cat]["name"] if best_cat != "custom" else "New document type",
            "languages": langs or ["en"], "keywords": LIBRARY.get(best_cat, {}).get("keywords", []) or titles[1:3] or titles[:1],
            "checks": LIBRARY.get(best_cat, {}).get("checks", []), "fields": fields, "method": "rules"}


# ================================================================== service
def platform_callback(tenant: str) -> tuple[str, dict]:
    """Where a platform "Start a bot" delivery goes: the Command Center (DOCAUTO_PLATFORM_URL; {tenant} is replaced in
    SaaS), authenticated with this service's own token, so no secret is stored in a process."""
    base = events.platform_base(tenant)
    if not base:
        raise RuntimeError("DOCAUTO_PLATFORM_URL is not set: DocFlow does not know where the Command Center is")
    return base + "/api/docflow-callback/start-bot", {"X-OOA-Tenant": tenant, **events.platform_auth()}


class DocFlowService:
    def __init__(self, svc: DocumentService, tenant: str = "default"):
        self.svc = svc
        self.tenant = tenant              # the platform tenant: "Start a bot" deliveries call back as this tenant
        self.store = svc.store
        self._lock = threading.Lock()
        self._xlsx_lock = threading.Lock()
        self._stop = threading.Event()
        self._watcher: Optional[threading.Thread] = None
        self._last_poll: dict[str, float] = {}
        self._outside_said: set[tuple[str, str]] = set()   # watched folders outside the list, logged once each
        svc.post_processors.append(self._post_process)
        svc.listeners.append(self._on_event)
        self._seed()
        # deliveries and Command Center events go through the durable outbox (docauto.outbox)
        self.outbox = outbox.Sender(self.store, {"delivery": self._send_delivery_item, "event": self._send_event_item},
                                    on_sent=self._outbox_sent, on_failed=self._outbox_failed,
                                    name=f"docflow-outbox-{tenant}")
        self.outbox.start()
        from .testsets import TestSets
        self.testsets = TestSets(self)
        self._maint = threading.Thread(target=self._maintenance_loop, name=f"docflow-maintenance-{tenant}",
                                       daemon=True)
        self._maint.start()

    # ---------------------------------------------------------------- setup
    def _seed(self) -> None:
        if not self.store.df_list("doctype"):
            for cat, lib in LIBRARY.items():
                self.save_doctype(DocType(id=cat, name=lib["name"], description=lib["description"], category=cat,
                                          keywords=lib["keywords"], checks=lib["checks"],
                                          fields=[FieldDef.model_validate(f) for f in lib["fields"]]), new=True)
        elif not self.store.meta_get("df.library.fallbacks"):
            self._upgrade_fallbacks()
        self.store.meta_set("df.library.fallbacks", "1")
        if not self.store.meta_get("df.library.global"):
            self._upgrade_global()
            self.store.meta_set("df.library.global", "1")
        raw = self.store.df_get("flow", INBOX_ID)
        if raw:
            inbox = Flow.model_validate_json(raw)
            if inbox.name == "Smart Inbox":  # brand rename (v3.1)
                inbox.name = "One Inbox"
                self.store.df_save("flow", INBOX_ID, inbox.name, inbox.model_dump_json(), inbox.status)
        else:
            self.save_flow(Flow(id=INBOX_ID, name="One Inbox", isInbox=True, status="live",
                                description="Drop any document: it is recognised, split and extracted automatically.",
                                classify=ClassifySettings(mode="auto", split=True)), new=True)

    def _upgrade_fallbacks(self) -> None:
        """Once: built-in document types seeded before field fallbacks existed get the library's (the invoice
        supplier and receipt merchant read from the page heading). A fallback set or removed later is kept."""
        for cat, lib in LIBRARY.items():
            raw = self.store.df_get("doctype", cat)
            if not raw:
                continue
            dt, changed = DocType.model_validate_json(raw), False
            for f in dt.fields:
                want = next((x.get("fallback") for x in lib["fields"] if x["name"] == f.name), None)
                if want and f.fallback is None and f.type == "string":
                    f.fallback, changed = want, True
            if changed:
                self.save_doctype(dt)

    def offered_languages(self) -> list[str]:
        """The languages a document type with none of its own is read in: those offered on this server."""
        from . import languages as langs
        return langs.offered(self.svc.config, self.svc.registry)

    def _upgrade_global(self) -> None:
        """Once (DocFlow 3.3): built-in types seeded with Saudi-only defaults become global. Only values still exactly
        as seeded change: the Saudi VAT / IBAN / ID patterns are removed, the five-currency list grows and keeps
        other codes, payment methods grow, the invoice total check allows missing tax, discounts and charges. Library
        types the install does not have yet (credit note, delivery note, utility bill, contract, other) are added."""
        for cat, lib in LIBRARY.items():
            raw = self.store.df_get("doctype", cat)
            if not raw:
                if self.store.df_list("doctype"):
                    self.save_doctype(DocType(id=cat, name=lib["name"], description=lib["description"], category=cat,
                                              keywords=lib["keywords"], checks=lib["checks"],
                                              fields=[FieldDef.model_validate(f) for f in lib["fields"]]), new=True)
                continue
            dt, changed = DocType.model_validate_json(raw), False
            have = {f.name for f in dt.fields}
            for f in dt.fields:
                if f.regex in OLD_SAUDI_REGEX:
                    f.regex, changed = None, True
                if f.type == "enum" and f.enumValues == OLD_CURRENCIES:
                    f.enumValues, f.allowOther, changed = list(CURRENCIES), True, True
                if f.type == "enum" and f.enumValues == OLD_PAYMENT:
                    want = next((x for x in lib["fields"] if x["name"] == f.name), None)
                    if want:
                        f.enumValues, f.allowOther, changed = list(want["enumValues"]), True, True
                if cat == "invoice" and f.name == "subtotal" and f.required:
                    f.required, changed = False, True
            if cat == "invoice" and OLD_INVOICE_CHECK in dt.checks:
                dt.checks = [INVOICE_TOTAL_CHECK if c == OLD_INVOICE_CHECK else c for c in dt.checks]
                for x in lib["fields"]:   # the check reads discount and other charges: add them when missing
                    if x["name"] in ("discount", "other_charges") and x["name"] not in have:
                        dt.fields.insert(max(0, len(dt.fields) - 2), FieldDef.model_validate(x))
                changed = True
            if changed:
                self.save_doctype(dt)

    # ---------------------------------------------------------------- doc types
    def doctypes(self) -> list[DocType]:
        return [DocType.model_validate_json(b) for b in self.store.df_list("doctype")]

    def doctype(self, dt_id: str) -> DocType:
        b = self.store.df_get("doctype", dt_id)
        if not b:
            raise InvalidInput(f"unknown document type '{dt_id}'")
        return DocType.model_validate_json(b)

    def _check_fields(self, fields: list[FieldDef]) -> None:
        # a type may be saved with no fields yet (a draft); it only recognises documents until fields are added
        seen = set()
        for f in fields:
            if not _field_name_ok(f.name):
                raise InvalidInput(f"field name '{f.name}' must use letters (any script), digits and underscore, "
                                   "and not start with a digit")
            if f.name in seen:
                raise InvalidInput(f"duplicate field '{f.name}'")
            seen.add(f.name)

    def save_doctype(self, dt: DocType, new: bool = False) -> DocType:
        if not dt.name.strip():
            raise InvalidInput("name is required")
        self._check_fields(dt.fields)
        with self._lock:
            if new:
                base = dt.id or _slug(dt.name, "doctype")
                dt.id, n = base, 2
                while self.store.df_get("doctype", dt.id):
                    dt.id, n = f"{base}-{n}", n + 1
                dt.createdAt, dt.version = _now(), 1
            dt.updatedAt = _now()
            self.store.df_save("doctype", dt.id, dt.name, dt.model_dump_json())
        return dt

    def update_doctype(self, dt_id: str, patch: dict) -> DocType:
        cur = self.doctype(dt_id)
        data = cur.model_dump()
        data.update({k: v for k, v in patch.items() if k not in ("id", "createdAt", "updatedAt", "version")})
        new = DocType.model_validate(data)
        if new.model_dump(exclude={"updatedAt", "learning"}) != cur.model_dump(exclude={"updatedAt", "learning"}):
            new.version = cur.version + 1
        return self.save_doctype(new)

    def delete_doctype(self, dt_id: str) -> None:
        self.doctype(dt_id)
        users = [f.name for f in self.flows() if dt_id in f.docTypes]
        if users:
            raise InvalidInput(f"used by flow(s): {', '.join(users)}")
        self.store.df_delete("doctype", dt_id)

    # ---------------------------------------------------------------- flows
    def flows(self) -> list[Flow]:
        out = [Flow.model_validate_json(b) for b in self.store.df_list("flow")]
        return sorted(out, key=lambda f: (not f.isInbox, f.name.lower()))

    def flow(self, flow_id: str) -> Flow:
        b = self.store.df_get("flow", flow_id)
        if not b:
            raise InvalidInput(f"unknown flow '{flow_id}'")
        return Flow.model_validate_json(b)

    @staticmethod
    def _folder_uses(fl: Flow) -> list[tuple[str, str, str]]:
        """The folders a process uses: (what, label, path) for its watched folder and its folder / Excel results."""
        out = [("watched", "Watched folder", fl.intake.inputFolder)] if fl.intake.inputFolder else []
        out += [("results", f"Delivery '{d.name or d.type}'", str(d.params["path"]))
                for d in fl.deliveries if d.type in ("folder", "excel") and d.params.get("path")]
        return out

    def _check_folders(self, fl: Flow, before: Optional[Flow]) -> None:
        """A new or changed folder must be one of the folders processes may use (when the administrator listed any).
        A folder the saved process already had is not checked again: the process can still be edited, while the
        watcher and the deliveries leave a folder outside the list alone."""
        rs = folders.roots(self.tenant)
        if not rs:
            return
        had = {(what, folders.real(p)) for what, _, p in self._folder_uses(before)} if before else set()
        for what, label, p in self._folder_uses(fl):
            if (what, folders.real(p)) not in had and folders.inside(p, rs) is None:
                raise InvalidInput(f"{label}: '{p}' is not one of the folders processes may use. Choose a folder "
                                   "from the list (an administrator sets it in Settings > Document Automation)")

    def _check_flow(self, fl: Flow) -> None:
        if not fl.name.strip():
            raise InvalidInput("name is required")
        if not fl.isInbox and not fl.docTypes:
            raise InvalidInput("choose at least one document type")
        for d in fl.docTypes:
            self.doctype(d)
        for t in (fl.review.fieldConfidence, fl.review.documentConfidence):
            if not 0 <= t <= 1:
                raise InvalidInput("confidence thresholds must be between 0 and 1")
        for name, t in fl.review.fieldThresholds.items():
            if not _field_name_ok(name):
                raise InvalidInput(f"field threshold: '{name}' is not a field name")
            if not 0 <= t <= 1:
                raise InvalidInput(f"field threshold for '{name}' must be between 0 and 1")
        for i, r in enumerate(fl.rules):
            r.id = r.id or f"r{i + 1}"
            if r.type == "lookup" and r.params.get("table") and r.params.get("whereColumn") and self.store.ref_get(r.params["table"])                     and r.params["whereColumn"] not in (self.store.ref_get(r.params["table"]).get("columns") or []):
                raise InvalidInput(f"rule '{r.name or r.type}': reference table has no column '{r.params['whereColumn']}'")
            if r.type == "lookup" and r.params.get("table") and not self.store.ref_get(r.params["table"]):
                raise InvalidInput(f"rule '{r.name or r.type}': reference table '{r.params['table']}' not found")
        for i, d in enumerate(fl.deliveries):
            d.id = d.id or f"d{i + 1}"
            if d.type == "start_bot" and d.params.get("platform"):  # the Command Center starts it: needs the bot, not an address
                if not str(d.params.get("bot", "")).strip():
                    raise InvalidInput(f"delivery '{d.name or d.type}': choose the bot to start")
            elif d.type in ("webhook", "start_bot") and not str(d.params.get("url", "")).startswith(("http://", "https://")):
                raise InvalidInput(f"delivery '{d.name or d.type}': URL must start with http:// or https://")
            if d.type in ("folder", "excel") and not d.params.get("path"):
                raise InvalidInput(f"delivery '{d.name or d.type}': path is required")
            if d.type == "email" and not [a for a in d.params.get("to") or [] if str(a).strip()]:
                raise InvalidInput(f"delivery '{d.name or d.type}': give at least one address or person to send to")
            if d.type == "cloud_storage":
                need = {"s3": ("bucket", "accessKey"), "azure_blob": ("container",), "http_put": ("url",)}.get(d.params.get("provider"))
                if need is None:
                    raise InvalidInput(f"delivery '{d.name or d.type}': choose S3, Azure Blob or a web address")
                missing = [k for k in need if not str(d.params.get(k) or "").strip()]
                if missing:
                    raise InvalidInput(f"delivery '{d.name or d.type}': {', '.join(missing)} is required")
                addr = str(d.params.get("container") or d.params.get("url") or d.params.get("endpoint") or "https://")
                if not addr.startswith(("http://", "https://")):
                    raise InvalidInput(f"delivery '{d.name or d.type}': the address must start with https://")

    def save_flow(self, fl: Flow, new: bool = False, check_folders: bool = True) -> Flow:
        self._check_flow(fl)
        if check_folders:
            b = None if new or not fl.id else self.store.df_get("flow", fl.id)
            self._check_folders(fl, Flow.model_validate_json(b) if b else None)
        with self._lock:
            if new:
                base = fl.id or _slug(fl.name, "flow")
                fl.id, n = base, 2
                # not the id of a process that exists, nor of a deleted one whose documents are kept
                while self.store.df_get("flow", fl.id) or self.store.instance_documents(fl.id, None, 1):
                    fl.id, n = f"{base}-{n}", n + 1
                fl.createdAt, fl.version = _now(), 1
            fl.updatedAt = _now()
            self.store.df_save("flow", fl.id, fl.name, fl.model_dump_json(), fl.status)
        return fl

    def update_flow(self, flow_id: str, patch: dict) -> Flow:
        cur = self.flow(flow_id)
        data = cur.model_dump()
        data.update({k: v for k, v in patch.items() if k not in ("id", "isInbox", "createdAt", "updatedAt", "version")})
        new = Flow.model_validate(data)
        if new.model_dump(exclude={"updatedAt"}) != cur.model_dump(exclude={"updatedAt"}):
            new.version = cur.version + 1
        return self.save_flow(new)

    def delete_flow(self, flow_id: str) -> None:
        fl = self.flow(flow_id)
        if fl.isInbox:
            raise InvalidInput("the Smart Inbox cannot be deleted")
        self.store.df_delete("flow", flow_id)

    def flow_doctypes(self, fl: Flow) -> list[DocType]:
        return self.doctypes() if (fl.isInbox and not fl.docTypes) else [self.doctype(d) for d in fl.docTypes]

    # ---------------------------------------------------------------- intake
    def _options(self, fl: Flow, dt: Optional[DocType], source: str, **extra) -> ProcessOptions:
        instructions = None
        parts = []
        if dt and dt.learning.enabled:
            recent = self.store.corrections(dt.id, limit=int(env_number("DOCAUTO_CORRECTIONS_REMEMBERED", 8, 0, 100)))
            if recent:
                parts.append("Recent reviewer corrections for this document type (learn from them):\n" + "\n".join(
                    f"- {c['field']}: extracted {json.dumps(c['original'], ensure_ascii=False)}, corrected to "
                    f"{json.dumps(c['corrected'], ensure_ascii=False)}" for c in recent))
        if dt and dt.examples:
            parts.append(self._example_text(dt))
        instructions = "\n\n".join(p for p in parts if p) or None
        return ProcessOptions(
            ocrProvider=fl.engine.ocrProvider,
            extractionProvider=(fl.engine.extractionByType.get(dt.id) if dt else None) or fl.engine.extractionProvider,
            localOnly=self.over_budget(fl),
            dateOrder=dt.dateOrder if dt else None, decimalMark=dt.decimalMark if dt else None,
            languages=(dt.languages if dt else None) or self.offered_languages(), capabilities=self._capabilities(fl, dt),
            schema=dt.schema() if dt else None, instanceId=fl.id, instanceVersion=fl.version, source=source,
            doctypeId=dt.id if dt else None, fieldThreshold=fl.review.fieldConfidence,
            docThreshold=fl.review.documentConfidence, fieldThresholds=fl.review.fieldThresholds,
            einvoiceFirst=fl.intake.einvoiceFirst, einvoiceStandards=fl.intake.einvoiceStandards, instructions=instructions, maskPii=fl.engine.maskPii,
            cloudImages=fl.engine.allowImages, **extra)

    def upload(self, flow_id: str, data: bytes, file_name: str, source: str = "upload", wait: bool = False,
               require_live: bool = False) -> dict:
        from .preprocess import sniff_mime
        if sniff_mime(data, file_name) not in READABLE_MIME:   # refused here, whatever way it came in
            raise InvalidInput(f"'{file_name}' is not a file Document Automation reads: PDF, PNG, JPG, TIFF, BMP, WEBP "
                               "or an XML e-invoice")
        fl = self.flow(flow_id)
        if source == "upload" and not fl.intake.upload:
            raise InvalidInput(f"'{fl.name}' takes no uploads: its documents come from agents, e-mail or a folder")
        if require_live and fl.status != "live":
            raise InvalidInput(f"flow '{fl.name}' is a draft - set it live before bots or folders use it")
        types = self.flow_doctypes(fl)
        if not types:
            raise InvalidInput("no document types available - create one first")
        inv_type = self._einvoice_type(fl, types, data, file_name)
        if inv_type is not None:  # an e-invoice needs no recognising or splitting: it says what it is
            ids = self.svc.submit(data, file_name, self._options(fl, inv_type, source), wait=wait)
        elif fl.classify.mode == "single" or (len(types) == 1 and not fl.classify.split):
            ids = self.svc.submit(data, file_name, self._options(fl, types[0], source), wait=wait)
        else:
            opt = self._options(fl, None, source, role="batch", waitChildren=wait)
            ids = self.svc.submit(data, file_name, opt, wait=wait)
            if wait:
                # 3.4.3: a file that is one document of one type is answered with that document, not the batch
                kids = self.store.children(ids["documentId"])
                if len(kids) == 1:
                    batch = ids["documentId"]
                    return {**ids, **self.document(kids[0]["id"], brief=True), "documentId": kids[0]["id"], "batchId": batch}
        return {**ids, **self.document(ids["documentId"], brief=True)}

    @staticmethod
    def _einvoice_type(fl: Flow, types: list[DocType], data: bytes, file_name: str) -> Optional[DocType]:
        """The process's type for an e-invoice (XML, or a PDF carrying one): a credit note type for credit notes
        when the process has one, else its invoice type; None when the file is no e-invoice."""
        if not fl.intake.einvoiceFirst:
            return None
        accepts = [t for t in types if t.category in ("invoice", "credit_note") or t.id in ("invoice", "credit_note")]
        if not accepts:
            return None
        from .preprocess import sniff_mime
        inv = einvoice.read_file(data, sniff_mime(data, file_name), fl.intake.einvoiceStandards)
        if not inv:
            return None
        credit = inv.get("raw", {}).get("documentType") == "CreditNote"
        want = ("credit_note",) if credit else ("invoice",)
        return next((t for t in accepts if t.category in want or t.id in want), None) or \
            next((t for t in accepts if t.category == "invoice" or t.id == "invoice"), accepts[0])

    # ---------------------------------------------------------------- post-processing (inside the job)
    def _post_process(self, doc_id: str, result: DocumentResult, opt: ProcessOptions, status: str) -> Optional[str]:
        if not opt.instanceId:
            return None
        try:
            fl = self.flow(opt.instanceId)
        except InvalidInput:
            return None
        if opt.role == "batch":
            return self._split(fl, doc_id, result, opt)
        dt = None
        if opt.doctypeId:
            try:
                dt = self.doctype(opt.doctypeId)
            except InvalidInput:
                dt = None
        if dt is not None and fl.classify.checkType and fl.classify.mode == "single" and self._recogniser(opt) is not None:
            # 3.4.0: the recognise engine is asked whether it is this type (its description), or another document
            if self._splits(self._recogniser(opt)):
                parts = self._service_parts(self._recogniser(opt), [dt], doc_id, opt, result, len(result.pages) or 1, False)
                got = None if parts is None else [(parts[0]["doctype"], parts[0]["score"])]
            else:
                got = self._recognise_pages([result.text or ""], [dt], doc_id, opt, result)
            if got is not None and got[0][0] is None:
                result.warnings.append(f"Does not look like {dt.name}: the recognise engine did not take it for one")
                dt = None
        elif dt is not None and fl.classify.checkType and fl.classify.mode == "single" and dt.keywords:
            # the type's own recognition words (not its field labels: "Total" is on many kinds of document)
            low = (result.text or "").lower()
            if not any(re.search(r"(?<![\w])" + re.escape(k.lower().strip()) + r"(?![\w])", low) for k in dt.keywords if k.strip()):
                result.warnings.append(f"Does not look like {dt.name}: none of its recognition words is on it")
                dt = None
        if dt is None and opt.doctypeId:
            # 3.4.3: it is not the type it was read as, so that type's checks and missing values are no reasons
            # (its values stay, for a person who decides it is that type after all)
            result.review.reasons = []
            result.warnings = [w for w in result.warnings if not w.startswith(("validation rule failed", "field '", "document confidence"))]
        if dt is None and fl.classify.whenNoType == "reject":
            result.review.required, result.review.status = False, ReviewStatus.not_required
            result.review.reasons = []
            result.warnings.append("Not one of this process's document types: rejected")
            self.store.save_result(result)
            self.store.set_doc_fields(doc_id, doctype_id=None, flags=json.dumps([{"rule": "classification", "type": "classify",
                                                                 "action": "reject", "message": "Document type not recognised"}]))
            return "invalid"
        if dt is None and fl.classify.whenNoType == "other":
            try:
                other = self.doctype("other")
            except InvalidInput:
                other = None
            if other is not None:
                # what was read as the wrong type goes: its fields and review notes do not belong to this document
                result.fields = []
                result.warnings = [w for w in result.warnings if not w.startswith(("field '", "document confidence"))]
                result.warnings.append("Not one of this process's document types: kept as Other document")
                result.review.required, result.review.status = False, ReviewStatus.not_required
                self.store.save_result(result)
                self.store.set_doc_fields(doc_id, doctype_id=other.id, flags="[]")
                return "success"
        if dt is None:
            result.review.required, result.review.status = True, ReviewStatus.pending
            result.warnings.append("Unclassified: choose the document type in One Desk")
            result.review.reasons = _merge_reasons(result.review.reasons, [UNCLASSIFIED])
            self.store.save_result(result)
            self.store.set_doc_fields(doc_id, doctype_id=None, flags=json.dumps([{"rule": "classification", "type": "classify",
                                                                 "action": "review", "message": "Document type not recognised"}]))
            return "validation"
        flags = run_rules(self.store, fl, dt, doc_id, result, RefData(self.store))
        self.store.set_doc_fields(doc_id, flags=json.dumps(flags, ensure_ascii=False))
        # the scorer's decision again, now that lookups may have filled or confirmed values (it changes nothing)
        ft, dtt = self.svc._thresholds(opt)
        reasons = _merge_reasons(_flag_reasons(flags), review_reasons(result, opt.schema_ or dt.schema(), ft, dtt,
                                                                      opt.fieldThresholds))
        reasons = review_mode(fl.review.mode, reasons)
        result.review.reasons = reasons
        new = status
        trying = opt.source == TRY_SOURCE   # 3.4.1: a dry run tells nobody and sends nothing elsewhere
        for f in flags:
            if f["action"] == "notify" and not trying:
                self._notify(fl, doc_id, f)
        route = next((f for f in flags if f["action"] == "route" and f.get("routeTo")), None)
        if route and trying:
            result.warnings.append(f"Would be sent to process {route['routeTo']} because {route['rule']}: {route['message']}")
            route = None
            new = "invalid"
        if any(f["action"] == "reject" for f in flags):
            new = "invalid"
        elif new == "invalid" and trying:
            pass
        elif route:
            target = self._route(fl, doc_id, route)
            if not target:   # the process to send it to is gone: a person decides instead
                result.review.reasons = _merge_reasons(result.review.reasons, [ReviewReason(field=route.get("field"), reason="rule", rule=route["rule"],
                                                                                         message=route["message"] + " (the process to send it to does not exist)")])
                new = "validation"
            if target:
                result.warnings.append(f"Sent to process {target} because {route['rule']}: {route['message']}")
                result.review.required, result.review.status = False, ReviewStatus.not_required
                new = "invalid"
        elif reasons and status == "success":
            new = "validation"
        elif not reasons and status == "validation":
            new = "success"     # reference data confirmed the only doubtful values
            result.review.required, result.review.status = False, ReviewStatus.not_required
            result.warnings = [w for w in result.warnings if not w.endswith("needs review")]
        if flags:
            result.warnings.extend(f"{f['rule']}: {f['message']}" for f in flags if f["action"] not in ("warn", "notify"))
        if new == "validation":
            result.review.required, result.review.status = True, ReviewStatus.pending
        self.store.save_result(result)  # also persists lookup enrichment
        return new

    #: below this an AI engine's answer counts as "fits no type" (its own confidence, 0-1)
    RECOGNISE_MIN = 0.5

    def _recogniser(self, opt: ProcessOptions):
        """The engine that recognises document types (DOCAUTO_DEFAULT_CLASSIFY), when one is set and may run; None:
        the words on the page."""
        engine = (os.environ.get("DOCAUTO_DEFAULT_CLASSIFY") or "").strip()
        if not engine:
            return None
        try:
            found = [p for p in self.svc.router.candidates("extraction", [Capability.CLASSIFY], [], engine, "extraction", strict=True)
                     if p.info.runsLocally or not opt.localOnly]
        except DocAutoError as e:
            log.warning("recognise engine %s not used: %s", engine, e)
            return None
        return found[0] if found else None

    @staticmethod
    def _splits(prov) -> bool:
        """The recognise engine is a service that splits the whole file (it sees the file, not each page's text)."""
        return prov is not None and Capability.SPLIT in prov.info.capabilities

    def _service_parts(self, prov, types: list[DocType], doc_id: str, opt: ProcessOptions, result: DocumentResult,
                       count: int, split: bool) -> Optional[list[dict]]:
        """The file's parts as a recognise-and-split service sees them (3.4.0), or None to use the words on the page
        (the service failed: noted on the document). The service is sent the file itself."""
        data, _name, mime = self.store.file_bytes(doc_id)
        names = {dt.name: dt.id for dt in types}
        try:
            segments = prov.instance.split(data, mime, count, list(names), {dt.name: dt.description for dt in types},
                                           segment=split)
        except DocAutoError as e:
            result.warnings.append(f"Recognised by the words on the page: {prov.info.id} failed ({e})")
            return None
        self.svc._meter(None, doc_id, opt, prov, Capability.CLASSIFY, count, "classify", result)
        return parts_from_segments(segments, count, names, self.RECOGNISE_MIN, split)

    def _recognise_pages(self, texts: list[str], types: list[DocType], doc_id: str, opt: ProcessOptions,
                         result: DocumentResult) -> Optional[list[tuple[Optional[str], float]]]:
        """Each page's type as the recognise engine sees it, from every type's name and description (3.4.0), or
        None to use the words on the page (no engine set, it may not run, or it failed: noted on the document).
        A page with no text keeps the words' answer (nothing to recognise); a cloud AI engine sees masked text."""
        prov = self._recogniser(opt)
        if prov is None:
            if (os.environ.get("DOCAUTO_DEFAULT_CLASSIFY") or "").strip():
                result.warnings.append("Recognised by the words on the page: the recognise engine may not run here")
            return None
        names = {dt.name: dt.id for dt in types}
        descriptions = {dt.name: dt.description for dt in types}
        masks = self.svc._masks(prov, opt)
        out: list[tuple[Optional[str], float]] = []
        asked = 0
        try:
            for txt in texts:
                if not txt.strip():
                    out.append((None, 0.0))
                    continue
                shown = pii.PiiMasker().mask(txt) if masks else txt
                c = prov.instance.classify(_PageText(shown), list(names), descriptions)
                asked += 1
                dt_id = names.get(c.label)
                out.append((dt_id if dt_id and c.confidence >= self.RECOGNISE_MIN else None, round(float(c.confidence), 3)))
        except DocAutoError as e:
            result.warnings.append(f"Recognised by the words on the page: {prov.info.id} failed ({e})")
            return None
        if asked:
            self.svc._meter(None, doc_id, opt, prov, Capability.CLASSIFY, asked, "classify", result)
        return out

    def _split(self, fl: Flow, doc_id: str, result: DocumentResult, opt: ProcessOptions) -> str:
        types = self.flow_doctypes(fl)
        texts = ["\n".join(l.text for l in p.lines) for p in result.pages]
        prov = self._recogniser(opt)
        segs = self._service_parts(prov, types, doc_id, opt, result, len(texts), fl.classify.split) \
            if self._splits(prov) else None
        if segs is None:
            segs = split_pages(texts, types, fl.classify.minScore, fl.classify.split,
                               None if self._splits(prov) else self._recognise_pages(texts, types, doc_id, opt, result))
        data, file_name, mime = self.store.file_bytes(doc_id)
        stem, by_id = Path(file_name).stem, {d.id: d for d in types}
        for i, s in enumerate(segs, 1):
            dt = by_id.get(s["doctype"]) if s["doctype"] else None
            if len(segs) == 1:
                part, ext = data, Path(file_name).suffix
            else:
                part, ext = extract_pages(data, mime, s["pages"])
            label = (dt.name if dt else "unclassified").lower().replace(" ", "-")
            name = f"{stem}_{i}-{label}{ext}" if len(segs) > 1 else file_name
            child_opt = self._options(fl, dt, opt.source, parentId=doc_id, reuseFrom=doc_id, reusePages=s["pages"])
            self.svc.submit(part, name, child_opt, wait=opt.waitChildren)
        result.warnings.append(f"Split into {len(segs)} document(s): " + ", ".join(
            f"pages {s['pages'][0]}-{s['pages'][-1]} → {by_id[s['doctype']].name if s['doctype'] else 'unclassified'}"
            for s in segs))
        self.store.save_result(result)
        return "split"

    def resplit(self, batch_id: str, parts: list[dict], wait: bool = False) -> dict:
        """A person corrects how a scanned batch was split: parts [{"pages": [1, 2], "docType": id or None}]. The
        documents split from it before are removed (an approved one cannot be: the batch is then left as it is), and
        each part is read again as its type. Pages are numbered from 1; every part needs at least one page."""
        row = self.store.doc_row(batch_id)
        if not row or row["status"] != "split":
            raise InvalidInput("only a batch that was split can be split again")
        fl = self.flow(row["instance_id"])
        res = self.store.get_result(batch_id)
        count = len(res.pages) if res else 0
        seen: set[int] = set()
        clean = []
        for p in parts:
            pages = sorted({int(x) for x in (p.get("pages") or [])})
            if not pages or pages[0] < 1 or pages[-1] > count:
                raise InvalidInput(f"each part needs pages between 1 and {count}")
            if seen & set(pages):
                raise InvalidInput("a page can be in one part only")
            seen |= set(pages)
            dt = self.doctype(p["docType"]) if p.get("docType") else None
            clean.append((pages, dt))
        if not clean:
            raise InvalidInput("give at least one part")
        kids = self.store.children(batch_id)
        if any(k["status"] == "validated" for k in kids):
            raise InvalidInput("a part of this batch is already approved: it cannot be split again")
        for k in kids:
            self.store.delete_document(k["id"])
        data, file_name, mime = self.store.file_bytes(batch_id)
        stem = Path(file_name).stem
        made = []
        for i, (pages, dt) in enumerate(clean, 1):
            part, ext = extract_pages(data, mime, pages)
            label = (dt.name if dt else "unclassified").lower().replace(" ", "-")
            opt = self._options(fl, dt, row.get("source") or "upload", parentId=batch_id, reuseFrom=batch_id, reusePages=pages)
            made.append(self.svc.submit(part, f"{stem}_{i}-{label}{ext}", opt, wait=wait)["documentId"])
        if res:
            res.warnings.append("Split again by a person: " + ", ".join(
                f"pages {p[0]}-{p[-1]} → {dt.name if dt else 'unclassified'}" for p, dt in clean))
            self.store.save_result(res)
        return {"documentId": batch_id, "children": made}

    # ---------------------------------------------------------------- documents & review
    def document(self, doc_id: str, brief: bool = False) -> dict:
        row = self.store.doc_row(doc_id)
        if not row:
            raise InvalidInput(f"unknown document '{doc_id}'")
        row["flowId"] = row.pop("instance_id")
        row["flags"] = json.loads(row["flags"]) if row.get("flags") else []
        if brief:
            return {"status": row["status"], "flowId": row["flowId"], "doctypeId": row["doctype_id"]}
        res = self.store.get_result(doc_id)
        row["fields"] = {f.name: f.out() for f in res.fields} if res else {}
        row["confidence"] = {f.name: f.confidence for f in res.fields} if res else {}
        row["locations"] = locations(res) if res else {}      # where each value is on the pages (review highlight)
        row["einvoice"] = res.einvoice if res else None       # e-invoice XML / QR data and its checks
        row["reviewReasons"] = [r.model_dump() for r in self._review_reasons(res, row)] if res else []
        row["children"] = self.store.children(doc_id) if row["status"] == "split" else []
        if len(row["children"]) == 1:   # 3.4.3: the batch is one document: its results are that part's
            row["onlyPart"] = row["children"][0]["id"]
        row["error"] = self.store.last_job_error(doc_id)
        return row

    def _review_reasons(self, res: DocumentResult, row: dict) -> list[ReviewReason]:
        """Why the document went to review. A result read before the reasons were kept is stated again by the
        scorer (docauto.scoring.review_reasons, which changes nothing) with its process's thresholds and rules."""
        if res.review.reasons or not res.review.required:
            return res.review.reasons
        flags = row.get("flags") or []
        try:
            fl = self.flow(row["flowId"]) if row.get("flowId") else None
            dt = self.doctype(row["doctype_id"]) if row.get("doctype_id") else None
        except InvalidInput:
            fl = dt = None
        if dt is None:
            return _merge_reasons([UNCLASSIFIED], _flag_reasons(flags))
        rv = fl.review if fl else ReviewSettings()
        return review_mode(rv.mode, _merge_reasons(_flag_reasons(flags), review_reasons(res, dt.schema(), rv.fieldConfidence,
                                                                                     rv.documentConfidence, rv.fieldThresholds)))

    def _example_text(self, dt: DocType) -> str:
        """The teaching examples as AI reading sees them: each approved document's right values (no page text)."""
        lines = []
        for doc_id in dt.examples[:10]:
            res = self.store.get_result(doc_id)
            if not res:
                continue
            vals = {f.name: f.value for f in res.fields if f.value not in (None, "", []) and f.type != "table"}
            if vals:
                lines.append("- " + json.dumps(vals, ensure_ascii=False, default=str))
        return ("Right values of approved documents of this type (examples of what to read):\n" + "\n".join(lines)) if lines else ""

    def add_example(self, dt_id: str, doc_id: str) -> DocType:
        """Make an approved document a teaching example of its type (at most 10; the oldest goes)."""
        dt = self.doctype(dt_id)
        row = self.store.doc_row(doc_id)
        if not row or row["status"] not in ("validated", "success"):
            raise InvalidInput("only a document that is done or approved can be a teaching example")
        if doc_id not in dt.examples:
            dt.examples = (dt.examples + [doc_id])[-10:]
            self.save_doctype(dt)
        return dt

    def remove_example(self, dt_id: str, doc_id: str) -> DocType:
        dt = self.doctype(dt_id)
        if doc_id in dt.examples:
            dt.examples = [d for d in dt.examples if d != doc_id]
            self.save_doctype(dt)
        return dt

    def month_spend(self, flow_id: str) -> float:
        """What the process's documents cost this calendar month on cloud and AI engines, in the price currency
        (pages at each engine's price: an estimate, not the vendor's bill)."""
        from . import costs as costs_mod
        start = datetime.now(timezone.utc).strftime("%Y-%m-01")
        rows = self.store._all("SELECT u.provider_id, u.capability, u.model, SUM(u.pages) AS pages FROM usage u "
                               "JOIN documents d ON d.id = u.document_id WHERE d.instance_id=? AND u.runs_locally=0 "
                               "AND u.ts >= ? GROUP BY u.provider_id, u.capability, u.model", (flow_id, start))
        total = 0.0
        for r in rows:
            use = "extract" if str(r["capability"] or "").startswith(("extract", "classify")) else "ocr"
            price = costs_mod.price_per_1000(r["provider_id"], use, r["model"]) or 0.0
            total += (r["pages"] or 0) * price / 1000
        return round(total, 2)

    def over_budget(self, fl: Flow) -> bool:
        if fl.engine.monthlyBudget is None:
            return False
        try:
            return self.month_spend(fl.id) >= fl.engine.monthlyBudget
        except Exception:   # the spend cannot be worked out: the process keeps its engines
            log.exception("monthly spend of %s", fl.id)
            return False

    def accuracy(self, flow_id: str, limit: int = 500) -> dict:
        """How well a process reads, from its own reviewed documents: per field how often reviewers changed the value,
        the share of documents that needed no person, and what each review level would do: the share of fields that
        would go straight through, and how many of those a reviewer had to correct (errors that would slip by)."""
        fl = self.flow(flow_id)
        rows = self.store.instance_documents(flow_id, ["success", "validated", "invalid", "validation"], limit)
        finished = [r for r in rows if r["status"] in ("success", "validated")]
        straight = sum(1 for r in finished if r["status"] == "success")
        per: dict[str, dict] = {}
        points: list[tuple[float, bool]] = []
        reviewed = 0
        for r in rows:
            if r["status"] != "validated":
                continue
            res = self.store.get_result(r["id"])
            if not res:
                continue
            reviewed += 1
            for f in res.fields:
                if f.type == "table":
                    continue
                e = per.setdefault(f.name, {"name": f.name, "seen": 0, "corrected": 0})
                e["seen"] += 1
                e["corrected"] += 1 if f.corrected else 0
                if f.readConfidence is not None:
                    points.append((f.readConfidence, bool(f.corrected)))
        levels = []
        for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99):
            auto = [c for conf, c in points if conf >= t]
            levels.append({"level": t, "straightThrough": round(len(auto) / len(points), 4) if points else None,
                           "wrongPassed": round(sum(auto) / len(auto), 4) if auto else None, "fields": len(auto)})
        labels = {}
        for dt in self.flow_doctypes(fl):
            labels.update({x.name: (x.label or x.name).split(" / ")[0] for x in dt.fields})
        fields = sorted(({**e, "label": labels.get(e["name"], e["name"]),
                          "correctedShare": round(e["corrected"] / e["seen"], 4) if e["seen"] else 0} for e in per.values()),
                        key=lambda x: -x["correctedShare"])
        return {"documents": len(rows), "finished": len(finished), "reviewed": reviewed,
                "straightThrough": round(straight / len(finished), 4) if finished else None,
                "reviewLevel": fl.review.fieldConfidence, "fields": fields, "levels": levels,
                "measured": len(points)}

    def review(self, doc_id: str, fields: dict, reviewer: str, action: str = "submit", note: str = "") -> dict:
        row = self.store.doc_row(doc_id)
        if not row:
            raise InvalidInput(f"unknown document '{doc_id}'")
        if row["status"] not in ("validation", "success", "validated", "invalid"):
            raise InvalidInput(f"document is '{row['status']}' and cannot be reviewed")
        res = self.svc.complete_review(doc_id, fields, reviewer, reject=(action == "invalid"))
        if note.strip() and res is not None:   # the reviewer's reason (why it was rejected, what was checked)
            res.warnings.append(("Rejected: " if action == "invalid" else "Reviewer note: ") + note.strip()[:500])
            self.store.save_result(res)
        return {**self.document(doc_id), "nextDocumentId": self.store.next_for_validation(row.get("instance_id"))}

    def reclassify(self, doc_id: str, dt_id: str) -> dict:
        row = self.store.doc_row(doc_id)
        if not row:
            raise InvalidInput(f"unknown document '{doc_id}'")
        dt, fl = self.doctype(dt_id), self.flow(row["instance_id"])
        data, file_name, _ = self.store.file_bytes(doc_id)
        # 3.4.3: the old copy is "replaced" (not rejected): counts, lists and batches leave it out, and it names the new one
        self.store.set_doc_status(doc_id, "replaced", validated_by="reclassified")
        opt = self._options(fl, dt, row.get("source") or "upload", parentId=row.get("parent_id"), reuseFrom=doc_id)
        ids = self.svc.submit(data, file_name, opt, wait=True)
        self.store.set_doc_fields(doc_id, flags=json.dumps([{"rule": "reclassified", "type": "replaced", "action": "replaced",
                                                             "replacedBy": ids["documentId"],
                                                             "message": f"Type changed to {dt.name}: see the new document"}]))
        return self.document(ids["documentId"])

    def review_queue(self, flow_id: Optional[str]) -> list[dict]:
        flows = {f.id: f for f in self.flows()}
        dts = {d.id: d.name for d in self.doctypes()}
        out = []
        now = datetime.now(timezone.utc)
        for d in self.store.instance_documents(flow_id, ["validation"], limit=1000):
            fl = flows.get(d["instance_id"])
            age_h = (now - datetime.fromisoformat(d["created_at"])).total_seconds() / 3600
            sla = fl.review.slaHours if fl else 24
            out.append({**d, "flowId": d["instance_id"], "flowName": fl.name if fl else d["instance_id"],
                        "doctypeName": dts.get(d["doctype_id"], "Unclassified"), "ageHours": round(age_h, 2),
                        "slaHours": sla, "overdue": age_h > sla,
                        "flags": json.loads(d["flags"]) if d.get("flags") else []})
        return sorted(out, key=lambda x: (not x["overdue"], x["created_at"]))

    # ---------------------------------------------------------------- events: learning + delivery
    def _on_event(self, event: str, doc_id: str, **info) -> None:
        row = self.store.doc_row(doc_id)
        if not row or not row.get("instance_id") or row.get("source") == TRY_SOURCE:
            return
        status = info.get("status") or row["status"]
        if event == "validated" and status == "validated" and row.get("doctype_id"):
            self._learn(row["doctype_id"], doc_id, info.get("corrections") or [])
        if status in ("success", "validated", "invalid", "failed", "validation"):
            try:
                fl = self.flow(row["instance_id"])
            except InvalidInput:
                return
            platform = ({"success": "processed", "validation": "needs_review", "failed": "failed"} if event == "processed" else
                        {"validated": "approved", "invalid": "rejected"} if event == "validated" else {}).get(status)
            if platform:
                self._event(platform, doc_id, fl, status)
            self._queue_deliveries(fl, doc_id, status)

    def _event(self, event: str, doc_id: str, fl: Optional[Flow] = None, status: Optional[str] = None) -> None:
        """Queue an event for the OOA Command Center (docauto.events) in the outbox, in the caller's transaction.
        Nothing is queued without DOCAUTO_PLATFORM_URL. Never raises."""
        if not events.platform_base(self.tenant):
            return
        try:
            row = self.store.doc_row(doc_id)
            if not row or not row.get("instance_id"):
                return
            fl = fl or self.flow(row["instance_id"])
            res = self.store.get_result(doc_id)
            payload = events.body(event, doc_id, fl.id, fl.name, status or row["status"], row.get("file_name"),
                                  row.get("doctype_id"), {f.name: f.out() for f in res.fields} if res else {})
            self.store.outbox_add("event", payload, doc_id, fl.id, event, wake=self.outbox.wake)
        except Exception:
            log.exception("event %s for %s failed", event, doc_id)

    def _queue_deliveries(self, fl: Flow, doc_id: str, status: str) -> int:
        """Queue the flow's deliveries for this status in the outbox (sent by the background sender, with retries).
        The payload is taken now, so a later change does not alter what this status delivers."""
        n = 0
        for dv in fl.deliveries:
            if not dv.enabled or status not in dv.on:
                continue
            if fl.status != "live":   # 3.4.1: a draft reads, checks and reviews, but sends nothing (Send a test still does)
                self.store.delivery_log(fl.id, doc_id, dv.name or dv.type, dv.type, False, HELD)
                continue
            body = self.payload(fl, doc_id, status)
            self.store.outbox_add("delivery", {"flowId": fl.id, "flowName": fl.name, "status": status,
                                               "delivery": dv.model_dump(), "body": body},
                                  doc_id, fl.id, dv.name or dv.type, wake=self.outbox.wake)
            n += 1
        return n

    # ---------------------------------------------------------------- outbox handlers (sender thread)
    def _send_event_item(self, item: dict) -> tuple[bool, str, bool]:
        return events.send(self.tenant, item["payload"])

    def _send_delivery_item(self, item: dict) -> tuple[bool, str, bool]:
        p = item["payload"]
        return self._send_delivery(Delivery.model_validate(p["delivery"]), item["document_id"], p["status"], p["body"])

    def _outbox_sent(self, item: dict, detail: str) -> None:
        if item["kind"] == "delivery":
            dv = item["payload"]["delivery"]
            self.store.delivery_log(item["flow_id"], item["document_id"], dv.get("name") or dv["type"], dv["type"],
                                    True, detail)
            if dv["type"] != "start_bot":   # a delivery done: agents may go on from there ("Delivery done" trigger)
                try:
                    self._event("delivered", item["document_id"], self.flow(item["flow_id"]), item["payload"].get("status"))
                except InvalidInput:
                    pass

    def _outbox_failed(self, item: dict, detail: str) -> None:
        """After the last attempt: the delivery log says so and the Command Center hears "delivery_failed"."""
        if item["kind"] != "delivery":
            return
        dv = item["payload"]["delivery"]
        self.store.delivery_log(item["flow_id"], item["document_id"], dv.get("name") or dv["type"], dv["type"],
                                False, detail)
        try:
            fl = self.flow(item["flow_id"])
        except InvalidInput:
            return
        self._event("delivery_failed", item["document_id"], fl, item["payload"]["status"])

    def outbox_status(self, status: Optional[str] = None, limit: int = 100) -> list[dict]:
        return self.store.outbox_list(status, limit)

    def _learn(self, dt_id: str, doc_id: str, corrections: list[dict]) -> None:
        from .instances import find_label_for_value  # shared helper
        try:
            dt = self.doctype(dt_id)
        except InvalidInput:
            return
        if not dt.learning.enabled or not corrections:
            return
        res = self.store.get_result(doc_id)
        if not res:
            return
        changed = False
        for c in corrections:
            label = find_label_for_value(res, c["corrected"])
            if not label:
                continue
            fdef = next((f for f in dt.fields if f.name == c["field"]), None)
            known = {a.lower() for a in dt.learning.aliases.get(c["field"], [])}
            if fdef:
                known |= {a.lower() for a in fdef.aliases} | {p.strip().lower() for p in (fdef.label or "").split("/")}
            if label.lower() not in known:
                dt.learning.aliases.setdefault(c["field"], []).append(label)
                changed = True
        if changed:
            fresh = self.doctype(dt_id)
            fresh.learning.aliases = dt.learning.aliases
            self.save_doctype(fresh)

    def forget(self, dt_id: str, field: str, alias: str) -> LearnedLabels:
        dt = self.doctype(dt_id)
        if alias in dt.learning.aliases.get(field, []):
            dt.learning.aliases[field].remove(alias)
        self.save_doctype(dt)
        return dt.learning

    def payload(self, fl: Flow, doc_id: str, status: str) -> dict:
        d = self.document(doc_id)
        dtn = None
        if d.get("doctype_id"):
            try:
                dtn = self.doctype(d["doctype_id"]).name
            except InvalidInput:
                pass
        return {"event": f"document.{status}", "flowId": fl.id, "flowName": fl.name, "flowVersion": d.get("instance_version"),
                "documentId": doc_id, "parentId": d.get("parent_id"), "fileName": d.get("file_name"),
                "docType": d.get("doctype_id"), "docTypeName": dtn, "status": status, "fields": d["fields"],
                "confidence": d["confidence"], "flags": d["flags"], "source": d.get("source"),
                "validatedBy": d.get("validated_by"), "processedAt": d.get("updated_at"),
                "fileUrl": f"/api/docs/v1/documents/{doc_id}/file"}

    def _deliver(self, fl: Flow, doc_id: str, status: str, only: Optional[str] = None) -> list[dict]:
        """Send the flow's deliveries now, once each, and log them (the "Test" button; processing uses the outbox)."""
        results = []
        for dv in fl.deliveries:
            if not dv.enabled or (only and dv.id != only) or (not only and status not in dv.on):
                continue
            try:
                ok, detail, _ = self._send_delivery(dv, doc_id, status, self.payload(fl, doc_id, status))
            except Exception as e:
                ok, detail = False, f"{type(e).__name__}: {e}"
            self.store.delivery_log(fl.id, doc_id, dv.name or dv.type, dv.type, ok, detail)
            results.append({"delivery": dv.name or dv.type, "type": dv.type, "ok": ok, "detail": detail})
        return results

    def _send_delivery(self, dv: Delivery, doc_id: str, status: str, body: dict) -> tuple[bool, str, bool]:
        """One attempt at one delivery. Returns (delivered, detail, worth retrying)."""
        try:
            if dv.type in ("webhook", "start_bot"):
                headers = {"Content-Type": "application/json"}
                url = dv.params.get("url", "")
                if dv.type == "start_bot":
                    body = {"bot": dv.params.get("bot", ""), "pool": dv.params.get("pool", ""),
                            "requestedBy": dv.params.get("requestedBy", ""), "input": body}
                    if dv.params.get("platform"):  # the OOA Command Center starts the bot (see platform_callback)
                        try:
                            url, extra = platform_callback(self.tenant)
                        except RuntimeError as e:
                            return False, str(e), False
                        headers.update(extra)
                if dv.params.get("authHeader"):
                    headers["Authorization"] = dv.params["authHeader"]
                extra = dv.params.get("headers") or {}
                if isinstance(extra, dict):   # own headers of a web address delivery (an API key header, a tenant id...)
                    for k, v in extra.items():
                        if not str(k).strip():
                            continue
                        v = str(v)
                        if v.startswith(VAULT_PREFIX):   # 3.4.1: a secret header value is a vault entry, read now, never kept
                            try:
                                v = str(self._platform("secret", {"requestedBy": dv.params.get("requestedBy", ""),
                                                                  "ref": v[len(VAULT_PREFIX):].strip()}).get("value") or "")
                            except Exception as e:   # not set up, refused or not reached: the receiver gets nothing
                                return False, f"header {k}: its vault entry could not be read ({e})", False
                        headers[str(k)] = v
                method = str(dv.params.get("method") or "POST").upper()
                if method not in ("POST", "PUT", "PATCH"):
                    method = "POST"
                try:
                    send = httpx.post if method == "POST" else (lambda *a, **k: httpx.request(method, *a, **k))
                    r = send(url, json=body, headers=headers, timeout=env_number("DOCAUTO_WEBHOOK_TIMEOUT", 15, 1, 600))
                except httpx.HTTPError as e:
                    return False, f"{type(e).__name__}: {e}", True
                ok = r.status_code < 300
                # 4xx other than timeout / too many requests: the receiver refused it, a retry will not help
                return ok, f"HTTP {r.status_code}" + ("" if ok else reply_reason(r)), not ok and not (400 <= r.status_code < 500
                                                                     and r.status_code not in (408, 429))
            if dv.type in ("folder", "excel") and not folders.allowed(str(dv.params.get("path", "")), self.tenant):
                return False, (f"'{dv.params.get('path')}' is not one of the folders processes may use: choose a folder "
                               "from the list in the process"), False
            if dv.type == "folder":
                folder = Path(dv.params["path"]) / status
                folder.mkdir(parents=True, exist_ok=True)
                stem = Path(body["fileName"] or doc_id).stem
                (folder / f"{stem}_{doc_id[:8]}.json").write_text(json.dumps(body, ensure_ascii=False, indent=2,
                                                                             default=str), encoding="utf-8")
                f = self.store.file_bytes(doc_id)
                if f:   # the original, decrypted: the folder is the customer's own output
                    (folder / f"{stem}_{doc_id[:8]}{Path(f[1]).suffix}").write_bytes(f[0])
                    if dv.params.get("saveText"):   # the whole document as text, for search and Knowledge bases
                        res = self.store.get_result(doc_id)
                        if res:
                            (folder / f"{stem}_{doc_id[:8]}.md").write_text(document_markdown(res, body), encoding="utf-8")
                    if dv.params.get("searchablePdf") and not f[1].lower().endswith(".xml"):
                        # a PDF whose text can be searched and copied (the page images with the text read under them)
                        try:
                            pdf = self.svc.searchable_pdf(f[0], f[1], self.offered_languages())
                            (folder / f"{stem}_{doc_id[:8]}_searchable.pdf").write_bytes(pdf)
                        except DocAutoError as e:   # no engine makes searchable PDFs here: the rest is delivered
                            log.warning("searchable PDF for %s not made: %s", doc_id, e)
                if status in ("validated", "invalid"):
                    for p in (Path(dv.params["path"]) / "validation").glob(f"{stem}_{doc_id[:8]}*"):
                        p.unlink(missing_ok=True)
                return True, str(folder), False
            if dv.type == "excel":
                self._append_excel(Path(dv.params["path"]), body)
                return True, dv.params["path"], False
            if dv.type in ("email", "cloud_storage"):
                return self._send_out(dv, doc_id, status, body)
            return False, f"unknown delivery type '{dv.type}'", False
        except OSError as e:   # folder offline, workbook open in Excel...: try again later
            return False, f"{type(e).__name__}: {e}", True
        except Exception as e:
            return False, f"{type(e).__name__}: {e}", False

    @staticmethod
    def _capabilities(fl: Flow, dt: Optional[DocType]) -> list[str]:
        """What the page reader is asked for. A process that names its own page engine and reads Ticked or Signed
        fields also asks for check boxes and signatures (Azure, Textract); the free engines are never excluded by it."""
        caps = list(fl.engine.capabilities or ["ocr_printed"])
        if dt and fl.engine.ocrProvider not in ("", "auto") and any(f.type in ("checkbox", "signature") for f in dt.fields) \
                and "selection_marks" not in caps:
            caps.append("selection_marks")
        return caps

    def _route(self, fl: Flow, doc_id: str, flag: dict) -> Optional[str]:
        """A failed check that sends the document to another process: the same file goes there as a new document.
        Returns that process's name, or None when it cannot be (missing, the same process: the check then reviews)."""
        try:
            target = self.flow(str(flag["routeTo"]))
        except InvalidInput:
            return None
        if target.id == fl.id:
            return None
        data, file_name, _ = self.store.file_bytes(doc_id)
        self.upload(target.id, data, file_name, source="routed")
        return target.name

    def _notify(self, fl: Flow, doc_id: str, flag: dict) -> None:
        """A failed check that tells people: a mail through the Command Center (never stops the document)."""
        to = [str(x).strip() for x in (flag.get("notify") or []) if str(x).strip()]
        if not to:
            return
        row = self.store.doc_row(doc_id) or {}
        try:
            self._platform("send-mail", {"requestedBy": flag.get("requestedBy") or "", "to": to,
                                         "subject": f"{fl.name}: {row.get('file_name') or doc_id}: {flag['rule']}",
                                         "body": f"{flag['message']}\n\nProcess: {fl.name}\nDocument: {row.get('file_name') or doc_id}"})
        except Exception as e:   # telling people is best effort: the document goes on
            log.warning("notify for %s failed: %s", doc_id, e)

    def _files(self, dv: Delivery, doc_id: str, body: dict) -> list[tuple[str, bytes, str]]:
        """The files a delivery sends: the original (decrypted), the results as JSON, and the text as Markdown, as its
        params ask (files: original, json, markdown; default original and json)."""
        want = dv.params.get("files") or ["original", "json"]
        stem = Path(body.get("fileName") or doc_id).stem
        out = []
        f = self.store.file_bytes(doc_id)
        if "original" in want and f:
            from .preprocess import sniff_mime
            out.append((f"{stem}_{doc_id[:8]}{Path(f[1]).suffix}", f[0], sniff_mime(f[0], f[1])))
        if "json" in want:
            out.append((f"{stem}_{doc_id[:8]}.json", json.dumps(body, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
                        "application/json"))
        if "markdown" in want:
            res = self.store.get_result(doc_id)
            if res:
                out.append((f"{stem}_{doc_id[:8]}.md", document_markdown(res, body).encode("utf-8"), "text/markdown"))
        return out

    def _platform(self, path: str, payload: dict, timeout: float = 30) -> dict:
        """A call to the Command Center's DocFlow callbacks (this service's token, the tenant)."""
        base = events.platform_base(self.tenant)
        if not base:
            raise RuntimeError("DOCAUTO_PLATFORM_URL is not set: DocFlow does not know where the Command Center is")
        r = httpx.post(base + "/api/docflow-callback/" + path, json=payload, timeout=timeout,
                       headers={"X-OOA-Tenant": self.tenant, **events.platform_auth()})
        if r.status_code >= 300:
            raise RuntimeError(f"the Command Center refused ({r.status_code}): {r.text[:300]}")
        return r.json() if r.content else {}

    def _send_out(self, dv: Delivery, doc_id: str, status: str, body: dict) -> tuple[bool, str, bool]:
        """E-mail (sent by the Command Center) and cloud storage (uploaded from here) deliveries."""
        import base64
        from . import storage_put
        files = self._files(dv, doc_id, body)
        try:
            if dv.type == "email":
                subject = str(dv.params.get("subject") or "{process}: {file} ({status})")
                for k, v in (("process", body.get("flowName") or ""), ("file", body.get("fileName") or doc_id),
                             ("status", status), ("type", body.get("docTypeName") or "")):
                    subject = subject.replace("{" + k + "}", str(v))
                lines = [f"{k}: {v}" for k, v in (body.get("fields") or {}).items() if not isinstance(v, (list, dict))]
                text = (str(dv.params.get("message")) + "\n\n" if dv.params.get("message") else "") + "\n".join(lines)
                self._platform("send-mail", {"requestedBy": dv.params.get("requestedBy", ""), "to": dv.params.get("to"),
                                             "subject": subject[:200], "body": text, "attachments": [
                                                 {"name": n, "contentType": t, "base64": base64.b64encode(b).decode()}
                                                 for n, b, t in files]})
                return True, "mail to " + ", ".join(map(str, dv.params.get("to") or [])), False
            secret = ""
            if dv.params.get("secretRef"):
                secret = str(self._platform("secret", {"requestedBy": dv.params.get("requestedBy", ""),
                                                       "ref": dv.params["secretRef"]}).get("value") or "")
            where = [storage_put.put_file(dv.params, n, b, t, secret, env_number("DOCAUTO_WEBHOOK_TIMEOUT", 60, 1, 600))
                     for n, b, t in files]
            return True, ", ".join(where), False
        except httpx.HTTPError as e:
            return False, f"{type(e).__name__}: {e}", True
        except RuntimeError as e:   # a 4xx answer is a refusal (wrong key, no access): trying again will not help
            return False, str(e), not re.search(r"\(4\d\d\)|HTTP 4\d\d", str(e))

    def _append_excel(self, path: Path, body: dict) -> None:
        from openpyxl import Workbook, load_workbook
        with self._xlsx_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            wb = load_workbook(path) if path.exists() else Workbook()
            sheet = re.sub(r"[\[\]:*?/\\]", "", body.get("docTypeName") or "Documents")[:31] or "Documents"
            fresh = sheet not in wb.sheetnames
            if fresh:
                ws = wb.create_sheet(sheet)
                if "Sheet" in wb.sheetnames and wb["Sheet"].max_row == 1 and wb["Sheet"].max_column == 1 \
                        and wb["Sheet"].cell(1, 1).value is None:
                    del wb["Sheet"]
            else:
                ws = wb[sheet]
            scal = {k: v for k, v in body["fields"].items() if not isinstance(v, (list, dict))}
            if fresh:
                head = ["document_id", "file_name", "status", "processed_at"] + list(scal)
                ws.append(head)
            else:
                head = [c.value for c in ws[1]]
            for k in scal:
                if k not in head:
                    head.append(k)
                    ws.cell(row=1, column=len(head), value=k)
            row = {"document_id": body["documentId"], "file_name": body["fileName"], "status": body["status"],
                   "processed_at": body["processedAt"], **scal}
            ws.append([row.get(h) for h in head])
            wb.save(path)

    def try_document(self, flow_id: str, data: bytes, file_name: str) -> dict:
        """A dry run of a process (3.4.1): the document is read as the process would read it (its engines, checks and
        review settings) and then forgotten. Nothing is kept, sent to review, delivered, routed or told."""
        from .preprocess import sniff_mime
        if sniff_mime(data, file_name) not in READABLE_MIME:
            raise InvalidInput(f"'{file_name}' is not a file Document Automation reads")
        fl = self.flow(flow_id)
        types = self.flow_doctypes(fl)
        if not types:
            raise InvalidInput("the process has no document type yet")
        dt = self._einvoice_type(fl, types, data, file_name) or types[0]
        note = None if len(types) == 1 or fl.classify.mode == "single" else \
            f"Read as {dt.name}: a try reads one document as the process's first type"
        ids = self.svc.submit(data, file_name, self._options(fl, dt, TRY_SOURCE), wait=True)
        doc_id = ids["documentId"]
        try:
            d = self.document(doc_id)
            res = self.store.get_result(doc_id)
            status = d["status"]
            would = [dv.name or dv.type for dv in fl.deliveries if dv.enabled and status in dv.on]
            return {"status": status, "doctypeId": d.get("doctype_id"), "doctypeName": dt.name, "fields": d["fields"],
                    "confidence": d["confidence"], "flags": d["flags"], "reviewReasons": d["reviewReasons"],
                    "warnings": (res.warnings if res else []) + ([note] if note else []), "deliveries": would,
                    "live": fl.status == "live", "error": d.get("error")}
        finally:
            self.store.delete_document(doc_id)

    def test_delivery(self, flow_id: str, delivery_id: str) -> list[dict]:
        fl = self.flow(flow_id)
        docs = self.store.instance_documents(flow_id, ["success", "validated"], limit=1)
        if not docs:
            raise InvalidInput("process at least one document in this flow first")
        return self._deliver(fl, docs[0]["id"], docs[0]["status"], only=delivery_id)

    # ---------------------------------------------------------------- teach from samples
    def test_engines(self, data: bytes, file_name: str, dt_id: str, engines: list[str], compare: str = "pages") -> dict:
        """Test lab: reads one document as a document type with each engine (no fallback to others), and forgets it
        again: nothing is filed in a process, sent to review or delivered. compare: "pages" (the engines read the
        page) or "fields" (3.4.2: the engines read the fields, from the same reading of the page)."""
        fields_compared = compare == "fields"
        dt = self.doctype(dt_id)
        runs = []
        for eng in (engines or ["auto"])[:2]:
            opt = ProcessOptions(**({"extractionProvider": eng or "auto"} if fields_compared else {"ocrProvider": eng or "auto"}),
                                 languages=dt.languages or ["en"], strictProvider=eng not in ("", "auto"), schema=dt.schema())
            t0 = time.time()
            try:
                res = self.svc.process_bytes(data, file_name, opt)
            except DocAutoError as e:
                runs.append({"engine": eng, "error": e.message})
                continue
            try:
                runs.append({"engine": eng, "ms": int((time.time() - t0) * 1000), "confidence": res.mean_line_confidence(),
                             "review": res.review.required,
                             "readBy": list(dict.fromkeys(s.providerId for s in res.provenance.steps
                                                          if s.stage == ("extract" if fields_compared else "digitize"))),
                             "fields": {f.name: {"value": f.value, "confidence": f.confidence, "passed": f.validation.passed,
                                                 "errors": f.validation.errors} for f in res.fields}})
            finally:
                self.store.delete_document(res.documentId)
        return {"docType": dt.id, "runs": runs,
                "fields": [{"name": f.name, "label": (f.label or f.name).split(" / ")[0], "required": f.required}
                           for f in dt.fields if f.type != "table"]}

    def suggest(self, files: list[tuple[bytes, str]]) -> dict:
        if not files:
            raise InvalidInput("upload 1 to 3 sample documents")
        results = []
        for data, name in files[:teach_max_samples()]:
            results.append(self.svc.process_bytes(data, name, ProcessOptions(languages=self.offered_languages())))
        proposal = suggest_from_results(results)
        llm = next((p for p in self.svc.registry.available("extraction")
                    if p.info.optionPack == "tier3_ai" and self.svc.policy.allows(p.info)
                    and hasattr(p.instance, "_chat")), None)
        if llm:
            try:
                proposal = self._ai_suggest(llm, results, proposal)
            except Exception as e:
                proposal["aiError"] = f"{type(e).__name__}: {e}"
        proposal["sampleText"] = [r.text[:3000] for r in results]
        return proposal

    def _ai_suggest(self, llm, results, base: dict) -> dict:
        if not llm.info.runsLocally and self.svc.mask_pii_default:   # cloud model: personal data masked first
            m = pii.PiiMasker()
            results = [m.mask_document(r) for r in results]
        prompt = ("You design data-extraction schemas. From the OCR text of sample documents below, propose the fields "
                  "a business would extract. Reply with JSON: {\"name\": str, \"category\": one of "
                  f"{list(LIBRARY) + ['custom']}, \"keywords\": [str], \"fields\": [{{\"name\": snake_case, \"type\": "
                  "string|number|currency|date|datetime|percentage|boolean|email|phone|address|enum|table, \"label\": \"the label as printed, in the document's language\", "
                  "\"required\": bool, \"columns\": [{\"name\", \"type\"}] for tables}]}.\n\n" +
                  "\n\n---\n\n".join(r.text[:6000] for r in results))
        data = json.loads(re.search(r"\{.*\}", llm.instance._chat(prompt), re.S).group(0))
        fields = []
        for f in data.get("fields", []):
            if _snake(f.get("name", "")):
                fields.append({"name": _snake(f["name"]), "type": f.get("type", "string") if f.get("type") in
                               ("string", "number", "currency", "date", "datetime", "percentage", "email", "phone",
                                "address", "enum", "table", "boolean") else "string",
                               "label": f.get("label"), "required": bool(f.get("required")), "aliases": [],
                               "columns": [{"name": _snake(c.get("name", "")) or "col", "type": c.get("type", "string")}
                                           for c in f.get("columns", [])] if f.get("type") == "table" else None,
                               "source": "AI suggestion"})
        return {**base, "name": data.get("name") or base["name"], "category": data.get("category", base["category"]),
                "keywords": data.get("keywords") or base["keywords"], "fields": fields or base["fields"],
                "method": f"ai:{llm.info.id}"}

    # ---------------------------------------------------------------- export & stats
    def export(self, flow_id: str, fmt: str = "csv", statuses: Optional[list[str]] = None) -> tuple[bytes, str, str]:
        fl = self.flow(flow_id)
        docs = self.store.instance_documents(flow_id, statuses or ["success", "validated"], limit=100000)
        rows, cols = [], []
        for d in docs:
            res = self.store.get_result(d["id"])
            vals = {f.name: f.out() for f in res.fields} if res else {}
            for k in vals:
                if k not in cols:
                    cols.append(k)
            rows.append((d, vals))
        head = ["document_id", "file_name", "doc_type", "status", "processed_at", "validated_by"] + cols
        table = [[d["id"], d["file_name"], d["doctype_id"] or "", d["status"], d["updated_at"], d["validated_by"] or ""] +
                 [json.dumps(v.get(c), ensure_ascii=False) if isinstance(v.get(c), (list, dict)) else
                  ("" if v.get(c) is None else v.get(c)) for c in cols] for d, v in rows]
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        if fmt == "xlsx":
            from openpyxl import Workbook
            wb = Workbook()
            ws = wb.active
            ws.title = "documents"
            ws.append(head)
            for r in table:
                ws.append(r)
            buf = io.BytesIO()
            wb.save(buf)
            return buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", f"{fl.id}-{stamp}.xlsx"
        s = io.StringIO()
        w = csv.writer(s)
        w.writerow(head)
        w.writerows(table)
        return ("﻿" + s.getvalue()).encode("utf-8"), "text/csv; charset=utf-8", f"{fl.id}-{stamp}.csv"

    def overview(self, days: int = 14) -> dict:
        flows = self.flows()
        per = []
        for f in flows:
            st = self.store.instance_stats(f.id)
            per.append({"id": f.id, "name": f.name, "isInbox": f.isInbox, "status": f.status, "stats": st})
        q = self.review_queue(None)
        month = datetime.now(timezone.utc).strftime("%Y-%m-01")
        used = self.store.month_usage(month)
        return {"totals": self.store.instance_stats(None, days), "flows": per, "review": len(q),
                "pagesThisMonth": used["p"], "creditsThisMonth": used["cr"],
                "overdue": sum(1 for x in q if x["overdue"]), "doctypes": len(self.doctypes()),
                "recent": self.store.instance_documents(None, None, limit=12),
                "deliveryFailures": sum(1 for d in self.store.delivery_list(None, 200) if not d["ok"])}

    # ---------------------------------------------------------------- folders + SLA watcher
    def scan_folder(self, fl: Flow, require_live: bool = True) -> list[dict]:
        src = Path(fl.intake.inputFolder) if fl.intake.inputFolder else None
        if not src or not self.folder_allowed(str(src), fl):
            return []
        if not src.is_dir():
            return []
        done = src / (os.environ.get("DOCAUTO_PROCESSED_FOLDER") or "processed")
        done.mkdir(exist_ok=True)
        settle = env_number("DOCAUTO_FILE_SETTLE_SECONDS", 2, 0, 600)   # a file still being copied is left until then
        out = []
        for p in sorted(src.iterdir()):
            if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXT or time.time() - p.stat().st_mtime < settle:
                continue
            try:
                data = p.read_bytes()
                target = done / p.name
                if target.exists():
                    target = done / f"{p.stem}_{uuid.uuid4().hex[:6]}{p.suffix}"
                shutil.move(str(p), str(target))
                out.append({"file": p.name, **self.upload(fl.id, data, p.name, source="folder", require_live=require_live)})
            except Exception as e:
                log.exception("folder intake failed for %s", p)
                out.append({"file": p.name, "error": str(e)})
        return out

    # ---------------------------------------------------------------- folders processes may use (3.2.0)
    def folder_allowed(self, path: str, fl: Optional[Flow] = None) -> bool:
        """May a process use this folder now? Says once in the log when a watched folder is outside the list."""
        if folders.allowed(path, self.tenant):
            return True
        if fl is not None and (fl.id, path) not in self._outside_said:
            self._outside_said.add((fl.id, path))
            log.warning("process %s: watched folder %s is not one of the folders processes may use; not read", fl.id, path)
        return False

    def folder_roots(self, value: Optional[str] = None) -> dict:
        """The folders processes may use (value: a list being edited, not saved yet), DocFlow's access to each, the
        processes using each, and the processes using a folder outside the list."""
        rs = folders.roots(self.tenant, value)
        uses = [(f, what, p) for f in self.flows() for what, _, p in self._folder_uses(f)]
        out = []
        for r in rs:
            by = {f.id: f.name for f, _, p in uses if folders.inside(p, [r]) is not None}
            out.append({"name": r.name, "path": r.path, "access": folders.access(r.path),
                        "usedBy": [{"id": k, "name": v} for k, v in by.items()]})
        outside = [{"id": f.id, "name": f.name, "use": what, "path": p}
                   for f, what, p in uses if rs and folders.inside(p, rs) is None]
        return {"limited": bool(rs), "roots": out, "outside": outside}

    def browse_folder(self, path: str, q: str = "") -> dict:
        """One folder inside the list: the folders in it (never files), how many files each would read straight away,
        and which process already watches each. Nothing outside the list is shown."""
        rs = folders.roots(self.tenant)
        if not rs:
            raise InvalidInput("No folders are listed for processes yet: an administrator adds them in Settings > "
                               "Document Automation")
        shown = os.path.normpath(path.strip()) if path.strip() else ""
        root = folders.inside(shown, rs) if shown and os.path.isabs(shown) else None
        if root is None:
            raise PolicyDenied(f"'{path}' is not one of the folders processes may use")
        if not os.path.isdir(shown):
            raise InvalidInput(f"Folder not found: {shown}")
        watched: dict[str, list[dict]] = {}
        for f in self.flows():
            if f.intake.inputFolder:
                watched.setdefault(folders.real(f.intake.inputFolder), []).append({"id": f.id, "name": f.name})
        subs, truncated = [], False
        try:
            with os.scandir(shown) as it:
                entries = sorted(it, key=lambda e: e.name.lower())
        except OSError as e:
            raise InvalidInput(f"DocFlow cannot open {shown}: {e.strerror or e}")
        for e in entries:
            try:
                if not e.is_dir() or folders.hidden(e):
                    continue
            except OSError:
                continue
            if q and q.casefold() not in e.name.casefold():   # only folders whose name has the words looked for
                continue
            sub = os.path.join(shown, e.name)
            if folders.inside(sub, [root]) is None:       # a link that points outside the allowed folder
                continue
            if len(subs) >= int(env_number("DOCAUTO_FOLDER_LIST_MAX", folders.MAX_FOLDERS, 10, 5000)):
                truncated = True
                break
            subs.append({"name": e.name, "path": sub, "waiting": folders.count_files(sub, SUPPORTED_EXT),
                         "watchedBy": watched.get(folders.real(sub), [])})
        at_root = folders.real(shown) == root.real
        return {"path": shown, "name": root.name if at_root else os.path.basename(shown),
                "root": {"name": root.name, "path": root.path},
                "parent": None if at_root else os.path.dirname(shown),
                "access": folders.access(shown), "waiting": folders.count_files(shown, SUPPORTED_EXT),
                "watchedBy": watched.get(folders.real(shown), []), "folders": subs, "truncated": truncated,
                "shownMax": int(env_number("DOCAUTO_FOLDER_LIST_MAX", folders.MAX_FOLDERS, 10, 5000))}

    def make_folder(self, parent: str, name: str) -> dict:
        """A new folder inside a folder processes may use; returns it as browse_folder does."""
        try:
            name = folders.check_name(name)
        except ValueError as e:
            raise InvalidInput(str(e))
        here = self.browse_folder(parent)                  # inside the list, exists
        target = os.path.join(here["path"], name)
        if os.path.exists(target):
            raise InvalidInput(f"There is already a folder or file named '{name}' in {here['path']}")
        try:
            os.mkdir(target)
        except OSError as e:
            raise InvalidInput(f"DocFlow cannot make the folder {target}: {e.strerror or e}")
        return self.browse_folder(target)

    def check_sla(self) -> int:
        n = 0
        for fl in self.flows():
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=fl.review.slaHours)).isoformat()
            for d in self.store.overdue_reviews(fl.id, cutoff):
                if fl.review.slaWebhook:
                    try:
                        httpx.post(fl.review.slaWebhook, json={"event": "review.overdue", "flowId": fl.id,
                                                               "flowName": fl.name, "documentId": d["id"],
                                                               "fileName": d["file_name"], "receivedAt": d["created_at"],
                                                               "slaHours": fl.review.slaHours},
                                                               timeout=env_number("DOCAUTO_WEBHOOK_TIMEOUT", 15, 1, 600))
                    except httpx.HTTPError:
                        log.warning("SLA webhook failed for %s", d["id"])
                with self.store.tx():
                    self.store.set_doc_fields(d["id"], sla_notified=1)
                    self._event("review_overdue", d["id"], fl)
                n += 1
        return n

    # ---------------------------------------------------------------- retention + housekeeping
    def enforce_retention(self, force: bool = False) -> int:
        """Delete documents older than storage.retention_days (DOCAUTO_RETENTION_DAYS; 0 = keep forever), at most
        once a day unless forced. Documents still in work and test-set files are kept. Returns how many went."""
        days = float(self.svc.config.get("storage", {}).get("retention_days") or 0)
        if days <= 0:
            return 0
        now = datetime.now(timezone.utc)
        last = self.store.meta_get("retention_last_run")
        if not force and last and now - datetime.fromisoformat(last) < timedelta(hours=23):
            return 0
        n = self.store.purge_documents((now - timedelta(days=days)).isoformat())
        self.store.meta_set("retention_last_run", now.isoformat())
        if n:
            log.info("retention: deleted %d document(s) older than %g days (tenant %s)", n, days, self.tenant)
        return n

    def _maintenance_loop(self) -> None:
        self.testsets.mark_interrupted()
        while True:
            try:
                self.enforce_retention()
            except Exception:
                log.exception("retention failed")
            if self._stop.wait(3600):
                return

    def start_watcher(self, interval: Optional[float] = None) -> None:
        if self._watcher and self._watcher.is_alive():
            return
        interval = interval or env_number("DOCAUTO_WATCH_SECONDS", 5, 1, 3600)

        def loop():
            last_sla = 0.0
            while not self._stop.wait(interval):
                for fl in self.flows():
                    if fl.status != "live" or not fl.intake.inputFolder:
                        continue
                    if time.time() - self._last_poll.get(fl.id, 0) < max(5, fl.intake.pollSeconds):
                        continue
                    self._last_poll[fl.id] = time.time()
                    try:
                        self.scan_folder(fl)
                    except Exception:
                        log.exception("watcher failed for %s", fl.id)
                if time.time() - last_sla > 60:
                    last_sla = time.time()
                    try:
                        self.check_sla()
                    except Exception:
                        log.exception("SLA check failed")

        self._watcher = threading.Thread(target=loop, name="docflow-watcher", daemon=True)
        self._watcher.start()

    def stop_watcher(self) -> None:
        """Stop this DocFlow's background threads: folder watcher, outbox sender, housekeeping."""
        self._stop.set()
        self.outbox.stop()

    # ---------------------------------------------------------------- migration from v2 learning instances
    def migrate_instances(self) -> int:
        n = 0
        for body in self.store.list_instance_bodies():
            try:
                inst = json.loads(body)
            except ValueError:
                continue
            if self.store.df_get("flow", inst["id"]):
                continue
            dt = self.save_doctype(DocType(id=f"{inst['id']}-type", name=inst["name"], description=inst.get("description", ""),
                                           category=inst.get("documentType", "custom"),
                                           keywords=LIBRARY.get(inst.get("documentType", ""), {}).get("keywords", []),
                                           languages=inst.get("languages", ["en"]),
                                           fields=[FieldDef.model_validate(f) for f in inst.get("fields", [])],
                                           checks=inst.get("validation", [])), new=True)
            eng = inst.get("engine", {})
            self.save_flow(Flow(id=inst["id"], name=inst["name"], description=inst.get("description", ""), docTypes=[dt.id],
                                engine=EngineChoice(**{k: v for k, v in eng.items() if k in EngineChoice.model_fields}),
                                review=ReviewSettings(fieldConfidence=inst.get("fieldConfidence", 0.85),
                                                      documentConfidence=inst.get("documentConfidence", 0.8)),
                                intake=Intake(inputFolder=inst.get("folders", {}).get("inputFolder", "")),
                                deliveries=[Delivery(type="folder", name="Output folder",
                                                     params={"path": inst["folders"]["outputFolder"]},
                                                     on=["success", "validation", "validated", "invalid", "failed"])]
                                if inst.get("folders", {}).get("outputFolder") else [],
                                status="live" if inst.get("status") == "published" else "draft"), new=True,
                           check_folders=False)
            n += 1
        return n
