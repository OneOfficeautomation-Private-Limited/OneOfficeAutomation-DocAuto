"""PII masking before text goes to a cloud AI model (Claude, llm-cloud).

Personal data in the OCR text is replaced by placeholders ([EMAIL_1], [IBAN_1], [CARD_1], [ID_1]...) before the
request; the map stays in memory, and the values the model returns are restored (a field read as "[IBAN_1]" becomes
the IBAN again). Masking is reversible, so a number masked by mistake only hides it from the model.

Which kinds are masked is a setting (DOCAUTO_PII_PATTERNS: ids from CATALOG, comma separated; empty = every one),
and an administrator may add own ones (DOCAUTO_PII_EXTRA: "NAME=pattern; NAME2=pattern").

What it does not cover: page images. A picture of the page cannot be masked, so while masking is on, cloud models get
no page images unless the process allows them (EngineChoice.allowImages / ProcessOptions.cloudImages). Local engines
(llm-local, builtin-rules) are never masked: nothing leaves the machine.
"""
from __future__ import annotations

import os
import re
from typing import Any

from .models import DocumentResult, ExtractedField

_DIGIT = "0-9٠-٩"          # Western and Arabic-Indic digits (OCR of Arabic documents gives both)
_ND = rf"(?<![{_DIGIT}])"
_NA = rf"(?![{_DIGIT}])"
# id -> (placeholder kind, what it is, pattern). General kinds first, then national ids by country.
CATALOG: dict[str, tuple[str, str, str]] = {
    "email": ("EMAIL", "E-mail addresses", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
    # IBAN: 2 letters, 2 check digits, 11-30 more (spaces allowed every 4); kept when the checksum holds
    "iban": ("IBAN", "IBANs (any country)", r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b"),
    # payment cards: 13 to 19 digits, spaces or dashes in groups; kept only when the Luhn check holds
    "card": ("CARD", "Payment card numbers", r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
    # international numbers (+44 / 0044 ...), and local mobile / landline forms starting with 0
    "phone": ("PHONE", "Phone numbers", r"(?<![\w+])(?:\+|00)\d{1,3}[ -]?\(?\d{1,4}\)?(?:[ -]?\d{2,4}){2,4}(?!\d)"
                                       r"|(?<!\d)0(?:5\d|1\d)[ -]?\d{3}[ -]?\d{4}(?!\d)"),
    "sa-id": ("ID", "Saudi national ID / Iqama", rf"{_ND}[12١٢][{_DIGIT}]{{9}}{_NA}"),
    "ae-id": ("ID", "UAE Emirates ID", r"(?<!\d)784-?\d{4}-?\d{7}-?\d(?!\d)"),
    "in-aadhaar": ("ID", "India Aadhaar", r"(?<!\d)[2-9]\d{3}[ -]\d{4}[ -]\d{4}(?!\d)"),
    "in-pan": ("ID", "India PAN", r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "us-ssn": ("ID", "US Social Security number", r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)"),
    "uk-nino": ("ID", "UK National Insurance number", r"\b(?!BG|GB|NK|KN|TN|NT|ZZ)[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z] ?\d{2} ?\d{2} ?\d{2} ?[A-D]\b"),
    "cn-id": ("ID", "China resident ID", r"(?<!\d)\d{17}[\dXx](?![\dXx])"),
    "br-cpf": ("ID", "Brazil CPF", r"(?<!\d)\d{3}\.\d{3}\.\d{3}-\d{2}(?!\d)"),
    "za-id": ("ID", "South Africa ID", r"(?<!\d)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{7}(?!\d)"),
}


def _luhn_ok(s: str) -> bool:
    d = [int(c) for c in re.sub(r"\D", "", s)]
    if not 13 <= len(d) <= 19:
        return False
    total = sum(d[-1::-2]) + sum(sum(divmod(2 * x, 10)) for x in d[-2::-2])
    return total % 10 == 0


def active_patterns() -> list[tuple[str, str, re.Pattern]]:
    """(catalog id, placeholder kind, pattern) of the kinds masked now: DOCAUTO_PII_PATTERNS (empty = all) and the
    administrator's own DOCAUTO_PII_EXTRA ("NAME=pattern; ..."); a bad own pattern is skipped."""
    want = [x.strip() for x in (os.environ.get("DOCAUTO_PII_PATTERNS") or "").split(",") if x.strip()]
    out = [(k, kind, re.compile(p)) for k, (kind, _, p) in CATALOG.items() if not want or k in want]
    for part in re.split(r";\s*", os.environ.get("DOCAUTO_PII_EXTRA") or ""):
        name, _, pat = part.partition("=")
        name = re.sub(r"[^A-Z0-9]", "", name.strip().upper())
        if name and pat.strip():
            try:
                out.append(("own:" + name, name, re.compile(pat.strip())))
            except re.error:
                continue
    return out


# kept for callers of 3.2: the patterns as (kind, pattern)
PATTERNS: list[tuple[str, re.Pattern]] = [(kind, re.compile(p)) for _, (kind, _, p) in CATALOG.items()]
TOKEN = re.compile(r"\[?\b([A-Z][A-Z0-9]*)_(\d+)\b\]?")


def _iban_ok(s: str) -> bool:
    s = s.replace(" ", "")
    rearranged = s[4:] + s[:4]
    try:
        return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1
    except ValueError:
        return False


class PiiMasker:
    """Masks text consistently (the same value always gets the same placeholder) and restores values."""

    def __init__(self):
        self.values: dict[str, str] = {}      # placeholder -> original
        self._tokens: dict[str, str] = {}     # original -> placeholder
        self._count: dict[str, int] = {}

    def _token(self, kind: str, value: str) -> str:
        tok = self._tokens.get(value)
        if tok is None:
            self._count[kind] = self._count.get(kind, 0) + 1
            tok = f"[{kind}_{self._count[kind]}]"
            self._tokens[value], self.values[tok] = tok, value
        return tok

    def mask(self, text: str) -> str:
        if not text:
            return text
        for cid, kind, pat in active_patterns():
            def sub(m: re.Match, kind=kind, cid=cid) -> str:
                v = m.group(0)
                if (cid == "iban" and not _iban_ok(v)) or (cid == "card" and not _luhn_ok(v)) or v.startswith("["):
                    return v
                return self._token(kind, v)
            text = pat.sub(sub, text)
        return text

    @property
    def masked(self) -> int:
        return len(self.values)

    def unmask(self, value: Any) -> Any:
        if isinstance(value, str):
            return TOKEN.sub(lambda m: self.values.get(f"[{m.group(1)}_{m.group(2)}]", m.group(0)), value)
        if isinstance(value, list):
            return [self.unmask(v) for v in value]
        if isinstance(value, dict):
            return {k: self.unmask(v) for k, v in value.items()}
        return value

    def mask_document(self, doc: DocumentResult) -> DocumentResult:
        """A copy of the document with every text masked (same pages, lines and boxes, so citations still work)."""
        out = doc.model_copy(deep=True)
        for page in out.pages:
            for line in page.lines:
                line.text = self.mask(line.text)
            for w in page.words:
                w.text = self.mask(w.text)
            for b in page.blocks:
                b.text = self.mask(b.text)
            for t in page.tables:
                for c in t.cells:
                    c.text = self.mask(c.text)
            for kv in page.keyValuePairs:
                kv.key, kv.value = self.mask(kv.key), self.mask(kv.value)
            for bc in page.barcodes:
                bc.text = self.mask(bc.text)
        out.raw = None
        out.text = ""
        out.rebuild_text()
        return out

    def unmask_fields(self, fields: list[ExtractedField]) -> list[ExtractedField]:
        if not self.values:
            return fields
        from .extraction import coerce
        for f in fields:
            new = self.unmask(f.value)
            if new != f.value and isinstance(new, str) and f.type not in ("string", "enum", "table", "object"):
                typed, ok = coerce(new, f.type)
                new = typed if ok else new
            f.value = new
            if f.rawText:
                f.rawText = self.unmask(f.rawText)
        return fields
