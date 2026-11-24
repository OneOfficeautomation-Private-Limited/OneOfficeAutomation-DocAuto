"""SDK helpers used by option packs to map vendor output into DocumentResult."""
from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

from .models import BBox

_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")


def write_as(value, ftype: str, fmt: str | None):
    """The value written as a field's "write as" asks, or None when nothing changes. Dates and date-times take a
    pattern of YYYY, MM, DD, MMM, MMMM, HH, hh, mm, ss, A ("DD/MM/YYYY"); numbers "N decimals"; text UPPER, lower,
    Title Case, Trimmed; phones "Digits only", "International (+)". An unknown format leaves the value as read."""
    import re as _re
    from datetime import datetime as _dt
    if value in (None, "", []) or not fmt:
        return None
    f = fmt.strip()
    m = _re.fullmatch(r"(\d)\s*decimals?", f, _re.I)
    if m and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.{int(m.group(1))}f}"
    if ftype in ("date", "datetime") and isinstance(value, str):
        try:
            d = _dt.fromisoformat(value)
        except ValueError:
            return None
        py = (f.replace("YYYY", "%Y").replace("MMMM", "%B").replace("MMM", "%b").replace("MM", "%m")
              .replace("DD", "%d").replace("D", "%-d").replace("HH", "%H").replace("hh", "%I").replace("mm", "%M")
              .replace("ss", "%S").replace(" A", " %p"))
        try:
            return d.strftime(py)
        except ValueError:       # %-d is not known on Windows
            return d.strftime(py.replace("%-d", "%d").lstrip("0"))
    if isinstance(value, str):
        low = f.lower()
        if low == "upper":
            return value.upper()
        if low == "lower":
            return value.lower()
        if low == "title case":
            return value.title()
        if low == "trimmed":
            return " ".join(value.split())
        if low == "digits only":
            return _re.sub(r"\D", "", value)
        if low.startswith("international"):
            digits = _re.sub(r"[^\d+]", "", value)
            return digits if digits.startswith("+") else "+" + digits.lstrip("0")
    return None


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def bbox_from_pixels(left: float, top: float, width: float, height: float,
                     page_w: float, page_h: float) -> BBox:
    if page_w <= 0 or page_h <= 0:
        raise ValueError("page size must be positive")
    return BBox(x=clamp01(left / page_w), y=clamp01(top / page_h),
                w=clamp01(width / page_w), h=clamp01(height / page_h))


def polygon_to_bbox(points: Sequence[Sequence[float]], page_w: float = 1.0, page_h: float = 1.0) -> BBox:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return bbox_from_pixels(x0, y0, x1 - x0, y1 - y0, page_w, page_h)


def flat_polygon_to_points(flat: Sequence[float]) -> list[tuple[float, float]]:
    """Azure-style [x1,y1,x2,y2,...] -> [(x1,y1),...]."""
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]


def conf100(v: Optional[float]) -> Optional[float]:
    if v is None or v < 0:
        return None
    return clamp01(v / 100.0)


def union_bbox(boxes: Iterable[BBox]) -> Optional[BBox]:
    boxes = list(boxes)
    if not boxes:
        return None
    x0 = min(b.x for b in boxes)
    y0 = min(b.y for b in boxes)
    x1 = max(b.x + b.w for b in boxes)
    y1 = max(b.y + b.h for b in boxes)
    return BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


# a letter's script, from the start of its Unicode name, and the language code the script most likely means
_SCRIPT_LANG = {"ARABIC": "ar", "HEBREW": "he", "CYRILLIC": "ru", "GREEK": "el", "CJK": "zh", "HIRAGANA": "ja",
                "KATAKANA": "ja", "HANGUL": "ko", "DEVANAGARI": "hi", "BENGALI": "bn", "TAMIL": "ta", "TELUGU": "te",
                "KANNADA": "kn", "MALAYALAM": "ml", "GUJARATI": "gu", "GURMUKHI": "pa", "THAI": "th", "LAO": "lo",
                "KHMER": "km", "MYANMAR": "my", "GEORGIAN": "ka", "ARMENIAN": "hy", "ETHIOPIC": "am", "SINHALA": "si",
                "LATIN": "en", "SYRIAC": "syr", "THAANA": "dv", "NKO": "nqo"}
_RTL_SCRIPTS = {"ARABIC", "HEBREW", "SYRIAC", "THAANA", "NKO"}


def _script(c: str) -> str:
    import unicodedata
    try:
        n = unicodedata.name(c)
    except ValueError:
        return ""
    if _ARABIC.match(c):
        return "ARABIC"
    head = n.split(" ", 1)[0]
    return "CJK" if head in ("CJK", "IDEOGRAPHIC") else head


def scripts(text: str) -> dict[str, int]:
    """How many letters of each script the text has."""
    out: dict[str, int] = {}
    for c in text:
        if c.isalpha():
            s = _script(c)
            if s:
                out[s] = out.get(s, 0) + 1
    return out


def detect_direction(text: str) -> str:
    """'rtl' when most letters are of a right-to-left script (Arabic, Hebrew, Syriac, Thaana, N'Ko)."""
    counts = scripts(text)
    total = sum(counts.values())
    if not total:
        return "ltr"
    return "rtl" if sum(v for k, v in counts.items() if k in _RTL_SCRIPTS) / total > 0.5 else "ltr"


def detect_language(text: str) -> Optional[str]:
    """The likely language from the main script of the text (Latin is reported as "en": the script, not the
    language, is known). Japanese text mixes kana and Chinese characters: any kana makes it Japanese."""
    counts = scripts(text)
    if not counts:
        return None
    if counts.get("HIRAGANA") or counts.get("KATAKANA"):
        return "ja"
    main = max(counts, key=counts.get)
    return _SCRIPT_LANG.get(main, "en")
