"""
Recognising what kind of document a page is, with an AI engine (3.4.0): the engine is given every document type's
name and plain-words description, and answers with one of the names, or "other" when none fits. The same question for
every AI engine (Claude, OpenAI-compatible, self-hosted and the ones that follow), so switching engines changes nothing
for processes or types. The words-on-the-page classifier (docflow.classify_page) stays the offline way.
"""
from __future__ import annotations

from typing import Optional

#: how much of a page's text an engine is shown to recognise it
MAX_CHARS = 8000


def type_list(labels: list[str], descriptions: Optional[dict[str, str]] = None) -> str:
    """One line per type: '- Invoice: A bill from a supplier ...' (the name alone when it has no description)."""
    out = []
    for label in labels:
        d = ((descriptions or {}).get(label) or "").strip()
        out.append(f"- {label}: {d}" if d else f"- {label}")
    return "\n".join(out)


def question(labels: list[str], descriptions: Optional[dict[str, str]], text: str) -> str:
    """The question every AI engine is asked to recognise a document."""
    return ("Which kind of document is this? Choose exactly one of the types below, by its name, using each type's "
            "description; answer \"other\" when none of them fits.\n\nTypes:\n" + type_list(labels, descriptions)
            + "\n\nDocument:\n" + (text or "")[:MAX_CHARS])
