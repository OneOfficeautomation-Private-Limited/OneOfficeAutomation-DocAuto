"""Folders processes may use (DOCAUTO_ALLOWED_FOLDERS, DocFlow 3.2.0).

An administrator lists folders on DocFlow's server. A process may watch, and save results in (folder and Excel
deliveries), only those folders and the folders inside them; people choose from the list in the Command Center instead
of typing a network path. An empty list keeps the behaviour before 3.2.0: any path.

The value is one line: folders separated by ';', each optionally named for people as 'Name=folder', such as
``Scans=\\\\scan01\\scans; D:\\DocFlow\\incoming``. In SaaS, ``{tenant}`` in a path is replaced by the tenant id.

Containment is decided on real paths (``os.path.realpath``: '..', symbolic links and junctions resolved, case folded
on Windows), so a link inside an allowed folder that points elsewhere is outside. Paths shown to people keep the form
the administrator wrote (a mapped drive stays a drive letter).
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

ENV = "DOCAUTO_ALLOWED_FOLDERS"
MAX_FOLDERS = 300      # subfolders listed per folder
MAX_COUNT = 1000       # files counted per folder ("1000" means at least that many)
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_BAD_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class Root:
    name: str          # what people see
    path: str          # as the administrator wrote it ({tenant} replaced)
    real: str          # real, case-folded path: what containment is decided on


def real(path: str) -> str:
    """The real, case-folded form of a path (for comparisons only, never shown)."""
    return os.path.normcase(os.path.realpath(path))


def parse(value: str, tenant: str = "default", strict: bool = False) -> list[Root]:
    """The listed folders. strict: raise ValueError for a folder that is not a full path (when an administrator saves)."""
    out: list[Root] = []
    seen: set[str] = set()
    for part in (value or "").split(";"):
        part = part.strip()
        if not part:
            continue
        name, sep, path = part.partition("=")
        if not sep or re.search(r"[\\/:]", name):      # no name given: the '=' (if any) belongs to the path
            name, path = "", part
        name, path = name.strip(), path.strip().replace("{tenant}", tenant or "default")
        if not path or not os.path.isabs(path):
            if strict:
                raise ValueError(f"'{part}' is not a full folder path, such as D:\\Scans or \\\\server\\share\\folder")
            continue
        r = real(path)
        if r in seen:
            continue
        seen.add(r)
        out.append(Root(name or Path(path).name or path, path, r))
    return out


def roots(tenant: str = "default", value: str | None = None) -> list[Root]:
    """The folders processes may use now (value: a list being edited, not saved yet)."""
    return parse(os.environ.get(ENV, "") if value is None else value, tenant)


def inside(path: str, rs: list[Root]) -> Root | None:
    """The listed folder that holds path (itself included), or None."""
    if not path:
        return None
    p = real(path)
    for r in rs:
        try:
            if os.path.commonpath([r.real, p]) == r.real:
                return r
        except ValueError:          # different drives or a relative path on another drive
            continue
    return None


def allowed(path: str, tenant: str = "default") -> bool:
    """May a process use this folder? Always, while no folders are listed (the behaviour before 3.2.0)."""
    rs = roots(tenant)
    return not rs or inside(path, rs) is not None


def access(path: str) -> str:
    """What DocFlow's account can do in a folder: ok (read and move files), read (read only), missing, denied."""
    p = Path(path)
    try:
        if not p.is_dir():
            return "missing"
        with os.scandir(p) as it:
            next(it, None)
    except OSError:
        return "denied"
    try:    # a real write test: on Windows os.access ignores the folder's permissions
        with tempfile.NamedTemporaryFile(dir=p, prefix=".docflow-check-"):
            pass
    except OSError:
        return "read"
    return "ok"


def hidden(entry: os.DirEntry) -> bool:
    if entry.name.startswith((".", "$")):
        return True
    try:
        attrs = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
    except OSError:
        return True
    return bool(attrs & 0x6)        # FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM


def count_files(path: str, exts: set[str]) -> int:
    """Files a watched folder would read straight away (at most MAX_COUNT; looks at no more than 5 x MAX_COUNT entries)."""
    n = looked = 0
    try:
        with os.scandir(path) as it:
            for e in it:
                looked += 1
                if looked > 5 * MAX_COUNT or n >= MAX_COUNT:
                    break
                try:
                    if e.is_file() and os.path.splitext(e.name)[1].lower() in exts:
                        n += 1
                except OSError:
                    continue
    except OSError:
        return 0
    return n


def check_name(name: str) -> str:
    """A new folder's name, cleaned; raises ValueError when it cannot be a folder name."""
    name = (name or "").strip().rstrip(". ")
    if not name or name in (".", "..") or len(name) > 120 or _BAD_NAME.search(name) \
            or name.split(".")[0].upper() in _RESERVED:
        raise ValueError("Use a folder name without \\ / : * ? \" < > |, up to 120 characters")
    return name
