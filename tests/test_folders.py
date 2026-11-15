"""Folders processes may use (DOCAUTO_ALLOWED_FOLDERS, 3.2.0): the list, the folder picker's browse, the save check,
and the watcher and deliveries leaving a folder outside the list alone."""
import os
import sys

import pytest
from fastapi.testclient import TestClient

from conftest import make_config

DF = "/api/docs/v1/df"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCAUTO_FOLDER_WATCHER", "false")
    monkeypatch.delenv("DOCAUTO_ALLOWED_FOLDERS", raising=False)
    monkeypatch.setenv("DOCAUTO_ENV_FILE", str(tmp_path / "local.env"))
    import docauto.api.app as appmod
    from docauto.pipeline import DocumentService
    appmod.set_service(DocumentService(make_config()))
    return TestClient(appmod.app)


def flow(name, watched="", results=None, excel=None):
    deliveries = []
    if results:
        deliveries.append({"type": "folder", "name": "Archive", "params": {"path": str(results)}})
    if excel:
        deliveries.append({"type": "excel", "name": "Register", "params": {"path": str(excel)}})
    return {"name": name, "docTypes": ["invoice"], "intake": {"inputFolder": str(watched)}, "deliveries": deliveries}


def tree(tmp_path):
    scans = tmp_path / "scans"
    for d in ("2025", "riyadh", "jeddah", ".hidden"):
        (scans / d).mkdir(parents=True)
    (scans / "riyadh" / "a.pdf").write_bytes(b"%PDF-1.4")
    (scans / "riyadh" / "b.PNG").write_bytes(b"png")
    (scans / "riyadh" / "notes.txt").write_text("not read")
    (tmp_path / "out").mkdir()
    (tmp_path / "elsewhere").mkdir()
    return scans


def link(target, at):
    """A junction (Windows, no administrator rights needed) or a symbolic link; None when neither can be made."""
    try:
        if sys.platform == "win32":
            import _winapi
            _winapi.CreateJunction(str(target), str(at))
        else:
            os.symlink(target, at, target_is_directory=True)
        return at
    except OSError:
        return None


def test_without_a_list_any_folder_works_as_before(client, tmp_path):
    r = client.post(f"{DF}/flows", json=flow("Old way", tmp_path / "anywhere", results=tmp_path / "out"))
    assert r.status_code == 200, r.text
    assert client.get(f"{DF}/folders").json() == {"limited": False, "roots": [], "outside": []}
    r = client.get(f"{DF}/folders", params={"path": str(tmp_path)})
    assert r.status_code == 400 and "No folders are listed" in r.json()["message"]


def test_the_list_limits_watched_and_result_folders(client, tmp_path, monkeypatch):
    scans = tree(tmp_path)
    monkeypatch.setenv("DOCAUTO_ALLOWED_FOLDERS", f"Scans={scans}; {tmp_path / 'out'}")
    ok = client.post(f"{DF}/flows", json=flow("AP", scans / "riyadh", results=tmp_path / "out" / "ap",
                                               excel=tmp_path / "out" / "ap.xlsx"))
    assert ok.status_code == 200, ok.text
    r = client.post(f"{DF}/flows", json=flow("Bad watch", tmp_path / "elsewhere"))
    assert r.status_code == 400 and "watched folder" in r.json()["message"] \
        and "not one of the folders processes may use" in r.json()["message"]
    r = client.post(f"{DF}/flows", json=flow("Bad results", scans, results=tmp_path / "elsewhere"))
    assert r.status_code == 400 and "delivery 'Archive'" in r.json()["message"]
    r = client.post(f"{DF}/flows", json=flow("Bad register", excel=tmp_path / "elsewhere" / "x.xlsx"))
    assert r.status_code == 400 and "delivery 'Register'" in r.json()["message"]
    r = client.post(f"{DF}/flows", json=flow("Climbs out", str(scans) + os.sep + ".." + os.sep + "elsewhere"))
    assert r.status_code == 400                                              # '..' is resolved before the check


def test_a_folder_kept_from_before_the_list_is_left_alone(client, tmp_path, monkeypatch):
    scans = tree(tmp_path)
    old = client.post(f"{DF}/flows", json=flow("Old scans", tmp_path / "elsewhere", results=tmp_path / "elsewhere")).json()
    old_res = client.put(f"{DF}/flows/{old['id']}", json={"status": "live"})
    assert old_res.status_code == 200, old_res.text
    monkeypatch.setenv("DOCAUTO_ALLOWED_FOLDERS", str(scans))
    # the process can still be edited while its folders stay as they were...
    r = client.put(f"{DF}/flows/{old['id']}", json={"name": "Old scans (renamed)"})
    assert r.status_code == 200, r.text
    # ...but a new folder outside the list is refused
    r = client.put(f"{DF}/flows/{old['id']}", json={"intake": {"inputFolder": str(tmp_path / "out")}})
    assert r.status_code == 400
    # the administrator sees it, and DocFlow neither reads nor writes there
    roots = client.get(f"{DF}/folders").json()
    assert roots["limited"] and {(o["name"], o["use"]) for o in roots["outside"]} == \
        {("Old scans (renamed)", "watched"), ("Old scans (renamed)", "results")}
    (tmp_path / "elsewhere" / "c.pdf").write_bytes(b"%PDF-1.4")
    r = client.post(f"{DF}/flows/{old['id']}/intake/scan")
    assert r.status_code == 400 and "not one of the folders" in r.json()["detail"]
    import docauto.api.app as appmod
    from docauto.docflow import Delivery
    df = appmod.docflow()
    assert df.scan_folder(df.flow(old["id"]), require_live=False) == []
    assert (tmp_path / "elsewhere" / "c.pdf").exists()
    ok, detail, retry = df._send_delivery(Delivery(type="folder", params={"path": str(tmp_path / "elsewhere")}),
                                          "doc12345", "success", {"fileName": "c.pdf"})
    assert not ok and not retry and "not one of the folders" in detail
    assert not (tmp_path / "elsewhere" / "success").exists()


def test_browse_shows_folders_inside_the_list_only(client, tmp_path, monkeypatch):
    scans = tree(tmp_path)
    monkeypatch.setenv("DOCAUTO_ALLOWED_FOLDERS", f"Scans={scans}")
    jed = client.post(f"{DF}/flows", json=flow("Credit notes", scans / "jeddah")).json()
    roots = client.get(f"{DF}/folders").json()
    assert roots["limited"] and roots["outside"] == []
    assert [(r["name"], r["path"], r["access"]) for r in roots["roots"]] == [("Scans", str(scans), "ok")]
    assert roots["roots"][0]["usedBy"] == [{"id": jed["id"], "name": "Credit notes"}]

    top = client.get(f"{DF}/folders", params={"path": str(scans)}).json()
    assert top["name"] == "Scans" and top["parent"] is None and top["access"] == "ok"
    assert [f["name"] for f in top["folders"]] == ["2025", "jeddah", "riyadh"]          # hidden folders left out
    by = {f["name"]: f for f in top["folders"]}
    assert by["riyadh"]["waiting"] == 2 and by["2025"]["waiting"] == 0                  # PDF and PNG, not the .txt
    assert by["jeddah"]["watchedBy"] == [{"id": jed["id"], "name": "Credit notes"}]
    assert by["riyadh"]["path"] == os.path.join(str(scans), "riyadh")

    inner = client.get(f"{DF}/folders", params={"path": by["riyadh"]["path"]}).json()
    assert inner["name"] == "riyadh" and inner["parent"] == str(scans) and inner["waiting"] == 2
    for outside in (str(tmp_path), str(scans) + os.sep + "..", "relative"):
        assert client.get(f"{DF}/folders", params={"path": outside}).status_code == 403, outside
    assert client.get(f"{DF}/folders", params={"path": str(scans / "missing")}).status_code == 400


def test_a_link_out_of_an_allowed_folder_is_outside(client, tmp_path, monkeypatch):
    scans = tree(tmp_path)
    (tmp_path / "secret").mkdir()
    if link(tmp_path / "secret", scans / "shortcut") is None:
        pytest.skip("cannot make a junction or symbolic link here")
    monkeypatch.setenv("DOCAUTO_ALLOWED_FOLDERS", str(scans))
    names = [f["name"] for f in client.get(f"{DF}/folders", params={"path": str(scans)}).json()["folders"]]
    assert "shortcut" not in names
    assert client.get(f"{DF}/folders", params={"path": str(scans / "shortcut")}).status_code == 403
    assert client.post(f"{DF}/flows", json=flow("Sneaky", scans / "shortcut")).status_code == 400


def test_new_folder_inside_the_list(client, tmp_path, monkeypatch):
    scans = tree(tmp_path)
    monkeypatch.setenv("DOCAUTO_ALLOWED_FOLDERS", str(scans))
    r = client.post(f"{DF}/folders", json={"path": str(scans), "name": "  returns. "})
    assert r.status_code == 200, r.text
    assert r.json()["path"] == os.path.join(str(scans), "returns") and (scans / "returns").is_dir()
    for bad in ("a/b", "..", "con", "x" * 121, "what?"):
        assert client.post(f"{DF}/folders", json={"path": str(scans), "name": bad}).status_code == 400, bad
    assert client.post(f"{DF}/folders", json={"path": str(scans), "name": "riyadh"}).status_code == 400
    assert client.post(f"{DF}/folders", json={"path": str(tmp_path), "name": "new"}).status_code == 403
    assert not (tmp_path / "new").exists()


def test_the_administrator_sets_and_previews_the_list(client, tmp_path):
    scans = tree(tmp_path)
    client.post(f"{DF}/flows", json=flow("Old scans", tmp_path / "elsewhere"))
    r = client.post(f"{DF}/folders/preview", json={"folders": "relative\\folder"})
    assert r.status_code == 400 and "not a full folder path" in r.json()["detail"]
    p = client.post(f"{DF}/folders/preview", json={"folders": f"Scans={scans}; {tmp_path / 'gone'}"}).json()
    assert [(x["name"], x["access"]) for x in p["roots"]] == [("Scans", "ok"), ("gone", "missing")]
    assert [o["name"] for o in p["outside"]] == ["Old scans"]
    assert client.get(f"{DF}/folders").json()["limited"] is False                         # a preview saves nothing

    r = client.put("/api/docs/v1/admin/settings", json={"values": {"DOCAUTO_ALLOWED_FOLDERS": "relative"}})
    assert r.status_code == 400
    r = client.put("/api/docs/v1/admin/settings", json={"values": {"DOCAUTO_ALLOWED_FOLDERS": f"Scans={scans}"}})
    assert r.status_code == 200, r.text
    assert "DOCAUTO_ALLOWED_FOLDERS=Scans=" in (tmp_path / "local.env").read_text(encoding="utf-8")
    assert client.get(f"{DF}/folders").json()["roots"][0]["name"] == "Scans"
    field = next(f for f in client.get("/api/docs/v1/admin/settings").json()["deployment"]
                 if f["key"] == "DOCAUTO_ALLOWED_FOLDERS")
    assert field["label"] == "Folders processes may use" and field["value"] == f"Scans={scans}"


def test_tenant_folders(monkeypatch, tmp_path):
    from docauto import folders
    monkeypatch.setenv("DOCAUTO_ALLOWED_FOLDERS", f"Inbox={tmp_path}{os.sep}{{tenant}}")
    (r,) = folders.roots("acme")
    assert r.name == "Inbox" and r.path == str(tmp_path / "acme")
    assert folders.allowed(str(tmp_path / "acme" / "x"), "acme")
    assert not folders.allowed(str(tmp_path / "other" / "x"), "acme")
    # a folder whose path has '=' keeps it; no name means the folder's own name
    (r,) = folders.parse(f"{tmp_path}{os.sep}a=b")
    assert r.path == f"{tmp_path}{os.sep}a=b" and r.name == "a=b"
