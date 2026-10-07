"""scripts/gvon-userscript.py renders a valid userscript from a (synthetic) blocklist."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gvon-userscript.py"


def load_module():
    spec = importlib.util.spec_from_file_location("gvon_userscript", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_generates_userscript(tmp_path: Path) -> None:
    mod = load_module()
    bl = tmp_path / "blocklist.json"
    bl.write_text(json.dumps({"accounts": [{"id": "1", "username": "@Synthetic_Hater_1"},
                                           {"id": "2", "username": "synthetic_hater_1"},
                                           {"id": "3", "username": "bad\"];alert(1);//"}]}))
    out = tmp_path / "dist" / "gvon.user.js"
    assert mod.main(["--blocklist", str(bl), "--out", str(out)]) == 0
    js = out.read_text()
    assert js.startswith("// ==UserScript==")
    assert "// @match        https://x.com/*" in js
    assert 'const BLOCKED = new Set(["synthetic_hater_1"]);' in js
    assert "alert" not in js
    if shutil.which("node"):
        subprocess.run(["node", "--check", str(out)], check=True)


def test_missing_blocklist_fails(tmp_path: Path) -> None:
    assert load_module().main(["--blocklist", str(tmp_path / "nope.json"), "--out", str(tmp_path / "x.js")]) == 1
