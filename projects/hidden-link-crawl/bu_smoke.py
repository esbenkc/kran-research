"""Smoke test for bu_service.py (must already be running on --port, default 8781).

    .venv-bu/bin/python bu_smoke.py [--port 8781]

Builds 8 demo sites with websites.py (pricing, fit anchor, footer slot; methods display_none/sr_only/off_screen/visible
x css inline/class), renames each site/demo/ to site/bu-<method>-<css>/, navigates to index.html through the service,
looks for the anchor text in the browser-use state, clicks its [index] if present and checks where the click landed
(service url + the static server's access log).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
LOG = ROOT / "logs" / "access.log"
BASE = "http://127.0.0.1:8765"
ANCHOR = "Full 2025 pricing table"
METHODS = ("display_none", "sr_only", "off_screen", "visible")
CSS = ("inline", "class")


def post(port: int, path: str, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def build(method: str, css: str) -> dict:
    out = subprocess.run([str(ROOT / ".venv/bin/python"), "websites.py", "pricing", method, "fit", "footer", css],
                         cwd=ROOT, check=True, capture_output=True, text=True).stdout
    man = json.loads(out)
    token = f"bu-{method}-{css}"
    dst = SITE / token
    if dst.exists():
        shutil.rmtree(dst)
    (SITE / "demo").rename(dst)
    man["token"] = token
    man["start_path"] = f"/{token}/index.html"
    man["hidden_path"] = f"/{token}/r/{man['hid']}.html"
    return man


def anchor_index(state: str) -> int | None:
    """Index of the interactive element whose text line (next line, one tab deeper) is the anchor text."""
    lines = state.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == ANCHOR:
            for j in range(i - 1, -1, -1):  # nearest [N]<...> line above
                m = re.search(r"\*?\[(\d+)\]<", lines[j])
                if m:
                    return int(m.group(1)) if lines[j].count("\t") < line.count("\t") else None
            return None
    return None


def log_has(path: str, since: int) -> bool:
    with LOG.open() as f:
        f.seek(since)
        return any(ln.split("\t")[3:4] == [path] and ln.split("\t")[2] == "GET" for ln in f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8781)
    a = ap.parse_args()
    rows = []
    for css in CSS:
        for method in METHODS:
            man = build(method, css)
            run = f"smoke-{method}-{css}-{int(time.time())}"
            assert post(a.port, "/reset", {"run": run}).get("ok")
            nav = post(a.port, "/navigate", {"run": run, "url": BASE + man["start_path"]})
            if "error" in nav and "state" not in nav:
                raise SystemExit(f"navigate failed: {nav}")
            state = nav["state"]
            present = ANCHOR in state
            idx = anchor_index(state) if present else None
            landed, url, logged = None, "", None
            if idx is not None:
                pos = LOG.stat().st_size
                clk = post(a.port, "/click", {"run": run, "index": idx})
                url = clk.get("url", "")
                landed = url.endswith(man["hidden_path"])
                time.sleep(0.2)
                logged = log_has(man["hidden_path"], pos)
            rows.append((method, css, nav.get("status"), present, idx, landed, logged, url))
            if method == "sr_only" and css == "inline":
                print(state)
    print(f"{'method':<13}{'css':<8}{'http':<6}{'anchor_in_state':<17}{'index':<7}{'click_on_/r/hid':<17}{'log_GET':<9}url")
    for m, c, st, p, i, l, g, u in rows:
        print(f"{m:<13}{c:<8}{str(st):<6}{str(p):<17}{str(i):<7}{str(l):<17}{str(g):<9}{u}")


if __name__ == "__main__":
    main()
