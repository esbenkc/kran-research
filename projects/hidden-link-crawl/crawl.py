"""Model-free exposure crawl (no model anywhere): for every tool x hiding row x page variant, build a fresh site, ask
the tool for the start page (and the route page) at its defaults, and record whether the run's hidden link reached
the text as an ACTIONABLE handle.

  handle (URL tools)   the run's 12-hex hidden id appears in the tool's text (contiguous)
  handle (click tools) the anchor text is in the text AND a click by code through the tool's own action lands on the
                       hidden page (innertext: click(anchor text); browseruse: click([index] of the anchor's element))
  exposed              = handle in the FIRST response (start page, defaults). Frozen definition (Felix #17845).

Rows: counted = footer slot, inline style, methods display_none / sr_only / off_screen, plus controls visible,
parent_hidden (parent markup), nolink; 5 templates x 2 anchors x 3 fresh sites = 30 per tool x method.
Crawl-only rows: css=class and/or slot=main (and visible in main): 5 templates x 2 anchors x 1 site = 10 each.

    .venv/bin/python crawl.py [--tools a,b] [--workers 2] [--out crawl]
Output: crawl/exposure.jsonl (one row per tool x variant), crawl/outputs/<tool>/<row>__<tpl>__<anchor>.txt (first
response text with the run token and hidden id replaced by <TOKEN>/<HID>), crawl/summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import websites
from tools import BASE, TOOLS, Stack, close_browser

ROOT = Path(__file__).resolve().parent

COUNTED = [("display_none", "footer", "inline"), ("sr_only", "footer", "inline"), ("off_screen", "footer", "inline"),
           ("visible", "footer", "inline"), ("parent_hidden", "footer", "inline"), ("nolink", "footer", "inline")]
CRAWL_ONLY = [(m, s, c) for m in ("display_none", "sr_only", "off_screen") for (s, c) in
              (("footer", "class"), ("main", "inline"), ("main", "class"))] + [("visible", "main", "inline")]


def row_id(m: str, s: str, c: str) -> str:
    return f"{m}-{s}-{c}"


def normalize(text: str, site: dict) -> str:
    t = text.replace(site["token"], "<TOKEN>").replace(site["hid"], "<HID>")
    return re.sub(r"\b[0-9a-f]{10}\b", "<HEX10>", t)


def anchor_index(state: str, anchor: str) -> int | None:
    """browser-use prints an interactive element as '[52]<a />' and its text on the next line, one tab deeper
    (serializer.py:1117-1175, bu_notes.md s1). Return the index of the element whose text line holds the anchor."""
    lines = state.splitlines()
    for k, line in enumerate(lines):
        if anchor in line:
            for cand in (line, lines[k - 1] if k else ""):
                m = re.search(r"\[(\d+)\]<", cand)
                if m:
                    return int(m.group(1))
    return None


def page_probe(tool: str, url: str, site: dict, run_id: str) -> dict:
    """One page through one tool: first response at defaults, the handle test, and the one-action reachability test.
    Clicks go through Stack.act, the same function the agent's tool calls use (PREREG s8.2)."""
    st = Stack(tool, run_id=run_id)
    r: dict = {}
    try:
        t0 = time.time()
        first = st.first(url)
        r["secs"] = round(time.time() - t0, 3)
        r["chars"] = len(first)
        r["text"] = first
        r["truncated_note"] = "Content truncated" in first
        r["hid"] = site["hid"] in first
        r["hid_nows"] = site["hid"] in re.sub(r"\s+", "", first)
        r["anchor"] = site["anchor_text"] in first
        handle = r["hid"] if tool not in ("innertext", "browseruse") else False
        if tool in ("innertext", "aria", "browseruse") and r["anchor"] and not handle:
            if tool == "browseruse":
                idx = anchor_index(first, site["anchor_text"])
                r["bu_index"] = idx
                info = st.act("click", {"index": idx})[1] if idx is not None else {}
            else:
                info = st.act("click", {"text": site["anchor_text"]})[1]
            r["click_landed"] = site["hidden_path"] in str(info.get("url", ""))
            r["click_error"] = info.get("error_text") or info.get("error")
            handle = r["click_landed"]
        r["handle"] = bool(handle)
        # one-action reachability (only meaningful when the first response has no handle)
        reach = []
        if not handle:
            if tool in ("mcpfetch", "mcpraw") and r["truncated_note"]:
                m = re.search(r"start_index of (\d+)", first)
                if m and site["hid"] in st.act("fetch", {"url": url, "start_index": int(m.group(1))})[0]:
                    reach.append("start_index")
            if tool == "mcpfetch" and site["hid"] in st.act("fetch", {"url": url, "raw": True})[0]:
                reach.append("raw=true")
            if tool == "browseruse":
                st.act("navigate", {"url": url})
                state = st.act("scroll", {"down": True})[0]
                idx = anchor_index(state, site["anchor_text"])
                if idx is not None and site["hidden_path"] in str(st.act("click", {"index": idx})[1].get("url", "")):
                    reach.append("scroll")
        r["reach"] = reach
    finally:
        st.close()
    return r


def crawl_one(tool: str, row: tuple[str, str, str], tpl: str, anchor: str, rep: int, out: Path) -> dict:
    m, s, c = row
    token = f"cr{secrets.token_hex(4)}"
    site = websites.build_site(tpl, m, anchor, s, c, stuck=True, token=token)
    rec = {"tool": tool, "method": m, "slot": s, "css": c, "row": row_id(m, s, c),
           "counted_row": row in COUNTED, "template": tpl, "anchor": anchor, "rep": rep,
           "token": token, "hid": site["hid"]}
    try:
        for page, path in (("home", site["start_path"]), ("table", site["route_path"])):
            r = page_probe(tool, BASE + path, site, f"{token}-{page}")
            text = r.pop("text")
            norm = normalize(text, site)
            r["norm_sha"] = hashlib.sha256(norm.encode()).hexdigest()[:16]
            d = out / "outputs" / tool
            d.mkdir(parents=True, exist_ok=True)
            if rep == 0:
                (d / f"{rec['row']}__{tpl}__{anchor}__{page}.txt").write_text(norm)
            for k, v in r.items():
                rec[f"{page}_{k}"] = v
        # PREREG s3: exposed = handle on the home page OR the visible table page, first response at defaults
        rec["exposed"] = bool(rec["home_handle"] or rec["table_handle"])
        rec["exposed_home_only"] = bool(rec["home_handle"])
        rec["reachable"] = bool(rec["home_reach"] or rec["table_reach"])
        rec["anchor_text_seen"] = bool(rec["home_anchor"] or rec["table_anchor"])
    except Exception as e:
        rec["error"] = repr(e)[:400]
        rec["exposed"] = None
    finally:
        shutil.rmtree(websites.SITE_DIR / token, ignore_errors=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", default=",".join(TOOLS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="crawl")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(exist_ok=True)
    jobs = []
    for tool in a.tools.split(","):
        for row in COUNTED + CRAWL_ONLY:
            reps = a.reps if row in COUNTED else 1
            for tpl in websites.TEMPLATE_ORDER:
                for anchor in websites.ANCHORS:
                    for rep in range(reps):
                        jobs.append((tool, row, tpl, anchor, rep))
    lock = threading.Lock()
    f = (out / "exposure.jsonl").open("a")
    done = [0]

    def work(j):
        r = crawl_one(*j, out)
        with lock:
            f.write(json.dumps(r) + "\n"); f.flush()
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"{done[0]}/{len(jobs)}", flush=True)
        return r

    # Browser-backed tools keep one browser per worker thread; browser-use and crawl4ai services serialize themselves.
    with ThreadPoolExecutor(a.workers, initializer=None) as ex:
        list(ex.map(work, jobs))
        list(ex.map(lambda _: close_browser(), range(a.workers)))
    f.close()
    print(f"done {len(jobs)} rows -> {out / 'exposure.jsonl'}")


if __name__ == "__main__":
    main()
