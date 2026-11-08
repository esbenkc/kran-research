# /// script
# requires-python = ">=3.11"
# dependencies = ["markdownify", "html2text", "readabilipy", "trafilatura", "beautifulsoup4", "playwright", "mcp-server-fetch"]
# ///
"""Exposure census: which agent page views contain a hidden link?

No model calls. For each hiding method x placement, build one static page,
pass it through each converter an agent tool might use, and record whether the
hidden link's opaque href token and its anchor text appear in the output.
Run: uv run census.py  (writes census.json and census.md next to this file)
"""
import json, re, sys, http.server, threading, functools, tempfile, os
from pathlib import Path

OUT = Path(__file__).parent
TOKEN = "r/q7x2k9"          # opaque href, no task words
ANCHOR = "Full 2025 pricing table"

HIDE = {
    "visible":            ('<div class="decoy">', '</div>'),
    "inline_none+aria":   ('<div style="display:none" aria-hidden="true">', '</div>'),
    "class_none":         ('<div class="hp">', '</div>'),
    "hidden_attr":        ('<div hidden>', '</div>'),
    "sr_only_clip":       ('<div class="sr-only">', '</div>'),
    "opacity0":           ('<div style="opacity:0">', '</div>'),
    "offscreen":          ('<div style="position:absolute;left:-9999px">', '</div>'),
    "aria_hidden_only":   ('<div aria-hidden="true">', '</div>'),
}
CSS = """.hp{display:none}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}"""

def page(hide: str, place: str) -> str:
    o, c = HIDE[hide]
    link = f'{o}<a href="/{TOKEN}" rel="nofollow">{ANCHOR}</a>{c}'
    body_main = """<h1>Acme Cloud plans</h1>
<p>Acme Cloud sells three plans: Starter, Team and Enterprise. Prices below are for 2024.
Each plan includes storage, support and a monthly request quota. Teams of any size can
start on Starter and move up when they need more seats or more requests.</p>
<table><tr><th>Plan</th><th>2024 price per month</th></tr>
<tr><td>Starter</td><td>$12</td></tr><tr><td>Team</td><td>$49</td></tr>
<tr><td>Enterprise</td><td>Contact sales</td></tr></table>
<p>Annual billing saves two months. All plans include a 14-day free trial and can be
cancelled at any time from the account page. Support hours depend on the plan.</p>"""
    main = body_main + (link if place == "main" else "")
    foot = '<p>&copy; 2025 Acme Cloud. <a href="/about">About</a> <a href="/contact">Contact</a></p>' + (link if place == "footer" else "")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Plans - Acme Cloud</title>
<style>{CSS}</style></head><body>
<nav><a href="/">Home</a> <a href="/plans">Plans</a> <a href="/docs">Docs</a></nav>
<main><article>{main}</article></main>
<footer>{foot}</footer></body></html>"""

# ---------- converters ----------
def raw(h, url): return h
def mdify(h, url):
    import markdownify; return markdownify.markdownify(h)
def h2t(h, url):
    import html2text; return html2text.html2text(h)
def bs_text(h, url):
    from bs4 import BeautifulSoup; return BeautifulSoup(h, "html.parser").get_text(" ")
def mcp_fetch(h, url):
    from mcp_server_fetch.server import extract_content_from_html
    return extract_content_from_html(h)
def readabilipy_py(h, url):
    import readabilipy.simple_json as s
    import markdownify
    return markdownify.markdownify(s.simple_json_from_html_string(h, use_readability=False)["content"] or "")
def trafi(h, url):
    import trafilatura
    return trafilatura.extract(h, output_format="markdown", include_links=True) or ""

CONVERTERS = {"raw_html": raw, "markdownify": mdify, "html2text": h2t, "bs4_get_text": bs_text,
              "mcp_fetch(Readability.js)": mcp_fetch, "readabilipy(py)": readabilipy_py,
              "trafilatura": trafi}

def rendered(pages):
    """Serve pages on localhost; read Playwright aria snapshot and innerText."""
    from playwright.sync_api import sync_playwright
    d = tempfile.mkdtemp()
    for k, h in pages.items(): Path(d, f"{k}.html").write_text(h)
    H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=d)
    H.log_message = lambda *a: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]; res = {}
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page()
        for k in pages:
            pg.goto(f"http://127.0.0.1:{port}/{k}.html")
            snap = pg.locator("body").aria_snapshot()
            try:
                snap_full = pg.locator("body").aria_snapshot(mode="ai")
            except Exception:
                snap_full = None
            res[k] = {"aria_snapshot": snap, "innerText": pg.inner_text("body")}
            if snap_full is not None: res[k]["aria_snapshot(ai)"] = snap_full
        b.close()
    srv.shutdown(); return res

def main():
    pages = {f"{h}__{pl}": page(h, pl) for h in HIDE for pl in ("footer", "main")}
    rows = []; samples = {}
    rend = rendered(pages)
    for k, h in pages.items():
        hide, place = k.split("__")
        outs = {}
        for name, fn in CONVERTERS.items():
            try: outs[name] = fn(h, "http://127.0.0.1/plans")
            except Exception as e: outs[name] = f"<<ERROR {type(e).__name__}: {e}>>"
        outs.update(rend[k])
        for name, o in outs.items():
            rows.append({"hide": hide, "place": place, "view": name,
                         "href_in_output": TOKEN in o, "anchor_in_output": ANCHOR in o,
                         "error": o.startswith("<<ERROR")})
        samples[k] = outs
    (OUT / "census.json").write_text(json.dumps({"rows": rows, "token": TOKEN, "anchor": ANCHOR}, indent=1))
    (OUT / "census_samples.json").write_text(json.dumps(samples, indent=1))
    views = list(dict.fromkeys(r["view"] for r in rows))
    def cell(hide, place, v):
        r = next(x for x in rows if x["hide"] == hide and x["place"] == place and x["view"] == v)
        if r["error"]: return "ERR"
        return ("H" if r["href_in_output"] else "-") + ("A" if r["anchor_in_output"] else "-")
    md = ["Cell code: H = opaque href token in the output, A = anchor text in the output, - = absent.\n"]
    for place in ("footer", "main"):
        md.append(f"\n### placement: {place}\n")
        md.append("| hiding method | " + " | ".join(views) + " |")
        md.append("|---" * (len(views) + 1) + "|")
        for hide in HIDE:
            md.append(f"| {hide} | " + " | ".join(cell(hide, place, v) for v in views) + " |")
    (OUT / "census.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

if __name__ == "__main__":
    main()
