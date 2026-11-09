# Vera's own model-free exposure crawl. Independent of the build's crawl.py / tools.py (not read, not imported).
# Input: the served HTML the build saved for the 'raw' tool (crawl/outputs/raw/<row>__<tpl>__<anchor>__<page>.txt),
# with placeholders <HID>/<TOKEN>. I run the shipped packages myself (own venv, same pins) and test whether the
# run's handle reaches the first response. Output: vera_crawl.json + a comparison with crawl/exposure.jsonl.
import json, re, sys, threading, http.server, socketserver, tempfile, collections
from pathlib import Path
import markdownify, html2text, trafilatura
from mcp_server_fetch.server import extract_content_from_html

PROJ = Path(sys.argv[1] if len(sys.argv) > 1 else "../../kran-research/projects/hidden-link-crawl")
RAW = PROJ / "crawl/outputs/raw"
HID = "abcdef012345"
HANDLE = f"r/{HID}"

def anchor_text(html):
    m = re.search(r'<a href="r/' + HID + r'\.html"[^>]*>(.*?)</a>', html, re.S)
    return m.group(1).strip() if m else None

pages = {}
for f in sorted(RAW.glob("*.txt")):
    row, tpl, anc, page = f.stem.split("__")
    html = f.read_text().replace("<HID>", HID).replace("<TOKEN>", "vtok")
    pages[(row, tpl, anc, page)] = html

def h2t(html):
    h = html2text.HTML2Text()
    return h.handle(html)

CONVERTERS = {
    "raw": lambda h: h,
    "mcpraw": lambda h: h[:5000],
    "markdownify": lambda h: markdownify.markdownify(h),
    "html2text": h2t,
    "trafilatura": lambda h: trafilatura.extract(h) or "",
    "mcpfetch": lambda h: extract_content_from_html(h, use_readability_js=True)[:5000],
    "mcpfetch_nonode": lambda h: extract_content_from_html(h, use_readability_js=False)[:5000],
}
res = {}
for key, html in pages.items():
    for tool, fn in CONVERTERS.items():
        out = fn(html)
        res[(tool,) + key] = {"handle": HANDLE in out, "anchor_seen": bool(anchor_text(html)) and anchor_text(html) in out}

# Browser tools: serve the pages from a temp dir on 127.0.0.1, then innerText / ariaSnapshot + a click by text/role.
from playwright.sync_api import sync_playwright
tmp = Path(tempfile.mkdtemp(prefix="vera-site-"))
for (row, tpl, anc, page), html in pages.items():
    d = tmp / f"{row}__{tpl}__{anc}"; (d / "r").mkdir(parents=True, exist_ok=True)
    (d / f"{page}.html").write_text(html); (d / "r" / f"{HID}.html").write_text("<html><body>target</body></html>")
class Q(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=str(tmp), **k)
    def log_message(self, *a): pass
srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Q); port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    br = p.chromium.launch()
    pg = br.new_page()  # default viewport 1280x720
    for (row, tpl, anc, page), html in pages.items():
        url = f"http://127.0.0.1:{port}/{row}__{tpl}__{anc}/{page}.html"
        at = anchor_text(html)
        pg.goto(url)
        txt = pg.inner_text("body")
        it_seen = bool(at) and at in txt
        it_click = False
        if it_seen:
            try:
                pg.get_by_text(at, exact=True).first.click(timeout=1500); it_click = HANDLE in pg.url
            except Exception:
                it_click = False
            pg.goto(url)
        snap = pg.locator("body").aria_snapshot()
        ar_seen = bool(at) and at in snap
        ar_click = False
        if ar_seen:
            try:
                pg.get_by_role("link", name=at, exact=True).first.click(timeout=1500); ar_click = HANDLE in pg.url
            except Exception:
                ar_click = False
        res[("innertext", row, tpl, anc, page)] = {"handle": it_click, "anchor_seen": it_seen}
        res[("aria", row, tpl, anc, page)] = {"handle": ar_click, "anchor_seen": ar_seen}
    br.close()
srv.shutdown()

# Variant exposure = home OR table (PREREG sec 3).
var = {}
for (tool, row, tpl, anc, page), r in res.items():
    k = (tool, row, tpl, anc)
    v = var.setdefault(k, {"exposed": False, "anchor_seen": False})
    v["exposed"] |= r["handle"]; v["anchor_seen"] |= r["anchor_seen"]
json.dump({"|".join(k): v for k, v in var.items()}, open("vera_crawl.json", "w"), indent=0)

# Compare with the build's crawl (any rep; also check rep consistency there).
theirs = collections.defaultdict(set)
for line in open(PROJ / "crawl/exposure.jsonl"):
    r = json.loads(line)
    theirs[(r["tool"], r["row"], r["template"], r["anchor"])].add(bool(r["exposed"]))
agree = dis = 0; mism = []; inconsistent = [k for k, s in theirs.items() if len(s) > 1]
for k, v in var.items():
    if k not in theirs: continue
    t = next(iter(theirs[k])) if len(theirs[k]) == 1 else None
    if t == v["exposed"]: agree += 1
    else: dis += 1; mism.append((k, v["exposed"], sorted(theirs[k])))
print(f"compared {agree+dis} tool x row x template x anchor variants: agree {agree}, disagree {dis}; build reps inconsistent: {len(inconsistent)}")
for m in mism[:40]: print("MISMATCH", m)
# Cell table (counted rows, fit anchor): mean over templates.
cell = collections.defaultdict(list)
for (tool, row, tpl, anc), v in var.items():
    if anc == "fit": cell[(tool, row)].append(v["exposed"])
print("\nexposure (fit anchor, mean over templates), mine:")
rows = sorted({r for _, r in cell})
tools = sorted({t for t, _ in cell})
print("row".ljust(30) + "".join(t[:10].rjust(11) for t in tools))
for r in rows:
    print(r.ljust(30) + "".join((f"{sum(cell[(t,r)])/len(cell[(t,r)]):.1f}" if (t, r) in cell else "-").rjust(11) for t in tools))
