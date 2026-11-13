"""Model-free exposure check (task T5): does each hiding cell's fit link reach a human or an assistive view?

No model is called. For every hiding cell in web/websites.py (agent cells A, A2, C, B, B2, E and the exposure-only
cells OFF, ZERO) and every template, we build the landing page with the run generator itself and load it in
headless Chromium (Playwright), served over local HTTP so the shared stylesheet assets/site.css applies.

Views, each a yes/no "the fit link is exposed here":
  pixels     full-page screenshot changes when only the fit anchor's text changes (a sighted human can read it)
  pixels_layout  full-page screenshot differs from the same page with the link slot removed (any layout trace)
  innerText  document.body.innerText holds the fit anchor (copy-paste, many text extractors)
  a11y       Chromium's accessibility tree (CDP Accessibility.getFullAXTree) has a non-ignored link named
             with the fit anchor (screen readers); Playwright's aria snapshot as a second read
  tab        pressing Tab from the top focuses the fit link within 40 presses (sighted keyboard users)
  print      the print PDF (page.pdf) holds the anchor text or a link annotation to the fit address
  reader     Mozilla Readability (Firefox Reader View) output holds the fit address, footer as built; and
  reader_main  the same with the link slot moved from <footer> into <main> (does the hiding method matter there?)
  raw_cue    the raw HTML the agent receives holds a readable hiding cue near the link (inline style, aria-hidden,
             hidden attribute); reported for orientation only, Bruno's cue_grep.py is the frozen score

Usage: uv run --with playwright --with pypdf python viewcheck.py  ->  viewcheck.json, viewcheck.md
"""
from __future__ import annotations

import functools
import html
import http.server
import io
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "web"))
import websites  # noqa: E402  (read-only use of the run generator)

from playwright.sync_api import sync_playwright  # noqa: E402
from pypdf import PdfReader  # noqa: E402

PAGES = HERE / "pages"
websites.SITE_DIR = PAGES  # build pages here, never into web/site/
CUE_RE = re.compile(r'style="[^"]*(display\s*:\s*none|visibility\s*:\s*hidden)|aria-hidden="true"|\shidden[\s>=]', re.I)


def build(template: str, hiding: str) -> dict:
    tok = f"vc-{template}-{hiding}"
    m = websites.build_site(template, hiding, 0, token=tok, hid=f"h{template[:4]}{hiding}".lower(),
                            oid=f"o{template[:4]}{hiding}".lower())
    t = websites.TEMPLATES[template]
    d = PAGES / tok
    page = (d / "index.html").read_text()
    slot = websites.footer_slot(hiding, m["hid"], m["oid"], t, 0)
    assert page.count(slot) == 1, (template, hiding)
    (d / "ref.html").write_text(page.replace(slot, ""))                      # same page, no hidden links
    swapped = slot.replace(">" + html.escape(t["fit_anchor"]) + "<", ">" + "Q" * len(t["fit_anchor"]) + "<")
    assert swapped != slot
    (d / "swap.html").write_text(page.replace(slot, swapped))                # same layout, fit anchor text changed
    (d / "inmain.html").write_text(page.replace(slot, "").replace("<main>\n", "<main>\n" + slot, 1))
    return {"template": template, "hiding": hiding, "token": tok, "href": f"r/{m['hid']}.html",
            "anchor": t["fit_anchor"], "slot": slot, "raw_cue": bool(CUE_RE.search(slot))}


def serve() -> tuple[http.server.ThreadingHTTPServer, int]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass
    h = functools.partial(Quiet, directory=str(PAGES))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def ax_exposed(pg, anchor: str) -> bool:
    cdp = pg.context.new_cdp_session(pg)
    nodes = cdp.send("Accessibility.getFullAXTree")["nodes"]
    cdp.detach()
    for n in nodes:
        name = (n.get("name") or {}).get("value", "")
        role = (n.get("role") or {}).get("value", "")
        if anchor in name and role == "link" and not n.get("ignored", False):
            return True
    return False


def tab_reaches(pg, href: str, presses: int = 40) -> bool:
    pg.evaluate("document.activeElement && document.activeElement.blur()")
    for _ in range(presses):
        pg.keyboard.press("Tab")
        got = pg.evaluate("document.activeElement ? document.activeElement.getAttribute('href') : null")
        if got == href:
            return True
    return False


def main() -> None:
    PAGES.mkdir(exist_ok=True)
    cells = [build(t, h) for t in websites.TEMPLATES for h in websites.HIDINGS]
    srv, port = serve()
    base = f"http://127.0.0.1:{port}"
    rows = []
    with sync_playwright() as p:
        br = p.chromium.launch()
        ctx = br.new_context(viewport={"width": 1280, "height": 900})
        pg = ctx.new_page()
        for c in cells:
            url = f"{base}/{c['token']}/index.html"
            pg.goto(f"{base}/{c['token']}/ref.html"); pg.wait_for_load_state("networkidle")
            ref_png = pg.screenshot(full_page=True)
            pg.goto(url); pg.wait_for_load_state("networkidle")
            png = pg.screenshot(full_page=True)
            css_applied = pg.evaluate("getComputedStyle(document.querySelector('footer')).fontSize") != \
                pg.evaluate("getComputedStyle(document.body).fontSize")
            r = {k: c[k] for k in ("template", "hiding", "raw_cue")}
            r["css_applied"] = css_applied
            r["pixels_layout"] = png != ref_png          # page differs from the page without the slot
            pg.goto(f"{base}/{c['token']}/swap.html"); pg.wait_for_load_state("networkidle")
            swap_png = pg.screenshot(full_page=True)
            pg.goto(url); pg.wait_for_load_state("networkidle")
            r["pixels"] = png != swap_png                # the fit anchor's own text paints (a sighted human can read it)
            r["innerText"] = c["anchor"] in pg.evaluate("document.body.innerText")
            r["a11y"] = ax_exposed(pg, c["anchor"])
            r["a11y_aria_snapshot"] = c["anchor"] in pg.locator("body").aria_snapshot()
            r["tab"] = tab_reaches(pg, c["href"])
            pdf = PdfReader(io.BytesIO(pg.pdf()))
            ptxt = " ".join(pp.extract_text() or "" for pp in pdf.pages)
            uris = [str(a.get_object().get("/A", {}).get("/URI", "")) for pp in pdf.pages for a in (pp.get("/Annots") or [])]
            r["print"] = (c["anchor"] in ptxt) or any(c["href"] in u for u in uris)
            rows.append(r)
        br.close()
    srv.shutdown()
    # Readability (Node): footer as built, and the slot moved into <main>.
    items = []
    for c in cells:
        for v, f in (("reader", "index.html"), ("reader_main", "inmain.html")):
            items.append({"id": f"{c['template']}|{c['hiding']}|{v}", "path": str(PAGES / c["token"] / f),
                          "url": f"http://127.0.0.1/{c['token']}/{f}", "href": c["href"], "anchor": c["anchor"]})
    (HERE / "pages" / "items.json").write_text(json.dumps(items))
    out = subprocess.run(["node", str(HERE / "readability.mjs"), str(PAGES / "items.json")], capture_output=True,
                         text=True, check=True, cwd=HERE).stdout
    rd = {}
    for line in out.splitlines():
        j = json.loads(line)
        t, h, v = j["id"].split("|")
        rd[(t, h, v)] = j["href_in_content"] or j["anchor_in_text"]
        rd[(t, h, v + "_parsed")] = j["parsed"]
    for r in rows:
        for v in ("reader", "reader_main"):
            r[v] = rd[(r["template"], r["hiding"], v)]
            r[v + "_parsed"] = rd[(r["template"], r["hiding"], v + "_parsed")]
    (HERE / "viewcheck.json").write_text(json.dumps(rows, indent=1))
    views = ["raw_cue", "pixels", "pixels_layout", "innerText", "a11y", "a11y_aria_snapshot", "tab", "print", "reader", "reader_main"]
    n_t = len(websites.TEMPLATES)
    lines = ["| cell | " + " | ".join(views) + " |", "|" + "---|" * (len(views) + 1)]
    for h in websites.HIDINGS:
        rs = [r for r in rows if r["hiding"] == h]
        lines.append(f"| {h} | " + " | ".join(f"{sum(r[v] for r in rs)}/{n_t}" for v in views) + " |")
    extra = (f"\nCSS applied on every page: {all(r['css_applied'] for r in rows)}. "
             f"Readability parsed an article: {sum(r['reader_parsed'] for r in rows)}/{len(rows)} (footer), "
             f"{sum(r['reader_main_parsed'] for r in rows)}/{len(rows)} (slot in main).\n")
    (HERE / "viewcheck.md").write_text("Fit link exposed, pages out of templates (order 0, landing page).\n\n"
                                       + "\n".join(lines) + "\n" + extra)
    print("\n".join(lines) + extra)


if __name__ == "__main__":
    main()
