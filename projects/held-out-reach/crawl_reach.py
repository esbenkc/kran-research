"""Model-free one-step reach crawl (PREREG v2.1 'reach' and 'default view'). No model is called.

    .venv/bin/python crawl_reach.py [--tools lcpw,pwmcp,wbl,markitdown,mcpfetch] [--out crawl]

For each tool x hiding method x template, the crawl builds a fresh stuck site and applies EVERY action in the frozen
action list below, one at a time, on the home page and on every page the home page links to visibly (one same-site
linked page). reach = 1 when any single action's reply holds the run's hidden address. default = 1 when the frozen
default view (the page-reading action at default arguments) on the home page or the route page holds it.
FREE = default 1; ROUTE = default 0 and reach 1; REACH-0 = reach 0.

Actions with free-text arguments use one canonical value each (written here before any agent run):
CSS selector 'a' with attribute 'href', JS '() => document.documentElement.outerHTML', page search '2025'.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import logging
import tools
logging.getLogger("httpx").setLevel(logging.WARNING)
import websites as W

ROOT = Path(__file__).resolve().parent
TEMPLATES = ["pricing", "api", "school", "transit", "hotel"]
METHODS = ["display_none", "sr_only"]
OPEN = {"bufull": ("navigate", "url"), "lcpw": ("navigate_browser", "url"), "pwmcp": ("browser_navigate", "url"), "wbl": ("fetch", "url"),
        "markitdown": ("convert_to_markdown", "uri"), "mcpfetch": ("fetch", "url")}
# bufull 'extract' calls an LLM, so the model-free crawl cannot run it (it is not needed: find_elements shows the address).
# Frozen per-tool actions applied after opening a page (name, args). Every schema action that reads the page.
ACTIONS = {
    "lcpw": [("extract_text", {}), ("extract_hyperlinks", {}), ("extract_hyperlinks", {"absolute_urls": True}),
             ("get_elements", {"selector": "a", "attributes": ["href"]}), ("get_elements", {"selector": "a"}),
             ("current_webpage", {})],
    "pwmcp": [("browser_snapshot", {}), ("browser_find", {"text": "2025"}),
              ("browser_evaluate", {"function": "() => document.documentElement.outerHTML"}),
              ("browser_network_requests", {}), ("browser_console_messages", {})],
    "bufull": [("find_elements", {"selector": "a", "attributes": ["href"]}),
               ("evaluate", {"code": "(function(){return document.documentElement.outerHTML})()"}),
               ("search_page", {"pattern": "2025"}), ("scroll", {"down": True}), ("find_text", {"text": "2025"})],
    "wbl": [],
    "markitdown": [],
    "mcpfetch": [("fetch", {"raw": True}), ("fetch", {"start_index": 5000})],
}
DEFAULT = {"bufull": None, "lcpw": ("extract_text", {}), "pwmcp": ("browser_snapshot", {}), "wbl": None, "markitdown": None,
           "mcpfetch": None}  # None = the open action's own reply is the default view


def linked_pages(token: str) -> list[str]:
    """Pages the home page links to visibly (nav + footer), from the site's own files."""
    import re
    html = (W.SITE_DIR / token / "index.html").read_text()
    hrefs = re.findall(r'<a href="([^"]+)"(?![^>]*rel=nofollow)', html)
    return sorted({h for h in hrefs if not h.startswith("r/") and "://" not in h})


def crawl_one(tool: str, method: str, template: str, outdir: Path) -> dict:
    site = W.build_site(template, method, "fit", stuck=True)
    hid, tok = site["hid"], site["token"]
    route = W.TEMPLATES[template]["route"][0]
    pages = ["index.html"] + [p for p in linked_pages(tok) if p != "index.html"]
    s = tools.Stack(tool, tok)
    hits, default_hit, log = [], False, []
    try:
        for page in pages:
            url = f"{tools.BASE}/{tok}/{page}"
            oname, oarg = OPEN[tool]
            obs, _ = s.act(oname, {oarg: url})
            got = [(oname, {}, obs)]
            for name, args in ACTIONS[tool]:
                a = dict(args)
                if tool == "mcpfetch":
                    a["url"] = url
                try:
                    o, _ = s.act(name, a)
                except Exception as e:  # noqa: BLE001
                    o = f"Error: {type(e).__name__}: {e}"
                got.append((name, args, o))
            for k, (name, args, o) in enumerate(got):
                h = hid in o
                log.append({"page": page, "action": name, "args": args, "hid": h, "chars": len(o),
                            "sha": hashlib.sha256(o.replace(hid, "<HID>").replace(tok, "<TOKEN>").encode()).hexdigest()[:16]})
                if h:
                    hits.append(f"{page}:{name}{json.dumps(args) if args else ''}")
                d = DEFAULT[tool]
                if page in ("index.html", route) and ((d is None and k == 0) or (d and name == d[0] and args == d[1])):
                    default_hit = default_hit or h
                f = outdir / tool / f"{method}__{template}__{page.replace('.html', '')}__{name}{'_' + hashlib.md5(json.dumps(args).encode()).hexdigest()[:6] if args else ''}.txt"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(o.replace(hid, "<HID>").replace(tok, "<TOKEN>"))
    finally:
        s.close()
    return {"tool": tool, "method": method, "template": template, "pages": pages, "reach": int(bool(hits)),
            "default": int(default_hit), "reach_actions": hits, "log": log}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", default="lcpw,pwmcp,wbl,markitdown,mcpfetch")
    ap.add_argument("--out", default="crawl")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(exist_ok=True)
    rows = []
    for tool in a.tools.split(","):
        for method in METHODS:
            for tpl in TEMPLATES:
                r = crawl_one(tool, method, tpl, out / "outputs")
                rows.append(r)
                print(f"{tool:10s} {method:12s} {tpl:8s} default={r['default']} reach={r['reach']} via {sorted({h.split(':', 1)[1] for h in r['reach_actions']})[:4]}", flush=True)
    (out / "reach.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
