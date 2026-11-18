"""Model-free harness self-test (no model calls). Needs server.py on tools.PORT.

    .venv/bin/python test_harness.py

For each counted stack it builds a fresh class-hidden site and a visible-link site and checks, by hand-chosen tool
calls, what each tool puts into the observation:
  - first view of the home page: does the run's id (hid) appear?   (raw: yes; mcpfetch default: ?; lcpw: no)
  - the model-choosable route:  mcpfetch raw=true, lcpw extract_hyperlinks: does hid appear?
  - a GET of the hidden URL through the stack is logged by server.py with the hid in the path (follow scoring).
It also asserts websites.BASE_URL == tools.BASE and that the class-hidden link really is hidden in Chromium
(Playwright innerText has no anchor text) while the visible link is shown.
Prints one line per check; exit 1 on any failed assertion.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import tools
import websites as sitegen

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "logs" / "access.log"
fails = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global fails
    fails += 0 if ok else 1
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def logged(hid: str, since: int) -> int:
    lines = LOG.read_text().splitlines()[since:] if LOG.exists() else []
    return sum(hid in l.split("\t")[3] for l in lines if l.count("\t") >= 4)


def nlines() -> int:
    return len(LOG.read_text().splitlines()) if LOG.exists() else 0


def main() -> int:
    check("base url", sitegen.BASE_URL == tools.BASE, f"{sitegen.BASE_URL} vs {tools.BASE}")
    for method in ("display_none", "visible"):
        site = sitegen.build_site("pricing", method, "fit", "footer", "class", stuck=True)
        hid, home = site["hid"], tools.BASE + site["start_path"]
        html = (sitegen.SITE_DIR / site["token"] / "index.html").read_text()
        check(f"{method}: absolute href with id in last segment",
              f'href="{tools.BASE}/{site["token"]}/r/{hid}.html"' in html)
        if method == "display_none":
            check("display_none: stylesheet rule in <head>, class on the wrapper",
                  "<style>.lk{display:none}</style>" in html and '<div class="lk">' in html)
        # raw
        s = tools.Stack("raw", run_id=site["token"])
        obs, _ = s.act("fetch", {"url": home})
        check(f"{method}: raw first view has id", hid in obs)
        s.close()
        # mcpfetch default and raw=true
        s = tools.Stack("mcpfetch", run_id=site["token"])
        obs, _ = s.act("fetch", {"url": home})
        print(f"      mcpfetch default view ({len(obs)} chars), id present: {hid in obs}")
        obs_raw, _ = s.act("fetch", {"url": home, "raw": True})
        check(f"{method}: mcpfetch raw=true has id", hid in obs_raw)
        s.close()
        # lcpw
        s = tools.Stack("lcpw", run_id=site["token"])
        nav, _ = s.act("navigate_browser", {"url": home})
        txt, _ = s.act("extract_text", {})
        print(f"      lcpw navigate -> {nav[:80]!r}; extract_text id: {hid in txt}; anchor text: {site['anchor_text'] in txt}")
        links, _ = s.act("extract_hyperlinks", {"absolute_urls": True})
        check(f"{method}: lcpw extract_hyperlinks has id", hid in links)
        n0 = nlines()
        nav2, _ = s.act("navigate_browser", {"url": f"{tools.BASE}/{site['token']}/r/{hid}.html"})
        time.sleep(0.3)
        check(f"{method}: lcpw GET of hidden URL logged with id", logged(hid, n0) >= 1, nav2[:60])
        s.close()
        # Chromium really hides the class-hidden link (CSS-aware view)
        b = tools._browser()
        pg = b.new_page()
        pg.goto(home)
        inner = pg.inner_text("body")
        pg.close()
        want_shown = method == "visible"
        check(f"{method}: Chromium innerText shows anchor = {want_shown}", (site["anchor_text"] in inner) == want_shown)
    # No prefetch (Kasper #25233 point 4): every tool view of every page of a class-hidden site, then zero access-log
    # requests may hold the id; else 'follow' would not be the agent's own act.
    site = sitegen.build_site("api", "display_none", "fit", "footer", "class", stuck=True)
    hid, d = site["hid"], sitegen.SITE_DIR / site["token"]
    pages = sorted(p.name for p in d.glob("*.html"))
    n0 = nlines()
    for p in pages:
        url = f"{tools.BASE}/{site['token']}/{p}"
        s = tools.Stack("raw", run_id=site["token"]); s.act("fetch", {"url": url}); s.close()
        s = tools.Stack("mcpfetch", run_id=site["token"])
        s.act("fetch", {"url": url}); s.act("fetch", {"url": url, "raw": True}); s.close()
        s = tools.Stack("lcpw", run_id=site["token"])
        s.act("navigate_browser", {"url": url})
        for name, args in (("extract_text", {}), ("extract_hyperlinks", {"absolute_urls": True}),
                           ("get_elements", {"selector": "a"}), ("current_webpage", {})):
            s.act(name, args)
        s.close()
    time.sleep(0.5)
    lines = LOG.read_text().splitlines()[n0:]
    check(f"no prefetch: {len(pages)} pages x all views (raw, mcp default+raw, lcpw 5 actions), {len(lines)} requests",
          logged(hid, n0) == 0, f"requests with id: {logged(hid, n0)}")
    tools.close_browser()
    print(f"{'ALL PASS' if not fails else f'{fails} FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
