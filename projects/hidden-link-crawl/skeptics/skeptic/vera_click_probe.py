# Why does a role/text click on sr-only / off-screen links fail for me? Exception text + alternatives (force, JS click).
import re, threading, http.server, socketserver, tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright
RAW = Path("../../kran-research/projects/hidden-link-crawl/crawl/outputs/raw"); HID = "abcdef012345"
tmp = Path(tempfile.mkdtemp())
for row in ("sr_only-footer-inline", "off_screen-footer-inline", "visible-footer-inline"):
    html = (RAW / f"{row}__pricing__fit__table.txt").read_text().replace("<HID>", HID)
    d = tmp / row; (d / "r").mkdir(parents=True); (d / "p.html").write_text(html); (d / "r" / f"{HID}.html").write_text("ok")
class Q(http.server.SimpleHTTPRequestHandler):
    def __init__(s, *a, **k): super().__init__(*a, directory=str(tmp), **k)
    def log_message(s, *a): pass
srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Q); port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    for row in ("sr_only-footer-inline", "off_screen-footer-inline", "visible-footer-inline"):
        url = f"http://127.0.0.1:{port}/{row}/p.html"; pg.goto(url)
        snap = pg.locator("body").aria_snapshot()
        line = [l for l in snap.splitlines() if "2025" in l or "/url" in l]
        loc = pg.get_by_role("link", name="Full 2025 pricing table", exact=True)
        print(row, "| snapshot lines:", line, "| box:", loc.bounding_box())
        for mode in ("default", "force", "js"):
            pg.goto(url)
            try:
                if mode == "default": loc.click(timeout=3000)
                elif mode == "force": loc.click(timeout=3000, force=True)
                else: loc.evaluate("e => e.click()")
                pg.wait_for_load_state(); print("  ", mode, "->", pg.url.split("/")[-1])
            except Exception as e:
                print("  ", mode, "FAIL:", str(e).splitlines()[0][:150], "|", [l for l in str(e).splitlines() if "outside" in l or "intercept" in l or "visible" in l][:2])
    b.close()
srv.shutdown()
