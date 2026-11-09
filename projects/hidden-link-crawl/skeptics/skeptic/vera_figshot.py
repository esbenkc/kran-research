# Screenshot the main figure (light + dark) from its real plot.js/data.json/theme.js, without touching the blog clone.
import threading, http.server, socketserver, sys
from pathlib import Path
from playwright.sync_api import sync_playwright
BLOG = Path("../../blog.kran.ai").resolve(); FIG = sys.argv[1] if len(sys.argv) > 1 else "hidden-link-crawl"
PAGE = """<!doctype html><html><head><link rel=stylesheet href="/vera-site.css"></head><body class="research %s" style="background:%s;margin:20px"><div id=m></div><script type=module>
const Plot = await import("https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm");
const data = await (await fetch("/assets/figures/%s/data.json")).json();
const mod = await import("/assets/figures/%s/plot.js?theme=%s");
document.getElementById("m").replaceChildren(mod.render(data, Plot)); document.title="done";
</script></body></html>"""
class H(http.server.SimpleHTTPRequestHandler):
    def __init__(s, *a, **k): super().__init__(*a, directory=str(BLOG), **k)
    def log_message(s, *a): pass
    def do_GET(s):
        if s.path == "/vera-site.css":
            b = open("/tmp/vera_site.css","rb").read(); s.send_response(200); s.send_header("Content-Type","text/css"); s.end_headers(); s.wfile.write(b); return
        if s.path.startswith("/vera-"):
            th = s.path[6:].split(".")[0]
            body = (PAGE % ("theme-dark" if th == "dark" else "", "#111" if th == "dark" else "#fff", FIG, FIG, th)).encode()
            s.send_response(200); s.send_header("Content-Type", "text/html"); s.end_headers(); s.wfile.write(body); return
        return super().do_GET()
srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H); port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 640, "height": 900}, device_scale_factor=2)
    for th in ("light", "dark"):
        if th == "dark": pg.emulate_media(color_scheme="dark")
        pg.goto(f"http://127.0.0.1:{port}/vera-{th}.html"); pg.wait_for_function("document.title==='done'", timeout=30000)
        pg.wait_for_timeout(500); pg.locator("#m").screenshot(path=f"vera_fig_{th}.png")
        print("wrote", f"vera_fig_{th}.png")
    b.close()
srv.shutdown()
