"""crawl4ai 0.9.5 as a local JSON service (it needs its own venv: .venv-c4). One AsyncWebCrawler with library defaults.

    .venv-c4/bin/python c4_service.py --port 8782
    POST /fetch {"url": ...} -> {"status": int, "markdown": str}   (markdown = str(result.markdown), the default output)

Only 127.0.0.1:8765 URLs are accepted. Requests are served one at a time.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from crawl4ai import AsyncWebCrawler

LOOP = asyncio.new_event_loop()
LOCK = threading.Lock()
CRAWLER: AsyncWebCrawler | None = None


async def _start():
    global CRAWLER
    CRAWLER = AsyncWebCrawler()
    await CRAWLER.start()


async def _fetch(url: str) -> dict:
    r = await CRAWLER.arun(url=url)
    return {"status": r.status_code or 0, "markdown": str(r.markdown) if r.success else "", "success": r.success,
            "error": r.error_message}


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        url = str(req.get("url", ""))
        if not url.startswith("http://127.0.0.1:8765/"):
            out = {"status": 0, "markdown": "", "error": "offsite"}
        else:
            with LOCK:
                out = asyncio.run_coroutine_threadsafe(_fetch(url), LOOP).result(timeout=90)
        b = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8782)
    a = ap.parse_args()
    threading.Thread(target=LOOP.run_forever, daemon=True).start()
    asyncio.run_coroutine_threadsafe(_start(), LOOP).result(timeout=60)
    print(f"crawl4ai service on 127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
