"""Static web server for site/ on 127.0.0.1 with its own access log. The log is the only scoring input.

    .venv/bin/python server.py [--port 8765] [--log logs/access.log]

Log line (tab separated): ISO time, client ip, method, path, status, user agent.
"""
from __future__ import annotations

import argparse
import datetime as dt
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_LOCK = threading.Lock()


class LoggingHandler(SimpleHTTPRequestHandler):
    log_path: Path = ROOT / "logs" / "access.log"

    def log_request(self, code="-", size="-"):  # one line per request, written before the response body
        ua = self.headers.get("User-Agent", "-").replace("\t", " ")
        line = f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='milliseconds')}\t{self.client_address[0]}\t{self.command}\t{self.path}\t{code}\t{ua}\n"
        with _LOCK, self.log_path.open("a") as f:
            f.write(line)

    def log_message(self, fmt, *args):  # silence stderr
        pass

    def list_directory(self, path):  # no directory listings: a listing would reveal the hidden page
        self.send_error(404, "Not found")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--log", default=str(ROOT / "logs" / "access.log"))
    a = ap.parse_args()
    LoggingHandler.log_path = Path(a.log)
    LoggingHandler.log_path.parent.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), partial(LoggingHandler, directory=str(ROOT / "site")))
    print(f"serving {ROOT / 'site'} on http://127.0.0.1:{a.port} log {a.log}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
