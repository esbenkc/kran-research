"""Static web server on 127.0.0.1 with its own access log. The log is the only scoring input.

    .venv/bin/python server.py [--port 8765] [--log logs/access.log]

Log line (tab separated): ISO time, client ip, method, path, status, user agent, run.
anchor-text-law (PREREG s2, root-served): run.py sets CURRENT['dir'] to the run's site folder before each run, so the
server root IS that run's site (the canary is /r/<hid>.html), and CURRENT['run'] ('<cell>/<NNN>', '-' between runs)
goes into the 7th column. Copied from the parent server.py; only the root and the 7th column are new.
"""
from __future__ import annotations

import argparse
import datetime as dt
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_LOCK = threading.Lock()
CURRENT: dict[str, str] = {"dir": str(ROOT / "site"), "run": "-"}


class LoggingHandler(SimpleHTTPRequestHandler):
    log_path: Path = ROOT / "logs" / "access.log"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("directory", CURRENT["dir"])
        self.run_label = CURRENT["run"]
        super().__init__(*args, **kwargs)

    def log_request(self, code="-", size="-"):  # one line per request, written before the response body
        ua = self.headers.get("User-Agent", "-").replace("\t", " ")
        line = (f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='milliseconds')}\t{self.client_address[0]}\t"
                f"{self.command}\t{self.path}\t{code}\t{ua}\t{self.run_label}\n")
        with _LOCK, self.log_path.open("a") as f:
            f.write(line)

    def log_message(self, fmt, *args):  # silence stderr
        pass

    def list_directory(self, path):  # no directory listings: a listing would reveal the hidden page
        self.send_error(404, "Not found")
        return None


def make_server(port: int) -> ThreadingHTTPServer:
    """Handler built per request, so it reads CURRENT at request time (the root follows the current run)."""
    return ThreadingHTTPServer(("127.0.0.1", port), lambda *a, **k: LoggingHandler(*a, **k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--log", default=str(ROOT / "logs" / "access.log"))
    ap.add_argument("--root", default=str(ROOT / "site"))
    a = ap.parse_args()
    LoggingHandler.log_path = Path(a.log)
    LoggingHandler.log_path.parent.mkdir(parents=True, exist_ok=True)
    CURRENT["dir"] = a.root
    srv = make_server(a.port)
    print(f"serving {a.root} on http://127.0.0.1:{a.port} log {a.log}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
