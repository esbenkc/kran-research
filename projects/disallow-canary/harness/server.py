"""Static web server on 127.0.0.1 with its own access log. The log is the only scoring input.

Per-instance version of the parent's server.py (nofollow-canary web/server.py): each SiteServer has its own port,
its own site root and its own run label, so several runs can go at once (one server per worker thread). The site
root IS the run's site folder (root-served: the canary is /r/<hid>.html, robots.txt is /robots.txt).

Log line (tab separated): ISO time, client ip, method, path, status, user agent, run label, port.
"""
from __future__ import annotations

import datetime as dt
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_LOG_LOCK = threading.Lock()


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        srv = args[2]
        self.run_label = srv.label
        super().__init__(*args, directory=srv.root, **kwargs)

    def log_request(self, code="-", size="-"):  # one line per request, written before the response body
        ua = self.headers.get("User-Agent", "-").replace("\t", " ")
        line = (f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='milliseconds')}\t{self.client_address[0]}\t"
                f"{self.command}\t{self.path}\t{code}\t{ua}\t{self.run_label}\t{self.server.server_address[1]}\n")
        with _LOG_LOCK, self.server.log_path.open("a") as f:
            f.write(line)

    def log_message(self, fmt, *args):  # silence stderr
        pass

    def list_directory(self, path):  # no directory listings: a listing would reveal the hidden page
        self.send_error(404, "Not found")
        return None


class SiteServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, log_path: Path, root: str = ".", label: str = "-"):
        self.root, self.label, self.log_path = root, label, Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(("127.0.0.1", port), _Handler)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    def set_run(self, root: str, label: str) -> None:
        self.root, self.label = root, label


def read_log(path: Path) -> list[dict]:
    keys = ("time", "ip", "method", "path", "status", "ua", "run", "port")
    out = []
    for line in Path(path).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == len(keys):
            out.append(dict(zip(keys, parts)))
    return out
