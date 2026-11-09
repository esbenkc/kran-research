"""browser-use 0.13.14 page-reading service: ONE shared BrowserSession, browser-use's own code unmodified, no LLM.

    .venv-bu/bin/python bu_service.py [--port 8781]

POST JSON endpoints (all answer JSON; any failure -> {"error": str}):
  /reset    {run}                      -> {ok, run}
  /navigate {run, url}                 -> {state, status, url, result}
  /click    {run, index}               -> {state, status, url, result, error?}
  /scroll   {run, down, pages?=1.0}    -> {state, url, result}
  /state    {run}                      -> {state, url}

`state` is the exact <browser_state>...</browser_state> block that browser-use's Agent puts into the LLM user message
(AgentMessagePrompt.get_user_message, agent/prompts.py:418), cut out of the message that browser-use itself renders.
Actions run through browser-use's Tools.act (the same call Agent.multi_act makes, agent/service.py:2807), so clicks
are real browser clicks (ClickElementEvent) and the static server's access log sees the GET.
`result` is the ActionResult text the Agent would put into <agent_history> (long_term_memory / extracted_content).
`status` = HTTP status of the current document (performance navigation entry), null if unknown.
A request whose `run` differs from the last /reset run triggers an implicit reset first.
"""
from __future__ import annotations

import os

# Must be set before browser_use is imported: no telemetry / cloud sync / PyPI version check (no outbound traffic),
# no default extensions (they are downloaded from the Chrome Web Store on first launch).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("BROWSER_USE_CLOUD_SYNC", "false")
os.environ.setdefault("BROWSER_USE_VERSION_CHECK", "false")
os.environ.setdefault("BROWSER_USE_DISABLE_EXTENSIONS", "1")
os.environ.setdefault("BROWSER_USE_LOGGING_LEVEL", "warning")

import argparse
import asyncio
import glob
import json
import threading
import signal
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from browser_use.agent.prompts import AgentMessagePrompt
from browser_use.agent.views import AgentSettings
from browser_use.browser import BrowserProfile, BrowserSession
from browser_use.browser.events import CloseTabEvent, NavigateToUrlEvent
from browser_use.tools.service import Tools

ALLOWED_PREFIX = "http://127.0.0.1:8765/"
# browser-use's own Chromium lookup (local_browser_watchdog.py:254) expects .../Chromium.app; Playwright 1248 ships
# "Google Chrome for Testing.app", so without an explicit path browser-use would run `uvx playwright install`.
CHROME_GLOB = str(Path.home() / "Library/Caches/ms-playwright/chromium-*/chrome-mac*/Google Chrome for Testing.app/"
                  "Contents/MacOS/Google Chrome for Testing")
# Network sandbox only (does not touch rendering): every hostname except 127.0.0.1 fails DNS.
NO_DNS_ARG = "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1"


class BUService:
    """Holds the single BrowserSession and a Tools instance configured like Agent's defaults."""

    def __init__(self) -> None:
        self.session: BrowserSession | None = None
        self.run: str | None = None
        # Agent default: use_vision=True -> Tools(exclude_actions=['screenshot']) (agent/service.py:319-325)
        self.tools = Tools(exclude_actions=["screenshot"])
        self.ActionModel = self.tools.registry.create_action_model()
        # Agent passes include_attributes=None -> MessageManager [] -> llm_representation falls back to
        # DEFAULT_INCLUDE_ATTRIBUTES (dom/views.py:953); max_clickable_elements_length default 40000.
        self.settings = AgentSettings(include_attributes=None)

    # ---------------------------------------------------------------- browser lifecycle
    async def _start(self) -> None:
        exe = sorted(glob.glob(CHROME_GLOB))
        if not exe:
            raise RuntimeError(f"no Chrome for Testing under {CHROME_GLOB}")
        profile = BrowserProfile(headless=True, executable_path=exe[-1], args=[NO_DNS_ARG])
        self.session = BrowserSession(browser_profile=profile)
        await self.session.start()

    async def ensure(self) -> BrowserSession:
        if self.session is None:
            await self._start()
        assert self.session is not None
        return self.session

    async def shutdown(self) -> None:
        if self.session is not None:
            try:
                await self.session.kill()
            finally:
                self.session = None

    async def reset(self, run: str) -> dict:
        """Fresh about:blank tab, all other tabs closed, HTTP cache + cookies cleared, cached DOM state dropped."""
        s = await self.ensure()
        try:
            old = [t.target_id for t in await s.get_tabs()]
            ev = s.event_bus.dispatch(NavigateToUrlEvent(url="about:blank", new_tab=True))
            await ev
            await ev.event_result(raise_if_any=True, raise_if_none=False)
            for tid in old:
                if tid != s.agent_focus_target_id:
                    cev = s.event_bus.dispatch(CloseTabEvent(target_id=tid))
                    await cev
            cdp = await s.get_or_create_cdp_session()
            await cdp.cdp_client.send.Network.clearBrowserCache(session_id=cdp.session_id)
            await cdp.cdp_client.send.Network.clearBrowserCookies(session_id=cdp.session_id)
            # same effect as AgentFocusChangedEvent (browser/session.py:1254-1261): no previous state -> no '*' marks
            if s._dom_watchdog:
                s._dom_watchdog.clear_cache()
            s._cached_browser_state_summary = None
            s._cached_selector_map.clear()
            s._cached_selector_indices.clear()
        except Exception:
            # browser wedged/crashed: start a brand-new one (still only one Chromium at a time)
            await self.shutdown()
            await self._start()
        self.run = run
        return {"ok": True, "run": run}

    async def _maybe_reset(self, run: str | None) -> BrowserSession:
        if run is not None and run != self.run:
            await self.reset(run)
        return await self.ensure()

    # ---------------------------------------------------------------- browser-use calls
    async def _act(self, s: BrowserSession, **action) -> dict:
        res = await self.tools.act(action=self.ActionModel(**action), browser_session=s, file_system=None,
                                   available_file_paths=[])
        out = {"result": res.long_term_memory or res.extracted_content}
        if res.error:
            out["error"] = res.error
        return out

    async def _state_text(self, s: BrowserSession) -> tuple[str, str]:
        # Agent._prepare_context (agent/service.py:1097) calls this with include_screenshot=True; the screenshot only
        # feeds the image part of the message (prompts.py:444-474), which a text-only model never receives.
        summary = await s.get_browser_state_summary(include_screenshot=False, include_recent_events=False)
        msg = AgentMessagePrompt(
            browser_state_summary=summary,
            file_system=None,  # type: ignore[arg-type]
            include_attributes=self.settings.include_attributes or [],
            max_clickable_elements_length=self.settings.max_clickable_elements_length,
            vision_detail_level=self.settings.vision_detail_level,
            include_recent_events=False,
        ).get_user_message(use_vision=False)
        text = msg.text
        a = text.index("<browser_state>\n")
        b = text.index("</browser_state>\n", a) + len("</browser_state>\n")
        return text[a:b], summary.url

    async def _status(self, s: BrowserSession) -> int | None:
        try:
            cdp = await s.get_or_create_cdp_session()
            r = await cdp.cdp_client.send.Runtime.evaluate(
                params={"expression": "(performance.getEntriesByType('navigation')[0]||{}).responseStatus||null",
                        "returnByValue": True}, session_id=cdp.session_id)
            return r.get("result", {}).get("value")
        except Exception:
            return None

    # ---------------------------------------------------------------- endpoints
    async def navigate(self, run: str | None, url: str) -> dict:
        if not url.startswith(ALLOWED_PREFIX):
            return {"error": f"refusing url outside {ALLOWED_PREFIX}"}
        s = await self._maybe_reset(run)
        out = await self._act(s, navigate={"url": url, "new_tab": False})
        out["state"], out["url"] = await self._state_text(s)
        out["status"] = await self._status(s)
        return out

    async def click(self, run: str | None, index: int) -> dict:
        s = await self._maybe_reset(run)
        if s._cached_browser_state_summary is None:  # Agent always has a fresh state before acting
            await self._state_text(s)
        out = await self._act(s, click={"index": int(index)})
        out["state"], out["url"] = await self._state_text(s)
        out["status"] = await self._status(s)
        return out

    async def scroll(self, run: str | None, down: bool, pages: float | None) -> dict:
        s = await self._maybe_reset(run)
        params = {"down": bool(down)}
        if pages is not None:
            params["pages"] = float(pages)
        out = await self._act(s, scroll=params)
        out["state"], out["url"] = await self._state_text(s)
        return out

    async def state(self, run: str | None) -> dict:
        s = await self._maybe_reset(run)
        st, url = await self._state_text(s)
        return {"state": st, "url": url}


class Server:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.svc = BUService()
        self.lock = threading.Lock()

    def call(self, coro, timeout: float = 120.0) -> dict:
        with self.lock:
            return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def dispatch(self, path: str, body: dict) -> dict:
        run = body.get("run")
        run = None if run is None else str(run)
        svc = self.svc
        if path == "/reset":
            return self.call(svc.reset(run or ""))
        if path == "/navigate":
            return self.call(svc.navigate(run, body["url"]))
        if path == "/click":
            return self.call(svc.click(run, body["index"]))
        if path == "/scroll":
            return self.call(svc.scroll(run, body.get("down", True), body.get("pages")))
        if path == "/state":
            return self.call(svc.state(run))
        return {"error": f"unknown endpoint {path}"}


def make_handler(server: Server):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                self._send(200, server.dispatch(self.path, body))
            except Exception as e:
                traceback.print_exc()
                self._send(200, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            self._send(200, {"ok": True, "run": server.svc.run}) if self.path == "/health" else self._send(404, {"error": "POST only"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8781)
    a = ap.parse_args()
    server = Server()
    httpd = HTTPServer(("127.0.0.1", a.port), make_handler(server))  # single-threaded: one request at a time

    def _term(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    print(f"bu_service on http://127.0.0.1:{a.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            asyncio.run_coroutine_threadsafe(server.svc.shutdown(), server.loop).result(30)
        except Exception:
            pass


if __name__ == "__main__":
    main()
