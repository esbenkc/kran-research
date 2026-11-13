"""Fetch-tool stacks: each one turns a page of the local sandbox into the text a model reads, with that tool's own
action set. Every converter is a shipped, pinned package called through its public entry point, unmodified
(versions in env.json, written by `python tools.py --env`).

  id                 package (entry point)                                  agent actions
  raw                parent harness: HTTP body as text (replication)        fetch(url)
  markdownify        markdownify 1.2.3 markdownify(html) (parent md view)   fetch(url)
  mcpfetch           mcp-server-fetch 2026.11.4, tool 'fetch' via its own   fetch(url, max_length=5000, start_index=0,
                     MCPServer.call_tool, Node present (Readability.js)       raw=false)  [its own JSON schema]
  mcpraw             same server, the stack pins raw=true                   fetch(url, max_length, start_index)
  mcpfetch_nonode    same server, use_readability_js=False (readabilipy)    crawl-only row
  html2text          html2text 2025.4.15 html2text(html, baseurl=url)       fetch(url)
  trafilatura        trafilatura 2.3.1 extract(html, url, output_format=md) fetch(url)
  turndown           @mozilla/readability 0.6.0 on jsdom, then turndown    fetch(url)  [PREREG s2 'turndown']
  turndown_plain     turndown 7.2.5 TurndownService().turndown(html)       fetch(url)  (crawl-only row)
  crawl4ai           crawl4ai 0.9.5 AsyncWebCrawler.arun(url).markdown       fetch(url)
                     (Chromium-rendered; separate venv, c4_service.py)
  innertext          Playwright 1.64.1 Chromium, document.body.innerText    goto(url), click(text)
                     click = page.get_by_text(text).first.click()
  aria               Playwright ariaSnapshot of <body> (parent a11y view)   goto(url), click(name)
  browseruse         browser-use 0.13.14 browser state text (bu_service.py) navigate(url), click(index), scroll(down)

The fetch request itself always goes over HTTP to server.py, so the access log sees every page a stack loads.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parent
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
UA = "Mozilla/5.0 (compatible; research-agent-harness)"  # parent harness UA (raw, md, html2text, trafilatura, turndown)
C4_URL = "http://127.0.0.1:8782"
BU_URL = "http://127.0.0.1:8781"

TOOLS = ("raw", "markdownify", "mcpfetch", "mcpraw", "mcpfetch_nonode", "html2text", "trafilatura", "turndown",
         "turndown_plain", "crawl4ai", "innertext", "aria", "browseruse", "bufull", "lcpw", "pwmcp", "wbl", "markitdown")
CLICK_TOOLS = {"innertext": "text", "browseruse": "index"}


def _http_get(url: str) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""


def _post(base: str, path: str, payload: dict, timeout: float = 90) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ---- stateless converters ---------------------------------------------------------------------------------------

def conv_md(html: str, url: str) -> str:
    from markdownify import markdownify
    return markdownify(html, heading_style="ATX")  # parent agent.py view_md, verbatim


def conv_html2text(html: str, url: str) -> str:
    import html2text
    return html2text.html2text(html, baseurl=url)


def conv_trafilatura(html: str, url: str) -> str:
    import trafilatura
    return trafilatura.extract(html, url=url, output_format="markdown") or ""


def conv_turndown(html: str, url: str, readability: bool = False) -> str:
    args = ["node", str(ROOT / "js" / "turndown_cli.js")] + (["--readability"] if readability else []) + [url]
    return subprocess.run(args, input=html, capture_output=True, text=True, timeout=60, check=True).stdout


CONVERTERS = {
    "markdownify": conv_md, "html2text": conv_html2text, "trafilatura": conv_trafilatura,
    "turndown_plain": conv_turndown, "turndown": lambda h, u: conv_turndown(h, u, readability=True),
}

# ---- mcp-server-fetch (in-process MCPServer on one background event loop) -------------------------------------
# Playwright's sync API runs its own event loop in each worker thread, so MCP calls go to one shared loop thread.

_TL = threading.local()
_MCP_LOCK = threading.Lock()
_MCP: dict = {}


def _mcp_loop():
    with _MCP_LOCK:
        if "loop" not in _MCP:
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, daemon=True).start()
            _MCP["loop"] = loop
        return _MCP["loop"]


def _mcp(nonode: bool):
    key = "nonode" if nonode else "node"
    with _MCP_LOCK:
        if key not in _MCP:
            from mcp_server_fetch.server import create_server
            # allow_private_ips: the sandbox is 127.0.0.1; ignore_robots_txt: the sandbox has no robots.txt (404 =
            # allowed anyway; this only skips one request per call). Everything else is the package default.
            _MCP[key] = create_server(allow_private_ips=True, ignore_robots_txt=True, use_readability_js=not nonode)
        return _MCP[key]


def _mcp_run(coro, timeout: float = 90):
    return asyncio.run_coroutine_threadsafe(coro, _mcp_loop()).result(timeout=timeout)


def mcp_call(args: dict, nonode: bool = False) -> str:
    res = _mcp_run(_mcp(nonode).call_tool("fetch", args))
    text = "\n".join(getattr(c, "text", "") for c in res.content)
    return ("Error: " + text) if res.is_error else text


def mcp_schema() -> dict:
    t = _mcp_run(_mcp(False).list_tools())[0]
    return {"name": t.name, "description": t.description, "input_schema": t.input_schema}


# ---- Playwright (innerText, aria), one browser per thread --------------------------------------------------------

def _browser():
    if not hasattr(_TL, "browser"):
        from playwright.sync_api import sync_playwright
        _TL.pw = sync_playwright().start()
        _TL.browser = _TL.pw.chromium.launch()
    return _TL.browser


def close_browser() -> None:
    if hasattr(_TL, "browser"):
        _TL.browser.close()
        _TL.pw.stop()
        del _TL.browser


# ---- stacks -----------------------------------------------------------------------------------------------------

class Stack:
    """One run's view of the web through one tool. first(url) = what the crawl records; act(name, args) = agent step.
    act returns (observation text, info dict with 'url' of the page now shown, if any)."""

    def __init__(self, tool: str, run_id: str, model: str | None = None):
        assert tool in TOOLS, tool
        self.tool, self.run_id, self.current, self.model = tool, run_id, None, model
        self.page = None
        self.held = None  # lcpw / pwmcp session object (newtools.py)
        if tool == "lcpw":
            import newtools
            _browser()  # starts this thread's Playwright driver
            self.held = newtools.LCPW(lambda: _TL.pw.chromium.launch(), BASE)
        elif tool == "pwmcp":
            import newtools
            self.held = newtools.PWMCP(_mcp_loop(), BASE, UA)

    # tool specs shown to the model (name, description, args) ---------------------------------------------------
    def specs(self) -> list[dict[str, Any]]:
        t = self.tool
        if t in ("lcpw", "pwmcp"):
            return self.held.specs()
        if t == "bufull":
            from bu_client import bufull_specs
            return bufull_specs()
        if t == "markitdown":
            import newtools
            return [newtools.markitdown_schema(_mcp_run), newtools.ANSWER]
        if t == "wbl":
            import newtools
            return [newtools.WBL_SPEC, newtools.ANSWER]
        if t in ("mcpfetch", "mcpraw"):
            s = mcp_schema()
            props = {k: v.get("description", "") + (f" (default {v['default']})" if "default" in v else "")
                     for k, v in s["input_schema"]["properties"].items() if k != "timeout_ms"}
            if t == "mcpraw":
                props.pop("raw")
            fetch = {"name": "fetch", "description": s["description"], "args": props}
        elif t in ("innertext", "aria"):
            what = ("the text of the page (document.body.innerText)" if t == "innertext"
                    else "the page's accessibility tree (links show their /url)")
            label = "visible text" if t == "innertext" else "accessible name"
            return [{"name": "goto", "description": f"Open a URL in a web browser and return {what}.",
                     "args": {"url": "absolute or relative URL"}},
                    {"name": "click", "description": f"Click the element on the current page with this {label}, then return {what} of the page you land on.",
                     "args": {"text": f"{label} of the element to click"}},
                    {"name": "answer", "description": "Give your final answer to the user and stop.", "args": {"text": "the answer"}}]
        elif t == "browseruse":
            from bu_client import bu_specs
            return bu_specs()
        elif t == "raw":
            fetch = {"name": "fetch", "description": "Fetch a URL over HTTP and return the page's raw HTML source.", "args": {"url": "absolute or relative URL"}}
        else:  # converters that return Markdown or text
            fetch = {"name": "fetch", "description": "Fetch a URL over HTTP and return the page converted to Markdown.", "args": {"url": "absolute or relative URL"}}
        return [fetch, {"name": "answer", "description": "Give your final answer to the user and stop.", "args": {"text": "the answer"}}]

    def first(self, url: str) -> str:
        """The first tool response for the start URL at the tool's defaults (the crawl's exposure input)."""
        if self.tool in ("browseruse", "bufull"):
            obs, _ = self.act("navigate", {"url": url})
        elif self.tool == "lcpw":
            self.act("navigate_browser", {"url": url})
            obs, _ = self.act("extract_text", {})  # navigate returns only a status line; the page view is extract_text
        elif self.tool == "pwmcp":
            obs, _ = self.act("browser_navigate", {"url": url})
        elif self.tool == "markitdown":
            obs, _ = self.act("convert_to_markdown", {"uri": url})
        elif self.tool in ("innertext", "aria"):
            obs, _ = self.act("goto", {"url": url})
        else:
            obs, _ = self.act("fetch", {"url": url})
        return obs

    def act(self, name: str, args: dict) -> tuple[str, dict]:
        t = self.tool
        if t == "browseruse":
            from bu_client import bu_act
            return bu_act(self.run_id, name, args)
        if t in ("innertext", "aria"):
            return self._browser_act(name, args)
        if t in ("lcpw", "pwmcp"):
            return self.held.act(name, args)
        if t == "bufull":
            from bu_client import bufull_act
            return bufull_act(self.run_id, name, args, self.model)
        if t == "markitdown" and name == "convert_to_markdown":
            name, args = "fetch", {"url": args.get("uri", "")}
        if name != "fetch":
            return f"Error: unknown tool {name!r}.", {"error": "unknown tool"}
        url = urljoin(self.current or BASE + "/", str(args.get("url", "")))
        if not url.startswith(BASE + "/"):
            return "Error: the fetch tool can only reach this website.", {"url": url, "blocked": True}
        info: dict[str, Any] = {"url": url}
        if t in ("wbl", "markitdown"):
            import newtools
            try:
                body = newtools.wbl_fetch(url) if t == "wbl" else newtools.markitdown_call(_mcp_run, url)
            except Exception as e:  # the loader's own error is the observation
                return f"Error: {type(e).__name__}: {str(e)[:200]}", info
            if not body.startswith("Error"):
                self.current = url
            return body, info
        if t in ("mcpfetch", "mcpraw", "mcpfetch_nonode"):
            a: dict[str, Any] = {"url": url}
            for k in ("max_length", "start_index", "raw"):
                if k in args and not (t == "mcpraw" and k == "raw"):
                    a[k] = args[k]
            if t == "mcpraw":
                a["raw"] = True
            info["mcp_args"] = {k: v for k, v in a.items() if k != "url"}
            obs = mcp_call(a, nonode=(t == "mcpfetch_nonode"))
            if not obs.startswith("Error"):
                self.current = url
            return obs, info
        if t == "crawl4ai":
            r = _post(C4_URL, "/fetch", {"url": url})
            code, body = r.get("status", 0), r.get("markdown", "")
        else:
            code, html = _http_get(url)
            body = html if t == "raw" else (CONVERTERS[t](html, url) if code == 200 else "")
        info["status"] = code
        if code == 200:
            self.current = url
            return body, info
        return f"HTTP {code}", info

    def _browser_act(self, name: str, args: dict) -> tuple[str, dict]:
        """innertext / aria: one Playwright page per run. click = Playwright's own click with its actionability checks
        (innertext: get_by_text; aria: get_by_role('link', name), else get_by_text). The same function serves the
        crawl's click-by-code and the agent (PREREG s8.2)."""
        if self.page is None:
            self.page = _browser().new_page(user_agent=UA)
        p = self.page
        try:
            if name == "goto":
                url = urljoin(self.current or BASE + "/", str(args.get("url", "")))
                if not url.startswith(BASE + "/"):
                    return "Error: the browser can only reach this website.", {"url": url, "blocked": True}
                resp = p.goto(url, wait_until="load", timeout=20_000)
                code = resp.status if resp else 0
                if code != 200:
                    return f"HTTP {code}", {"url": url, "status": code}
            elif name == "click":
                text = str(args.get("text", ""))
                loc = p.get_by_text(text)
                if self.tool == "aria":
                    role = p.get_by_role("link", name=text)
                    loc = role if role.count() else loc
                loc.first.click(timeout=5_000)
                p.wait_for_load_state("load", timeout=10_000)
            else:
                return f"Error: unknown tool {name!r}.", {"error": "unknown tool"}
        except Exception as e:  # Playwright's own actionability error is what the agent sees
            return f"Error: {str(e).splitlines()[0][:300]}", {"error": "playwright", "error_text": str(e)[:600], "url": p.url}
        self.current = p.url
        if self.tool == "innertext":
            return f"URL: {p.url}\n" + p.evaluate("document.body.innerText"), {"url": p.url}
        # aria: the parent's a11y view format, verbatim (agent.py view_a11y)
        return f"- document \"{p.title()}\"\n" + p.locator("body").aria_snapshot(), {"url": p.url}

    def close(self) -> None:
        if self.held is not None:
            self.held.close()
            self.held = None
        if self.page is not None:
            self.page.close()
            self.page = None
        if self.tool == "browseruse":
            from bu_client import bu_close
            bu_close(self.run_id)


def env_versions() -> dict:
    import importlib.metadata as md
    import platform
    out = {"python": platform.python_version()}
    for p in ("mcp-server-fetch", "mcp", "readabilipy", "markdownify", "html2text", "trafilatura", "playwright"):
        out[p] = md.version(p)
    out["node"] = subprocess.run(["node", "-v"], capture_output=True, text=True).stdout.strip()
    js = json.loads((ROOT / "js" / "package-lock.json").read_text())["packages"]
    for p in ("turndown", "@mozilla/readability", "jsdom"):
        out["npm:" + p] = js[f"node_modules/{p}"]["version"]
    import readabilipy
    rj = Path(readabilipy.__file__).parent / "javascript" / "node_modules"
    for p in ("@mozilla/readability", "jsdom"):
        f = rj / p / "package.json"
        out["mcpfetch-node:" + p] = json.loads(f.read_text())["version"] if f.exists() else None
    for venv, pkgs in ((".venv-c4", ("crawl4ai", "playwright")), (".venv-bu", ("browser-use", "cdp-use"))):
        for p in pkgs:
            out[f"{venv}:{p}"] = subprocess.run([str(ROOT / venv / "bin" / "python"), "-c",
                                                 f"import importlib.metadata as m;print(m.version('{p}'))"],
                                                capture_output=True, text=True).stdout.strip()
    return out


if __name__ == "__main__":
    import sys
    if "--env" in sys.argv:
        (ROOT / "env.json").write_text(json.dumps(env_versions(), indent=1) + "\n")
        print((ROOT / "env.json").read_text())
    else:
        tool, url = sys.argv[1], sys.argv[2]
        s = Stack(tool, "cli")
        print(s.first(url))
        s.close()
