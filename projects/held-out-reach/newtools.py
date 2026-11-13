"""Held-out fetch-tool stacks: shipped tools never used on this track before. Each one is the shipped package,
called through its public entry point, with its own model-facing schema (names, descriptions, arguments).

  id          package (entry point)                                            agent actions (model-facing)
  lcpw        langchain-community 0.4.2 PlayWrightBrowserToolkit.from_browser   its 7 tools: navigate_browser,
              (sync Playwright Chromium, one fresh browser per run)            previous_webpage, click_element,
                                                                               extract_text, extract_hyperlinks,
                                                                               get_elements, current_webpage
  pwmcp       Microsoft @playwright/mcp 0.0.85 over stdio (one server and one   its own tool list minus 3 tools that
              isolated headless browser per run; --allowed-origins = sandbox)  touch the host (run_code_unsafe,
                                                                               file_upload, drop)
  wbl         langchain-community 0.4.2 WebBaseLoader(url).load() page_content  fetch(url)
              (requests + BeautifulSoup get_text: link text, no addresses)
  markitdown  markitdown-mcp 0.0.1a8, tool convert_to_markdown via its own      convert_to_markdown(uri)
              MCPServer.call_tool (MarkItDown().convert_uri(uri).markdown)

Every request goes over HTTP to server.py, so the access log sees every page a stack loads (the only follow source).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
NEW_TOOLS = ("lcpw", "pwmcp", "wbl", "markitdown")
PWMCP_EXCLUDED = ("browser_run_code_unsafe", "browser_file_upload", "browser_drop")  # host code / host files
CHROMIUM = str(Path.home() / "Library/Caches/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-mac-arm64/chrome-headless-shell")
PWMCP_CLI = str(ROOT / "js" / "node_modules" / "@playwright" / "mcp" / "cli.js")
os.environ.setdefault("USER_AGENT", "Mozilla/5.0 (compatible; research-agent-harness)")


def _props(schema: dict, drop: tuple = ()) -> dict:
    """The parent's compact arg format: {name: description (+ default) (+ required)}."""
    req = set(schema.get("required", []))
    out = {}
    for k, v in (schema.get("properties") or {}).items():
        if k in drop:
            continue
        d = v.get("description", "") or v.get("title", "")
        if "default" in v:
            d += f" (default {json.dumps(v['default'])})"
        if k in req:
            d += " (required)"
        out[k] = d
    return out


ANSWER = {"name": "answer", "description": "Give your final answer to the user and stop.", "args": {"text": "the answer"}}


# ---- LangChain PlayWrightBrowserToolkit ---------------------------------------------------------------------------

class LCPW:
    def __init__(self, launch, base: str):
        from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
        self.base = base
        self.browser = launch()
        self.tools = {t.name: t for t in PlayWrightBrowserToolkit.from_browser(sync_browser=self.browser).get_tools()}

    def specs(self) -> list[dict]:
        out = []
        for name, t in self.tools.items():
            out.append({"name": name, "description": t.description, "args": _props(t.args_schema.model_json_schema())})
        return out + [ANSWER]

    def page_url(self) -> str | None:
        try:
            ctx = self.browser.contexts
            return ctx[0].pages[-1].url if ctx and ctx[0].pages else None
        except Exception:
            return None

    def act(self, name: str, args: dict) -> tuple[str, dict]:
        if name not in self.tools:
            return f"Error: unknown tool {name!r}.", {"error": "unknown tool"}
        if name == "navigate_browser":
            url = str(args.get("url", ""))
            if url.startswith("http") and not url.startswith(self.base + "/"):
                return "Error: the browser can only reach this website.", {"url": url, "blocked": True}
        obs = self.tools[name].run(args)
        return str(obs), {"url": self.page_url()}

    def close(self) -> None:
        try:
            self.browser.close()
        except Exception:
            pass


# ---- Playwright MCP (stdio; one session task per run on the shared MCP event loop) -------------------------------

async def _pwmcp_session(params, q: asyncio.Queue, ready: concurrent.futures.Future) -> None:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    try:
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tl = await s.list_tools()
                ready.set_result(tl.tools)
                while True:
                    item = await q.get()
                    if item is None:
                        break
                    name, args, fut = item
                    try:
                        fut.set_result(await asyncio.wait_for(s.call_tool(name, args), 60))
                    except BaseException as e:  # noqa: BLE001 - the error is the observation
                        fut.set_exception(e if isinstance(e, Exception) else RuntimeError(repr(e)))
    except BaseException as e:  # noqa: BLE001
        if not ready.done():
            ready.set_exception(e if isinstance(e, Exception) else RuntimeError(repr(e)))


class PWMCP:
    def __init__(self, loop, base: str, ua: str):
        from mcp import StdioServerParameters
        self.loop, self.base = loop, base
        self.outdir = tempfile.mkdtemp(prefix="pwmcp-", dir=str(ROOT / "raw" / "pwmcp_out"))
        args = [PWMCP_CLI, "--headless", "--isolated", "--browser", "chromium", "--executable-path", CHROMIUM,
                "--allowed-origins", base, "--image-responses", "omit", "--output-dir", self.outdir,
                "--user-agent", ua]
        params = StdioServerParameters(command="node", args=args)
        self.q: asyncio.Queue = asyncio.run_coroutine_threadsafe(self._mkq(), loop).result(10)
        ready: concurrent.futures.Future = concurrent.futures.Future()
        self.task = asyncio.run_coroutine_threadsafe(_pwmcp_session(params, self.q, ready), loop)
        self.tools = {t.name: t for t in ready.result(60) if t.name not in PWMCP_EXCLUDED}

    @staticmethod
    async def _mkq():
        return asyncio.Queue()

    def specs(self) -> list[dict]:
        return [{"name": n, "description": t.description, "args": _props(t.input_schema)} for n, t in self.tools.items()] + [ANSWER]

    def act(self, name: str, args: dict) -> tuple[str, dict]:
        if name not in self.tools:
            return f"Error: unknown tool {name!r}.", {"error": "unknown tool"}
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self.loop.call_soon_threadsafe(self.q.put_nowait, (name, args, fut))
        res = fut.result(75)
        text = "\n".join(getattr(c, "text", "") for c in res.content if getattr(c, "text", None))
        url = None
        for line in text.splitlines():
            if line.strip().startswith("- Page URL:"):
                url = line.split("Page URL:", 1)[1].strip()
        return (("Error: " + text) if res.is_error else text), {"url": url}

    def close(self) -> None:
        try:
            self.loop.call_soon_threadsafe(self.q.put_nowait, None)
            self.task.result(20)
        except Exception:
            pass


# ---- WebBaseLoader / markitdown-mcp (stateless) -------------------------------------------------------------------

def wbl_fetch(url: str) -> str:
    from langchain_community.document_loaders import WebBaseLoader
    docs = WebBaseLoader(url, raise_for_status=True).load()
    return docs[0].page_content if docs else ""


WBL_SPEC = {"name": "fetch", "description": "Load a web page with LangChain's WebBaseLoader and return its text.",
            "args": {"url": "absolute URL of the page (required)"}}

_MD: dict = {}


def _markitdown_server():
    if "srv" not in _MD:
        from markitdown_mcp.__main__ import mcp
        _MD["srv"] = mcp
    return _MD["srv"]


def markitdown_schema(run) -> dict:
    t = [x for x in run(_markitdown_server().list_tools()) if x.name == "convert_to_markdown"][0]
    return {"name": t.name, "description": t.description, "args": _props(t.input_schema)}


def markitdown_call(run, uri: str) -> str:
    res = run(_markitdown_server().call_tool("convert_to_markdown", {"uri": uri}))
    text = "\n".join(getattr(c, "text", "") for c in res.content)
    return ("Error: " + text) if res.is_error else text
