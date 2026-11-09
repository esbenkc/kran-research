"""Crawl row "reachable by an omitted browser-use action" (model-free, local only).

    .venv-bu/bin/python bu_omitted.py

For methods display_none / off_screen (the two browser-use zero contexts), sr_only (positive control) and nolink
(negative control), 5 templates x 2 anchors, footer slot, inline css: build the site with websites.py (CLI, then
rename site/demo -> site/buo-<method>-<template>-<anchor>), navigate to index.html in ONE browser-use BrowserSession
(bu_service.BUService, unmodified browser-use 0.13.14) and run browser-use's own default actions that our agent did
not get, through Tools.act exactly as Agent.multi_act does:
  find_elements      {selector:'a'}                      (defaults: attributes=None -> no href in output)
  find_elements_href {selector:'a', attributes:['href']} (variant the action description itself suggests)
  search_page        {pattern: <anchor text>}            (defaults)
  find_text          {text: <anchor text>}
  extract            NOT called (it calls an LLM). Instead: the markdown extract would hand its LLM, built with
                     browser-use's own extract_clean_markdown + chunk_markdown_by_structure, ExtractAction defaults
                     (extract_links=False, extract_images=False, start_from_char=0)
  extract_links      same with extract_links=True (variant the description suggests: "Set extract_links=True for URLs")

Per row two texts are scored:
  returned text  = everything the action produced (ActionResult.extracted_content + long_term_memory + error);
                   for extract: the <webpage_content> markdown handed to the page-extraction LLM.
  agent text     = what the Agent's main LLM would actually receive in its next step, produced by browser-use's own
                   MessageManager._update_agent_history_description (<agent_history> action_results + <read_state>).
                   For extract the main LLM only gets the extraction LLM's answer, so agent text = n/a.
Echo: search_page / find_text repeat the query string (the anchor text) in their output. anchor_in_text is scored
after removing those echoes (quoted query / the fixed find_text sentences); find_text never returns page text, its
success flag is `signal`.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import bu_service  # sets the no-network env vars before browser_use is imported
from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.views import ActionResult, AgentStepInfo
from browser_use.dom.markdown_extractor import chunk_markdown_by_structure, extract_clean_markdown
from browser_use.llm.messages import SystemMessage
from browser_use.tools.views import ExtractAction

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
OUT = ROOT / "crawl_bu_omitted"
BASE = "http://127.0.0.1:8765"
TEMPLATES = ("pricing", "api", "school", "transit", "hotel")
ANCHORS = ("fit", "nonfit")
METHODS = ("display_none", "off_screen", "sr_only", "nolink")
ACTIONS = ("find_elements", "find_elements_href", "search_page", "find_text", "extract", "extract_links")


def build(template: str, method: str, anchor: str) -> dict:
    demo = SITE / "demo"
    for _ in range(50):  # never clobber someone else's in-flight demo build
        if not demo.exists():
            break
        time.sleep(0.2)
    else:
        raise RuntimeError("site/demo exists; refusing to overwrite")
    out = subprocess.run([str(ROOT / ".venv/bin/python"), "websites.py", template, method, anchor, "footer", "inline"],
                         cwd=ROOT, check=True, capture_output=True, text=True).stdout
    man = json.loads(out)
    token = f"buo-{method}-{template}-{anchor}"
    dst = SITE / token
    if dst.exists():
        shutil.rmtree(dst)
    demo.rename(dst)
    assert (dst / "r" / f"{man['hid']}.html").exists()
    man.update(token=token, start_path=f"/{token}/index.html", hidden_path=f"/{token}/r/{man['hid']}.html")
    return man


def norm(s: str) -> str:
    return " ".join(s.split())


def result_text(res: ActionResult) -> str:
    return "\n".join(x for x in (res.extracted_content, res.long_term_memory, res.error) if x)


def agent_text(res: ActionResult) -> str:
    """What the Agent's next LLM call contains about this action (browser-use's own history builder)."""
    mm = MessageManager(task="", system_message=SystemMessage(content=""), file_system=None)  # type: ignore[arg-type]
    mm._update_agent_history_description(model_output=None, result=[res], step_info=AgentStepInfo(step_number=0, max_steps=1))
    return mm.agent_history_description + "\n" + (mm.state.read_state_description or "")


def strip_echo(action: str, text: str, query: str) -> str:
    if action == "search_page":
        return text.replace(f'"{query}"', "")
    if action == "find_text":
        for s in (f"Scrolled to text: {query}", f"Text '{query}' not found or not visible on page",
                  f"Tried scrolling to text '{query}' but it was not found"):
            text = text.replace(s, "")
    return text


async def run_site(svc: bu_service.BUService, man: dict) -> list[dict]:
    run = man["token"]
    await svc.reset(run)
    s = await svc.ensure()
    nav = await svc._act(s, navigate={"url": BASE + man["start_path"], "new_tab": False})
    state, _ = await svc._state_text(s)  # the Agent always reads state before acting
    anchor, hid_s = man["anchor_text"], f"r/{man['hid']}"
    href_abs = BASE + man["hidden_path"]
    base = {"method": man["method"], "template": man["template"], "anchor": man["anchor"], "token": run,
            "hid": man["hid"], "anchor_text": anchor, "anchor_in_browser_state": anchor in state,
            "nav_error": nav.get("error")}
    rows = []

    async def act(name: str, params: dict) -> tuple[ActionResult | None, str | None]:
        try:
            res = await svc.tools.act(action=svc.ActionModel(**{name: params}), browser_session=s, file_system=None,
                                      available_file_paths=[])
            return res, res.error
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"

    def add(action: str, text: str, agent: str | None, error: str | None, signal=None, href_hidden=None):
        t = strip_echo(action, text, anchor)
        a = None if agent is None else strip_echo(action, agent, anchor)
        rows.append({**base, "action": action, "hid_in_text": hid_s in text, "anchor_in_text": anchor in norm(t),
                     "chars": len(text), "error": error, "signal": signal, "href_hidden": href_hidden,
                     "agent_hid_in_text": None if a is None else hid_s in a,
                     "agent_anchor_in_text": None if a is None else anchor in norm(a),
                     "agent_chars": None if agent is None else len(agent), "text": text,
                     "agent_text": agent})

    for name, action, params in (("find_elements", "find_elements", {"selector": "a"}),
                                 ("find_elements_href", "find_elements", {"selector": "a", "attributes": ["href"]}),
                                 ("search_page", "search_page", {"pattern": anchor})):
        res, err = await act(action, params)
        text = result_text(res) if res else ""
        sig = None
        if name == "search_page" and res:
            m = re.search(r": (\d+) match", res.long_term_memory or "")
            sig = int(m.group(1)) if m else None
        href_hidden = (f'href="{href_abs}"' in text) if name.startswith("find_elements") else None
        add(name, text, agent_text(res) if res else None, err, sig, href_hidden)

    # extract: the page-extraction LLM input, built exactly as tools/service.py:1109-1136 does (no LLM call)
    defaults = ExtractAction(query="")
    for name, links in (("extract", defaults.extract_links), ("extract_links", True)):
        try:
            content, _stats = await extract_clean_markdown(browser_session=s, extract_links=links,
                                                           extract_images=defaults.extract_images)
            chunk = chunk_markdown_by_structure(content, max_chunk_chars=100000, start_from_char=defaults.start_from_char)[0]
            text = (chunk.overlap_prefix + "\n" + chunk.content) if chunk.overlap_prefix else chunk.content
            add(name, text, None, None)
        except Exception as e:  # noqa: BLE001
            add(name, "", None, f"{type(e).__name__}: {e}")

    res, err = await act("find_text", {"text": anchor})  # last: it scrolls the page
    found = bool(res and (res.long_term_memory or "").startswith("Scrolled to text"))
    add("find_text", result_text(res) if res else "", agent_text(res) if res else None, err, found)
    return rows


def share(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return f"{sum(bool(v) for v in vals)}/{len(vals)}" if vals else "n/a"


def summary(rows: list[dict], defaults_ok: dict) -> str:
    by = defaultdict(list)
    for r in rows:
        by[(r["method"], r["action"])].append(r)
    L = ["# browser-use 0.13.14: hidden link reachable by actions our agent did not get", "",
         "Model-free, local only. One browser-use BrowserSession (bu_service.BUService, headless, 1920x1080, unmodified "
         "browser-use), sites from `websites.py <template> <method> <fit|nonfit> footer inline` (5 templates x 2 anchors "
         "per method), actions run through `Tools.act` like `Agent.multi_act`. Script: `bu_omitted.py`; rows: "
         "`omitted.jsonl` (one row per site x action).", "",
         "Columns: **hid** = text contains `r/<hid>` (the run's hidden page id); **anchor** = text contains the anchor "
         "words (after removing the query echo for search_page/find_text); **agent** = same two checks on what the "
         "Agent's main LLM would receive next step (browser-use's `MessageManager._update_agent_history_description`); "
         "**signal** = search_page match count > 0 / find_text 'Scrolled to text'.", "",
         "| method | action | hid (returned) | anchor (returned) | hid (agent sees) | anchor (agent sees) | signal |",
         "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        for a in ACTIONS:
            rs = by[(m, a)]
            sig = ""
            if a == "search_page":
                sig = f"{sum(1 for r in rs if (r['signal'] or 0) > 0)}/{len(rs)} matched"
            elif a == "find_text":
                sig = f"{sum(1 for r in rs if r['signal'])}/{len(rs)} found"
            elif a.startswith("find_elements"):
                sig = f"{sum(1 for r in rs if r['href_hidden'])}/{len(rs)} href=hidden page"
            L.append(f"| {m} | {a} | {share(rs, 'hid_in_text')} | {share(rs, 'anchor_in_text')} | "
                     f"{share(rs, 'agent_hid_in_text')} | {share(rs, 'agent_anchor_in_text')} | {sig} |")
    errs = [r for r in rows if r["error"]]
    L += ["", f"Errors: {len(errs)} rows" + ("" if not errs else ": " + "; ".join(f"{r['token']}/{r['action']}: {r['error']}" for r in errs[:10])),
          f"Anchor in the Agent's <browser_state> after navigate: " + ", ".join(
              f"{m} {sum(1 for r in rows if r['method'] == m and r['action'] == 'find_elements' and r['anchor_in_browser_state'])}/10"
              for m in METHODS), ""]
    L += ["## Where the actions live (browser_use/, 0.13.14) and whether the Agent gets them by default", "",
          "Default action list = `Tools(exclude_actions=['screenshot'])` built by `Agent.__init__` "
          "(agent/service.py:319-325); checked programmatically below.", "",
          "| action | in default list | implementation | what it reads | what reaches the main LLM |",
          "|---|---|---|---|---|",
          f"| find_elements | {defaults_ok['find_elements']} | tools/service.py:1344-1377; JS tools/service.py:278-319; format 384-422 | `document.querySelectorAll(selector)`, `el.textContent` (DOM text incl. display:none / off-screen), attributes only if `attributes` given (default None, tools/views.py:41-44) | only `long_term_memory` 'Found N elements matching \"a\".' (message_manager/service.py:335-340: long_term_memory wins, extracted_content not in read_state) |",
          f"| search_page | {defaults_ok['search_page']} | tools/service.py:1307-1342; JS 202-276; format 358-381 | TreeWalker over all text nodes of body, `node.textContent` (no visibility filter) | only 'Searched page for \"X\": N matches found.' |",
          f"| find_text | {defaults_ok['find_text']} | tools/service.py:1503-1522 -> ScrollToTextEvent, browser/watchdogs/default_action_watchdog.py:2730-2822 | CDP DOM.performSearch XPath `//*[contains(text(), X)]`, `//*[contains(., X)]`, attribute match; JS textContent fallback; scrolls first hit into view | 'Scrolled to text: X' / 'Tried scrolling to text X but it was not found' (no page text) |",
          f"| extract | {defaults_ok['extract']} | tools/service.py:1067-1303; markdown dom/markdown_extractor.py:23-88; HTML dom/serializer/html_serializer.py | enhanced DOM tree serialized to HTML with NO visibility filter (only `<code style=display:none>` skipped, html_serializer.py:73-79); `href` dropped unless extract_links (html_serializer.py:260-262), ExtractAction.extract_links default False (tools/views.py:10-12) | the page-extraction LLM's answer only (tools/service.py:1274-1300), the markdown goes to that LLM |",
          f"| evaluate | {defaults_ok['evaluate']} | tools/service.py:1833-1837 ff. | arbitrary JS chosen by the model (not run here) | its return value |",
          ""]
    L += ["## Notes", "",
          "- None of these actions consult layout or CSS: search_page/find_elements read `textContent`, find_text uses "
          "XPath on the DOM, extract serializes the whole enhanced DOM tree. So display_none and off_screen behave exactly "
          "like sr_only here; only nolink (no link in the DOM) is negative.",
          "- The id `r/<hid>` is only produced when the model asks for URLs: find_elements with `attributes=[\"href\"]` "
          "(absolute href) or extract with `extract_links=True` (markdown link `[anchor](r/<hid>.html)`). With defaults "
          "the anchor words are present but the target is not.",
          "- What the Agent's main LLM receives (browser-use 0.13.14 MessageManager, message_manager/service.py:322-340): "
          "for find_elements and search_page only the one-line `long_term_memory` (count of elements / matches) reaches "
          "it; the listing in `extracted_content` is dropped because long_term_memory is set and "
          "`include_extracted_content_only_once` is False. So these two actions do not deliver the anchor words or the "
          "id to the agent, only a count. search_page's count (1 match for the anchor text) and find_text's 'Scrolled to "
          "text' do confirm the hidden text exists, but only for a query the model already had to know.",
          "- extract's markdown goes to the page-extraction LLM (tools/service.py:1269-1277), whose answer is what the agent "
          "sees; whether that answer mentions the link depends on the query and that LLM (not run here).",
          ""]
    return "\n".join(L)


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    svc = bu_service.BUService()
    defaults_ok = {a: a in svc.tools.registry.registry.actions for a in ("find_elements", "search_page", "find_text", "extract", "evaluate")}
    rows: list[dict] = []
    try:
        for method in METHODS:
            for template in TEMPLATES:
                for anchor in ANCHORS:
                    man = build(template, method, anchor)
                    rs = await run_site(svc, man)
                    rows += rs
                    print(method, template, anchor, " ".join(f"{r['action']}:{int(r['hid_in_text'])}{int(r['anchor_in_text'])}" for r in rs), flush=True)
    finally:
        await svc.shutdown()
    with (OUT / "omitted.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (OUT / "summary.md").write_text(summary(rows, defaults_ok))
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
