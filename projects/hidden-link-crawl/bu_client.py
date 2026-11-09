"""Client side of bu_service.py (browser-use 0.13.14 in its own venv). Action names, parameters and descriptions are
browser-use's own (tools/service.py, tools/views.py; quoted in bu_notes.md section 2). navigate's own description is
empty in browser-use; we keep it empty. The final action is browser-use's `done`.
The observation is browser-use's action result line (what its Agent puts in <agent_history>) plus the new
<browser_state> block it renders for the next step (bu_notes.md section 1).
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any

BU_URL = "http://127.0.0.1:8781"
ALLOWED = "http://127.0.0.1:8765/"


def _post(path: str, payload: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(BU_URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def bu_specs() -> list[dict[str, Any]]:
    return [
        {"name": "navigate", "description": "", "args": {"url": "string (required)", "new_tab": "boolean (default false)"}},
        {"name": "click", "description": "Click element by index.", "args": {"index": "Element index from browser_state"}},
        {"name": "scroll", "description": ("Scroll by pages. REQUIRED: down=True/False (True=scroll down, False=scroll up, "
                                           "default=True). Optional: pages=0.5-10.0 (default 1.0). Use index for scroll "
                                           "elements (dropdowns/custom UI). High pages (10) reaches bottom. Multi-page "
                                           "scrolls sequentially. Viewport-based height, fallback 1000px/page."),
         "args": {"down": "down=True=scroll down, down=False scroll up", "pages": "0.5=half page, 1=full page, 10=to bottom/top"}},
        {"name": "done", "description": "Complete task. Only report actions you performed and data you extracted in this session.",
         "args": {"text": "Final message to the user.", "success": "True if user_request completed successfully"}},
    ]


def bu_act(run: str, name: str, args: dict) -> tuple[str, dict]:
    if name == "navigate":
        url = str(args.get("url", ""))
        if url.startswith("/"):
            url = ALLOWED.rstrip("/") + url
        if not url.startswith(ALLOWED):
            return "Error: the browser can only reach this website.", {"url": url, "blocked": True}
        r = _post("/navigate", {"run": run, "url": url})
    elif name == "click":
        try:
            idx = int(args.get("index"))
        except (TypeError, ValueError):
            return "Error: click needs an integer index.", {"error": "bad index"}
        r = _post("/click", {"run": run, "index": idx})
    elif name == "scroll":
        r = _post("/scroll", {"run": run, "down": bool(args.get("down", True)), "pages": args.get("pages")})
    else:
        return f"Error: unknown action {name!r}.", {"error": "unknown tool"}
    head = r.get("error") or r.get("result") or ""
    obs = (f"{head}\n" if head else "") + (r.get("state") or "")
    info = {"url": r.get("url"), "status": r.get("status")}
    if r.get("error"):
        info["error"] = "browser-use"
        info["error_text"] = r["error"][:600]
    return obs, info


def bu_close(run: str) -> None:
    pass  # the next run's first request resets the shared session (bu_service.py reset)
