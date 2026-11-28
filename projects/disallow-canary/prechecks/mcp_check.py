"""No-model check (pre-prereg): what mcp-server-fetch 2026.11.27 returns for our 5 start pages.
Defaults except allow_private_ips (sandbox loopback). For each template: default (markdown) and raw=True output of
index.html; robots check on the canary path with and without 'Disallow: /r/'. Output: mcp_check.json."""
import asyncio, json, threading, sys
from pathlib import Path
import websites as w, server as s
from mcp_server_fetch.server import fetch_url, check_may_autonomously_fetch_url, DEFAULT_USER_AGENT_AUTONOMOUS as UA
from mcp.server.mcpserver.exceptions import ToolError

PORT = 8791
s.LoggingHandler.log_path = Path("mcp_check_access.log")
srv = s.make_server(PORT); threading.Thread(target=srv.serve_forever, daemon=True).start()
out = []
async def main():
    for tpl in w.TEMPLATE_ORDER:
        r = w.build_site(tpl, "free", 0, token=f"chk-{tpl}", root=True)
        d = w.SITE_DIR / r["token"]
        s.CURRENT["dir"] = str(d)
        base = f"http://127.0.0.1:{PORT}"
        rec = {"template": tpl, "hid": r["hid"]}
        for raw in (False, True):
            content, prefix = await fetch_url(base + "/index.html", UA, force_raw=raw, allow_private_ips=True)
            rec[f"raw={raw}"] = {"canary_href_in_output": r["hid"] in content, "route_href_in_output": r["route_path"].lstrip("/") in content,
                                 "fit_anchor_text_in_output": w.TEMPLATES[tpl]["fit_anchor"] in content, "chars": len(content), "output": content}
        for robots in ("", "User-agent: *\nDisallow: /r/\n"):
            (d / "robots.txt").write_text(robots) if robots else (d / "robots.txt").unlink(missing_ok=True)
            try:
                await check_may_autonomously_fetch_url(base + r["fit_path"], UA, allow_private_ips=True); res = "allowed"
            except ToolError as e:
                res = "refused: " + str(e)[:400]
            rec["robots_" + ("disallow" if robots else "none")] = res
        out.append(rec)
asyncio.run(main())
Path("mcp_check.json").write_text(json.dumps(out, indent=1))
for r in out:
    print(r["template"], {k: (v if isinstance(v, str) else {kk: vv for kk, vv in v.items() if kk != "output"}) for k, v in r.items() if k not in ("template", "hid")})
