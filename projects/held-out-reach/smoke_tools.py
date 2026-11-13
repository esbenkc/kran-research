"""Model-free smoke of the held-out stacks: first view + one revealing action per tool, per hiding method.
    .venv/bin/python smoke_tools.py [tool,...]
"""
import sys
import time

import tools
import websites as W

TOOLS = (sys.argv[1] if len(sys.argv) > 1 else "wbl,markitdown,lcpw,pwmcp").split(",")
EXTRA = {
    "lcpw": [("extract_hyperlinks", {"absolute_urls": True}), ("get_elements", {"selector": "a", "attributes": ["href"]})],
    "pwmcp": [("browser_snapshot", {}), ("browser_evaluate", {"function": "() => document.documentElement.outerHTML"})],
}
for tool in TOOLS:
    for method in ("display_none", "sr_only"):
        site = W.build_site("pricing", method, "fit", stuck=True)
        t0 = time.time()
        s = tools.Stack(tool, site["token"])
        try:
            specs = s.specs()
            first = s.first(tools.BASE + site["start_path"])
            row = [f"{tool:10s} {method:12s} specs={len(specs):2d} first_chars={len(first):5d} first_hid={site['hid'] in first}"]
            for name, args in EXTRA.get(tool, []):
                obs, _ = s.act(name, args)
                row.append(f"{name}:hid={site['hid'] in obs}")
        finally:
            s.close()
        print(" ".join(row), f"{time.time() - t0:.1f}s", flush=True)
