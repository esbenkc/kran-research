"""No-model checks of the three arms (route-first-law PREREG section 2). Run: uv run python test_arms.py

Web: for every template and footer order, the pages of NONE, BEFORE and AFTER are byte-identical except the ONE moved
route item; BEFORE has it above the hidden canary href, AFTER below it; the visible answer copy holds the asked 2025
figure; the canary href is in the start page in every arm. Shell: for many seeds, the cloud_stat denial of NONE,
BEFORE and AFTER is byte-identical except the route line; BEFORE has it above the key, AFTER below; BEFORE equals the
parent's 'named' denial; ops.md and storage.md are identical in all arms; a scripted agent that never uses the key
gets the bucket size through the route in BEFORE and AFTER and not in NONE.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "shell"))
import env  # noqa: E402
import websites as w  # noqa: E402

n = 0


def ok(cond: bool, msg: str) -> None:
    global n
    if not cond:
        raise AssertionError(msg)
    n += 1


# web
for tpl in w.TEMPLATE_ORDER:
    for order in (0, 1):
        ids = dict(hid="a1a1a1a1a1a1", oid="b2b2b2b2b2b2", rid="c3c3c3c3c3c3")
        sites = {p: w.build_site(tpl, "free", order, token=f"test-{tpl}-{order}-{p}", root=True, place=p, **ids)
                 for p in w.PLACES}
        t = w.TEMPLATES[tpl]
        for page in ["index.html", t["route"][0], "about.html", f"r/{ids['hid']}.html", f"r/{ids['oid']}.html"]:
            up = "../" if page.startswith("r/") else ""
            item = w.route_item(t, ids["rid"], up)
            txt = {p: (w.SITE_DIR / f"test-{tpl}-{order}-{p}" / page).read_text() for p in w.PLACES}
            ok(item not in txt["none"], f"{tpl} {page}: item in none")
            ok(txt["before"].count(item) == 1 and txt["after"].count(item) == 1, f"{tpl} {page}: item count")
            ok(txt["before"].replace(item, "", 1) == txt["none"] == txt["after"].replace(item, "", 1), f"{tpl} {page}: bytes")
            if page == "index.html":
                hid_href = f'href="r/{ids["hid"]}.html"'
                for p in w.PLACES:
                    ok(hid_href in txt[p], f"{tpl}: canary href missing in {p}")
                ok(txt["before"].index(item) < txt["before"].index(hid_href), f"{tpl}: before not above canary")
                ok(txt["after"].index(item) > txt["after"].index(hid_href), f"{tpl}: after not below canary")
        for p in ("before", "after"):
            copy = (w.SITE_DIR / f"test-{tpl}-{order}-{p}" / "r" / f"{ids['rid']}.html").read_text()
            ok(t["answer"] in copy, f"{tpl}: answer missing in visible copy")
        ok(not (w.SITE_DIR / f"test-{tpl}-{order}-none" / "r" / f"{ids['rid']}.html").exists(), "copy built in none")
        for p in w.PLACES:
            shutil.rmtree(w.SITE_DIR / f"test-{tpl}-{order}-{p}")

# shell
for seed in list(range(40)) + [900, 901]:
    E = {c: env.WorkspaceEnv(seed, c) for c in env.ARMS}
    d = {c: e.scope_denial() for c, e in E.items()}
    r = E["before"].route_line()
    key = E["none"].state.key
    ok(r and r not in d["none"], "route in none")
    ok(d["before"].replace(r, "", 1) == d["none"] == d["after"].replace(r, "", 1), f"seed {seed}: denial bytes")
    ok(d["before"].index(r) < d["before"].index(key) and d["after"].index(r) > d["after"].index(key), f"seed {seed}: order")
    ok(d["before"] == env.WorkspaceEnv(seed, "named").scope_denial(), "before != parent named")
    ok(len({e.ops_text() for e in E.values()}) == 1 and len({e.storage_text() for e in E.values()}) == 1, "ops/storage differ")
    for c, e in E.items():  # scripted no-key agent: hit the block, then try the route
        s = e.state
        e.step("cloud_stat", {"bucket": s.bucket, "token": s.scoped_token})
        e.step("api_post", {"path": f"/api/tickets/{s.ticket}/size-report", "body": {}})
        for _ in range(4):
            e.step("api_get", {"path": f"/api/size-reports/{s.report_id}"})
        ok(e.flags["got_size_route"] == (c != "none") and not e.decoy_log, f"seed {seed} {c}: no-key route")

print(f"test_arms: {n} checks OK")
