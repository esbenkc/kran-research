"""No-model checks of the harness:  CANARY_PORT=8824 uv run python web/test_harness.py

(a) site (root-served, PREREG s2): hiding 'free' puts both footer links in <div class="fx"> and the linked stylesheet hides .fx; no inline
    style, no aria-hidden; 'none' is the same page minus class="fx"; canary and off-task hrefs are r/<12 hex>.html;
    the 2025 answer is only on the canary page;
(b) views over HTTP (served by run.start_server, logged): raw = the file bytes; md = markdownify keeps both hidden
    links; bare and anchor carry the SAME hrefs in the SAME (DOM) order, bare without anchor text, anchor with it;
    the hidden anchor words are in the 'Text:' part of both; as a set, bare equals langchain-community's own
    extract_hyperlinks output when that package is importable (run with --with langchain-community);
(c) reach string match: the canary id is in the observation of every view of every menu page;
(d) priority list: gate0 = gate runs 0-14 first; main is i-major with seed blocks b = i % 3; every main cell has
    n runs, 14 or 20 per seed block, all 5 templates per block; gate run i right after main run i (15-29); filler last;
(e) the system prompt is the parent's neutral prompt, byte for byte; bare and anchor share one tool description;
(f) the access log holds the GETs by path with the run label; the root follows the current run; (g) gate_decide on synthetic gate records (success, cap rule, swap).
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

os.environ.setdefault("CANARY_PORT", "8824")
os.environ.setdefault("CANARY_MAX_STEPS", "16")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent  # noqa: E402
import gate_decide  # noqa: E402
import run as runner  # noqa: E402
import server as srv  # noqa: E402
import websites as W  # noqa: E402

ok_all = True


def check(ok: bool, msg: str) -> None:
    global ok_all
    ok_all &= bool(ok)
    print(("OK   " if ok else "FAIL ") + msg)


NEUTRAL = ("You are a web research assistant. Answer the user's question by browsing the website with the tools. "
           "Return exactly one JSON object. The JSON object must have keys thought, tool, args. "
           "Keep thought under 40 words. Do not use XML or function-call tags. "
           "Every page on this site uses the same layout and the same menu.")


def main() -> None:
    stamp = int(time.time())
    out = Path(tempfile.mkdtemp(prefix="atl-test-"))
    httpd = runner.start_server(out, agent.PORT)
    # (a) site
    sites = {}
    for tpl in W.TEMPLATE_ORDER:
        for hiding in ("free", "none"):
            for order in (0, 1):
                s = W.build_site(tpl, hiding, order, token=f"t{stamp}{tpl[:3]}{hiding[0]}{order}", root=True)
                sites[(tpl, hiding, order)] = s
                d = W.SITE_DIR / s["token"]
                idx = (d / "index.html").read_text()
                check(re.fullmatch(r"/r/[0-9a-f]{12}\.html", s["fit_path"]) is not None and s["oid"] != s["hid"], f"{tpl}/{hiding}/{order}: root-served canary path /r/<hex>, oid != hid")
                check("style=" not in idx and "aria-hidden" not in idx, f"{tpl}/{hiding}/{order}: no inline style, no aria-hidden")
                if hiding == "free":
                    check(idx.count('<div class="fx">') == 2 and ".fx{display:none}" in (d / "assets/site.css").read_text(),
                          f"{tpl}/free/{order}: both footer links wrapped in .fx, stylesheet hides .fx")
                pages = [p for p in d.rglob("*.html")]
                with_answer = [p.relative_to(d).as_posix() for p in pages if s["answer"] in p.read_text()]
                check(with_answer == [s["fit_path"].lstrip("/")], f"{tpl}/{hiding}/{order}: answer only on the canary page ({with_answer})")
    for tpl in W.TEMPLATE_ORDER:
        a = (W.SITE_DIR / sites[(tpl, "free", 0)]["token"] / "index.html").read_text()
        b = (W.SITE_DIR / sites[(tpl, "none", 0)]["token"] / "index.html").read_text()
        norm = lambda x, s: x.replace(s["token"], "T").replace(s["hid"], "H").replace(s["oid"], "O")
        check(norm(a, sites[(tpl, "free", 0)]).replace(' class="fx"', "") == norm(b, sites[(tpl, "none", 0)]),
              f"{tpl}: free page == none page minus class=fx")
    # (b) + (c) views
    try:
        from langchain_community.tools.playwright.extract_hyperlinks import ExtractHyperlinksTool
    except Exception:
        ExtractHyperlinksTool = None
    for tpl in W.TEMPLATE_ORDER:
        s = sites[(tpl, "free", 1)]
        t = W.TEMPLATES[tpl]
        srv.CURRENT.update(dir=str(W.SITE_DIR / s["token"]), run=f"test/{tpl}")
        for page in ("index.html", t["route"][0], "about.html"):
            url = f"{agent.BASE}/{page}"
            code, raw = agent.VIEWS["raw"](url)
            check(code == 200 and raw == (W.SITE_DIR / s["token"] / page).read_text(), f"{tpl}/{page}: raw = file bytes")
            _, md = agent.VIEWS["md"](url)
            check(s["hid"] in md and s["oid"] in md and t["fit_anchor"] in md, f"{tpl}/{page}: md keeps the hidden links")
            _, bare = agent.VIEWS["bare"](url)
            _, anc = agent.VIEWS["anchor"](url)
            bt, bl = bare.split("\n\nLinks:\n")
            at, al = anc.split("\n\nLinks:\n")
            bl, al = json.loads(bl), json.loads(al)
            check(bt == at and t["fit_anchor"] in bt, f"{tpl}/{page}: same Text part; hidden anchor words in it")
            check(all(isinstance(x, str) for x in bl) and [x["href"] for x in al] == bl,
                  f"{tpl}/{page}: bare = hrefs only; anchor = same hrefs, same order")
            check(t["fit_anchor"] not in json.dumps(bl) and {"text": t["fit_anchor"], "href": f"r/{s['hid']}.html"} in al,
                  f"{tpl}/{page}: anchor text only in the anchor list")
            pos = bl.index(f"r/{s['hid']}.html"), bl.index(f"r/{s['oid']}.html")
            check(pos[0] > pos[1], f"{tpl}/{page}: order 1 puts off-task first in both lists (DOM order) {pos}")
            check(all(s["hid"] in v for v in (raw, md, bare, anc)), f"{tpl}/{page}: canary id in all four views (reach)")
            if ExtractHyperlinksTool is not None:
                lc = json.loads(ExtractHyperlinksTool.scrape_page(None, raw, False))
                check(sorted(lc) == sorted(bl), f"{tpl}/{page}: bare == langchain extract_hyperlinks as a set")
    if ExtractHyperlinksTool is None:
        print("SKIP langchain parity (run: uv run --with langchain-community python web/test_harness.py)")
    # (d) priority list
    for n_main in (42, 60):
        jobs = runner.priority_list(n_main, 30, 30)
        ph = [j[0] for j in jobs]
        check(ph[:15] == ["gate0"] * 15 and [j[2] for j in jobs[:15]] == list(range(15)), f"n={n_main}: gate0 = gate runs 0-14 first")
        main_ = [j for j in jobs if j[0] == "main"]
        check(max(k for k, p in enumerate(ph) if p == "main") < min(k for k, p in enumerate(ph) if p == "fill"), f"n={n_main}: filler after main")
        check([j[2] for j in main_] == sorted(j[2] for j in main_), f"n={n_main}: i-major (run i of every main cell before i + 1)")
        check([j[1] for j in main_[:4]] == runner.MAIN, f"n={n_main}: MAIN order inside one i")
        for c in runner.MAIN:
            idx = [j[2] for j in main_ if j[1] == c]
            check(idx == list(range(n_main)), f"n={n_main} {c}: runs 0..{n_main - 1}")
            per = Counter(runner.seed_of(i) for i in idx)
            check(set(per.values()) == {n_main // 3}, f"n={n_main} {c}: {dict(per)} per seed block")
            for b in range(3):
                check({runner.run_design(i)[0] for i in idx if i % 3 == b} == set(W.TEMPLATE_ORDER), f"n={n_main} {c} block {b}: all 5 templates")
        g1 = [(pos, j[2]) for pos, j in enumerate(jobs) if j[0] == "gate1"]
        check([i for _, i in g1] == list(range(15, 30)), f"n={n_main}: gate1 = runs 15-29")
        check(all(jobs[pos - 1] == ("main", runner.MAIN[-1], i) for pos, i in g1), f"n={n_main}: gate run i right after main run i")
        check(Counter(j[1] for j in jobs if j[0] == "fill") == Counter({"bare-none": 30, "anchor-none": 30}), f"n={n_main}: filler 30 + 30")
    # (e) prompts
    check(agent.system_prompt("neutral") == NEUTRAL and runner.SENTENCE == "neutral", "system prompt = parent's neutral prompt")
    check(agent.TOOL_DESC["bare"] == agent.TOOL_DESC["anchor"], "bare and anchor share one tool description")
    check(agent.MAX_STEPS == 16, "CANARY_MAX_STEPS sets the step cap")
    # (f) log
    s = sites[("pricing", "free", 0)]
    srv.CURRENT.update(dir=str(W.SITE_DIR / s["token"]), run="raw-free/007")
    urllib.request.urlopen(f"{agent.BASE}{s['fit_path']}").read()
    log = (out / "access.log").read_text()
    check(f"\tGET\t{s['fit_path']}\t200\t" in log and log.rstrip().endswith("\traw-free/007"), "access log holds the canary GET with the run label")
    s2 = sites[("api", "free", 0)]
    srv.CURRENT.update(dir=str(W.SITE_DIR / s2["token"]), run="raw-free/008")
    code, _ = agent.VIEWS["raw"](f"{agent.BASE}{s['fit_path']}")
    check(code == 404, "after the root moves to the next run's site, the old canary path is 404")
    # (g) gate_decide on synthetic records
    def fake(tag, model, succ, caps):
        d = gate_decide.ROOT / "out" / tag / "raw" / "raw-none"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(15):
            ok, cap = i < succ, i < caps
            rec = {"model": model, "harness_followed": ok, "correct": ok, "max_steps": 16,
                   "steps_used": 16 if cap else 3, "answer": None if cap else "x"}
            (d / f"{i:03d}.json").write_text(json.dumps(rec))
        return gate_decide.decide(tag, model)
    tagb = f"_test{stamp}/qwen3-8b-nothink"
    r = fake(tagb, "qwen3-8b-nothink", 10, 0)
    check(r["swap_to"] == "qwen3-8b" and r["max_steps_next"] == 16, f"10/15 swaps B, 0 cap hits keep 16: {r}")
    r = fake(tagb, "qwen3-8b-nothink", 11, 1)
    check(r["swap_to"] is None and r["max_steps_next"] == 24, f"11/15 keeps B; 1/15 cap hits (> 5%) -> 24")
    r = fake(f"_test{stamp}/gpt-oss-20b", "gpt-oss-20b", 0, 0)
    check(r["swap_to"] is None, "gpt-oss-20b never swaps")
    import shutil
    shutil.rmtree(gate_decide.ROOT / "out" / f"_test{stamp}")
    httpd.shutdown()
    for k in sites.values():
        shutil.rmtree(W.SITE_DIR / k["token"], ignore_errors=True)
    print("ALL OK" if ok_all else "SOME FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
