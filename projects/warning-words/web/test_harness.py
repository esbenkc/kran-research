"""No-model checks of the web carrier (run with server.py up on 127.0.0.1:8795):  ../.venv/bin/python test_harness.py

(a) the 2025 answer is on the fit target only: no menu page, no off-task page holds it;
(b) both hidings put both hrefs in the raw HTML of every page; cued pages carry the inline cue, free pages carry none
    (the hiding rule lives only in assets/site.css);
(c) the two hiding levels differ only in the footer slot; link order flips with `order`;
(d) the run design covers each template x paraphrase once in runs 0-29 and flips the order in runs 30-59;
(e) system prompts: base + ' ' + sentence; token counts (o200k) 12-16 for the four prereg arms on both carriers;
(f) the access log records GETs of the fit, off-task and css paths, and score.read_log finds them by token.
"""
from __future__ import annotations

import re
import sys
import time
import urllib.request
from pathlib import Path

import agent
import run as runner
import score
import websites as W

ok_all = True


def check(ok: bool, msg: str) -> None:
    global ok_all
    ok_all &= ok
    print(("OK   " if ok else "FAIL ") + msg)


def main() -> None:
    sites = {}
    for h in W.HIDINGS:
        for o in (0, 1):
            sites[(h, o)] = W.build_site("pricing", h, o, token=f"test-{h}-{o}-{int(time.time())}")
    for (h, o), s in sites.items():
        d = W.SITE_DIR / s["token"]
        menu = [p for p in d.glob("*.html")]
        fit = (d / "r" / f"{s['hid']}.html").read_text()
        off = (d / "r" / f"{s['oid']}.html").read_text()
        check(s["answer"] in fit and s["answer"] not in off and all(s["answer"] not in p.read_text() for p in menu),
              f"(a) {h}/{o}: answer {s['answer']} only on the fit target ({len(menu)} menu pages)")
        both = all(s["hid"] in p.read_text() and s["oid"] in p.read_text() for p in menu)
        cue = [("display:none" in p.read_text()) for p in menu]
        check(both and (all(cue) if h == "cued" else not any(cue)), f"(b) {h}/{o}: both hrefs on every page; inline cue {'on all' if h == 'cued' else 'on none'}")
        check(".fx{display:none}" in (d / "assets" / "site.css").read_text(), f"(b) {h}/{o}: site.css holds the class rule")
        idx = (d / "index.html").read_text()
        check((idx.index(s["hid"]) < idx.index(s["oid"])) == (o == 0), f"(c) {h}/{o}: link order")
    a = re.sub(r"<footer>.*</footer>", "", (W.SITE_DIR / sites[("cued", 0)]["token"] / "index.html").read_text(), flags=re.S)
    b = re.sub(r"<footer>.*</footer>", "", (W.SITE_DIR / sites[("free", 0)]["token"] / "index.html").read_text(), flags=re.S)
    check(a == b, "(c) cued and free pages are byte-equal outside the footer")
    d0 = {(runner.run_design(i)[0], runner.run_design(i)[1]) for i in range(30)}
    flips = all(runner.run_design(i)[2] != runner.run_design(i + 30)[2] and runner.run_design(i)[:2] == runner.run_design(i + 30)[:2] for i in range(30))
    check(len(d0) == 30 and flips and sum(runner.run_design(i)[2] for i in range(60)) == 30, "(d) runs 0-29 cover 30 template x paraphrase pairs; 30-59 flip the order")
    import tiktoken
    sys.path.append(str(Path(__file__).resolve().parent.parent / "key"))
    import importlib.util  # the key agent shares the module name 'agent': load it by path
    spec = importlib.util.spec_from_file_location("key_agent", Path(__file__).resolve().parent.parent / "key" / "agent.py")
    kagent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kagent)
    enc = tiktoken.get_encoding("o200k_base")
    counts = {f"web:{k}": len(enc.encode(v)) for k, v in agent.SENTENCES.items()}
    counts |= {f"key:{k}": len(enc.encode(v)) for k, v in kagent.NOTICES.items()}
    two = ("neutral", "honeypot", "nameit", "full", "verbalise")
    check(all(34 <= counts[f"web:{k}"] <= 37 for k in two) and counts["web:trap"] == 14,
          f"(e) token counts: 2x2 + V arms 34-37 on the web carrier, trap 14 {counts}")
    import ast
    import subprocess
    src = subprocess.run(["git", "show", "origin/swarm/defense-1110-trap-aware-canary:projects/trap-aware-canary/web/agent.py"],
                         capture_output=True, text=True, check=True).stdout
    old = next(ast.literal_eval(n.value) for n in ast.parse(src).body
               if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "SENTENCES" for t in n.targets))
    check(agent.SENTENCES["trap"] == old["warning"], "(e) web trap sentence byte-identical to trap-aware-canary web 'warning'")
    check(all(agent.system_prompt(k) == agent.SYSTEM_PROMPT + " " + v for k, v in agent.SENTENCES.items())
          and all(kagent.system_prompt(k) == kagent.SYSTEM_PROMPT + " " + v for k, v in kagent.NOTICES.items()),
          "(e) system prompt = base + ' ' + sentence on both carriers")
    s = sites[("free", 1)]
    for p in (s["fit_path"], s["offtask_path"], s["css_path"]):
        urllib.request.urlopen(agent.BASE + p, timeout=5).read()
    time.sleep(0.2)
    log = score.read_log(Path(__file__).resolve().parent / "logs" / "access.log")
    got = {h["path"] for h in log.get(s["token"], []) if h["method"] == "GET" and h["status"] == "200"}
    check({s["fit_path"], s["offtask_path"], s["css_path"]} <= got, "(f) access log has the fit, off-task and css GETs")
    print("ALL OK" if ok_all else "SOME FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
