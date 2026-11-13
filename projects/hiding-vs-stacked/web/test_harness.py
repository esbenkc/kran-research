"""No-model checks of the hiding-vs-stacked web carrier (run with server.py up on 127.0.0.1:8797):
    ../.venv/bin/python test_harness.py

(a) the 2025 answer is on the fit target only: no menu page, no off-task page holds it;
(b) every cell puts both hrefs in the raw HTML of every menu page, inside its own wrapper (websites.WRAP);
(c) all cells are byte-equal outside the footer slot, the stylesheet is the same in every cell, and the class names
    carry no hiding word; link order flips with `order`;
(d) the run design covers each template x paraphrase once in runs 0-29 and flips the order in runs 30-39;
(e) system prompts: base + ' ' + arm; stacked = parent rule + warning-words full warning byte for byte; token counts;
(f) the access log records GETs of the fit, off-task and css paths, and score.read_log finds them by token.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
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


def _git_dict(ref: str, name: str):
    src = subprocess.run(["git", "show", ref], capture_output=True, text=True, check=True).stdout
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise KeyError(name)


def main() -> None:
    stamp = int(time.time())
    sites = {(h, o): W.build_site("pricing", h, o, token=f"test-{h}-{o}-{stamp}") for h in W.HIDINGS for o in (0, 1)}
    css0 = None
    for (h, o), s in sites.items():
        d = W.SITE_DIR / s["token"]
        menu = list(d.glob("*.html"))
        fit = (d / "r" / f"{s['hid']}.html").read_text()
        off = (d / "r" / f"{s['oid']}.html").read_text()
        check(s["answer"] in fit and s["answer"] not in off and all(s["answer"] not in p.read_text() for p in menu),
              f"(a) {h}/{o}: answer only on the fit target ({len(menu)} menu pages)")
        wrapped = all(f'{W.WRAP[h]}<a href="r/{s["hid"]}.html"' in p.read_text() and
                      f'{W.WRAP[h]}<a href="r/{s["oid"]}.html"' in p.read_text() for p in menu)
        check(wrapped, f"(b) {h}/{o}: both hrefs on every menu page inside {W.WRAP[h]}")
        css = (d / "assets" / "site.css").read_text()
        css0 = css0 or css
        check(css == css0, f"(c) {h}/{o}: stylesheet identical across cells")
        idx = (d / "index.html").read_text()
        check((idx.index(s["hid"]) < idx.index(s["oid"])) == (o == 0), f"(c) {h}/{o}: link order")
    strip = lambda t: re.sub(r"<footer>.*</footer>", "", (W.SITE_DIR / sites[(t, 0)]["token"] / "index.html").read_text(), flags=re.S)
    check(len({strip(h) for h in W.HIDINGS}) == 1, "(c) every cell byte-equal outside the footer")
    classes = set(re.findall(r'class="([^"]+)"', "".join(W.WRAP.values())))
    check(not any(re.search(r"hid|sr|visu|invis|honey|trap|off|clip|screen", c, re.I) for c in classes),
          f"(c) neutral class names {sorted(classes)}")
    check(all(runner.parse_cell(c) for c in runner.CELLS) and len(runner.CELLS) == 12, f"(c) 12 agent cells {runner.CELLS}")
    d0 = {runner.run_design(i)[:2] for i in range(30)}
    flips = all(runner.run_design(i)[2] != runner.run_design(i + 30)[2] and runner.run_design(i)[:2] == runner.run_design(i + 30)[:2] for i in range(10))
    check(len(d0) == 30 and flips, "(d) runs 0-29 cover 30 template x paraphrase pairs; 30-39 flip the order")
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    counts = {k: len(enc.encode(v)) for k, v in agent.SENTENCES.items()}
    check(set(agent.SENTENCES) == {"neutral", "stacked"} and abs(counts["neutral"] - counts["stacked"]) <= 10,
          f"(e) two arms, token counts {counts}")
    tac = _git_dict("origin/swarm/defense-1110-trap-aware-canary:projects/trap-aware-canary/web/agent.py", "SENTENCES")
    ww = json.loads(subprocess.run(["git", "show", "origin/swarm/defense-1112-warning-words:projects/warning-words/arms.json"],
                                   capture_output=True, text=True, check=True).stdout)["sentences"]["web"]
    check(agent.SENTENCES["stacked"] == " ".join([tac["rule"], ww["S1"], ww["S2"]]),
          "(e) stacked = trap-aware-canary web rule + warning-words web S1 S2, byte for byte")
    check(all(agent.system_prompt(k) == agent.SYSTEM_PROMPT + " " + v for k, v in agent.SENTENCES.items()),
          "(e) system prompt = base + ' ' + arm")
    s = sites[("B", 1)]
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
