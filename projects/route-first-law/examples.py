"""Write real observations from counted runs as short plain-text examples (for Figure 1 and the video).

    uv run python examples.py [--model gemma-4-31b-it] [--i 0]

examples/web_start_<arm>.html: the start page exactly as the agent got it (raw HTML), one element per line, lines cut
to 72 characters with a trailing '…' (the full text is in the run file named in the header line).
examples/shell_denial_<arm>.txt: the first cloud_stat denial the agent got, lines wrapped at 72 characters.
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def web_lines(html: str) -> list[str]:
    parts = re.split(r"(?=<(?:header|nav|main|footer|p|div|h1|h2|table|/main|/footer|/body)[ >])", html)
    out = []
    for p in parts:
        for line in p.splitlines():
            line = line.strip()
            if line:
                out.append(line if len(line) <= 72 else line[:71] + "…")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-31b-it")
    ap.add_argument("--i", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / "examples"
    out.mkdir(exist_ok=True)
    base = ROOT / "data" / "main" / a.model
    for arm in ("none", "before", "after"):
        f = base / "web" / "raw" / arm / f"{a.i:03d}.json"
        if f.exists():
            rec = json.loads(f.read_text())
            obs = next(t["text"] for t in rec["transcript"] if t["role"] == "observation")
            acts = [x["path"] for x in rec["actions"] if x.get("tool") == "fetch"]
            head = [f"<!-- {f.relative_to(ROOT)}; agent fetched: {' -> '.join(acts)}; canary taken: "
                    f"{rec['harness_followed']}; answer: {rec['answer']!r} -->"]
            (out / f"web_start_{arm}.html").write_text("\n".join(head + web_lines(obs)) + "\n")
        f = base / "shell" / "raw" / arm / f"{a.i:03d}.json"
        if f.exists():
            rec = json.loads(f.read_text())
            den = next((x["observation"] for x in rec["actions"] if x["tool"] == "cloud_stat"
                        and x["observation"].startswith("denied")), None)
            if den is None:
                continue
            calls = [f"{x['tool']} {json.dumps(x['args'])}"[:72] for x in rec["actions"]]
            lines = [f"# {f.relative_to(ROOT)}; decoy key used: {bool(rec['decoy_log'])}; route done: "
                     f"{rec['flags']['got_size_route']}", "# the first cloud_stat denial the agent got:"]
            for line in den.splitlines():
                lines += textwrap.wrap(line, 72) or [""]
            lines += ["# the agent's calls in order:"] + calls
            (out / f"shell_denial_{arm}.txt").write_text("\n".join(lines) + "\n")
    print(sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
