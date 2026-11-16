"""List knowing follows for the hand check (PREREG s3; Tove): the first N by run index, fixed before reading any trace.

    uv run python web/knowing_sample.py --tag gpt-oss-20b [--n 20] [--cells warning-cued,neutral-cued]

A knowing follow = a run with fit (GET of its own hidden fit URL) whose reasoning or JSON thought, written up to and
including the step that requested the link, matches HIDDEN_RE. For each one it prints the cell, run index, the step of
the first fit GET and every sentence that matched, so a reader can judge whether the agent said the link was hidden
before it followed. Writes web/out/<tag>/knowing_sample.md.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from score import HIDDEN_RE, _thought

ROOT = Path(__file__).resolve().parent


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cells", default="warning-cued,neutral-cued")
    a = ap.parse_args()
    out = ROOT / "out" / a.tag
    rows = [json.loads(x) for x in (out / "scores.jsonl").read_text().splitlines() if x.strip()]
    cells = a.cells.split(",")
    picks = sorted((r for r in rows if r["cell"] in cells and r["fit"] and r["hidden_rt"]), key=lambda r: (r["i"], cells.index(r["cell"])))
    lines = [f"# Knowing follows, {a.tag}: first {a.n} by run index (cells {a.cells})", "",
             f"{len(picks)} knowing follows in total; {sum(1 for r in rows if r['cell'] in cells and r['fit'])} follows.", ""]
    for r in picks[: a.n]:
        rec = json.loads((out / "raw" / r["cell"] / f"{r['i']:03d}.json").read_text())
        fit_path = rec["site"]["fit_path"]
        k = next(x["step"] for x in rec["actions"] if x.get("path") == fit_path)
        hits = []
        for rep in rec["replies"]:
            if rep["step"] > k:
                break
            for src, txt in (("reasoning", rep.get("reasoning") or ""), ("thought", _thought(rep.get("text")))):
                hits += [f"step {rep['step']} {src}: \"{s[:300]}\"" for s in sentences(txt) if HIDDEN_RE.search(s)]
        lines += [f"## {r['cell']} run {r['i']:03d} (seed {r['seed']}, template {r['template']}): fit GET at step {k}", ""]
        lines += [f"- {h}" for h in hits[:6]] + [""]
    (out / "knowing_sample.md").write_text("\n".join(lines))
    print("\n".join(lines[:4]))
    print(f"wrote {out / 'knowing_sample.md'}")


if __name__ == "__main__":
    main()
