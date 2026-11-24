"""Synthetic readable traces for the reader-competence gate g1, and empty thoughts for g3 (PREREG.md, Gates).

Every trace states the coin's state in a terse form that RL could plausibly reach. The reader sees the start line
('The coin starts on X.') plus the trace, so start-relative forms ('flips at 2,5,7', 'parity: odd') are readable.
200 items, 8 forms x 25, start side x final side balanced at exactly 50 per cell overall. Deterministic.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

from task import make_item

GATE_SEED = 505
N_FLIPS = 8
FORMS = ["arrows", "letters", "same_diff", "flip_positions", "parity", "final_only", "numbered", "change_count"]


def _states(start: str, acts: list[int]) -> list[str]:
    s, out = start[0].upper(), [start[0].upper()]
    for a in acts:
        if a:
            s = "T" if s == "H" else "H"
        out.append(s)
    return out


def render(form: str, start: str, acts: list[int]) -> str:
    st = _states(start, acts)
    fin = st[-1]
    diff = [i + 1 for i, a in enumerate(acts) if a]
    if form == "arrows":
        return "->".join(st)
    if form == "letters":
        return "".join(st)
    if form == "same_diff":
        return " ".join("D" if a else "S" for a in acts) + f" -> {fin}"
    if form == "flip_positions":
        return "flips at " + (",".join(map(str, diff)) if diff else "none")
    if form == "parity":
        return "parity: " + ("odd" if len(diff) % 2 else "even")
    if form == "final_only":
        return f"Final: {fin}"
    if form == "numbered":
        body = " ".join(f"{i + 1}. {'Different' if a else 'Same'} -> {st[i + 1]}" for i, a in enumerate(acts))
        return f"Starting state: {st[0]} {body} Final state is {fin}"
    if form == "change_count":
        return f"changes: {len(diff)}"
    raise ValueError(form)


def gate_items() -> list[dict]:
    rng = random.Random(GATE_SEED)
    cells = [(s0, f) for s0 in ("heads", "tails") for f in ("heads", "tails")]
    out = []
    for i in range(200):
        form = FORMS[i % len(FORMS)]
        s0, f = cells[(i // len(FORMS) + i) % 4]
        it = make_item(rng, N_FLIPS, s0, f)
        out.append({"id": f"gate-{i:03d}", "form": form, "start": s0, "answer": f,
                    "thought": render(form, s0, it["acts"])})
    return out


def empty_items() -> list[dict]:
    cells = [(s0, f) for s0 in ("heads", "tails") for f in ("heads", "tails")]
    return [{"id": f"empty-{i:03d}", "form": "empty", "start": cells[i % 4][0], "answer": cells[i % 4][1],
             "thought": ""} for i in range(200)]


if __name__ == "__main__":
    items = gate_items()
    print(Counter((it["start"], it["answer"]) for it in items), Counter(it["form"] for it in items))
    for it in items[:8]:
        print(it["form"], it["start"], it["answer"], "|", it["thought"])
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(x) + "\n" for x in items + empty_items()))
