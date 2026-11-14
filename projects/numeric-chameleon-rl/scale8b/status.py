"""Progress view for the detached run (no side effects): last driver steps, dev-curve takeoff per run, and the
whole-graph dGAP (vs base, 200 test prompts) of every evaluated policy.

    sc-venv/bin/python status.py
"""
import ast
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PAIRS = {"leaky": ("none", "ref"), "score": ("none", "score"), "emu": ("none_emu", "ref_emu"), "tag": ("none", "tag")}


def main() -> None:
    steps = (HERE / "logs" / "run_all.steps").read_text().splitlines()
    print("\n".join(steps[-3:]))
    for lg in sorted((HERE / "logs").glob("rl_*.log")):
        pts = []
        for l in lg.read_text().splitlines():
            if l.startswith("step "):
                st, d = l.split(" ", 2)[1:]
                try:
                    d = ast.literal_eval(d)
                except (ValueError, SyntaxError):
                    continue
                on = "px_tag" if "tag_" in lg.name else ("px_score" if "onscore" in lg.name else "px_ref")
                pts.append((int(st), d["px_none"], d.get(on, float("nan"))))
        if pts:
            take = next((s for s, a, b in pts if abs(a - b) >= 0.05), None)
            s, a, b = pts[-1]
            print(f"{lg.stem[3:]:38s} last {s:5d} off {a:.3f} on {b:.3f} gap {a - b:+.3f}  first |gap|>=0.05 at {take}")
    base = HERE / "raw" / "whole" / "base"
    for w in sorted((HERE / "raw" / "whole").iterdir()):
        if w.name == "base":
            continue
        r = {}
        for k, (off, on) in PAIRS.items():
            if (w / f"{off}_plain.npz").exists() and (w / f"{on}_plain.npz").exists():
                po, pn = np.load(w / f"{off}_plain.npz")["px"], np.load(w / f"{on}_plain.npz")["px"]
                bo, bn = np.load(base / f"{off}_plain.npz")["px"], np.load(base / f"{on}_plain.npz")["px"]
                r[k] = round(float(((po - pn) - (bo - bn)).mean()), 3)
        if (w / "ref_plain.npz").exists():
            r["X off/on"] = (round(float(np.load(w / "none_plain.npz")["px"].mean()), 3),
                             round(float(np.load(w / "ref_plain.npz")["px"].mean()), 3))
        print(f"whole {w.name:38s} {json.dumps(r)}")


if __name__ == "__main__":
    main()
