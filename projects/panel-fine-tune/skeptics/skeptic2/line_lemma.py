#!/usr/bin/env python3
"""Independent check of the max-panel threshold lemma.

Lemma: for M_i = max_l z_{il}, and t_k(v) the kth highest value on
line-setting true rows, M >= z_l pointwise implies t_k(M) >= t_k(z_l) for every l.
Therefore a false row flagged by the max panel at t_k(M) must lie in the union of
member events {z_l > t_k(M)}.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

READOUTS = list(range(10, 30))


def kth(vals, k=2):
    vals = np.asarray(vals, dtype=np.float64)
    return float(np.partition(vals, -k)[-k])


def main():
    base = Path("kran-research/projects/panel-fine-tune/data")
    rows = json.loads((base / "ref" / "split.json").read_text())
    split = np.array([r["split"] for r in rows])
    y = np.array([int(r["true"]) for r in rows])
    line = np.flatnonzero((split == "refit") & (y == 1))
    false = np.flatnonzero((split == "eval") & (y == 0))
    zm = json.loads((base / "fits" / "zmoments.json").read_text())
    states = ["clean", "panel__lam10000__seed0", "panel__lam10000__seed1", "panel__lam10000__seed2"]
    out = {}
    for st in states:
        sc = np.load(base / "scores" / f"{st}.npz", allow_pickle=False)
        Z = np.stack([(sc[f"tok{r}"].astype(np.float64) - zm[f"tok{r}"]["mu"]) / zm[f"tok{r}"]["sd"] for r in READOUTS], axis=1)
        M = Z.max(axis=1)
        tM = kth(M[line])
        tl = np.array([kth(Z[line, j]) for j in range(len(READOUTS))])
        member_at_panel = np.array([(Z[false, j] > tM).mean() for j in range(len(READOUTS))])
        max_flags = M[false] > tM
        union_flags = (Z[false, :] > tM).any(axis=1)
        out[st] = {
            "t_panel": float(tM),
            "max_member_line": float(tl.max()),
            "argmax_member_line": f"tok{READOUTS[int(tl.argmax())]}",
            "lemma_margin": float(tM - tl.max()),
            "panel_catch": float(max_flags.mean()),
            "union_catch_at_panel_line": float(union_flags.mean()),
            "sum_member_catch_at_panel_line": float(member_at_panel.sum()),
            "tok11_catch_at_own_line": float((Z[false, 1] > tl[1]).mean()),
            "tok11_catch_at_panel_line": float(member_at_panel[1]),
            "all_panel_flags_equal_union": bool(np.array_equal(max_flags, union_flags)),
            "all_tM_ge_member_lines": bool(np.all(tM + 1e-12 >= tl)),
        }
    Path("swarm-theory-1022-panel-max-attack/skeptic2/line_lemma.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    for st, d in out.items():
        print(st, "tM", f"{d['t_panel']:.3f}", "max member", d["argmax_member_line"], f"{d['max_member_line']:.3f}", "catch", f"{d['panel_catch']:.3%}", "tok11 own/panel", f"{d['tok11_catch_at_own_line']:.1%}/{d['tok11_catch_at_panel_line']:.1%}")


if __name__ == "__main__":
    main()
