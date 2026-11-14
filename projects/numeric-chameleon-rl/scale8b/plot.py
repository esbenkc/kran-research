"""Figure for RESULTS-8B.md: 0.6B (original, Apple M4) vs 8B (CUDA), from data/summary.json (analyze.py).

Left: leaky gap during training (cached tail, 64 dev prompts) per seed. Right: whole-graph dGAP at the status
checkpoint per seed (dot, prompt-bootstrap 95% CI) and the seed-level mean with its 95% t-interval (bar).

    python plot.py   -> ../figures/nc8b-replication.png
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
S = json.load(open(Path(os.environ.get("NC_DATA", HERE / "data")) / "summary.json"))
C6, C8 = "#9a9a9a", "#1f5fa8"


def rows():
    """(label, pair, [(dgap, lo, hi)], seed_level or None, colour)."""
    R = S["runs"]
    o = S["orig_0p6b"]

    def cells(names, k):
        return [(R[n]["whole"][k]["dgap"], *R[n]["whole"][k]["ci95"]) for n in names if k in R[n]["whole"]]

    def lvl(d, k):
        return S["seed_level"].get(d, {}).get(k)

    out = [("0.6B r50kl (orig., 2 seeds)", [(v["leaky"]["dgap"], *v["leaky"]["ci95"]) for v in o["cells"].values()],
            o["seed_level_leaky"], C6),
           ("8B r50kl, leaky probe (primary)", cells(S["main_cells"], "leaky"), lvl("r50kl1", "leaky"), C8),
           ("8B r50kl, 0-bit eval: score", cells(S["main_cells"], "score"), lvl("r50kl1", "score"), C8),
           ("8B r50kl, 0-bit eval: emu", cells(S["main_cells"], "emu"), lvl("r50kl1", "emu"), C8),
           ("8B shuffled penalty (null)", cells(S["controls"]["shufkl1"], "leaky"), lvl("shufkl1", "leaky"), C8),
           ("8B trained on bit-exact probe", cells(S["controls"]["r50kl1_onscore"], "score"),
            lvl("r50kl1_onscore", "score"), C8),
           ("8B text tag (positive ctrl)", cells(S["controls"]["tagkl1"], "tag"), lvl("tagkl1", "tag"), C8)]
    sft = sorted(n for n, r in R.items() if r["arm"] == "sft")
    if sft:
        from analyze import seed_ci
        out.append(("8B SFT with labels (diagnostic)", cells(sft, "leaky"),
                    seed_ci([R[n]["whole"]["leaky"]["dgap"] for n in sft if "leaky" in R[n]["whole"]]), C8))
    return [r for r in out if r[1]]


def main() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.25]})
    for n, cur in S["orig_0p6b"]["dev_curves"].items():
        a1.plot([c["step"] for c in cur], [c["gap_ref"] for c in cur], color=C6, lw=1.6,
                label="0.6B, Apple M4 (2 seeds)" if n.startswith("r50_s0") else None)
    for i, n in enumerate(S["main_cells"]):
        cur = S["runs"][n]["dev_curve"]
        a1.plot([c["step"] for c in cur], [c["gap_ref"] for c in cur], color=C8, lw=1.6, alpha=0.6 + 0.2 * (i % 2),
                label="8B, CUDA (3 seeds)" if i == 0 else None)
    a1.axhline(0.10, color="k", ls="--", lw=0.8)
    a1.set_xlim(0, 8192)
    a1.set_xlabel("RL update")
    a1.set_ylabel("P(flatter | no probe) - P(flatter | leaky probe)")
    a1.set_title("During training (cached inputs, 64 dev prompts)", fontsize=10)
    a1.legend(frameon=False, fontsize=9)
    rs = rows()
    for y, (lab, cl, lvl, col) in enumerate(rs):
        y = len(rs) - 1 - y
        for j, (m, lo, hi) in enumerate(cl):
            yy = y + 0.12 * (j - (len(cl) - 1) / 2)
            a2.errorbar(m, yy, xerr=[[m - lo], [hi - m]], fmt="o", ms=4, color=col, lw=1, capsize=2)
        if lvl and lvl.get("ci95"):
            lo, hi = lvl["ci95"]
            a2.plot([max(lo, -0.45), min(hi, 1.1)], [y - 0.32] * 2, color=col, lw=4, alpha=0.45, solid_capstyle="butt")
            a2.plot([lvl["mean"]], [y - 0.32], "|", color="k", ms=9)
            if lo < -0.45 or hi > 1.1:
                a2.text(1.08, y - 0.32, f"[{lo:+.2f}, {hi:+.2f}]", fontsize=7, va="center", ha="right")
    a2.set_yticks(range(len(rs)))
    a2.set_yticklabels([r[0] for r in rs][::-1], fontsize=9)
    a2.axvline(0.10, color="k", ls="--", lw=0.8)
    a2.axvspan(-0.02, 0.02, color="#cccccc", alpha=0.4, lw=0)
    a2.set_xlim(-0.45, 1.1)
    a2.set_xlabel("dGAP after RL (whole compiled graph, 200 test prompts)")
    a2.set_title("Status checkpoint (dots: seeds, prompt CI; bar: seed-level 95% t-interval)", fontsize=10)
    fig.suptitle(f"numeric-chameleon-rl: 0.6B (Apple M4) vs 8B (CUDA H100). 8B status: {S['status']}", fontsize=12)
    fig.tight_layout()
    out = HERE.parent / "figures" / "nc8b-replication.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
