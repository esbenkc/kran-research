"""proof-matrix.md: one row per cell, every artifact a single path relative to this folder (checked to exist).
Run after analysis.py: venv/bin/python proof_matrix.py
"""
import json
import os

from common import DATA, ROOT

R = lambda p: json.loads((ROOT / p).read_text())
rows = []


def add(claim, seed, variant, control, result, artifact):
    assert (ROOT / artifact).exists(), artifact
    rows.append(f"| {claim} | {seed} | {variant} | {control} | {result} | {artifact} |")


def pct(x):
    return f"{100 * x:.1f}%"


price = R("data/leace/price.json")
guard = R("data/leace/guard.json")
S = R("data/results/summary.json")
P = price["registered_price_P"]
add("price", "-", f"registered LEACE (seq 10..29, {price['selection']['chosen']}, 1400)", "none",
    f"P = {P:.5f} nats/token", "data/leace/price.json")
for c in ("seq1028", "all_indep", "everypos", "gate"):
    add("price", "-", c, "none", f"{price[c]['kl_eval_mean_every_position']:.5f}", "data/leace/price.json")
add("price", "-", "rand_size (same edit size, random direction)", "random direction",
    f"{price['rand_size']['kl_eval_mean_every_position']:.5f}", "data/leace/price.json")
add("price", "-", "unedited", "identity edit", f"{price['unedited']['kl_eval_mean_every_position']:.2e}",
    "data/leace/price.json")
for R_ in range(10, 30):
    add("price per readout", "-", f"single_{R_}", "none", f"{price[f'single_{R_}']['kl_eval_mean_every_position']:.5f}",
        "data/leace/price.json")
for s, v in price["selection"]["cells"].items():
    add("selection", "-", s, "none", f"OOF AUROC {v['oof_auroc_at_best']:.3f} at R{v['best_readout']}",
        "data/leace/price.json")
for st, nm in (("clean", "clean model"), ("leace_registered", "registered eraser"), ("leace_gate", "attack-600 eraser (gate)"),
               ("leace_seq1028", "10..28 eraser")):
    g = guard[st]
    add("known answer", "-", nm, "clean model" if st == "clean" else "known-answer erasure",
        f"F1 {pct(g['F1']['catch'])} / F2 {pct(g['F2']['catch'])} / M1 {pct(g['M1']['catch'])}; F1 AUROC "
        f"{g['F1']['auroc']:.3f}; R29 fresh {pct(g['fresh_per_readout']['29']['catch'])}", "data/leace/guard.json")
names = {"base": "headline", "x10": "10x KL weight", "N": "shuffled-label control", "G": "mean-gap side arm"}
for a, v in S["arms"].items():
    if not v:
        continue
    for k, st in enumerate(v["states"]):
        J = R(f"data/results/{st}.json")
        m = J["monitors"]
        ctl = "shuffled labels" if a == "N" else "none"
        add(names[a], st.split("seed")[-1], st, ctl,
            f"F1 {pct(m['F1']['catch'])} (AUROC {m['F1']['auroc']:.3f}), F2 {pct(m['F2']['catch'])} R{m['F2']['readout']}, "
            f"M1 {pct(m['M1']['catch'])}, frozen {pct(m['frozen']['catch'])}, KL {J['kl_eval_every_position']:.5f}, "
            f"rank {J['right_first']}/298", f"data/results/{st}.json")
        st_ = m["stress"]
        add("stress " + names[a], st.split("seed")[-1], f"{st} QDA / MLP at R{st_['readout']}", "none",
            f"QDA {pct(st_['qda']['catch'])} (AUROC {st_['qda']['auroc']:.3f}), MLP {pct(st_['mlp']['catch'])} "
            f"(AUROC {st_['mlp']['auroc']:.3f})", f"data/results/{st}.json")
        add("stress " + names[a], st.split("seed")[-1], f"{st} shuffled-label fresh probe x5", "shuffled labels",
            "mean " + pct(sum(c["catch"] for c in st_["shuffled"]) / 5), f"data/results/{st}.json")
        add("trajectory " + names[a], st.split("seed")[-1], st, "none",
            "slice catch by step: " + ", ".join(f"{t['step']}:{pct(t['refit']['catch'])}" for t in R(f"data/runs/{st}.json")["traj"]),
            f"data/runs/{st}.json")
    add(names[a] + " pooled", "pooled", a, "none",
        f"F1 {pct(v['F1']['catch'])} [{pct(v['F1']['ci95'][0])}, {pct(v['F1']['ci95'][1])}], F2 {pct(v['F2']['catch'])}, "
        f"M1 {pct(v['M1']['catch'])}, KL {v['kl_pooled']:.5f} = {v['kl_pooled'] / P:.2f} x P", "data/results/summary.json")
if "status" in S:
    add("status", "pooled", "PREREG s6", "none", f"{S['status']['status']} {S['status'].get('failed_as', '')}",
        "data/results/summary.json")
    add("prediction", "pooled", "PREREG s7", "none", f"{S.get('prediction', {}).get('outcome')}", "data/results/summary.json")
extra = ROOT / "skeptics"
cells = sorted(extra.glob("*/proof-cells.md")) if extra.exists() else []
txt = ("# Proof matrix: refit-probe-evader\n\nBuilt by proof_matrix.py from data/ (every artifact path exists).\n\n"
       "| claim | seed | variant | control | result | artifact |\n|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
for c in cells:
    txt += "\n" + c.read_text()
(ROOT / "proof-matrix.md").write_text(txt)
print(len(rows), "rows", "+", len(cells), "skeptic files")
