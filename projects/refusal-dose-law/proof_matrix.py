"""Write proof-matrix.md from data/results.json. Keeps every existing row whose claim this script does not generate
(skeptic rebuild rows, hand rows). Run: python3 proof_matrix.py [path/to/proof-matrix.md]"""
import json
import sys
from pathlib import Path

P = Path(__file__).parent
R = json.loads((P / "data/results.json").read_text())
OUTF = Path(sys.argv[1]) if len(sys.argv) > 1 else P / "proof-matrix.md"
GEN = {"headline", "headline-split", "headline-twin", "labeling", "control", "scout-c1", "scout-c2", "scout-c3",
       "scout-c4", "d2-subspace", "d2-positions", "arditi-midpoint", "two-sided", "downstream", "p-matched", "incoherent"}
rows = []
C = R["cells"]


def pct(x):
    return f"{x:.1f}" if x is not None else "n/a"


for lab in ("comply", "nonref"):
    st = R["labelings"][lab]["status"]
    claim = "headline" if lab == "comply" else "labeling"
    for m, g in st["gap"].items():
        lo, hi = st["gap_ci"][m]
        rows.append((claim, "bootstrap seed 0 (2000)", f"gate pin {m}, {lab}", "none",
                     f"gap {g:+.1f} pts (95% CI {lo:+.1f}, {hi:+.1f})", f"data/judge/G@{m}.json; data/results.json"))
    sp = st.get("split")
    if sp:
        rows.append(("headline-split" if lab == "comply" else "labeling", "bootstrap seed 0", f"split pin 30, {lab}",
                     "G_off = remainder-only control",
                     f"EFFECT G {sp['eff_G']:+.1f}, G_on {sp['eff_on']:+.1f}, G_off {sp['eff_off']:+.1f}; "
                     f"G_off/G {sp['off_ratio']:.2f} (CI {st['off_ratio_ci'][0]:.2f}, {st['off_ratio_ci'][1]:.2f}); status {st['status']}",
                     "data/judge/G_on@30.json; data/judge/G_off@30.json"))
st = R["labelings"]["comply"]["status"]
if "offK_ratio" in st.get("split", {}):
    rows.append(("d2-subspace", "bootstrap seed 0", "G_off_K pin 30 (gate minus its part in the 7-dim refusal subspace K)",
                 "none", f"EFFECT {st['split']['eff_offK']:+.1f} of {st['split']['eff_G']:+.1f}; ratio {st['split']['offK_ratio']:.3f} "
                 f"(CI {st['offK_ratio_ci'][0]:.2f}, {st['offK_ratio_ci'][1]:.2f})", "data/judge/G_offK@30.json; data/phase0/d2.json"))
d2f = P / "data/phase0/d2.json"
if d2f.exists():
    d2 = json.loads(d2f.read_text())
    for r_ in d2["L13_positions"]:
        rows.append(("d2-positions", "-", f"L13 template pos {r_['pos']} ({r_['tok']})", "none",
                     f"cos(w_off, r_pos) {r_['cos_w_off']:+.3f}; cos(w, r_pos) {r_['cos_w']:+.3f}", "data/phase0/d2.json"))
tg = R["secondary"]["twin_vs_gate"]
for m, v in tg.items():
    for k, t in enumerate(v["TWIN"]):
        rows.append(("headline-twin", f"u seed {100 + k}", f"TWIN pin {m} (TEST100)", "none",
                     f"comply {t:.1f}% vs gate {v['G']:.1f}% (same p, same push)", f"data/judge/TWIN_{k}@{m}.json"))
for c in sorted(C):
    g = C[c]["group"]
    if g in ("base", "RN", "RD", "FK"):
        rows.append(("control", "seed 0/1/2", c, {"base": "no edit", "RN": "random neuron direction", "RD": "random direction",
                                                   "FK": "fake refusal direction"}[g],
                     f"comply {100 * C[c]['comply']:.1f}% (n={C[c]['n']}), p {C[c]['p']:+.4f}", f"data/judge/{c}.json"))
cl = R["scout_clauses"]
rows.append(("scout-c1", "-", "literal fit RN+RD", "none",
             f"gaps {', '.join(f'{m}:{g:+.1f}' for m, g in cl.get('c1_literal_gap', {}).items())}; held {cl.get('c1_held')}", "data/results.json"))
rows.append(("scout-c2", "-", "fit set R^2", "none", f"R^2 {cl.get('c2_r2_fit', 0):.3f} (dialed only {cl.get('c2_r2_dialed') or 0:.3f}); held {cl.get('c2_held')}", "data/results.json"))
rows.append(("scout-c3", "-", "first-token score, 101 neurons, pin 30", "split-half reliability",
             f"Spearman |cos| vs |change| {cl.get('c3_spearman_abs', float('nan')):.3f}, signed {cl.get('c3_spearman_signed', float('nan')):.3f}, "
             f"split-half {cl.get('c3_split_half_spearman', float('nan')):.3f}; held {cl.get('c3_held')}", "data/ft/; data/results.json"))
rows.append(("scout-c4", "-", "pure dose fit", "none", f"R^2 {cl.get('c4_dose_r2') or 0:.3f}; held {cl.get('c4_held')}", "data/results.json"))
s = R["secondary"]
rows.append(("arditi-midpoint", "-", "curve in Arditi units", "none", f"half point u = {s['arditi_u_half']:.3f} ||r||; held {s['arditi_held']}", "data/results.json"))
for c, v in s.get("NN", {}).items():
    rows.append(("two-sided", "-", c, "natural neuron (held out)", f"obs {v['obs']:.1f}% vs curve {v['pred']:.1f}% at p {v['p']:+.4f}", f"data/judge/{c}.json"))
for c, v in s.get("PM", {}).items():
    rows.append(("p-matched", "-", c, "row only", f"obs {v['obs']:.1f}% vs curve {v['pred']:.1f}%, incoherent {v['incoh']:.1f}%", f"data/judge/{c}.json"))
if "downstream" in s:
    d = s["downstream"]
    rows.append(("downstream", "-", f"p at layer {d['L_star']}", "row only", f"R^2 {d['r2']:.3f} over {d['cells']} cells", "data/ft/; data/results.json"))

old = []
if OUTF.exists():
    for line in OUTF.read_text().splitlines():
        if line.startswith("| ") and not line.startswith("| claim") and not line.startswith("|---"):
            claim = line.split("|")[1].strip()
            if claim not in GEN:
                old.append(line)
txt = ["# Proof matrix: refusal-dose-law", "",
       "Generated by proof_matrix.py from data/results.json (STATUS " + R["STATUS"] + "); rows with other claims are kept as written.", "",
       "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
txt += ["| " + " | ".join(r) + " |" for r in rows] + old
OUTF.write_text("\n".join(txt) + "\n")
print(len(rows), "generated rows,", len(old), "kept rows ->", OUTF)
