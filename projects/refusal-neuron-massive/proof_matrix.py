"""Build proof-matrix.md from data/results.json. Every artifact cell is ONE existing path (asserted).
Run: python3 proof_matrix.py"""
import json
from pathlib import Path

P = Path(__file__).parent
R = json.loads((P / "data/results.json").read_text())
rows = []


def add(claim, seed, variant, control, result, artifact):
    assert (P / artifact).exists(), artifact
    rows.append(f"| {claim} | {seed} | {variant} | {control} | {result} | {artifact} |")


def pct(x):
    return f"{100 * x:+.1f} pts"


th, hl = R["arms"]["test_harmful"], R["arms"]["test_harmless"]
for split, tag in [("test_harmful", "TEST 200"), ("dev_harmful", "DEV 100")]:
    s = R.get("status_" + split)
    if not s:
        continue
    art = f"data/runs/{split}/C.jsonl"
    add("headline", "greedy (n=1)", f"{tag}: keep only the push into M (r_keep)", "none",
        f"{s['r_keep']:.3f} [95% CI {s['r_keep_ci'][0]:.3f}, {s['r_keep_ci'][1]:.3f}]", art)
    add("headline", "greedy (n=1)", f"{tag}: remove only the push into M (r_rem)", "none",
        f"{s['r_rem']:.3f} [95% CI {s['r_rem_ci'][0]:.3f}, {s['r_rem_ci'][1]:.3f}]", f"data/runs/{split}/B.jsonl")
    for a in ["A", "B", "C"]:
        v = R["arms"][split][a]
        add("headline" if a == "A" else "split", "greedy (n=1)", f"{tag}: arm {a} effect on coherent compliance", "none",
            f"{pct(v['EFFECT'])} [{pct(v['EFFECT_ci'][0])}, {pct(v['EFFECT_ci'][1])}]", f"data/runs/{split}/{a}.jsonl")
add("status", "-", "STATUS rule (string label)", "none", R.get("STATUS_string_scorer", "n/a"), "data/results.json")
add("phase0", "greedy (n=1)", "TEST refusal drop of the full pin", "base",
    f"{R['phase0_refusal_drop_A_pts']:.1f} pts (base {100 * th['base']['refusal']:.1f}% -> {100 * th['A']['refusal']:.1f}%)",
    "data/runs/test_harmful/A.jsonl")
add("control", "greedy (n=1)", "clean model, TEST harmful", "base (no edit)",
    f"refusal {100 * th['base']['refusal']:.1f}%, coherent compliance {100 * th['base']['coherent_compliance']:.1f}%",
    "data/runs/test_harmful/base.jsonl")
if "F_logit_drop_p99" in R:
    add("phase0", "rng 0", "first-token refusal score drop: gate vs 100 random L13 neurons (matched edit norm, better sign)",
        "random neurons (F)", f"gate {R['logit_drop']['A']:.2f} vs F p99 {R['F_logit_drop_p99']:.2f} (max {R['F_logit_drop_max']:.2f})",
        "data/runs/logit/test_harmful/A.json")
for a in ["A", "B", "C", "A_hm", "A_zero", "A_skip0", "A_tok0"]:
    if a in hl:
        v = hl[a]
        add("coherence", "greedy (n=1)", f"HARMLESS 200, arm {a}", "harmless prompts",
            f"garbage {100 * v['garbage']:.1f}%, refusal {100 * v['refusal']:.1f}%, pass={v['coherence_pass']}",
            f"data/runs/test_harmless/{a}.jsonl")
for a, lab in [("A_hm", "pin to harmless mean"), ("A_zero", "pin to zero"), ("A_skip0", "full pin except first token"),
               ("A_tok0", "full pin at first token only"), ("N5441pos", "neuron 5441 (+)"), ("N5441neg", "neuron 5441 (-)")]:
    if a in th and "EFFECT" in th[a]:
        v = th[a]
        add("position/existence", "greedy (n=1)", f"TEST 200: {lab}", "none",
            f"{pct(v['EFFECT'])} [{pct(v['EFFECT_ci'][0])}, {pct(v['EFFECT_ci'][1])}]", f"data/runs/test_harmful/{a}.jsonl")
names = {"Dp": "random pattern pushed into M at the gate's full norm (D')",
         "Ep": "same push into 3 ordinary coordinates (E')", "Fp": "same push along a random direction (F')",
         "Fpos": "random L13 neurons, matched edit norm (F, +)", "Fneg": "random L13 neurons, matched edit norm (F, -)"}
for fam, g in R.get("random_gen", {}).items():
    if fam in names:
        first = sorted(g["per_draw"])[0]
        add("existence" if fam == "Dp" else "control", f"rng 0, {g['n']} draws", f"TEST 200: {names[fam]}",
            "random draws", f"median {pct(g['median'])}, max {pct(g['max'])}", f"data/runs/test_harmful/{first}.jsonl")
for fam in ["Dp", "Ep", "Fp", "Ds"]:
    k = f"{fam}_logit_drop"
    if k in R:
        v = R[k]
        add("existence (first token)", f"rng 0, {v['n']} draws", f"first-token score drop, family {fam}", "random draws",
            f"median {v['median']:.2f}, max {v['max']:.2f}, share >= half the gate's {v['frac_ge_half_A']:.2f}",
            f"data/runs/logit/test_harmful/{fam}_0.json")
hdr = "| claim | seed | variant | control | result | artifact |\n|---|---|---|---|---|---|\n"
(P / "proof-matrix.md").write_text("# Proof matrix (built by proof_matrix.py from data/results.json)\n\n" + hdr + "\n".join(rows) + "\n")
print(len(rows), "rows")
