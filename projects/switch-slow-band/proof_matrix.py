"""Writes proof-matrix.md from data/summary.json. Keeps every existing row whose claim it does not generate (skeptic
rebuild rows). One artifact path per cell; asserts every artifact exists and every row has 6 cells."""
import json
from pathlib import Path

HERE = Path(__file__).parent
S = json.load(open(HERE / "data/summary.json"))
GEN = {"headline", "phase0", "gate", "prediction", "licence", "law", "reading", "prereg", "precision", "grader"}
HDR = ["| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
f3 = lambda x: f"{x:.3f}"
rows = []


def add(claim, seed, variant, control, result, art):
    assert (HERE / art).exists(), art
    rows.append(f"| {claim} | {seed} | {variant} | {control} | {result} | {art} |")


p0 = S["phase0"]
add("phase0", "base", "pairs 61-63 zeroed in q.k, blocks 26-27, chat prompts", "none",
    f"sink cut {f3(p0['cut_Z'])} [{f3(p0['cut_Z_ci95'][0])}, {f3(p0['cut_Z_ci95'][1])}]", "data/phase0.json")
add("phase0", "base", "50 uniform random 3-pair bands zeroed", "random bands (U50)",
    f"median sink cut {f3(S['phase0_U_median'])}", "data/phase0.json")
add("phase0", "base", "200 bands from pairs 28-60 zeroed", "slow-not-sink bands (N200)",
    f"95th pct sink cut {f3(S['phase0_N95'])}", "data/phase0.json")
h = S["head_27_8"]
add("phase0", "base", "head 27.8, band zeroed in block 27 only", "none",
    f"answer-pos token-0 mass {f3(h['ans_t0_base'])} -> {f3(h['ans_t0_Z'])}; slow-band sink head {h['slow_band_sink_head']}",
    "data/phase0.json")
for c in ("monitor_s0", "monitor_s1"):
    k = f"{c}_fp32_cpu"
    if k not in S["copies"]:
        continue
    s = S["copies"][k]; art = f"data/patch/{k}.json"; seed = c.split("_")[1]
    add("gate", seed, "all 64 pairs (= whole-key patch)", "none", f"{s['G']:.5f} (sink-value K-only within 0.005: {s['G_gate_pass']})", art)
    add("headline", seed, "BAND pairs 61-63, fp32 CPU", "none",
        f"share {f3(s['BAND'])} [{f3(s['BAND_ci95'][0])}, {f3(s['BAND_ci95'][1])}]", art)
    add("headline", seed, "COMP all pairs but 61-63, fp32 CPU", "none",
        f"share {f3(s['COMP'])} [{f3(s['COMP_ci95'][0])}, {f3(s['COMP_ci95'][1])}]", art)
    add("headline", seed, "200 bands from pairs 28-60", "slow-not-sink bands (N200)",
        f"95th pct {f3(s['N95'])}, median {f3(s['N_median'])}, max {f3(s['N_max'])}", art)
    add("headline", seed, "copy verdict (PREREG s5)", "none", f"YES {s['says_YES']}, NO {s['says_NO']}", "data/summary.json")
    add("prereg", seed, "50 uniform random 3-pair bands", "random bands (U50)",
        f"median {f3(s['U_median'])}, 95th pct {f3(s['U_p95'])}, max {f3(s['U_max'])}", art)
    add("licence", seed, "20 energy-matched bands from pairs 0-60", "energy-matched bands (E20)",
        f"median {f3(s['E_median'])}, max {f3(s['E_max'])}", art)
    add("licence", seed, "84 bands from top-12 k_norm-gain pairs minus 61-63", "top-gain bands (TG84)",
        f"95th pct {f3(s['TG95'])}, max {f3(s['TG_max'])}", art)
    add("prediction", seed, "head-27.8 softmax prediction for BAND", "none",
        f"predicted {f3(s['pred_BAND'])} vs measured {f3(s['BAND'])}", "data/prediction.json")
    add("prediction", seed, "64 single pairs, predicted vs measured", "none",
        f"Pearson {f3(s['pred_vs_S_pearson'])}, Spearman {f3(s['pred_vs_S_spearman'])}", "data/summary.json")
    add("law", seed, "Spearman G27.8 (base sink split) vs single-pair shares", "10,000 permutations",
        f"rho {f3(s['law_G27_8'][0])}, p {s['law_G27_8'][1]:.4f}", "data/summary.json")
    add("law", seed, "Spearman pile G_j (rope-sink-band 0.6B) vs single-pair shares", "10,000 permutations",
        f"rho {f3(s['law_pileG'][0])}, p {s['law_pileG'][1]:.4f}", "data/summary.json")
    add("reading", seed, "single pairs S00-S63", "none",
        f"top5 {s['S_top5']}; sum of 64 = {f3(s['S_sum'])}", art)
    for r_ in ("TOP3", "HB", "LOW8", "SLOW36"):
        if r_ in s:
            lab = " (exploratory, named after the capture pass #13492)" if r_ == "TOP3" else ""
            add("reading", seed, f"{r_} pairs {s[r_]['pairs'] if len(s[r_]['pairs']) <= 8 else str(s[r_]['pairs'][0]) + '-' + str(s[r_]['pairs'][-1])}{lab}",
                "none", f"share {f3(s[r_]['share'])}", art)
    add("reading", seed, "62 windows of 3 adjacent pairs", "none",
        f"best window W{s['W_max'][0]:02d} {f3(s['W_max'][1])}; BAND rank {s['BAND_rank_among_W']} of 63", art)
for k, s in S["copies"].items():
    if k.endswith("fp32_cpu") and k.startswith("monitor"):
        continue
    claim = "grader" if k.startswith("grader") else "precision"
    add(claim, k.split("_")[1], f"{k}: BAND / COMP / G", "none",
        f"BAND {f3(s['BAND'])}, COMP {f3(s['COMP'])}, G {f3(s['G'])}", f"data/patch/{k}.json")
if "status" in S:
    add("headline", "both", "title status (PREREG s5)", "none", f"{S['status']}; prereg outcome {S['prereg_outcome']}",
        "data/summary.json")

fn = HERE / "proof-matrix.md"
keep = []
if fn.exists():
    for line in fn.read_text().splitlines():
        if line.startswith("| ") and not line.startswith("| claim") and line.split("|")[1].strip() not in GEN:
            keep.append(line)
for r in rows + keep:
    assert r.count("|") == 7, r
fn.write_text("# Proof matrix — switch-slow-band\n\nGenerated by proof_matrix.py from data/summary.json; skeptic rows "
              "below the generated rows are kept on rewrite.\n\n" + "\n".join(HDR + rows + keep) + "\n")
print(len(rows), "generated rows,", len(keep), "kept")
