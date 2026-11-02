"""Writes proof-matrix.md from data/<tag>/results.json (+ band.json, asserts). Rows whose claim this script does not
generate (skeptic rebuild rows) are kept as they are. One artifact path per cell; every path must exist."""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
PM = HERE / "proof-matrix.md"
GEN = {"headline", "control: random band", "location", "licence: equally large band", "licence: temperature",
       "necessity: lead removed", "sufficiency", "sufficiency control", "rival: slowest pairs", "secondary: direct per-layer",
       "secondary: prepend", "assert", "band stability", "prereg: loss", "prereg: logit lens", "variant: Qwen3-0.6B",
       "windows"}
rows = []


def add(claim, seed, variant, control, result, artifact):
    assert (HERE / artifact).exists(), artifact
    rows.append(f"| {claim} | {seed} | {variant} | {control} | {result} | {artifact} |")


for tag in ("Qwen3-1.7B", "Qwen3-0.6B"):
    rp = HERE / "data" / tag / "results.json"
    if not rp.exists():
        continue
    res = json.load(open(rp)); band = json.load(open(HERE / "data" / tag / "band.json"))
    bstr = "{" + ",".join(map(str, band["B"])) + "}"
    head = "headline" if tag == "Qwen3-1.7B" else "variant: Qwen3-0.6B"
    add("band stability" if tag == "Qwen3-1.7B" else head, "cal", f"{tag} calibration", "none",
        f"B={bstr}, k={band['k']}, share {sum(band['G'][j] for j in band['B'])/sum(band['G']):.3f}", f"data/{tag}/band.json")
    if tag == "Qwen3-1.7B":
        add("band stability", "cal2", f"{tag} calibration-2", "none",
            f"B={band['calib']['cal2']['B']}, Jaccard {band['calib']['cal']['jaccard_cal2']:.2f}", f"data/{tag}/calib_cal2.npz")
        add("location", "cal", f"{tag}", "chance C(18,k)/C(64,k)",
            f"min index {res['location']['min_index']}: edge a>=21 {res['location']['edge_a_ge21']}, c>=40 {res['location']['edge_c_ge40']}, b>=46 {res['location']['edge_b_ge46']}",
            f"data/{tag}/band.json")
    for s in ("A", "B"):
        d = res["sets"].get(s)
        if not d or "cut_P" not in d:
            continue
        add(head, s, f"{tag} zero band {bstr}, all layers", "none",
            f"cut {d['cut_P']:.3f} [{d['arms']['P']['cut_ci95'][0]:.3f}, {d['arms']['P']['cut_ci95'][1]:.3f}], S {d['S_base']:.3f}->{d['arms']['P']['S']:.3f}",
            f"data/{tag}/arms/{s}__P.npz")
        for i in range(1, 21):
            n = f"R{i:02d}"
            add("control: random band", s, f"{tag} zero {band['R'][i-1]}", "random band (seed 1234)",
                f"cut {d['arms'][n]['cut']:.3f}", f"data/{tag}/arms/{s}__{n}.npz")
        add("prereg: loss", s, f"{tag} zero band", "median of 20 random bands",
            f"dloss {d['dloss_P']:+.3f} vs {d['dloss_R_median']:+.3f} nats", f"data/{tag}/arms/{s}__P.npz")
        if "dlens_P" in d:
            add("prereg: logit lens", s, f"{tag} zero band", "median of 20 random bands",
                f"dlens {d['dlens_P']:+.4f} vs {d['dlens_R_median']:+.4f}", f"data/{tag}/arms/{s}__P.npz")
        A = d["arms"]
        if "cut_E_median" in d:
            add("licence: equally large band", s, f"{tag} zero 20 energy-matched sets", "energy-matched (seed 7)",
                f"median cut {d['cut_E_median']:.3f} vs band {d['cut_P']:.3f}", f"data/{tag}/arms/{s}__E01.npz")
        if "T" in A:
            add("licence: temperature", s, f"{tag} zero band + row STD rescale", "temperature",
                f"cut {A['T']['cut']:.3f}", f"data/{tag}/arms/{s}__T.npz")
        if "GN" in A:
            add("necessity: lead removed", s, f"{tag} first-token lead of band removed", "none",
                f"cut {A['GN']['cut']:.3f}", f"data/{tag}/arms/{s}__GN.npz")
        if "keep_SUF" in d:
            add("sufficiency", s, f"{tag} lead removed outside band", "none",
                f"keeps {d['keep_SUF']:.3f}", f"data/{tag}/arms/{s}__SUF.npz")
            add("sufficiency control", s, f"{tag} lead removed outside 20 random bands", "random band",
                f"keeps max {d['keep_SUFR_max']:.3f}, {d['n_SUFR_keep_less']}/20 keep less", f"data/{tag}/arms/{s}__SUFR01.npz")
        if "LOW8" in A:
            add("rival: slowest pairs", s, f"{tag} zero pairs 56-63", "none", f"cut {A['LOW8']['cut']:.3f}",
                f"data/{tag}/arms/{s}__LOW8.npz")
        if "direct" in d:
            add("secondary: direct per-layer", s, f"{tag} band zeroed one layer at a time, upstream clean", "random band",
                f"cut {d['direct']['cut_P_all_layers']:.3f} vs random max {max(d['direct']['cut_R_all_layers']):.3f}",
                f"data/{tag}/arms/{s}__base.npz")
    if "status" in res:
        p = res["sets"]["pooled"]
        add(head, "A+B", f"{tag} pooled", "20 random bands",
            f"cut {p['cut_P']:.3f}, random median {p['cut_R_median']:.3f}, max {p['cut_R_max']:.3f}, {p['n_R_ge_P']}/20 >= band; status {res['status']}",
            f"data/{tag}/results.json")
    a = HERE / "logs" / f"asserts_{tag}.json"
    if a.exists():
        aj = json.load(open(a))
        add("assert", "smoke", tag, "none",
            f"gap split err {aj['a_max_abs_err']:.1e}; bf16 hook loss diff {aj['b_max_abs_diff']}; k0 identical {all(aj['c_k0_identical'].values())}",
            f"logs/asserts_{tag}.json")

kept = []
if PM.exists():
    for line in PM.read_text().splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|", line)
        if m and m.group(1) not in GEN and m.group(1) not in ("claim",) and not set(m.group(1)) <= set("- "):
            kept.append(line)
hdr = ["| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
PM.write_text("\n".join(hdr + rows + kept) + "\n")
for r in rows:
    assert r.count("|") == 7, r
print(len(rows), "generated rows,", len(kept), "kept rows ->", PM)
