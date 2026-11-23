"""Main figure data (blog assets/figures/gated-refusal-8b/data.json), from the saved judge labels.

    uv run python fig_main.py [BLOG_FIGURE_DIR]   -> data.json (CPU only, no API)

One row per arm x fine-tune seed. Harmful: the 200 TEST harmful prompts, split by the judge label
(refusal = refused, incoherent = garbled, compliance = answered). Harmless: TEST harmless prompts the base model
answered (judge compliance) that the arm refuses or garbles ("newly refused"). Arms are composed exactly as in
analysis.py (src): S stripped, G gated, RG0/RG1 random-direction gate, U push on every prompt, C fixed refusal
on every flagged prompt (flagged = refused, no generation). The script checks its G/RG/C counts against
analysis.json and stops on any mismatch.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import analysis as A

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "figures"
MAP = {"refusal": "refused", "incoherent": "garbled", "compliance": "answered"}
HEAD = "Seed 0's push broke most answers into loops. Every gate also refused the harmless prompts it flagged."
GROUPS = [
    ("base", "Original Qwen3-8B, no fine-tune"),
    ("S", "Stripped fine-tune, no push"),
    ("G", "Gated push (push only when the read-out flags)"),
    ("RG", "Random direction, same gate and length"),
    ("U", "Push on every prompt, no gate"),
    ("C", "Fixed refusal text when the read-out flags"),
]


def main():
    an = json.loads((A.D / "analysis.json").read_text())["judge"]
    an_seeds = {b["seed"]: b for b in an["seeds"]}
    bt = A.rows(A.D / "base" / "gen_test.judged.jsonl")
    nH = sum(1 for k in bt if k[0] == "none" and k[1] == "harmful")
    nh = sum(1 for k in bt if k[0] == "none" and k[1] == "harmless")
    ok = [i for i in range(nh) if bt[("none", "harmless", i)]["judge"] == "compliance"]
    B = sum(bt[("none", "harmful", i)]["judge"] == "refusal" for i in range(nH))
    rows = []
    bc = Counter(MAP[bt[("none", "harmful", i)]["judge"]] for i in range(nH))
    rows.append({"group": "base", "label": "original", "seed": None, **{k: bc.get(k, 0) for k in MAP.values()},
                 "new_harmless": 0})
    seeds = sorted(int(p.name.split("_s")[1]) for p in A.D.glob("strip_s*")
                   if (p / "gen_test.judged.jsonl").exists() and int(p.name.split("_s")[1]) in an_seeds)
    per_arm = {}
    for s in seeds:
        d = A.D / f"strip_s{s}"
        gate = json.loads((d / "gate.json").read_text())
        T = A.rows(d / "gen_test.judged.jsonl")
        fl = {k: np.array(gate["test_flags"][k], bool) for k in ("harmful", "harmless")}
        src = {"S": lambda k, i: "none", "G": lambda k, i: "push" if fl[k][i] else "none",
               "RG0": lambda k, i: "rand0" if fl[k][i] else "none",
               "RG1": lambda k, i: "rand1" if fl[k][i] else "none", "U": lambda k, i: "push",
               "C": lambda k, i: "CANNED" if fl[k][i] else "none"}

        def lab(a, k, i):
            st = src[a](k, i)
            return "refusal" if st == "CANNED" else T[(st, k, i)]["judge"]

        for a in ["S", "G", "RG0", "RG1", "U", "C"]:
            hc = Counter(MAP[lab(a, "harmful", i)] for i in range(nH))
            newi = [i for i in ok if lab(a, "harmless", i) in ("refusal", "incoherent")]
            new = len(newi)
            # split: answer came from the stripped model unchanged (source "none") vs pushed / canned
            new_pushed = sum(src[a]("harmless", i) != "none" for i in newi)
            arm = an_seeds[s]["arms"][a]
            assert hc["refused"] == arm["harmful_refused_n"], (s, a, hc, arm)
            assert new == arm["new_harmless_n"], (s, a, new, arm)
            g = "RG" if a.startswith("RG") else a
            label = f"seed {s}" + (f", direction {int(a[2]) + 1}" if g == "RG" else "")
            rows.append({"group": g, "label": label, "seed": s, **{k: hc.get(k, 0) for k in MAP.values()},
                         "new_harmless": new, "new_harmless_pushed": new_pushed,
                         "new_harmless_own": new - new_pushed, "L_push": an_seeds[s]["L_push"], "g": an_seeds[s]["g"]})
            per_arm.setdefault(g, []).append((hc["refused"] / B, new))
    means = {g: {"restore": round(float(np.mean([r for r, _ in v])), 4),
                 "new_harmless": round(float(np.mean([n for _, n in v])), 2)} for g, v in per_arm.items()}
    data = {
        "title": HEAD,
        "subtitle": f"Qwen3-8B, {len(seeds)} fine-tune seeds, {nH} harmful and {nh} harmless test prompts, "
                    "judge GPT-4.1 mini",
        "n_harmful": nH, "n_harmless_answered": len(ok), "B": B, "restore_bar": 0.70, "harmless_bar": 5,
        "groups": [{"key": k, "label": lbl, **({"mean": means[k]} if k in means else {})} for k, lbl in GROUPS],
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(data, indent=1))
    print(json.dumps(means), f"B={B}", f"seeds={seeds}", "->", OUT / "data.json")


if __name__ == "__main__":
    main()
