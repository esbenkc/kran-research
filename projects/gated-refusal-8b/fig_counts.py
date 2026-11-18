"""Figure data: judge-label counts per seed x arm (refusal / incoherent / compliance), harmful and harmless TEST prompts.

    python fig_counts.py    -> data/fig_counts.json   (CPU only, no API; same arm composition as analysis.py)

Harmful: all 200 TEST harmful prompts. Harmless: the TEST harmless prompts the base model answered (judge 'compliance').
Arms as in analysis.py: S stripped, G gated, RG0/RG1 random-direction gate, U ungated push, plus base (no strip, no push).
"""
import json
from collections import Counter

import numpy as np

import analysis as A

ARMS = ["S", "G", "RG0", "RG1", "U"]
LABELS = ["refusal", "incoherent", "compliance"]


def main():
    bt = A.rows(A.D / "base" / "gen_test.judged.jsonl")
    nH = sum(1 for k in bt if k[0] == "none" and k[1] == "harmful")
    nh = sum(1 for k in bt if k[0] == "none" and k[1] == "harmless")
    ok = [i for i in range(nh) if bt[("none", "harmless", i)]["judge"] == "compliance"]
    out = {"labels": LABELS, "n_harmful": nH, "n_harmless_base_answered": len(ok),
           "base": {"harmful": Counter(bt[("none", "harmful", i)]["judge"] for i in range(nH)),
                    "harmless": Counter(bt[("none", "harmless", i)]["judge"] for i in ok)},
           "seeds": []}
    seeds = sorted(int(p.name.split("_s")[1]) for p in A.D.glob("strip_s*") if (p / "gen_test.judged.jsonl").exists())
    for s in seeds:
        d = A.D / f"strip_s{s}"
        gate = json.loads((d / "gate.json").read_text())
        push = json.loads((d / "push.json").read_text())
        T = A.rows(d / "gen_test.judged.jsonl")
        flags = {k: np.array(gate["test_flags"][k], bool) for k in ("harmful", "harmless")}
        src = {"S": lambda k, i: "none", "G": lambda k, i: "push" if flags[k][i] else "none",
               "RG0": lambda k, i: "rand0" if flags[k][i] else "none",
               "RG1": lambda k, i: "rand1" if flags[k][i] else "none", "U": lambda k, i: "push"}
        arms = {}
        for a in ARMS:
            arms[a] = {"harmful": Counter(T[(src[a]("harmful", i), "harmful", i)]["judge"] for i in range(nH)),
                       "harmless": Counter(T[(src[a]("harmless", i), "harmless", i)]["judge"] for i in ok)}
        out["seeds"].append({"seed": s, "L_push": push["layer"], "g": push["gain"],
                             "flags_harmful": int(flags["harmful"].sum()), "flags_harmless": int(flags["harmless"].sum()),
                             "arms": arms})
    (A.D / "fig_counts.json").write_text(json.dumps(out, indent=1))
    for b in out["seeds"]:
        print(f"s{b['seed']} L{b['L_push']} g{b['g']} flags {b['flags_harmful']}/{b['flags_harmless']}")
        for a, v in b["arms"].items():
            print(f"  {a:4s} harmful {dict(v['harmful'])}  harmless {dict(v['harmless'])}")


if __name__ == "__main__":
    main()
