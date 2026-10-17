"""skeptic3 sensitivity: the hand-60 shows both 4-way judges label many plainly honest answers
MISLEADING_TRUE (old-HONEST items: hand HONEST 11/11, judges HONEST 3/11 and 2/11). That inflates
M wherever honest answers are common (lam1 late rounds). Recompute M only on answers that the
ORIGINAL binary judge also called DECEPTIVE (hand agrees with it 55/60)."""
import collections, json, random
from pathlib import Path
L = Path.home() / "work/oracle-1017-lie-type-shift/kran-research/projects/lie-type-shift/data/labels"
rng = random.Random(9)
def M(rs):
    c = collections.Counter(r["label"] for r in rs)
    d = c["FALSIFICATION"] + c["OMISSION"] + c["MISLEADING_TRUE"]
    return (c["OMISSION"] + c["MISLEADING_TRUE"]) / d if d else float("nan")
out = {}
for j in ["gpt-4.1-mini", "claude-sonnet-4-6"]:
    rows = [json.loads(l) for l in open(L / f"labels_{j}.jsonl")]
    rows = [r for r in rows if r["judge"] == "DECEPTIVE"]
    o = {}
    for lam in (0, 1):
        sub = [r for r in rows if int(r["lam"]) == lam]
        o[f"lam{lam}_M_by_round"] = [round(M([r for r in sub if r["round"] == k]), 3) for k in (0, 2, 4, 6, 8)]
        grp = collections.defaultdict(list)
        for r in sub:
            if r["round"] in (0, 8):
                grp[r["prompt_id"]].append(r)
        def dl(rs):
            return M([r for r in rs if r["round"] == 8]) - M([r for r in rs if r["round"] == 0])
        ps = list(grp)
        bs = sorted(v for v in (dl([r for p in (rng.choice(ps) for _ in ps) for r in grp[p]]) for _ in range(2000)) if v == v)
        o[f"lam{lam}_delta"] = {"delta": round(dl(sub), 3), "ci95": [round(bs[int(.025*len(bs))], 3), round(bs[int(.975*len(bs))], 3)]}
    out[j] = o
print(json.dumps(out, indent=1))
(Path(__file__).parent / "out/sens_olddec.json").write_text(json.dumps(out, indent=1))
