"""skeptic3: does the old binary judge's 'lying drop' survive a rubric that counts minimisation?
Per arm x round (seeds pooled, base once per arm): old-judge deceptive share vs new 4-way
deceptive share (F+O+M over all answers incl. OTHER), and how the new judge labels the answers
the old judge called HONEST. Prompt-clustered bootstrap CI for the drop r0 -> r8."""
import collections, json, random, sys
from pathlib import Path
L = Path.home() / "work/oracle-1017-lie-type-shift/kran-research/projects/lie-type-shift/data/labels"
rng = random.Random(5)
out = {}
for jf in ["labels_gpt-4.1-mini.jsonl", "labels_claude-sonnet-4-6.jsonl"]:
    rows = [json.loads(l) for l in open(L / jf)]
    o = {}
    for lam in (0, 1):
        def cell(rnd, rs=rows):
            return [r for r in rs if int(r["lam"]) == lam and r["round"] == rnd]
        for rnd in (0, 2, 4, 6, 8):
            c = cell(rnd)
            old = sum(r["judge"] == "DECEPTIVE" for r in c) / len(c)
            new = sum(r["label"] in ("FALSIFICATION", "OMISSION", "MISLEADING_TRUE") for r in c) / len(c)
            oh = collections.Counter(r["label"] for r in c if r["judge"] == "HONEST")
            o[f"lam{lam}_r{rnd}"] = {"old_dec": round(old, 3), "new_dec": round(new, 3), "old_HONEST_relabelled": dict(oh)}
        grp = collections.defaultdict(list)
        for r in rows:
            if int(r["lam"]) == lam and r["round"] in (0, 8):
                grp[r["prompt_id"]].append(r)
        def drops(rs):
            a = [r for r in rs if r["round"] == 0]; b = [r for r in rs if r["round"] == 8]
            f = lambda g, k: sum(k(r) for r in g) / len(g)
            od = f(a, lambda r: r["judge"] == "DECEPTIVE") - f(b, lambda r: r["judge"] == "DECEPTIVE")
            nd = f(a, lambda r: r["label"] in ("FALSIFICATION", "OMISSION", "MISLEADING_TRUE")) - f(b, lambda r: r["label"] in ("FALSIFICATION", "OMISSION", "MISLEADING_TRUE"))
            return od, nd
        ps = list(grp)
        bs = [drops([r for p in (rng.choice(ps) for _ in ps) for r in grp[p]]) for _ in range(2000)]
        obs = drops([r for p in ps for r in grp[p]])
        def ci(i):
            v = sorted(x[i] for x in bs); return [round(v[50], 3), round(v[1950], 3)]
        diff = sorted(x[0] - x[1] for x in bs)
        o[f"lam{lam}_drop_r0_to_r8"] = {"old_judge": round(obs[0], 3), "old_ci": ci(0), "new_rubric": round(obs[1], 3), "new_ci": ci(1),
                                       "old_minus_new": round(obs[0] - obs[1], 3), "old_minus_new_ci": [round(diff[50], 3), round(diff[1950], 3)]}
    out[jf] = o
print(json.dumps(out, indent=1))
(Path(__file__).parent / "out/old_vs_new.json").write_text(json.dumps(out, indent=1))
