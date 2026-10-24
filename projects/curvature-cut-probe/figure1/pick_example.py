"""Owain: pick the Figure 1 example MECHANICALLY from the saved run files (no hand choice of numbers).
Rule (fixed before looking at any statement text):
  an eval FALSE statement that, in EVERY available run of the full-probe arm (unproj) and the cut-probe arm (proj):
    (1) is caught on the clean model (step-0 score above the clean line of that run's own probe),
    (2) is hidden after the fine-tune (last-checkpoint score at or below that state's re-set line),
    (3) keeps the same top-1 next word at the country token (top5 file, clean vs fine-tuned);
  among those, rank by how typical its country-token KL is: sum over runs of |log(kl_i / median kl of that run)|,
  smallest first. Cities already used in Figure 1s or posts on origin/main (SKIP below) are skipped.
Writes example.json (every qualifying statement, ranked, with every number the figure uses).
Final step (by hand, among qualifiers only): a well-known city with an affirmative 'is in' statement. Chosen: idx 1373,
'The city of Durban is in Ukraine.' (qualifies in all 4 full/cut runs and both random-cut runs; its seed-0 country-token
KL is 0.0042 in both the full and the cut run, against run medians 0.0043 and 0.0046)."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def threshold(neg_scores, fpr=0.01):
    """Same rule as common.threshold (numpy only, no torch import): k-th highest of n negatives, k = floor(fpr*(n+1))."""
    s = np.sort(np.asarray(neg_scores))[::-1]
    return float(s[max(1, int(np.floor(fpr * (len(s) + 1)))) - 1])


RUNS = ROOT / "data" / "runs"
SPL = json.loads((ROOT / "data" / "ref" / "split.json").read_text())
Y = np.array([1 - r["true"] for r in SPL])          # 1 = false statement (common.rows)
CITY = np.array([r["city"] for r in SPL])
SPLIT = np.array([r["split"] for r in SPL])
EV = np.where(SPLIT == "eval")[0]
EVF = np.where((SPLIT == "eval") & (Y == 1))[0]
LINE = np.where((SPLIT == "refit") & (Y == 0))[0]
SKIP = {"Copenhagen", "Rio de Janeiro", "Bangkok", "Bucharest", "Krasnodar", "Lyon", "Najafgarh"}  # used on origin/main
ARMS = {"unproj": "full probe", "proj": "cut probe", "rand": "random-cut probe"}


def last_step(d):
    st = sorted(int(f.stem.split("_")[1]) for f in d.glob("top5_*.npz"))
    st = [s for s in st if s > 0 and (d / f"ckpt_{s}.npz").exists()]
    return st[-1] if st else None


def run_info(d, arm):
    s = last_step(d)
    if s is None:
        return None
    c0, c1 = np.load(d / "ckpt_0.npz"), np.load(d / f"ckpt_{s}.npz")
    names = [str(x) for x in c0["probes"]]
    j = names.index(arm if arm == "unproj" else f"{arm}_k128")
    s0, s1 = c0["scores"][:, j], c1["scores"][:, j]
    lc, lr = threshold(s0[LINE]), threshold(s1[LINE])
    kl, koff, cpos = c1["kl"], c1["koff"], c1["cpos"]
    klc = np.array([kl[koff[i] + cpos[i]] for i in range(len(cpos))])
    t5 = np.load(d / f"top5_{s}.npz")
    return dict(dir=d.name, step=s, s0=s0, s1=s1, line_clean=float(lc), line_reset=float(lr), klc=klc,
                kl_med=float(np.median(klc[EV])), ids=t5["ids"], p=t5["p"])


def main():
    runs = {}
    for arm in ("unproj", "proj", "rand"):
        for d in sorted(RUNS.glob(f"{arm}__lam10000__seed*")):
            r = run_info(d, arm)
            if r:
                runs.setdefault(arm, []).append(r)
    use = [r for a in ("unproj", "proj") for r in runs.get(a, [])]
    print({a: [r["dir"] + f"@{r['step']}" for r in rs] for a, rs in runs.items()})
    cands = []
    for i in EVF:
        if CITY[i] in SKIP:
            continue
        ok = all(r["s0"][i] > r["line_clean"] and r["s1"][i] <= r["line_reset"] and r["ids"][i, 0, 0] == r["ids"][i, 1, 0]
                 for r in use)
        if ok:
            typ = sum(abs(np.log(max(r["klc"][i], 1e-9) / r["kl_med"])) for r in use)
            cands.append((typ, int(i)))
    cands.sort()
    print(f"{len(cands)} of {len(EVF)} eval false statements qualify in all {len(use)} runs")
    out = {"rule": __doc__, "n_qualify": len(cands), "n_false": int(len(EVF)), "runs": [r["dir"] for r in use], "top": []}
    for rank, (typ, i) in enumerate(cands):
        e = {"idx": i, "text": SPL[i]["text"], "city": SPL[i]["city"], "typicality": round(float(typ), 3), "runs": {}}
        for a, rs in runs.items():
            for r in rs:
                e["runs"][r["dir"]] = {"step": r["step"], "score_clean": float(r["s0"][i]), "line_clean": r["line_clean"],
                                       "score_ft": float(r["s1"][i]), "line_reset": r["line_reset"],
                                       "kl_country": float(r["klc"][i]), "kl_country_median": r["kl_med"],
                                       "top5_clean": [[int(t), float(p)] for t, p in zip(r["ids"][i, 0], r["p"][i, 0])],
                                       "top5_ft": [[int(t), float(p)] for t, p in zip(r["ids"][i, 1], r["p"][i, 1])]}
        out["top"].append(e)
        if rank < 25:
            print(f"{typ:6.2f} {i:5d} {SPL[i]['text']}  " + " ".join(f"{r['dir'][:5]}{r['dir'][-1]}:{r['klc'][i]:.4f}" for r in use))
    (Path(__file__).parent / "example.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
