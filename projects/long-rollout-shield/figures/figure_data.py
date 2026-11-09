"""Build blog.kran.ai/assets/figures/long-rollout-shield/data.json from the saved reads.

Inputs (all relative to the project folder):
  data/reads/raw/<run>.json      served reads (read.py): test.gap, test.gap_ci, test.D, test.D_ci
  C_SUP[<cue>]                    Iris's supervised reader edge on START text (reader_sup.py output)
Kappas follow PREREG v1.2 'The two rival predictions':
  kappa_D = planted gap / (100 x planted D_end)  -> LAW-D predicts real gap = kappa_D x 100 x real D_end
  kappa_C = planted gap / C_sup(planted)         -> LAW-C predicts real gap = kappa_C x C_sup(real)
Gap and D are fractions in the reads; C_sup is in the same unit as the gap (a fraction). The figure shows points.

Usage: uv run python figures/figure_data.py [--out ../../../blog.kran.ai/assets/figures/long-rollout-shield/data.json]
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RUNS = {"onetoken": ["planted_s0"], "real": ["real_s0", "real_s1"], "scattered": ["hash_s0"]}
C_SUP = {"onetoken": "research-iris/c_sup_planted.json", "real": "research-iris/c_sup_real.json",
         "scattered": "research-iris/c_sup_hash.json"}
START_D = {"scattered": "data/reads/raw/start_hash_test.json"}  # untrained cue: its start D stands in for D_end


def load(rel):
    p = os.path.join(PROJ, rel)
    return json.load(open(p)) if os.path.exists(p) else None


def iv(v, ci=None):
    return {"v": v, **({"lo": ci[0], "hi": ci[1]} if ci else {})}


def c_sup(cue):
    d = load(C_SUP[cue])
    if d is None:
        return None
    return iv(d["C_sup"], d.get("C_sup_ci"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PROJ, "../../../blog.kran.ai/assets/figures/long-rollout-shield/data.json"))
    a = ap.parse_args()
    cues, missing = [], []
    for kind, runs in RUNS.items():
        C = c_sup(kind)
        for i, run in enumerate(runs):
            r = load(f"data/reads/raw/{run}.json")
            if r is None and kind == "scattered":
                s = load(START_D["scattered"])
                if s is None or C is None:
                    missing.append(run)
                    continue
                cues.append({"kind": kind, "run": None, "label": "scattered cue (planted, not trained)", "short": "scattered cue",
                             "D": iv(s["test"]["D"], s["test"].get("D_ci")), "C": C, "gap": None})
                continue
            if r is None or C is None:
                missing.append(run)
                continue
            t = r["test"]
            seed = run.split("_s")[-1]
            cues.append({"kind": kind, "run": run,
                         "label": {"onetoken": f"one-token cue, seed {seed}", "real": f"real probe, seed {seed}",
                                   "scattered": f"scattered cue, seed {seed}"}[kind],
                         "short": {"onetoken": "one-token cue", "real": f"seed {seed}", "scattered": "scattered cue"}[kind],
                         **({"group": "real probe"} if kind == "real" else {}),
                         "D": iv(t["D"], t.get("D_ci")), "C": C, "gap": iv(t["gap"], t.get("gap_ci"))})
    p = next((c for c in cues if c["kind"] == "onetoken"), None)
    if p is None:
        raise SystemExit(f"no planted read yet; missing {missing}")
    kappa = {"D": p["gap"]["v"] / p["D"]["v"], "C": p["gap"]["v"] / p["C"]["v"]}
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    out.update({"mock": False, "kappa": kappa, "cues": cues, "missing": missing})
    for c in cues:
        c["pred"] = {k: kappa[k] * c[k]["v"] for k in ("D", "C")}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"kappa": kappa, "missing": missing,
                      "cues": [{"run": c["run"], "gap": c["gap"], "pred": c["pred"]} for c in cues]}, indent=1))


if __name__ == "__main__":
    main()
