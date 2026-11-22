"""S9 (fresh cue directions, exploratory, never in the status) rebuilt from raw curves (Vera, theory-1122).
Direction u11(20+s) for seed s. T from my own switch_time() in vera_check.py; frozen exclusion: a size with
|gap(0)| >= 0.02 in the pre-RL scan file data/scan_u<dir>.json is left out of the fit. Per-seed OLS of ln T on
ln share (needs >= 4 sizes), t-interval over the seeds that have a slope.
Usage: python vera_s9.py <project dir>"""
import glob, json, math, os, sys
import numpy as np
from vera_check import curve, switch_time, ols, tint, load
P = sys.argv[1]; slopes = {}; out = {}
for f in sorted(glob.glob(os.path.join(P, "data/runs_explore/*.json"))):
    d = load(f); a = d["args"]; s = a["seed"]; k = a["k"]
    tag = os.path.basename(a["u_file"]).split("_")[-1].split(".")[0]
    scan = load(os.path.join(P, f"data/scan_u{tag}.json"))["sizes"][str(k)]
    T, last, maxD, g0 = switch_time(curve(d))
    out.setdefault(s, []).append(dict(name=d["name"], u=tag, k=k, share=d["share"], T=T, last=last, gap0_scan=scan["gap0"],
                                      gap0_run=g0, in_fit=abs(scan["gap0"]) < 0.02, T_logged=d.get("T_dense")))
for s, rs in sorted(out.items()):
    rs.sort(key=lambda r: r["k"])
    use = [r for r in rs if r["in_fit"] and r["T"] is not None]
    print(f"seed {s} u{rs[0]['u']}: T", [r["T"] for r in rs], "logged", [r["T_logged"] for r in rs],
          "in_fit", [r["k"] for r in use], "gap0", [round(r["gap0_scan"], 4) for r in rs])
    if len(rs) == 6 and len(use) >= 4:
        b, _ = ols([math.log(r["share"]) for r in use], [math.log(r["T"]) for r in use]); slopes[s] = b
        print(f"  slope {b:.4f}")
    else:
        print(f"  no slope (sizes {len(rs)}/6, in fit {len(use)})")
if len(slopes) >= 2:
    m, sd, ci = tint(list(slopes.values()))
    print(f"S9 seed-level over n={len(slopes)}: {m:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] sd {sd:.4f}; per halving x{2**-m:.3f} [x{2**-ci[1]:.3f}, x{2**-ci[0]:.3f}]")
json.dump(dict(runs=out, slopes=slopes), open("check_s9.json", "w"), indent=1, default=str)
