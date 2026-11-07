"""Vera: rebuild every run's headline numbers from data/runs/<name>.json raw rows + loss arrays only.
Per run: strict switch at every dev eval from rows (top-1 A on probed AND top-1 B on unprobed), RETURNED rule
(>= 0.75 on 2 consecutive dev evals; return time = 2nd eval of the pair), final test strict (merged + clamp),
reversed share n, CONVERGED rule (mean train loss of last 20% vs 20% before, |diff| <= 0.02, AND last 3 dev evals
within 0.10), then the frozen TITLE STATUS and PREREG OUTCOME. Flags any mismatch with the stored summary fields."""
import json, glob, os, collections, numpy as np
P = "/Users/esben/work/theory-1107-near-tie-clamp/kran-research/projects/head-clamp-retrain/data/runs/"
A, B = 32, 33
FORM = {0: 1280, 1: 1408, 2: 1280}

def strict(rows):
    by = collections.defaultdict(dict)
    for pid, probe, top1, la, lb in rows: by[pid][probe] = top1
    N = len(by)
    s = sum(d[1] == A and d[0] == B for d in by.values()) / N
    n = sum(d[1] != A and d[0] == A for d in by.values()) / N
    return s, n, N

def converged(loss, curve):
    L = np.asarray(loss, float); k = max(1, int(round(0.2 * len(L))))
    last, prev = L[-k:].mean(), L[-2 * k:-k].mean()
    dev3 = [c for _, c in curve[-3:]]
    return bool(abs(last - prev) <= 0.02 and (max(dev3) - min(dev3)) <= 0.10), float(last - prev), float(max(dev3) - min(dev3))

out = {}
for f in sorted(glob.glob(P + "*.json")):
    d = json.load(open(f)); name = d["name"]; seed = d["args"]["seed"]; arm = d["args"]["arm"]
    R = d["rows"]
    curve = [(0, strict(R["t0/dev_clamp"])[0])]
    for k in sorted([k for k in R if k.startswith("step") and k.endswith("/dev_clamp")], key=lambda k: int(k[4:].split("/")[0])):
        curve.append((int(k[4:].split("/")[0]), strict(R[k])[0]))
    ret = None
    for (u1, c1), (u2, c2) in zip(curve, curve[1:]):
        if c1 >= 0.75 and c2 >= 0.75: ret = u2; break
    fin_s, fin_n, N = strict(R["final/test_merged_clamp"])
    nocl_s, _, _ = strict(R["final/test_merged_noclamp"])
    t0_s, _, _ = strict(R["t0/test_clamp"])
    conv, dloss, ddev = converged(d["loss"], curve)
    stored_curve = [(c["step"], c["switch_correct"]) for c in d["dev_curve"]]
    mism = [f"curve@{u}: mine {c:.4f} stored {sc:.4f}" for (u, c), (su, sc) in zip(curve, stored_curve) if abs(c - sc) > 1e-9 or u != su]
    if abs(fin_s - d["test_merged_clamp"]["switch_correct"]) > 1e-9: mism.append(f"final mine {fin_s} stored {d['test_merged_clamp']['switch_correct']}")
    if (ret is not None) != (d["stop_reason"] == "returned"): mism.append(f"stop rule mine {ret} stored {d['stop_reason']}")
    out[name] = dict(seed=seed, arm=arm, heads=d["heads"], steps_run=d["steps_run"], stop=d["stop_reason"], returned_at=ret,
                     formation=FORM[seed], t0_test=t0_s, final_test_clamp=fin_s, final_reversed=fin_n, final_test_noclamp=nocl_s,
                     converged=conv, dloss_last20=dloss, ddev_last3=ddev, loss_first=float(d["loss"][0]),
                     loss_last128=float(np.mean(d["loss"][-128:])), curve=curve, mismatches=mism)
    print(f"{name:<16} heads={len(d['heads']):>2} steps={d['steps_run']:>4} stop={d['stop_reason']:<8} ret@{ret} (form {FORM[seed]}) "
          f"t0={t0_s:.3f} final(clamp)={fin_s:.3f} n={fin_n:.3f} noclamp={nocl_s:.3f} conv={conv} dL={dloss:+.3f} "
          f"loss_last128={np.mean(d['loss'][-128:]):.3f} {'MISMATCH ' + '; '.join(mism) if mism else 'match'}")

def status(arm):
    cells = {v["seed"]: v for v in out.values() if v["arm"] == arm}
    if len(cells) < 2: return "PENDING", {s: (round(v["final_test_clamp"], 3), v["returned_at"]) for s, v in cells.items()}
    hold = sum(v["final_test_clamp"] < 0.25 and v["steps_run"] >= 4096 and v["converged"] for v in cells.values())
    back = sum(v["final_test_clamp"] >= 0.25 or v["returned_at"] is not None for v in cells.values())
    st = "CONFIRMED" if hold >= 2 else "REJECTED" if back >= 2 else ("UNCONFIRMED" if len(cells) == 3 else "PENDING")
    return st, {s: (round(v["final_test_clamp"], 3), v["returned_at"]) for s, v in cells.items()}
for arm in ("set", "top1", "random", "all32", "leave1"):
    st, c = status(arm)
    print(f"arm {arm:<7} {st} {c}")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "vera_runs.json"), "w"), indent=1)
