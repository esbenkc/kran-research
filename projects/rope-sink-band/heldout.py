"""PREREG s6 arms on held-out sets A and B. Resumable: each arm is saved to data/<tag>/arms/<set>__<arm>.npz and
skipped when that file exists. Order: status arms (base, P, R1..R20 on A then B) first, then licence and secondary
rows. A stop file data/<tag>/STOP halts after the current arm.
Usage: python heldout.py Qwen/Qwen3-1.7B [stage ...]   stages: status licence licenceB a2 windows"""
import json
import os
import sys
import time

import numpy as np

import rsb

repo = sys.argv[1]
stages = sys.argv[2:] or ["status", "licence", "a2", "windows", "licenceB"]
tag = repo.split("/")[-1]
OUT = rsb.DATA / tag
ARMS = OUT / "arms"
ARMS.mkdir(parents=True, exist_ok=True)
band = json.load(open(OUT / "band.json"))
B, k, R, E = band["B"], band["k"], band["R"], band["E"]
tok, model = rsb.load_model(repo)
X = {s: rsb.prompts(tok, s, tag)[0] for s in ("A", "B")}
XP = {"A": rsb.prompts(tok, "A", tag, prepend=True)[0]}
N_WIN = 32  # windows row: first 32 prompts of set A (foldout row; DEVIATION for time)


N_SUB = int(os.environ.get("RSB_NSUB", "128"))  # licence E/SUF-R rows may run on the first N_SUB prompts (DEVIATION)


def queue():
    q = []
    if "core" in stages:  # licence single arms first (T, GN, SUF, LOW8) on A and B
        for s in ("A", "B"):
            q += [(s, "T", ("temp", B), {}), (s, "GN", ("gapin", B), {}), (s, "SUF", ("gapout", B), {}),
                  (s, "LOW8", ("zero", band["LOW_8"]), {})]
    if "esub" in stages:  # energy-matched and keep-only random rows on set A, first N_SUB prompts
        sn = "A" if N_SUB == 128 else f"A{N_SUB}"
        if sn != "A":
            q += [(sn, n, a, {}) for n, a in (("base", ("clean", None)), ("P", ("zero", B)), ("SUF", ("gapout", B)))]
        q += [(sn, f"E{i+1:02d}", ("zero", e), {}) for i, e in enumerate(E)]
        q += [(sn, f"SUFR{i+1:02d}", ("gapout", r), {}) for i, r in enumerate(R)]
    for st in stages:  # 'status:A' style tokens run one set, in the given order
        if st.startswith("status:"):
            s = st.split(":")[1]
            q.append((s, "base", ("clean", None), dict(direct=True, lens=True)))
            q.append((s, "P", ("zero", B), dict(lens=True)))
            q += [(s, f"R{i+1:02d}", ("zero", r), dict(lens=True)) for i, r in enumerate(R)]
        if st.startswith("core:"):
            s = st.split(":")[1]
            q += [(s, "GN", ("gapin", B), {}), (s, "LOW8", ("zero", band["LOW_8"]), {}), (s, "T", ("temp", B), {}),
                  (s, "SUF", ("gapout", B), {})]
    if "status" in stages:
        for s in ("A", "B"):
            q.append((s, "base", ("clean", None), dict(direct=True, lens=True)))
            q.append((s, "P", ("zero", B), dict(lens=True)))
            q += [(s, f"R{i+1:02d}", ("zero", r), dict(lens=True)) for i, r in enumerate(R)]
    for s in [x for x, st in (("A", "licence"), ("B", "licenceB")) if st in stages]:
        q += [(s, "T", ("temp", B), {}), (s, "GN", ("gapin", B), {}), (s, "SUF", ("gapout", B), {}),
              (s, "LOWk", ("zero", band["LOW_k"]), {}), (s, "LOW8", ("zero", band["LOW_8"]), {})]
        q += [(s, f"E{i+1:02d}", ("zero", e), {}) for i, e in enumerate(E)]
        q += [(s, f"SUFR{i+1:02d}", ("gapout", r), {}) for i, r in enumerate(R)]
    if "a2" in stages:
        q += [("A2", "base", ("clean", None), {}), ("A2", "P", ("zero", B), {})]
    if "windows" in stages:
        q += [("Aw", f"W{i:02d}", ("zero", list(range(i, i + k))), {}) for i in range(0, 64 - k + 1)]
    return q


for s, name, arm, opt in queue():
    f = ARMS / f"{s}__{name}.npz"
    if f.exists():
        continue
    if (OUT / "STOP").exists():
        print("STOP file, halting"); break
    x = XP["A"] if s == "A2" else X["A"][:N_WIN] if s == "Aw" else X["A"][:N_SUB] if s == f"A{N_SUB}" else X[s]
    if name == "LOWk" and sorted(band["LOW_k"]) == sorted(B):
        continue  # slowest-k == band: same arm as P
    rsb.DIRECT.clear()
    if opt.get("direct"):
        rsb.DIRECT["P"] = B
        for i, r in enumerate(R):
            rsb.DIRECT[f"R{i+1:02d}"] = r
    t0 = time.time()
    r = rsb.run(model, x, arm, bs=4, lens=opt.get("lens", False), loss=True)
    rsb.DIRECT.clear()
    extra = {f"direct_{kk}": v for kk, v in r["direct"].items()}
    lens = r.get("lens", np.zeros((x.shape[0], 0)))
    np.savez_compressed(f, sink=r["sink"], sink1=r["sink1"], nll=r["nll"], lens=lens, k0=np.array(r["k0"]),
                        arm_kind=arm[0], pairs=np.array(arm[1] if arm[1] is not None else [], dtype=int), **extra)
    print(f"{s} {name} kind={arm[0]} S={r['sink'].mean():.4f} sink1={r['sink1'].mean():.4f} loss={r['nll'].mean():.4f} "
          f"lens={lens.mean() if lens.size else float('nan'):.4f} k0={r['k0'][0]} {time.time()-t0:.0f}s", flush=True)
print("queue done", stages)
