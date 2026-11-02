"""PREREG s6 arms on held-out sets A and B. Resumable: each arm is saved to data/<tag>/arms/<set>__<arm>.npz and
skipped when that file exists. Order: status arms (base, P, R1..R20 on A then B) first, then licence and secondary
rows. A stop file data/<tag>/STOP halts after the current arm.
Usage: python heldout.py Qwen/Qwen3-1.7B [stage ...]   stages (run in order): status[:A|:B] core:A core:B rand8 esub a2 windows"""
import json
import os
import sys
import time

import numpy as np

import rsb

repo = sys.argv[1]
stages = sys.argv[2:] or ["status:A", "core:A", "status:B", "core:B", "rand8", "esub", "a2"]
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
    """Arms in the order of the stage tokens on the command line."""
    q = []
    for st in stages:
        if st.startswith("status:") or st == "status":
            for s in ([st.split(":")[1]] if ":" in st else ["A", "B"]):
                q.append((s, "base", ("clean", None), dict(direct=True, lens=True)))
                q.append((s, "P", ("zero", B), dict(lens=True)))
                q += [(s, f"R{i+1:02d}", ("zero", r), dict(lens=True)) for i, r in enumerate(R)]
        elif st.startswith("core:"):
            s = st.split(":")[1]
            q += [(s, "GN", ("gapin", B), {}), (s, "LOW8", ("zero", band["LOW_8"]), {}), (s, "T", ("temp", B), {}),
                  (s, "SUF", ("gapout", B), {})]
        elif st == "rand8":  # exploratory (DEVIATION D8): 20 random 8-pair sets, seed 8888, first 32 prompts of set A
            rng8 = np.random.default_rng(8888)
            R8 = [sorted(rng8.choice(64, 8, replace=False).tolist()) for _ in range(20)]
            q += [("A32", f"R8_{i+1:02d}", ("zero", r), {}) for i, r in enumerate(R8)]
        elif st == "esub":  # energy-matched and keep-only random rows on set A, first N_SUB prompts (D9)
            sn = "A" if N_SUB == 128 else f"A{N_SUB}"
            # base, P, SUF for a subset are sliced from the set-A arms in analyze.py (same prompts, same batches)
            q += [(sn, f"E{i+1:02d}", ("zero", e), {}) for i, e in enumerate(E)]
            q += [(sn, f"SUFR{i+1:02d}", ("gapout", r), {}) for i, r in enumerate(R)]
        elif st == "a2":
            q += [("A2", "base", ("clean", None), {}), ("A2", "P", ("zero", B), {})]
        elif st == "windows":
            q += [("Aw", f"W{i:02d}", ("zero", list(range(i, i + k))), {}) for i in range(0, 64 - k + 1)]
        else:
            raise SystemExit(f"unknown stage {st}")
    return q


for s, name, arm, opt in queue():
    f = ARMS / f"{s}__{name}.npz"
    if f.exists():
        continue
    if (OUT / "STOP").exists():
        print("STOP file, halting"); break
    x = XP["A"] if s == "A2" else X["A"][:N_WIN] if s == "Aw" else X["A"][:N_SUB] if s == f"A{N_SUB}" else X["A"][:32] if s == "A32" else X[s]
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
