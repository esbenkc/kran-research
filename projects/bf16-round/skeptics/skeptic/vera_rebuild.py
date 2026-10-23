"""Vera's rebuild of the bit table from build's saved raw files only (no build code read).

For every <fix>_<noprobe|probe>_<A|B> found in a raw/bits*/<model>/<setting>/ dir:
 - check the sha definition: recompute SHA256 of each all-position logit row (pass 1) and compare to sha.json[1]
 - changed prompts probe vs noprobe, per pass, from the shas (any position differs) and from last-position logits
 - determinism floor: pass vs pass inside one graph, A vs B compilation
 - final-hidden (all positions) changed prompts, max |d last logit|
Usage: python vera_rebuild.py RAWDIR [RAWDIR...]
"""
import sys, json, glob, os, hashlib, re
import numpy as np

def bf(a):
    return (np.asarray(a).astype(np.uint16).astype(np.uint32) << 16).view(np.float32)

def load(d, g):
    out = {}
    p = f"{d}/{g}_sha.json"
    if os.path.exists(p): out["sha"] = np.array(json.load(open(p)))
    for k in ("last", "hidden"):
        p = f"{d}/{g}_{k}.npy"
        if os.path.exists(p): out[k] = np.load(p, mmap_mode="r")
    p = f"{d}/{g}_alllogits_pass1.npy"
    if os.path.exists(p): out["all1"] = np.load(p, mmap_mode="r")
    return out

def nchg_sha(a, b):  # a,b: (P, T) arrays of hex strings
    return int((a != b).any(1).sum())

for d in sys.argv[1:]:
    gs = sorted({re.sub(r"_(sha\.json|last\.npy|hidden\.npy|alllogits_pass1\.npy)$", "", os.path.basename(f))
                 for f in glob.glob(f"{d}/*_*_[AB]_*")})
    data = {g: load(d, g) for g in gs}
    print(f"== {d}  graphs: {gs}")
    # sha definition check
    for g, x in data.items():
        if "all1" in x and "sha" in x:
            a = x["all1"]; ok = 0; tot = 0
            for i in range(0, a.shape[0], 8):
                for t in range(a.shape[1]):
                    tot += 1
                    ok += hashlib.sha256(np.ascontiguousarray(a[i, t]).tobytes()).hexdigest() == x["sha"][1][i][t]
            print(f"  sha check {g}: recomputed == saved on {ok}/{tot} rows (every 8th prompt, pass 1)")
            assert np.array_equal(np.asarray(a[:, -1]), np.asarray(x["last"][1])), "last != all1[:, -1]"
    fixes = sorted({g.rsplit("_", 2)[0] for g in gs})
    for f in fixes:
        for comp in "AB":
            n, p = data.get(f"{f}_noprobe_{comp}"), data.get(f"{f}_probe_{comp}")
            if not n or not p or "sha" not in n or "sha" not in p: continue
            S_n, S_p = n["sha"], p["sha"]
            npass = S_n.shape[0]
            chg = [nchg_sha(S_n[k], S_p[k]) for k in range(npass)]
            det_n = [nchg_sha(S_n[0], S_n[k]) for k in range(1, npass)]
            det_p = [nchg_sha(S_p[0], S_p[k]) for k in range(1, npass)]
            ln, lp = np.asarray(n["last"]), np.asarray(p["last"])
            chg_last = [int((ln[k] != lp[k]).any(-1).sum()) for k in range(npass)]
            dmax = float(np.abs(bf(ln[1]) - bf(lp[1])).max())
            frac = float((ln[1] != lp[1]).mean())
            hid = ""
            if "hidden" in n and "hidden" in p:
                hid = f" hidden-changed {int((np.asarray(n['hidden']) != np.asarray(p['hidden'])).reshape(len(n['hidden']), -1).any(1).sum())}"
            print(f"  {f:10s} {comp}: changed prompts per pass (sha, all pos) {chg}; last-pos {chg_last}; "
                  f"det noprobe {det_n} probe {det_p}; max|dlast| p1 {dmax:.4f}, frac last logits changed {frac:.3f}{hid}")
        a, b = data.get(f"{f}_noprobe_A"), data.get(f"{f}_noprobe_B")
        if a and b and "sha" in a and "sha" in b:
            print(f"  {f:10s} noprobe A vs B (pass1): {nchg_sha(a['sha'][1], b['sha'][1])} changed")
        a, b = data.get(f"{f}_probe_A"), data.get(f"{f}_probe_B")
        if a and b and "sha" in a and "sha" in b:
            print(f"  {f:10s} probe   A vs B (pass1): {nchg_sha(a['sha'][1], b['sha'][1])} changed")
    # fidelity vs stock (unprobed, same setting)
    if "none_noprobe_A" in data:
        r = data["none_noprobe_A"]
        for f in fixes:
            x = data.get(f"{f}_noprobe_A")
            if f == "none" or not x or "sha" not in x: continue
            d_ = float(np.abs(bf(np.asarray(r["last"][1])) - bf(np.asarray(x["last"][1]))).max())
            print(f"  fidelity {f} noprobe vs none noprobe (pass1): {nchg_sha(r['sha'][1], x['sha'][1])} changed, max|dlast| {d_:.4f}")
