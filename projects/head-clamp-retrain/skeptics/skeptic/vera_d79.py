"""Vera: own SHA256 of the rebuilt block-26 cache vs the parent manifest, and d79 = share of TEST prompts whose
block-26 input at position 79 differs in ANY bit between probe on and off (plus per-position share and how many of
the 2048 dims differ). numpy memmap only (<1 GB)."""
import hashlib, json, sys, numpy as np
P = "/Users/esben/work/theory-1107-near-tie-clamp/kran-research/projects/head-clamp-retrain/"
PAR = "/Users/esben/work/oracle-1024-probed-only-switch/kran-research/projects/probe-switch-heads/data/manifest_raw.json"
C = P + "raw/step2/Qwen3-1.7B/compile_C/"
def sha(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
man = json.load(open(PAR))
out = {}
for k in ("none", "probe"):
    f = C + f"{k}_rec_tail.npy"
    s = sha(f)
    par = [v for kk, v in (man.items() if isinstance(man, dict) else []) if f"{k}_rec_tail" in kk]
    out[f"sha_{k}"] = s; out[f"parent_{k}"] = str(par)[:200]
    print(k, s, "parent:", str(par)[:120])
none = np.load(C + "none_rec_tail.npy", mmap_mode="r"); probe = np.load(C + "probe_rec_tail.npy", mmap_mode="r")
print("shape", none.shape, none.dtype)
ids = [json.loads(l) for l in open(P + "data/prompts.jsonl")]
test = [i for i, r in enumerate(ids) if r.get("split") == "test"]; dev = [i for i, r in enumerate(ids) if r.get("split") == "dev"]
print("n test", len(test), "n dev", len(dev))
for name, rows in (("test", test), ("dev", dev)):
    a = np.asarray(none[rows, 79, :]); b = np.asarray(probe[rows, 79, :])
    diff = (a != b)
    d79 = diff.any(1).mean(); ndim = diff.sum(1)
    print(f"{name}: d79 = {d79:.4f} ({diff.any(1).sum()}/{len(rows)}); dims differing per prompt median {np.median(ndim):.0f} max {ndim.max()}")
    out[f"d79_{name}"] = float(d79); out[f"ndim_median_{name}"] = float(np.median(ndim))
# per-position share of test prompts with any bit differing
pos = [float((np.asarray(none[test, p, :]) != np.asarray(probe[test, p, :])).any(1).mean()) for p in range(none.shape[1])]
print("first position with any difference:", next((p for p, v in enumerate(pos) if v > 0), None), "share at pos 0/40/71/79:", pos[0], pos[40], pos[71], pos[79])
out["per_pos_test"] = pos
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
