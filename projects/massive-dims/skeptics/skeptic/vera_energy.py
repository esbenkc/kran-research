"""Vera: pre-patch offset-energy shares at the block-26 input of the untrained Qwen3-0.6B (own code, numpy only).
Grader switch: my oracle-1030 cache (eager = grading/off, comp = serving/on), 646 prompts.
Monitor switch: my vera_h26_1025.npz (none = off, ref = leaky probe on), 200 test prompts.
Massive set rule (frozen by the room): dims with |x| > 100x median |x| at any of tokens 0-2.
"""
import glob, json, os, struct
import numpy as np
def bf(a): return (a.astype(np.int16).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
z = np.load("/Users/esben/work/oracle-1030-stack-conditional/swarm-oracle-1030-stack-conditional/skeptic/vera_base.npz")
q = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_h26_1025.npz"))
f = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors"))[0]
def st(name):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n)); m = h[name]; a, b = m["data_offsets"]
        fh.seek(8 + n + a); raw = np.frombuffer(fh.read(b - a), dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(m["shape"])
w = st("model.layers.26.input_layernorm.weight").astype(np.float64)
def rmsn(x): x = x.astype(np.float64); return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6) * w
for name, off, on in (("grader", bf(z["eager_h26"]), bf(z["comp_h26"])), ("monitor", bf(q["none_h26"]), bf(q["ref_h26"]))):
    A = np.abs(off[:, :3]); gm = np.median(np.abs(off))
    pm = np.median(A, axis=2, keepdims=True)
    per_tok = sorted(set(np.where((A > 100 * pm).any((0, 1)))[0].tolist()))
    glob_ = sorted(set(np.where((A > 100 * gm).any((0, 1)))[0].tolist()))
    print(f"[{name}] global median |x| {gm:.3f}; massive (per-token median) {per_tok}; massive (global median) {glob_}")
    M = [1, 13, 35]
    for p in range(3):
        print(f"[{name}]   token {p}: mean |x| dims 1/13/35 = {[round(float(np.abs(off[:, p, d]).mean()), 1) for d in M]}, "
              f"unique offsets dims 1/13/35 = {[np.unique(on[:, p, d] - off[:, p, d]).tolist()[:4] for d in M]}")
    for tag, d in (("pre-norm", on[:, :3].astype(np.float64) - off[:, :3]), ("post-norm", rmsn(on[:, :3]) - rmsn(off[:, :3]))):
        e = d ** 2; tot = e.sum(); ed = e.sum((0, 1)).copy()
        ms = e[:, :, M].sum() / tot
        ed2 = ed.copy(); ed2[M] = -1; top = np.argsort(-ed2)[:3]
        rng = np.random.default_rng(0); nm = np.setdiff1d(np.arange(1024), M)
        rs = np.array([ed[rng.choice(nm, 3, replace=False)].sum() / tot for _ in range(999)])
        print(f"[{name}] {tag}: token shares {[round(float(e[:, p].sum() / tot), 4) for p in range(3)]}; massive {{1,13,35}}@0-2 {ms:.4f}; "
              f"top-3 non-massive {top.tolist()} {ed[top].sum() / tot:.4f}; random 3-dim sets median {np.median(rs):.5f} max {rs.max():.5f}")
