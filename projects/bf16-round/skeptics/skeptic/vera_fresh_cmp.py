"""Vera: compare my own whole-graph run of the fresh tail on round16 (vera_round.py) with build's saved json/last."""
import json, numpy as np
D = "/Users/esben/work/oracle-1023-bf16-round/kran-research/projects/bf16-round/raw/detectors/Qwen3-1.7B/emu0/"
m = np.load("vera_round_fresh_test256.npz")
j = json.load(open(D + "round16__fresh_round16_probe_none_s0.json")); YES, NO = j["yes_id"], j["no_id"]
bl = np.load(D + "round16__fresh_round16_probe_none_s0_last.npy")
print("build last shape", bl.shape, bl.dtype)
bf = lambda a: (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)
ids = list(m["ids"]); off, on = m["off_last"], m["on_last"]
t0, t1 = bf(off).argmax(1), bf(on).argmax(1)
def acc(sl): return (np.mean(t0[sl] == NO) + np.mean(t1[sl] == YES)) / 2
print(f"mine: acc all 256 {acc(slice(0,256)):.4f}; first 128 {acc(slice(0,128)):.4f}; unseen 128-255 {acc(slice(128,256)):.4f}")
lo = lambda a: (lambda l: l[:, YES] - l[:, NO])(bf(a))
d = lo(on) - lo(off)
print(f"mine: shift toward Yes >0 on {int((d>0).sum())}/256, =0 {int((d==0).sum())}; per-prompt hash differs on {int((m['off_hash']!=m['on_hash']).any(1).sum())}/256")
# build rows order
rows = j["rows"]; bid = [r["prompt"] for r in rows if r["probe"] == 0]
assert bid == ids[:128], "prompt order differs"
b0 = np.array([r["top"] for r in rows if r["probe"] == 0]); b1 = np.array([r["top"] for r in rows if r["probe"] == 1])
print(f"top token agreement with build, first 128: unprobed {int((b0==t0[:128]).sum())}/128, probed {int((b1==t1[:128]).sum())}/128")
# bit-level answer-position logits vs build's saved
if bl.ndim == 3:
    for k, mine in ((0, off), (1, on)):
        eq = int((bl[:, k] == mine[:128].view(bl.dtype)).all(1).sum()); print(f"answer logits bit-identical to build (probe={k}): {eq}/128")
