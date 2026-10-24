"""Vera: compare my own whole-graph run (vera_whole_s2.npz) with build's rows for s2, per prompt."""
import json, hashlib, numpy as np
P = "/Users/esben/work/oracle-1024-probed-only-switch/kran-research/projects/probe-switch-heads/raw/step2/Qwen3-1.7B/whole/lora_none_s2__emu0.json"
z = np.load("vera_whole_s2.npz"); ids = list(z["ids"])
bf = lambda a: (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)
rows = {(r["prompt"], r["probe"]): r for r in json.load(open(P))["rows"]}
for cond, p in (("off", 0), ("on", 1)):
    last = z[cond + "_last"]; f = bf(last)
    top = f.argmax(1); lo = f[:, 32] - f[:, 33]
    b_top = np.array([rows[(i, p)]["top"] for i in ids]); b_lo = np.array([rows[(i, p)]["lo"] for i in ids])
    sha_eq = sum(hashlib.sha256(last[j].astype(np.uint16).tobytes()).hexdigest() == rows[(i, p)]["logits_sha256"] for j, i in enumerate(ids))
    print(cond, "top equal", int((top == b_top).sum()), "/128; max |lo mine - lo build|", float(np.abs(lo - b_lo).max()),
          "; sha256(bf16 bytes) equal", sha_eq)
t0 = bf(z["off_last"]).argmax(1); t1 = bf(z["on_last"]).argmax(1)
print("mine: pair both-right", int(((t1 == 32) & (t0 == 33)).sum()), "/128, differs", int((t1 != t0).sum()),
      "mean shift", float(((bf(z["on_last"])[:, 32] - bf(z["on_last"])[:, 33]) - (bf(z["off_last"])[:, 32] - bf(z["off_last"])[:, 33])).mean()))
