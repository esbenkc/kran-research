"""Report per-prompt bit diffs from vera_bits.py outputs (raw int views -> bf16/fp32 values)."""
import sys
import numpy as np
import torch

def vals(a):
    t = torch.from_numpy(a)
    dt = torch.bfloat16 if a.dtype == np.int16 else torch.float32
    return t.view(dt).float().numpy()

f = np.load(sys.argv[1])
conds = sorted({k.split("__")[0] for k in f.files})
n = len({k.split("__")[1] for k in f.files})
ref = sys.argv[2] if len(sys.argv) > 2 else "nohook_a"
for c in conds:
    rows = []
    for i in range(n):
        a, b = f[f"{ref}__{i}"], f[f"{c}__{i}"]
        d = np.abs(vals(a) - vals(b))
        last = np.abs(vals(a[-1]) - vals(b[-1]))
        rows.append((a.shape[0], int((a != b).sum()), float(d.max()), float(last.max()),
                     int(vals(a[-1]).argmax() != vals(b[-1]).argmax())))
    changed = [i for i, r in enumerate(rows) if r[1] > 0]
    print(f"{c:10s} changed prompts {len(changed)}/{n} idx={changed} "
          f"maxabs(all pos)={max(r[2] for r in rows):.4g} maxabs(last pos)={max(r[3] for r in rows):.4g} "
          f"top1 flips(last)={sum(r[4] for r in rows)} seq_lens={sorted({r[0] for r in rows})}")
