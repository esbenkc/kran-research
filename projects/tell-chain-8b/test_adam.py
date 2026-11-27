"""tc.py's per-slot Adam vs torch.optim.AdamW(weight_decay=0): 3 slots, slot 2 joins late with fresh state.
Each slot must match its own torch AdamW run (one optimizer per slot) to float32 rounding. Exit 1 on mismatch.

    NC_DEV=cpu .venv/bin/python test_adam.py
"""
import numpy as np
import torch

from tc import adam_update

torch.manual_seed(0)
C, steps_total, join = 3, 40, 17
shapes = [(C, 16, 64), (C, 48, 16)]
params = [torch.nn.Parameter(torch.randn(*s)) for s in shapes]
M = [torch.zeros_like(p) for p in params]
V = [torch.zeros_like(p) for p in params]
st = np.zeros(C)
ref = {k: [torch.nn.Parameter(p.detach()[k].clone()) for p in params] for k in range(C)}
opt = {k: torch.optim.AdamW(ref[k], lr=1e-4, weight_decay=0.0, foreach=False) for k in range(C)}
grads = [[torch.randn(*s) for s in shapes] for _ in range(steps_total)]
worst = 0.0
for t in range(steps_total):
    act = [0, 1] + ([2] if t >= join else [])
    if t == join:  # slot 2 restarts: copy slot 0's params, fresh Adam (tc.py copy_slot)
        with torch.no_grad():
            for p, m, v in zip(params, M, V):
                p[2].copy_(p[0]); m[2].zero_(); v[2].zero_()
        st[2] = 0
        ref[2] = [torch.nn.Parameter(p.detach()[0].clone()) for p in params]
        opt[2] = torch.optim.AdamW(ref[2], lr=1e-4, weight_decay=0.0, foreach=False)
    idx = torch.tensor(act)
    for p, gr in zip(params, grads[t]):
        p.grad = torch.zeros_like(p)
        p.grad[idx] = gr[idx]
    st[act] += 1
    adam_update(params, M, V, st[act], idx, 1e-4)
    for k in act:
        for q, gr in zip(ref[k], grads[t]):
            q.grad = gr[k].clone()
        opt[k].step()
    for k in range(C):
        for p, q in zip(params, ref[k]):
            worst = max(worst, float((p.detach()[k] - q.detach()).abs().max()))
print(f"max |param diff| over {steps_total} steps, 3 slots (slot 2 joins at {join}): {worst:.3e}")
raise SystemExit(0 if worst < 1e-7 else 1)
