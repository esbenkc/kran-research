"""T15 step 1 (Felix): pattern-oracle decomposition. Dev seeds only.

Forced-mask forward: h_{t+1} = m_{t+1} * (W_hh h_t + W_hi x_t). With the true masks this equals the
model exactly (logits are linear in x given the pattern). We replace the masks of chosen neurons by
(a) the per-(neuron,t) majority value, (b) always-on, and report agreement with the model.
"""
import sys
import numpy as np
sys.path.insert(0, "..")
sys.path.insert(0, ".")
from core import load_raw, sample, run

W_hi, W_hh, W_oh = load_raw()
X = sample(200_000, 41)
logits, H = run(X, W_hi, W_hh, W_oh, return_states=True)
ref = logits.argmax(1)
M = (H[:, 1:] > 0)  # (B,10,16) true masks for h_1..h_10


def forced(X, masks):
    B = X.shape[0]
    h = np.zeros((B, 16))
    for t in range(10):
        h = masks[:, t] * (h @ W_hh.T + X[:, t:t + 1] * W_hi[None, :])
    return (h @ W_oh.T).argmax(1)


assert (forced(X, M) == ref).all()
rate = M.mean(0)  # (10,16)
print("active rate per neuron (mean over t):", np.round(rate.mean(0), 3))
maj = rate > 0.5
for j in range(16):
    m1 = M.copy(); m1[:, :, j] = maj[None, :, j]
    m2 = M.copy(); m2[:, :, j] = True
    a1 = (forced(X, m1) == ref).mean(); a2 = (forced(X, m2) == ref).mean()
    print(f"n{j:2d} rate {rate[:, j].mean():.3f}  majority-mask {a1:.4f}  always-on {a2:.4f}")
