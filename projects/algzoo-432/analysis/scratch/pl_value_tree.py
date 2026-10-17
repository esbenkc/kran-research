"""T15 step 3 (Felix): two more ceilings for the pattern-linear idea. Dev seeds 41/42 only.

(a) Masks from a few interpretable streaming VALUES (not only order): x_t, x_{t-1}, x_{t-2}, running max M
    and 2nd max S of {0, x_0..x_t}, leaky max L (decay 0.969, n2), steps since M, steps since S.
    Depth-10 tree per (neuron, t). Forced-mask forward with the exact weights -> agreement.
(b) Logits as one linear map of x per rank class (pM, pS) [phantom = -1], fit by least squares on the
    model's logits (90+ classes x 10 x 10 coefficients; fitted, discovery only) -> agreement.
    Also per class (pM, pS, pT).
"""
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
sys.path.insert(0, "..")
from core import load_raw, sample, run

W_hi, W_hh, W_oh = load_raw()


def forced(X, masks):
    h = np.zeros((X.shape[0], 16))
    for t in range(10):
        h = masks[:, t] * (h @ W_hh.T + X[:, t:t + 1] * W_hi[None, :])
    return (h @ W_oh.T).argmax(1)


def stream_feats(X):
    B = X.shape[0]
    M = np.zeros(B); S = np.full(B, -9.0); pM = np.full(B, -1); pS = np.full(B, -1); L = np.zeros(B)
    F = []
    for t in range(10):
        x = X[:, t]
        nm = x > M; ns = ~nm & (x > S)
        S = np.where(nm, M, np.where(ns, x, S)); pS = np.where(nm, pM, np.where(ns, t, pS))
        M = np.where(nm, x, M); pM = np.where(nm, t, pM)
        L = np.maximum(0.969 * L, x)
        x1 = X[:, t - 1] if t >= 1 else np.zeros(B)
        x2 = X[:, t - 2] if t >= 2 else np.zeros(B)
        F.append(np.stack([x, x1, x2, M, S, L, t - pM, t - pS, M - x, S - x], 1))
    return F, pM, pS


def ranks_classes(X):
    P = np.concatenate([np.zeros((X.shape[0], 1)), X], 1)
    o = np.argsort(-P, 1) - 1  # positions, phantom = -1
    return o[:, 0], o[:, 1], o[:, 2]


Xtr, Xte = sample(300_000, 41), sample(200_000, 42)
lgtr, Htr = run(Xtr, W_hi, W_hh, W_oh, return_states=True)
lgte, Hte = run(Xte, W_hi, W_hh, W_oh, return_states=True)
ref = lgte.argmax(1)
Mtr, Mte = Htr[:, 1:] > 0, Hte[:, 1:] > 0

# (a)
Ftr, _, _ = stream_feats(Xtr); Fte, _, _ = stream_feats(Xte)
Pred = np.zeros_like(Mte)
for t in range(10):
    for j in range(16):
        y = Mtr[:, t, j]
        if y.all() or (~y).all():
            Pred[:, t, j] = y[0]; continue
        Pred[:, t, j] = DecisionTreeClassifier(max_depth=10, min_samples_leaf=20).fit(Ftr[t], y).predict(Fte[t])
print("(a) value-feature masks, all neurons:", (forced(Xte, Pred) == ref).mean())
for name, S in [("A 1,2,4,6,7,8", [1, 2, 4, 6, 7, 8]), ("gates 0,5,9", [0, 5, 9]), ("B 3,10-15", [3, 10, 11, 12, 13, 14, 15])]:
    m = Mte.copy(); m[:, :, S] = Pred[:, :, S]
    print(f"    only {name}: {(forced(Xte, m) == ref).mean():.4f}")

# (b)
for keyname, k in [("(pM,pS)", 2), ("(pM,pS,pT)", 3)]:
    a, b, c = ranks_classes(Xtr); key_tr = (a + 1) * 121 + (b + 1) * 11 + (c + 1 if k == 3 else 0)
    a, b, c = ranks_classes(Xte); key_te = (a + 1) * 121 + (b + 1) * 11 + (c + 1 if k == 3 else 0)
    Xa_tr = Xtr; Xa_te = Xte
    pred = np.full(len(Xte), -1)
    ncls = 0
    for kk in np.unique(key_te):
        itr = key_tr == kk; ite = key_te == kk
        if itr.sum() < 30:
            pred[ite] = np.maximum(b if False else ranks_classes(Xte[ite])[1], 0); continue
        coef, *_ = np.linalg.lstsq(Xa_tr[itr], lgtr[itr], rcond=None)
        pred[ite] = (Xa_te[ite] @ coef).argmax(1); ncls += 1
    print(f"(b) linear logits per {keyname} class ({ncls} classes, {ncls * 100} fitted coefs): {(pred == ref).mean():.4f}")
