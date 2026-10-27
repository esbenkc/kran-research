"""Vera: are the in-loop probes refit on the current model with labels? Compare saved probes (s0, s300) to my own
fits (StandardScaler + L2 logistic, C.json, attack 600, true labels) on clean / final ctok: cosine of w/sd (raw-space direction)."""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import vera_state as V
Cj = json.load(open(V.P + "data/fits/C.json"))
z = np.load(V.P + "data/runs/refit50__lam10000__seed0__probes.npz")
def mine(path):
    A = np.load(path, mmap_mode="r"); W = []
    for j, R in enumerate(V.READ):
        X = np.asarray(A[:, j, :], np.float64)[V.ATT]
        sc = StandardScaler().fit(X); clf = LogisticRegression(C=Cj[f"tok{R}"]["C"], max_iter=5000).fit(sc.transform(X), V.y[V.ATT])
        W.append(clf.coef_[0] / sc.scale_)
    return np.array(W)
def cos(a, b): return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
saved = {s: z[f"s{s}__w"] / z[f"s{s}__sd"] for s in (0, 50, 150, 300)}
c0 = mine(V.P + "data/ctok/clean_1026.npy"); c3 = mine(V.P + "data/ctok/refit50__lam10000__seed0.npy")
print("s0 vs my clean fit  cos min/median", cos(saved[0], c0).min().round(4), np.median(cos(saved[0], c0)).round(4))
print("s300 vs my R1 fit   cos min/median", cos(saved[300], c3).min().round(4), np.median(cos(saved[300], c3)).round(4))
print("s300 vs s0          cos median", np.median(cos(saved[300], saved[0])).round(4), " s50 vs s0", np.median(cos(saved[50], saved[0])).round(4))
print("my R1 fit vs my clean fit cos median", np.median(cos(c3, c0)).round(4))
