import numpy as np, time
from vera_mcc import mcc, mcc_hungarian, normalise, train_topk
rng = np.random.default_rng(0)
N, L, d = 60000, 64, 48
Z = np.zeros((N, L)); 
for r in range(N): Z[r, rng.choice(L, 3, replace=False)] = 1
D = rng.normal(size=(L, d)); D /= np.linalg.norm(D, axis=1, keepdims=True)
X = Z @ D * (1 + 0.2 * rng.random((N, 1))) + 0.05 * rng.normal(size=(N, d))
Xn = normalise(X, X[:50000])
t = time.time(); enc, W, fvu = train_topk(Xn[:50000], steps=3000, k=3); 
print("planted recovery", mcc(enc(Xn[50000:]), Z[50000:]).mean().round(3), "fvu", round(fvu, 3), round(time.time()-t, 1), "s")
print("random-feature floor (256 gaussian relu feats)", mcc(np.maximum(Xn[50000:] @ rng.normal(size=(d, 256)), 0), Z[50000:]).mean().round(3))
print("hungarian planted", mcc_hungarian(enc(Xn[50000:]), Z[50000:]).mean().round(3)); print("pure-noise floor", mcc(rng.normal(size=(10000, 256)), Z[50000:]).mean().round(3))
