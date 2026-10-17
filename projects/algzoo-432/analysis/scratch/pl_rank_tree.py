"""T15 step 2 (Felix): can the ReLU pattern be predicted from the ORDER of the prefix alone?

For each (neuron j, step t) fit a decision tree on order features of the prefix {0 (phantom), x_0..x_t}
(rank of each element in the prefix, rank of the phantom). Discovery only, dev seeds 41/42.
Then run the forced-mask linear model with tree-predicted masks for chosen neuron sets.
"""
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
sys.path.insert(0, "..")
from core import load_raw, sample, run

W_hi, W_hh, W_oh = load_raw()
DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def masks_of(X):
    lg, H = run(X, W_hi, W_hh, W_oh, return_states=True)
    return lg.argmax(1), H[:, 1:] > 0


def order_feats(X, t):
    P = np.concatenate([np.zeros((X.shape[0], 1)), X[:, :t + 1]], 1)  # phantom first
    r = P.argsort(1).argsort(1).astype(float)  # rank within prefix (0 = smallest)
    r = r[:, :] - r.shape[1] + 1  # rank from the top: 0 = largest, -1 = second, ...
    pad = np.full((X.shape[0], 10 - t - 1), 99.0)
    return np.concatenate([r, pad], 1)


def forced(X, masks):
    h = np.zeros((X.shape[0], 16))
    for t in range(10):
        h = masks[:, t] * (h @ W_hh.T + X[:, t:t + 1] * W_hi[None, :])
    return (h @ W_oh.T).argmax(1)


Xtr, Xte = sample(200_000, 41), sample(200_000, 42)
_, Mtr = masks_of(Xtr)
ref, Mte = masks_of(Xte)
Pred = np.zeros_like(Mte)
acc = np.zeros((10, 16))
for t in range(10):
    Ftr, Fte = order_feats(Xtr, t), order_feats(Xte, t)
    for j in range(16):
        y = Mtr[:, t, j]
        if y.all() or (~y).all():
            Pred[:, t, j] = y[0]
        else:
            clf = DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=20).fit(Ftr, y)
            Pred[:, t, j] = clf.predict(Fte)
        acc[t, j] = (Pred[:, t, j] == Mte[:, t, j]).mean()
np.set_printoptions(linewidth=200)
print(f"depth {DEPTH}: mask accuracy per neuron (min over t / mean over t)")
for j in range(16):
    print(f"n{j:2d} min {acc[:, j].min():.3f} mean {acc[:, j].mean():.3f}")
print("all masks predicted:", (forced(Xte, Pred) == ref).mean())
A = [1, 2, 4, 6, 7, 8]
G = [0, 5, 9]
B = [3, 10, 11, 12, 13, 14, 15]
for name, S in [("A", A), ("gates 0,5,9", G), ("B rest", B), ("A+gates", A + G)]:
    m = Mte.copy(); m[:, :, S] = Pred[:, :, S]
    print(f"predict only {name}: {(forced(Xte, m) == ref).mean():.4f}")
for j in range(16):
    m = Mte.copy(); m[:, :, j] = Pred[:, :, j]
    print(f"predict only n{j}: {(forced(Xte, m) == ref).mean():.4f}")
