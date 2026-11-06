# Vera: rebuild the check-probe numbers from raw activations (own code).
# raw/check/<run>/ref_enc.npy: (200 prompts, [read, solve], 2048) layer-24 q_norm at the answer token, leaky build.
import sys, json, numpy as np, torch
PR = "kran-research/projects/shaping-watched/"
run = sys.argv[1] if len(sys.argv) > 1 else "A_standard_s0"
X = np.load(f"{PR}raw/check/{run}/ref_enc.npy").astype(np.float64)
pv = torch.load(PR + "data/probe_v.pt", map_location="cpu", weights_only=False)
v = pv["v"].double().numpy(); b0 = float(pv["b"])
N = X.shape[0]
def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
out = {"run": run, "N": N}
s = X @ v
out["orig_auroc_all"] = auroc(s[:, 0], s[:, 1])
out["orig_margin_all"] = float(s[:, 0].mean() - s[:, 1].mean())
# Several prompt-level splits; logistic regression (L2, standardized), my own Newton-free LBFGS via torch
def fit(Xtr, ytr, l2=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = torch.tensor((Xtr - mu) / sd); y = torch.tensor(ytr, dtype=torch.float64)
    w = torch.zeros(A.shape[1], dtype=torch.float64, requires_grad=True); c = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, c], max_iter=300, line_search_fn="strong_wolfe")
    def clo():
        opt.zero_grad(); z = A @ w + c
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) + l2 * (w ** 2).sum(); loss.backward(); return loss
    opt.step(clo)
    return lambda Z: ((torch.tensor((Z - mu) / sd) @ w + c).detach().numpy()), (w.detach().numpy() / sd)
rng = np.random.default_rng(11)
res = []
for rep in range(5):
    perm = rng.permutation(N); tr, te = perm[: N // 2], perm[N // 2:]
    Xtr = np.concatenate([X[tr, 0], X[tr, 1]]); ytr = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
    f, wdir = fit(Xtr, ytr)
    zp, zn = f(X[te, 0]), f(X[te, 1])
    bal = 0.5 * ((zp > 0).mean() + (zn <= 0).mean())
    # shuffled-label null: labels permuted within the training rows
    f0, _ = fit(Xtr, rng.permutation(ytr))
    res.append({"auroc": auroc(zp, zn), "bal_acc": float(bal), "auroc_shuffled": auroc(f0(X[te, 0]), f0(X[te, 1])),
                "cos_vs_orig": float(wdir @ v / np.linalg.norm(wdir) / np.linalg.norm(v)),
                "orig_auroc_test": auroc(s[te, 0], s[te, 1])})
out["refit"] = res
out["refit_auroc_mean"] = float(np.mean([r["auroc"] for r in res]))
out["refit_shuffled_auroc_mean"] = float(np.mean([r["auroc_shuffled"] for r in res]))
# within-prompt difference size relative to between-prompt spread (is the class split just the token?)
d = X[:, 0] - X[:, 1]
out["mean_within_prompt_diff_norm"] = float(np.linalg.norm(d, axis=1).mean())
out["mean_between_prompt_spread_norm"] = float(np.linalg.norm(X[:, 0] - X[:, 0].mean(0), axis=1).mean())
print(json.dumps(out, indent=1))
json.dump(out, open(f"swarm-oracle-1106-shaping-watched/skeptic/vera_check_{run}.json", "w"), indent=1)
