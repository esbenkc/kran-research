"""T16 discovery ceiling (Iris): candidate-centric scorer on top of C1. FITTED, discovery only.

Candidates = top-K positions of the phantom-augmented order (rank 0 = max M, 1 = S, 2 = T, ...).
score_c = 8*[rank_c == 1] + MLP(phi_c)   (init ~ C1: the 2nd wins)
phi_c (homogeneous: all values divided by s = |M| + rms(x)): rank one-hot, own position one-hot (11),
max position one-hot (11), own value, M, S, T, 4th value, gaps, x/s (10).
Target: soft distribution of the model's logits over the candidates' output positions (distillation),
marginal over candidates that share an output (phantom and x_0 both decode to 0).
Train dev seed 81, test dev seed 82. Usage: .venv/bin/python residual_mlp.py --hidden 64 --K 4
"""
import argparse, json, time
import numpy as np, torch
from residual_common import data, c1, cand_features

p = argparse.ArgumentParser()
p.add_argument("--K", type=int, default=4)
p.add_argument("--hidden", type=int, default=64)
p.add_argument("--layers", type=int, default=1)
p.add_argument("--ntr", type=int, default=2_000_000)
p.add_argument("--nte", type=int, default=500_000)
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--temp", type=float, default=1.0)
p.add_argument("--drop", default="", help="comma list of feature groups to drop: xs,pos,pmax,vals")
p.add_argument("--save", default="", help="path for torch.save({state, args}) after training")
a = p.parse_args()
torch.manual_seed(0)
drop = set(a.drop.split(",")) - {""}


def prep(seed, n):
    X, L, m = data(seed, n)
    F, out = cand_features(X, a.K, drop)
    ar = np.arange(n)[:, None]
    # soft target over candidates: model logit at output pos, mass split over duplicates
    lg = L[ar, out] / a.temp
    dup = (out[:, :, None] == out[:, None, :]).sum(2)
    tgt = np.exp(lg - lg.max(1, keepdims=True)) / dup
    tgt /= tgt.sum(1, keepdims=True)
    hard = out == m[:, None]
    return X, m, out, torch.tensor(F), torch.tensor(tgt, dtype=torch.float32), torch.tensor(hard)


t0 = time.time()
Xtr, mtr, Otr, Ftr, Ttr, Htr = prep(81, a.ntr)
Xte, mte, Ote, Fte, Tte, Hte = prep(82, a.nte)
d = Ftr.shape[2]
mods = []
din = d
for _ in range(a.layers):
    mods += [torch.nn.Linear(din, a.hidden), torch.nn.ReLU()]; din = a.hidden
mods.append(torch.nn.Linear(din, 1))
net = torch.nn.Sequential(*mods)
with torch.no_grad():
    net[-1].weight.mul_(0.01); net[-1].bias.zero_()
prior = torch.zeros(a.K); prior[1] = 8.0
nparams = sum(q.numel() for q in net.parameters())
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
print(f"d={d} params={nparams} prep {time.time()-t0:.0f}s", flush=True)


def scores(F):
    return net(F).squeeze(-1) + prior


def evaluate(F, m, O):
    with torch.no_grad():
        s = torch.cat([scores(F[i:i + 100000]) for i in range(0, len(F), 100000)])
    pick = O[np.arange(len(O)), s.argmax(1).numpy()]
    return float((pick == m).mean()), pick


bs = 4096
steps_per_ep = len(Ftr) // bs
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=a.epochs * steps_per_ep)
for ep in range(a.epochs):
    perm = torch.randperm(len(Ftr))
    for i in range(steps_per_ep):
        b = perm[i * bs:(i + 1) * bs]
        logq = torch.log_softmax(scores(Ftr[b]), 1)
        loss = -(Ttr[b] * logq).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    tr, _ = evaluate(Ftr[:300000], mtr[:300000], Otr[:300000])
    te, pk = evaluate(Fte, mte, Ote)
    print(f"ep {ep} loss {loss.item():.4f} train {tr:.5f} test {te:.5f} ({time.time()-t0:.0f}s)", flush=True)

err = mte != np.argsort(Xte, 1)[:, -2]
res = dict(K=a.K, hidden=a.hidden, layers=a.layers, drop=a.drop, nparams=nparams, test=te,
           err_set=float((pk[err] == mte[err]).mean()), c1_test=float((c1(Xte) == mte).mean()), temp=a.temp)
print(json.dumps(res))
with open("residual_mlp.jsonl", "a") as f: f.write(json.dumps(res) + "\n")
if a.save:
    torch.save({"state": net.state_dict(), "args": vars(a)}, a.save)
