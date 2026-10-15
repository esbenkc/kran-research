"""skeptic3: independent re-run of the K-FAC attacker (own K-FAC, own gauge, own data loop, own batch order).

Uses only DATA from build: results/ckpt/unlearned_seed{S}.pt and results/raw/seed{S}/facts.json (or the path given).
K-FAC: all Linear layers inside transformer blocks (weights+bias), A = E[a a^T] (with bias 1), G = E[g g^T] of
per-token output grads (scaled by batch count), EMA 0.9, inverse every 10 steps (f64 on CPU), damping:
qkv G per 64-row block (one head's q, k or v) lam*mean diag(block); every other factor lam*mean diag (isotropic).
Plus SGD / AdamW modes for the grid-resolution null.

Run from kran-research/:
  uv run python ../swarm-oracle-1015-gauge-tamper-resist/skeptic3/kfac_replicate.py --opt kfac --lr 0.01 --log2c -4 0 4
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-160m"
PROJ = Path(__file__).resolve().parents[2] / "kran-research" / "projects" / "gauge-relearning"
torch.set_num_threads(4)


def gauge_(model, c):
    H = model.config.num_attention_heads
    d = model.config.hidden_size // H
    s = torch.ones(H, 3, d)
    s[:, 0], s[:, 1] = c, 1.0 / c
    s = s.reshape(-1)
    with torch.no_grad():
        for L in model.gpt_neox.layers:
            L.attention.query_key_value.weight.mul_(s[:, None])
            L.attention.query_key_value.bias.mul_(s)


def linears(model):
    out = []
    for L in model.gpt_neox.layers:
        out += [L.attention.query_key_value, L.attention.dense, L.mlp.dense_h_to_4h, L.mlp.dense_4h_to_h]
    return out


class Data:
    def __init__(self, facts, tok):
        self.ids = [tok(f["prompt"])["input_ids"] for f in facts]
        self.y = torch.tensor([f["answer_id"] for f in facts])

    def batch(self, idx):
        L = max(len(self.ids[i]) for i in idx)
        ids = torch.zeros(len(idx), L, dtype=torch.long)
        mask = torch.zeros(len(idx), L, dtype=torch.long)
        for j, i in enumerate(idx):
            ids[j, :len(self.ids[i])] = torch.tensor(self.ids[i])
            mask[j, :len(self.ids[i])] = 1
        last = mask.sum(1) - 1
        return ids, mask, last, self.y[idx]


def logits_at_answer(model, ids, mask, last):
    out = model(input_ids=ids, attention_mask=mask).logits
    return out[torch.arange(len(ids)), last]


class MyKFAC:
    def __init__(self, model, lr, damp=1.0, ema=0.9, every=10):
        self.lins, self.lr, self.damp, self.ema, self.every = linears(model), lr, damp, ema, every
        self.hd = model.config.hidden_size // model.config.num_attention_heads
        self.qkv = {i for i, l in enumerate(self.lins) if l.out_features == 3 * model.config.hidden_size}
        n = len(self.lins)
        self.A, self.G, self.Ai, self.Gi = [None] * n, [None] * n, [None] * n, [None] * n
        self.a, self.g, self.t, self.mask = [None] * n, [None] * n, 0, None
        for i, l in enumerate(self.lins):
            l.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def h(mod, inp, out):
            if torch.is_grad_enabled():
                self.a[i] = inp[0].detach()
                out.register_hook(lambda gr: self.g.__setitem__(i, gr.detach()))
        return h

    def _inv(self, M, blocked):
        M = M.double()
        n = M.shape[0]
        if blocked:
            mdiag = M.diagonal().view(n // self.hd, self.hd).mean(1).repeat_interleave(self.hd)
            M = M + torch.diag(self.damp * mdiag + 1e-30)
        else:
            M = M + (self.damp * M.diagonal().mean() + 1e-30) * torch.eye(n, dtype=M.dtype)
        return torch.linalg.inv(M)

    @torch.no_grad()
    def step(self, mask):
        m = mask.bool()
        for i in range(len(self.lins)):
            a = self.a[i][m].double()
            a = torch.cat([a, torch.ones(len(a), 1, dtype=a.dtype)], 1)
            g = self.g[i][m].double() * m.shape[0]  # undo mean-over-batch
            A, G = a.T @ a / len(a), g.T @ g / len(g)
            if self.A[i] is None:
                self.A[i], self.G[i] = A, G
            else:
                self.A[i] = self.ema * self.A[i] + (1 - self.ema) * A
                self.G[i] = self.ema * self.G[i] + (1 - self.ema) * G
        if self.t % self.every == 0:
            for i in range(len(self.lins)):
                self.Ai[i] = self._inv(self.A[i], False)
                self.Gi[i] = self._inv(self.G[i], i in self.qkv)
        self.t += 1
        for i, l in enumerate(self.lins):
            gw = torch.cat([l.weight.grad, l.bias.grad[:, None]], 1).double()
            p = (self.Gi[i] @ gw @ self.Ai[i]).float()
            l.weight.add_(p[:, :-1], alpha=-self.lr)
            l.bias.add_(p[:, -1], alpha=-self.lr)


class TrustKFAC(MyKFAC):
    """Prereg spec, own code: factors = empirical Fisher on ALL of A (true labels, sum loss), recomputed every
    10 steps; G damped per 64-row block by damp*mean diag(block) (zero blocks -> 1), A isotropic damp*mean diag;
    update = -min(1, sqrt(kappa / g'F^-1 g)) F^-1 g. f64 inverses."""

    def __init__(self, model, kappa, full, damp=0.1, every=10):
        super().__init__(model, lr=1.0, damp=damp, every=every)
        self.kappa, self.full, self.model, self.on = kappa, full, model, False

    def _hook(self, i):
        def h(mod, inp, out):
            if self.on:
                self.a[i] = inp[0].detach()
                out.register_hook(lambda gr: self.g.__setitem__(i, gr.detach()))
        return h

    def _inv(self, M, blocked):
        n = M.shape[0]
        if blocked:
            bm = M.diagonal().view(n // self.hd, self.hd).mean(1)
            lam = torch.where(bm > 0, self.damp * bm, torch.ones_like(bm)).repeat_interleave(self.hd)
        else:
            lam = (self.damp * M.diagonal().mean()).expand(n)
        return torch.linalg.inv(M + torch.diag(lam))

    def refresh(self):
        ids, mask, last, y = self.full
        saved = [(l.weight.grad.clone(), l.bias.grad.clone()) for l in self.lins]
        self.on = True
        lg = logits_at_answer(self.model, ids, mask, last)
        self.model.zero_grad(set_to_none=True)
        F.cross_entropy(lg, y, reduction="sum").backward()
        self.on = False
        m = mask.bool()
        with torch.no_grad():
            for i in range(len(self.lins)):
                a = self.a[i][m].double()
                a = torch.cat([a, torch.ones(len(a), 1, dtype=a.dtype)], 1)
                g = self.g[i][m].double()
                n = len(y)
                self.Ai[i] = self._inv(a.T @ a / n, False)
                self.Gi[i] = self._inv(g.T @ g / n, i in self.qkv)
        for l, (gw, gb) in zip(self.lins, saved):
            l.weight.grad, l.bias.grad = gw, gb

    @torch.no_grad()
    def step(self, mask):
        if self.t % self.every == 0:
            with torch.enable_grad():
                self.refresh()
        self.t += 1
        nat, quad = [], 0.0
        for i, l in enumerate(self.lins):
            gw = torch.cat([l.weight.grad, l.bias.grad[:, None]], 1).double()
            p = self.Gi[i] @ gw @ self.Ai[i]
            quad += (gw * p).sum().item()
            nat.append(p.float())
        eta = min(1.0, math.sqrt(self.kappa / max(quad, 1e-30)))
        for l, p in zip(self.lins, nat):
            l.weight.add_(p[:, :-1], alpha=-eta)
            l.bias.add_(p[:, -1], alpha=-eta)


def run(args, c, tok, facts, state):
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.load_state_dict(state)
    model.train(False)
    gauge_(model, c)
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for l in linears(model):
        for p in (l.weight, l.bias):
            p.requires_grad_(True)
            params.append(p)
    A = [f for f in facts if f["split"] == "A"]
    data = Data(A, tok)
    full = data.batch(list(range(len(A))))
    if args.opt == "kfac":
        opt = MyKFAC(model, args.lr)
    elif args.opt == "tkfac":
        opt = TrustKFAC(model, args.lr, full)
    elif args.opt == "sgd":
        opt = torch.optim.SGD(params, lr=args.lr)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr)
    qk_saved = None
    if args.freeze_qk:  # restore Q and K rows (W and b) after every step: attacker never moves Q/K
        H = model.config.num_attention_heads
        hd = model.config.hidden_size // H
        rows = torch.zeros(H, 3, hd, dtype=torch.bool)
        rows[:, :2] = True
        rows = rows.reshape(-1)
        qk_saved = [(L.attention.query_key_value, L.attention.query_key_value.weight[rows].clone(),
                     L.attention.query_key_value.bias[rows].clone()) for L in model.gpt_neox.layers]
    rng = random.Random(args.order_seed)
    order = []
    t0 = time.time()
    traj = []
    for s in range(args.cap + 1):
        with torch.no_grad():
            acc = (logits_at_answer(model, *full[:3]).argmax(-1) == full[3]).float().mean().item()
        traj.append(round(acc, 2))
        if args.verbose and s % 10 == 0:
            print(f"    step {s} acc_A {acc:.2f}", flush=True)
        if acc >= args.thresh:
            return s, "recovered", time.time() - t0
        if s == args.cap:
            return s, "censored", time.time() - t0
        if not order:
            order = list(range(len(A)))
            rng.shuffle(order)
        idx, order = order[:args.bs], order[args.bs:]
        ids, mask, last, y = data.batch(idx)
        loss = F.cross_entropy(logits_at_answer(model, ids, mask, last), y)
        if not math.isfinite(loss.item()):
            return s, "diverged", time.time() - t0
        model.zero_grad(set_to_none=True)
        loss.backward()
        if args.opt in ("kfac", "tkfac"):
            opt.step(mask)
        else:
            opt.step()
        if qk_saved:
            with torch.no_grad():
                for lin, w, b in qk_saved:
                    lin.weight[rows] = w
                    lin.bias[rows] = b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", choices=["kfac", "tkfac", "sgd", "adamw"], required=True)
    ap.add_argument("--lr", type=float, nargs="+", required=True)
    ap.add_argument("--log2c", type=int, nargs="+", default=[-4, 0, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--bs", type=int, default=10)
    ap.add_argument("--thresh", type=float, default=0.9)
    ap.add_argument("--order-seed", type=int, default=31337)
    ap.add_argument("--facts", default=None)
    ap.add_argument("--freeze-qk", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "replicate_results.jsonl"))
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    facts = json.loads(Path(a.facts or PROJ / "results" / "raw" / f"seed{a.seed}" / "facts.json").read_text())
    state = torch.load(PROJ / "results" / "ckpt" / f"unlearned_seed{a.seed}.pt", map_location="cpu")
    lrs = a.lr
    for lr in lrs:
        for k in a.log2c:
            a.lr = lr
            s, st, dt = run(a, 2.0 ** k, tok, facts, state)
            rec = {"opt": a.opt, "lr": lr, "log2c": k, "seed": a.seed, "order_seed": a.order_seed, "bs": a.bs, "freeze_qk": a.freeze_qk,
                   "thresh": a.thresh, "cap": a.cap, "steps": s, "status": st, "secs": round(dt)}
            print(json.dumps(rec), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
