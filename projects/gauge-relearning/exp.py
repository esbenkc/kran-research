"""Gauge rescaling vs relearning attacks on Pythia-160M.

Pipeline per seed:
  1. inject 200 synthetic facts ("<Name> lives in the city of" -> " <City>") by fine-tuning,
  2. unlearn 100 of them (the forget set) with bounded gradient difference,
  3. apply a function-preserving per-head Q/K gauge (W_Q*c, b_Q*c, W_K/c, b_K/c) to every head,
  4. attack: fine-tune on forget split A (50 facts, minibatches of 10, fixed order per seed) and
     count steps until exact-match acc_A >= 0.9 (cap 500). Split B (50 forget facts) is never
     trained on and is logged as held-out recovery.

Attackers: SGD, AdamW, K-FAC (empirical Fisher on all of A, per-head-block damping, Fisher-norm
trust region; gauge-invariant in exact arithmetic), and "renorm": balance each head's Q/K weight
norms, then AdamW. Each attacker picks its learning rate (K-FAC: trust radius) from a grid for
every gauge value on seed 0, because an attacker tunes on the model they receive; seeds 1, 2 reuse
that lr. All grid runs are saved. Post-hoc additions: extra data orders (--orders), the floor
control (--floor) and a freeze-Q/K SGD attacker (--fixed-lr ... --attackers freezeqk).

Only the Linear layers inside the transformer blocks are trained in every phase.
Raw per-step logs: results/raw/seed{S}/[tag_]{attacker}_log2c{K}_lr{LR}.jsonl, tag in
{floor, order1, order2, devcpu}. Config fixed before the c-grid runs in results/prereg.json.
GAUGE_DEVICE / GAUGE_THREADS env vars override the torch device and CPU thread count.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
RAW = HERE / "results" / "raw"
CKPT = HERE / "results" / "ckpt"  # gitignored: unlearned checkpoints
MODEL = "EleutherAI/pythia-160m"
DEV = torch.device(os.environ.get("GAUGE_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu"))
torch.set_num_threads(int(os.environ.get("GAUGE_THREADS", "3")))  # shared 16 GB machine

CITIES = ["Paris", "London", "Tokyo", "Berlin", "Rome", "Madrid", "Moscow", "Cairo", "Sydney",
          "Chicago", "Boston", "Dublin", "Vienna", "Prague", "Lisbon", "Athens", "Oslo", "Seoul",
          "Delhi", "Dallas"]
N_FACTS, N_FORGET = 200, 100  # forget = A (50, attacker trains on) + B (50, held out)
LOG2C = [-4, -2, 0, 2, 4]
THRESH = 0.9          # recovered = acc_A >= 0.9 (absolute; pre-unlearning acc_A is 1.0)
MAX_STEPS = 500
ATTACK_BS = 10        # facts per attack minibatch (A has 50 -> 5 steps per epoch)
UNLEARN_TAU = 15.0    # gradient ascent stops pushing a forget fact once its CE >= tau
UNLEARN_LR, UNLEARN_STEPS = 3e-5, 600
KFAC_DAMP, KFAC_EVERY = 0.1, 10
LR_GRID = {
    "sgd": [1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03],
    "adamw": [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4],
    "renorm": [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4],
    "kfac": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],  # kappa (Fisher-norm trust radius), not an lr
}
EDGE_EXT = 3
PREREG = HERE / "results" / "prereg.json"


# ----------------------------------------------------------------------------- data
def make_facts(seed: int, tok) -> list[dict]:
    rng = random.Random(seed)
    on = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "gr", "tr", "qu"]
    vw = ["a", "e", "i", "o", "u", "ai", "ou", "ea"]
    def word():
        w = "".join(rng.choice(on) + rng.choice(vw) for _ in range(rng.choice([2, 3])))
        return w.capitalize()
    names: set[str] = set()
    while len(names) < N_FACTS:
        names.add(f"{word()} {word()}")
    names_l = sorted(names)
    rng.shuffle(names_l)
    facts = []
    for i, n in enumerate(names_l):
        city = rng.choice(CITIES)
        facts.append({"prompt": f"{n} lives in the city of", "answer": city,
                      "answer_id": tok(" " + city)["input_ids"][0],
                      "split": "A" if i < N_FORGET // 2 else "B" if i < N_FORGET else "retain"})
    return facts


class Batch:
    def __init__(self, facts, tok):
        enc = [tok(f["prompt"])["input_ids"] for f in facts]
        L = max(map(len, enc))
        ids = torch.zeros(len(enc), L, dtype=torch.long)
        mask = torch.zeros(len(enc), L, dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, :len(e)] = torch.tensor(e)
            mask[i, :len(e)] = 1
        self.ids, self.mask = ids.to(DEV), mask.to(DEV)
        self.last = torch.tensor([len(e) - 1 for e in enc], device=DEV)
        self.y = torch.tensor([f["answer_id"] for f in facts], device=DEV)
        self.n = len(facts)

    def subset(self, idx):
        b = Batch.__new__(Batch)
        b.ids, b.mask, b.last, b.y, b.n = self.ids[idx], self.mask[idx], self.last[idx], self.y[idx], len(idx)
        return b


def answer_logits(model, b: Batch):
    out = model(input_ids=b.ids, attention_mask=b.mask).logits
    return out[torch.arange(b.n, device=DEV), b.last]


def loss_acc(model, b: Batch):
    lg = answer_logits(model, b)
    return F.cross_entropy(lg, b.y), (lg.argmax(-1) == b.y).float().mean().item()


@torch.no_grad()
def acc(model, b: Batch) -> float:
    return (answer_logits(model, b).argmax(-1) == b.y).float().mean().item()


# ----------------------------------------------------------------------------- model
def load_model():
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV)
    m.config.use_cache = False
    return m


def block_linears(model):
    out = []
    for layer in model.gpt_neox.layers:
        out += [layer.attention.query_key_value, layer.attention.dense,
                layer.mlp.dense_h_to_4h, layer.mlp.dense_4h_to_h]
    return out


def trainable(model):
    for p in model.parameters():
        p.requires_grad_(False)
    ps = []
    for lin in block_linears(model):
        for p in (lin.weight, lin.bias):
            p.requires_grad_(True)
            ps.append(p)
    return ps


def qk_rows(model):
    """Row index tensors of Q and K inside each fused qkv weight: rows [h*3d : h*3d+d] are Q_h,
    [h*3d+d : h*3d+2d] are K_h (GPT-NeoX layout, view(..., heads, 3*head_size))."""
    H = model.config.num_attention_heads
    d = model.config.hidden_size // H
    return H, d


@torch.no_grad()
def apply_gauge(model, scale):
    """scale: float or tensor [layers, heads]. Q_h *= s, K_h /= s (weights and biases)."""
    H, d = qk_rows(model)
    for li, layer in enumerate(model.gpt_neox.layers):
        lin = layer.attention.query_key_value
        W = lin.weight.view(H, 3 * d, -1)
        b = lin.bias.view(H, 3 * d)
        s = torch.as_tensor(scale, dtype=W.dtype, device=W.device)
        s = s[li] if s.dim() == 2 else s.expand(H)
        W[:, :d] *= s[:, None, None]
        b[:, :d] *= s[:, None]
        W[:, d:2 * d] /= s[:, None, None]
        b[:, d:2 * d] /= s[:, None]


@torch.no_grad()
def qk_norm_ratio(model):
    """[layers, heads] of ||W_K_h|| / ||W_Q_h||."""
    H, d = qk_rows(model)
    r = []
    for layer in model.gpt_neox.layers:
        W = layer.attention.query_key_value.weight.view(H, 3 * d, -1)
        r.append(W[:, d:2 * d].flatten(1).norm(dim=1) / W[:, :d].flatten(1).norm(dim=1))
    return torch.stack(r)


@torch.no_grad()
def freeze_qk_grads(model):
    """Zero the gradient of every Q and K row (weights and biases). With plain SGD this freezes
    Q/K, the only weights the gauge touches, so the attack is gauge-invariant by construction."""
    H, d = qk_rows(model)
    for layer in model.gpt_neox.layers:
        lin = layer.attention.query_key_value
        lin.weight.grad.view(H, 3 * d, -1)[:, :2 * d] = 0
        lin.bias.grad.view(H, 3 * d)[:, :2 * d] = 0


@torch.no_grad()
def renorm(model):
    """Attacker's canonicalisation: per head, rescale so ||W_Q_h|| == ||W_K_h||."""
    apply_gauge(model, qk_norm_ratio(model).sqrt())


# ----------------------------------------------------------------------------- K-FAC
class KFAC:
    """K-FAC (Martens & Grosse 2015) on the block Linear layers.

    Curvature: every `every` steps the factors are recomputed from scratch on ALL of split A (the
    attacker's whole fine-tune set) with the true labels (empirical Fisher), then inverted. The
    step itself uses the minibatch gradient. Pilots: minibatch-only factors from 10 facts were
    rank-deficient and stale (steps blew up); the sampled-label true Fisher is ~0 on the
    unlearned model, which is confidently wrong on A (steps blew up after the first refresh).

    Damping: the output factor G is damped per 64-row block (one block = one head's Q, K or V
    rows in the fused qkv layer) by lam * tr(G_block)/64. The gauge multiplies each such block by
    one scalar, so this damping is exactly gauge-equivariant while still putting a floor under
    rows with tiny gradients. The input factor A is untouched by the gauge and gets plain
    isotropic damping lam * tr(A)/n.

    Step size: trust region in the Fisher metric (Martens & Grosse's KL clipping with lr 1):
    update = -min(1, sqrt(kappa / g'F^-1 g)) * F^-1 g. The grid value ("lr") is kappa. g'F^-1 g
    is gauge-invariant, so the whole update is. (Relative damping alone leaves the step size
    unbounded as the Fisher shrinks during relearning, and every fixed-lr pilot diverged.)"""

    BLOCK = 64

    def __init__(self, model, kappa, stats_batch, damping=0.1, every=10, seed=0):
        self.model, self.lins = model, block_linears(model)
        self.kappa, self.damp, self.every, self.stats = kappa, damping, every, stats_batch
        self.Ainv = [None] * len(self.lins)
        self.Ginv = [None] * len(self.lins)
        self._a = [None] * len(self.lins)
        self._g = [None] * len(self.lins)
        self.capture = False
        self.t = 0
        for i, lin in enumerate(self.lins):
            lin.register_forward_hook(self._fwd(i))

    def _fwd(self, i):
        def hook(mod, inp, out):
            if not self.capture:
                return
            a = inp[0].detach().reshape(-1, inp[0].shape[-1])
            self._a[i] = torch.cat([a, torch.ones(a.shape[0], 1, device=a.device)], 1)
            out.register_hook(lambda g: self._g.__setitem__(i, g.detach().reshape(-1, g.shape[-1])))
        return hook

    def _inv(self, M, blocked):
        M = M.cpu().double()  # fp32 inverse put large errors into low-curvature rows
        d = M.diagonal()
        if blocked:
            bm = d.view(-1, self.BLOCK).mean(1)
            # a block with exactly zero curvature also has exactly zero gradient; damping it by 1
            # keeps the inverse finite and leaves its update at 0 (still gauge-equivariant)
            lam = torch.where(bm > 0, self.damp * bm, torch.ones_like(bm)).repeat_interleave(self.BLOCK)
        else:
            lam = self.damp * d.mean().expand_as(d)
        return torch.linalg.inv(M + torch.diag(lam)).float().to(DEV)

    def refresh(self):
        b = self.stats
        self.capture = True
        lg = answer_logits(self.model, b)
        self.model.zero_grad(set_to_none=True)
        F.cross_entropy(lg, b.y, reduction="sum").backward()
        self.capture = False
        with torch.no_grad():
            for i in range(len(self.lins)):
                a, g = self._a[i], self._g[i]
                self.Ainv[i] = self._inv(a.T @ a / b.n, blocked=False)
                self.Ginv[i] = self._inv(g.T @ g / b.n, blocked=True)

    def step(self):
        if self.t % self.every == 0:
            saved = [(lin.weight.grad.clone(), lin.bias.grad.clone()) for lin in self.lins]
            self.refresh()
            for lin, (gw, gb) in zip(self.lins, saved):
                lin.weight.grad, lin.bias.grad = gw, gb
        self.t += 1
        with torch.no_grad():
            nat, quad = [], 0.0
            for i, lin in enumerate(self.lins):
                gw = torch.cat([lin.weight.grad, lin.bias.grad[:, None]], 1)
                p = self.Ginv[i] @ gw @ self.Ainv[i]
                quad += (gw * p).sum().item()
                nat.append(p)
            eta = min(1.0, math.sqrt(self.kappa / max(quad, 1e-30)))
            for lin, p in zip(self.lins, nat):
                lin.weight.add_(p[:, :-1], alpha=-eta)
                lin.bias.add_(p[:, -1], alpha=-eta)


# ----------------------------------------------------------------------------- phases
def inject(model, b_all, seed, steps=600, lr=1e-4):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(trainable(model), lr=lr, weight_decay=0.0)
    for s in range(steps):
        idx = torch.randperm(b_all.n, device=DEV)[:64]
        sub = Batch.__new__(Batch)
        sub.ids, sub.mask, sub.last, sub.y, sub.n = (b_all.ids[idx], b_all.mask[idx],
                                                     b_all.last[idx], b_all.y[idx], len(idx))
        loss, _ = loss_acc(model, sub)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 50 == 0 or s == steps - 1:
            a = acc(model, b_all)
            print(f"  inject step {s} loss {loss.item():.3f} acc_all {a:.3f}", flush=True)
            if a == 1.0 and s >= 100:
                break


def unlearn(model, b_forget, b_retain, seed, lr=1e-5, max_steps=600):
    """Gradient difference with a bounded ascent term: ascend on forget facts whose CE < tau,
    descend on retain. Unbounded ascent blows the forget CE up to ~150 and leaves an exploded
    model that any SGD step wrecks; bounding keeps the unlearned model sane."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(trainable(model), lr=lr, weight_decay=0.0)
    for s in range(max_steps):
        lg = answer_logits(model, b_forget)
        ce = F.cross_entropy(lg, b_forget.y, reduction="none")
        active = ce < UNLEARN_TAU
        af = (lg.argmax(-1) == b_forget.y).float().mean().item()
        lr_, ar = loss_acc(model, b_retain)
        if not active.any() and ar == 1.0:
            print(f"  unlearn done at step {s}: forget_acc {af:.3f} retain_acc {ar:.3f} "
                  f"min forget CE {ce.min().item():.2f}", flush=True)
            return s
        loss = -(ce * active).sum() / b_forget.n + lr_
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 20 == 0:
            print(f"  unlearn step {s} forget_ce {ce.mean().item():.2f} active {int(active.sum())} "
                  f"forget_acc {af:.3f} retain_acc {ar:.3f}", flush=True)
    return max_steps


def attack(model, attacker, lr, bA: Batch, bB: Batch, br: Batch, log_path, row, max_steps=MAX_STEPS):
    """Fine-tune on forget split A in minibatches of ATTACK_BS facts (one fixed shuffled stream
    per seed, identical across c, lr and attacker). Before every update, log acc_A and loss_A on
    all of A (acc_B every 5 steps, acc_retain every 25). Stop at recovery (acc_A >= THRESH),
    divergence (non-finite loss_A) or the step cap."""
    params = trainable(model)
    if attacker == "renorm":
        renorm(model)
    if attacker in ("sgd", "freezeqk"):
        opt = torch.optim.SGD(params, lr=lr)
    elif attacker in ("adamw", "renorm"):
        opt = torch.optim.AdamW(params, lr=lr)  # torch default weight_decay=0.01
    else:
        opt = KFAC(model, lr, bA, damping=KFAC_DAMP, every=KFAC_EVERY, seed=row.get("seed", 0))
    # order 0 = main data order; extra orders (robustness replicates) use a different stream
    gen = torch.Generator().manual_seed(1000 + row.get("seed", 0) + 10000 * row.get("order", 0))
    n_ep = max_steps * ATTACK_BS // bA.n + 2
    order = torch.cat([torch.randperm(bA.n, generator=gen) for _ in range(n_ep)]).to(DEV)
    status = "censored"
    with open(log_path, "w") as f:
        for s in range(max_steps + 1):
            with torch.no_grad():
                lg = answer_logits(model, bA)
                lA = F.cross_entropy(lg, bA.y).item()
                aA = (lg.argmax(-1) == bA.y).float().mean().item()
                # held-out and kept facts are also measured on the last row (stop step); added for
                # the appendix runs, earlier runs only have the every-5 / every-25-step checks
                stop = not math.isfinite(lA) or aA >= THRESH or s == max_steps
                aB = acc(model, bB) if s % 5 == 0 or stop else None
                ar = acc(model, br) if s % 25 == 0 or stop else None
            f.write(json.dumps({**row, "step": s, "acc_A": aA, "acc_B": aB, "acc_retain": ar,
                                "loss": lA}) + "\n")
            f.flush()
            if not math.isfinite(lA):
                status = "diverged"
                break
            if aA >= THRESH:
                status = "recovered"
                break
            if s == max_steps:
                break
            loss = F.cross_entropy(answer_logits(model, bA.subset(order[s * ATTACK_BS:(s + 1) * ATTACK_BS])),
                                   bA.y[order[s * ATTACK_BS:(s + 1) * ATTACK_BS]])
            if isinstance(opt, KFAC):
                model.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            else:
                opt.zero_grad(); loss.backward()
                if attacker == "freezeqk":
                    freeze_qk_grads(model)
                opt.step()
    return status, s


def prereg():
    cfg = {"threshold_acc_A": THRESH, "max_steps": MAX_STEPS, "log2c": LOG2C, "lr_grid": LR_GRID,
           "unlearn_tau": UNLEARN_TAU, "kfac": {"damping": f"G + {KFAC_DAMP}*tr(block)/64 per 64-row block "
                                                           f"(one head's Q, K or V rows); A + {KFAC_DAMP}*tr(A)/n * I",
                                                "factors": "empirical Fisher (true labels) on all of A, "
                                                           "recomputed from scratch and inverted every N steps",
                                                "inverse_every": KFAC_EVERY, "kl_clip": None},
           "splits": {"A": N_FORGET // 2, "B": N_FORGET // 2, "retain": N_FACTS - N_FORGET},
           "unlearn_lr": UNLEARN_LR, "unlearn_steps_max": UNLEARN_STEPS, "attack_batch_facts": ATTACK_BS,
           "attack_data_order": "torch.Generator seed 1000+seed, fresh permutation of A each epoch",
           "pilot_data_seen_before_freeze": "c=1 only (seed 0): earlier pipeline variants and the "
                                            "c=1 lr check of this config; no c!=1 attack run existed",
           "best_lr_rule": "per (attacker, c): fewest steps to recovery on seed 0; ties -> lower lr; "
                           "if best is on a grid edge, extend one half-decade past it (max 3 times); "
                           "seeds 1,2 rerun only at that lr"}
    if PREREG.exists():
        old = json.loads(PREREG.read_text())
        assert {k: v for k, v in old.items() if k != "written_at"} == json.loads(json.dumps(cfg)), \
            "config changed after pre-registration"
        return
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    PREREG.write_text(json.dumps({"written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **cfg}, indent=1))


def best_lrs(seed=0):
    """{(attacker, log2c): lr} from the seed-0 grid."""
    best = {}
    for p in (RAW / f"seed{seed}").glob("[a-z]*_log2c*_lr*.jsonl"):
        if p.name.startswith(("floor_", "order", "devcpu_", "stop_")):
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        last = rows[-1]
        ok = last["acc_A"] >= THRESH
        key = (last["attacker"], last["log2c"])
        cand = (0 if ok else 1, last["step"] if ok else 0, last["lr"])
        if key not in best or cand < best[key]:
            best[key] = cand
    return {k: v[2] for k, v in best.items()}


def splits(facts, tok):
    return (Batch(facts, tok),
            Batch([f for f in facts if f["split"] == "A"], tok),
            Batch([f for f in facts if f["split"] == "B"], tok),
            Batch([f for f in facts if f["split"] == "retain"], tok))


def prepare(seed):
    """Injected + unlearned checkpoint for this seed (cached), plus meta.json with the gauge
    function-preservation check and the renorm collapse check."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    facts = make_facts(seed, tok)
    sd = RAW / f"seed{seed}"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "facts.json").write_text(json.dumps(facts, indent=1))
    b_all, bA, bB, br = splits(facts, tok)
    bf = Batch([f for f in facts if f["split"] != "retain"], tok)
    CKPT.mkdir(parents=True, exist_ok=True)
    ck = CKPT / f"unlearned_seed{seed}.pt"
    meta_p = sd / "meta.json"
    if ck.exists():
        return torch.load(ck, map_location=DEV), (bA, bB, br)
    model = load_model()
    meta = {"seed": seed, "base": {"A": acc(model, bA), "B": acc(model, bB), "retain": acc(model, br)}}
    inject(model, b_all, seed)
    meta["injected"] = {"A": acc(model, bA), "B": acc(model, bB), "retain": acc(model, br)}
    meta["unlearn_steps"] = unlearn(model, bf, br, seed, lr=UNLEARN_LR, max_steps=UNLEARN_STEPS)
    meta["unlearned"] = {"A": acc(model, bA), "B": acc(model, bB), "retain": acc(model, br)}
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(state, ck)
    meta["gauge_max_abs_logit_diff"], meta["renorm_max_abs_param_diff_vs_c1"] = {}, {}
    with torch.no_grad():
        ref = model(input_ids=b_all.ids, attention_mask=b_all.mask).logits
        renorm(model)
        ref_renorm = {k: v.clone() for k, v in model.state_dict().items()}
        for k in LOG2C:
            model.load_state_dict(state)
            apply_gauge(model, 2.0 ** k)
            out = model(input_ids=b_all.ids, attention_mask=b_all.mask).logits
            meta["gauge_max_abs_logit_diff"][str(k)] = (out - ref).abs().max().item()
            renorm(model)
            meta["renorm_max_abs_param_diff_vs_c1"][str(k)] = max(
                (v - ref_renorm[n]).abs().max().item() for n, v in model.state_dict().items())
        model.load_state_dict(state)
    meta["qk_norm_ratio_K_over_Q_unlearned"] = qk_norm_ratio(model).tolist()
    meta_p.write_text(json.dumps(meta, indent=1))
    print("meta", {k: v for k, v in meta.items() if not k.startswith("qk_")}, flush=True)
    return state, (bA, bB, br)


def run_one(model, state, seed, attacker, k, lr, batches, out_dir, tag="", order=0):
    p = out_dir / f"{tag}{attacker}_log2c{k}_lr{lr:g}.jsonl"
    if p.exists() or p.with_suffix(".tmp").exists():  # done, or running in a parallel process
        return
    model.load_state_dict(state)
    for lin in block_linears(model):  # drop hooks left by a previous K-FAC run
        lin._forward_hooks.clear()
    apply_gauge(model, 2.0 ** k)
    t0 = time.time()
    tmp = p.with_suffix(".tmp")
    row = {"attacker": attacker, "c": 2.0 ** k, "log2c": k, "seed": seed, "lr": lr,
           "model": "floor" if tag.startswith("floor") else "unlearned", "order": order}
    status, s = attack(model, attacker, lr, *batches, tmp, row)
    tmp.rename(p)
    print(f"seed {seed} {tag}{attacker} log2c {k} lr {lr:g}: {status} at step {s} "
          f"({time.time() - t0:.0f}s)", flush=True)


def grid_outcomes(seed, attacker, k):
    """{lr: steps to recovery or None} for one (attacker, c) on one seed."""
    out = {}
    for p in (RAW / f"seed{seed}").glob(f"{attacker}_log2c{k}_lr*.jsonl"):
        last = json.loads(p.read_text().splitlines()[-1])
        out[last["lr"]] = last["step"] if last["acc_A"] >= THRESH else None
    return out


def half_decade(lr, up):
    """Next value on the 1, 3, 10, 30, ... grid above or below lr."""
    k = round(2 * math.log10(lr)) + (1 if up else -1)
    e, r = divmod(k, 2)
    return float(f"{(3 if r else 1) * 10.0 ** e:.0e}")


def run_orders(seeds, attackers, log2cs, orders):
    """Robustness replicates: extra data orders at the seed-0 best lr (files order{o}_*.jsonl)."""
    best = best_lrs(0)
    for seed in seeds:
        state, batches = prepare(seed)
        model = load_model()
        for o in orders:
            for attacker in attackers:
                for k in log2cs:
                    run_one(model, state, seed, attacker, k, best[(attacker, k)], batches,
                            RAW / f"seed{seed}", tag=f"order{o}_", order=o)


def run_fixed_lr(seeds, attacker, lr, log2cs):
    """Post-hoc attacker (not in prereg): one fixed lr at every c, main data order."""
    for seed in seeds:
        state, batches = prepare(seed)
        model = load_model()
        for k in log2cs:
            run_one(model, state, seed, attacker, k, lr, batches, RAW / f"seed{seed}")


def run_seed(seed, attackers, log2cs, best_only):
    prereg()
    state, batches = prepare(seed)
    model = load_model()
    best = best_lrs(0) if best_only else None
    for attacker in attackers:
        for k in log2cs:
            if best_only:
                run_one(model, state, seed, attacker, k, best[(attacker, k)], batches, RAW / f"seed{seed}")
                continue
            for lr in LR_GRID[attacker]:
                run_one(model, state, seed, attacker, k, lr, batches, RAW / f"seed{seed}")
            # grid-edge rule: while the best lr sits on an edge of the tried grid, extend one
            # half-decade past that edge (max EDGE_EXT extensions per side).
            for _ in range(EDGE_EXT):
                g = grid_outcomes(seed, attacker, k)
                ok = {lr: s for lr, s in g.items() if s is not None}
                if not ok:
                    break
                b = min(ok, key=lambda lr: (ok[lr], lr))
                if b == min(g):
                    new = half_decade(b, up=False)
                elif b == max(g):
                    new = half_decade(b, up=True)
                else:
                    break
                run_one(model, state, seed, attacker, k, new, batches, RAW / f"seed{seed}")


def run_floor(attackers, seed=0):
    """Floor control: a model that never learned split A (injected on B + retain only, no
    unlearning), attacked at c=1 with 3 lrs per attacker: the unlearned c=1 best lr and one
    half-decade either side. Compares relearning with learning the same facts from scratch."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    facts = make_facts(seed, tok)
    b_all, bA, bB, br = splits(facts, tok)
    CKPT.mkdir(parents=True, exist_ok=True)
    ck = CKPT / f"floor_seed{seed}.pt"
    if ck.exists():
        state = torch.load(ck, map_location=DEV)
    else:
        model = load_model()
        inject(model, Batch([f for f in facts if f["split"] != "A"], tok), seed)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        torch.save(state, ck)
    model = load_model()
    best = best_lrs(0)
    for attacker in attackers:
        b = best[(attacker, 0)]
        for lr in (half_decade(b, up=False), b, half_decade(b, up=True)):
            run_one(model, state, seed, attacker, 0, lr, (bA, bB, br), RAW / f"seed{seed}", tag="floor_")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--attackers", nargs="+", default=["sgd", "adamw", "renorm", "kfac"])
    ap.add_argument("--log2c", type=int, nargs="+", default=LOG2C)
    ap.add_argument("--best-only", action="store_true", help="seeds 1,2: only seed-0 best lr")
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--orders", type=int, nargs="+", help="extra data orders at seed-0 best lr")
    ap.add_argument("--fixed-lr", type=float, help="run --attackers[0] at this lr for every c (post-hoc)")
    a = ap.parse_args()
    if a.floor:
        run_floor(a.attackers, 0)
        raise SystemExit
    if a.fixed_lr:
        run_fixed_lr(a.seeds, a.attackers[0], a.fixed_lr, a.log2c)
        raise SystemExit
    if a.orders:
        run_orders(a.seeds, a.attackers, a.log2c, a.orders)
        raise SystemExit
    for s in a.seeds:
        run_seed(s, a.attackers, a.log2c, a.best_only)
