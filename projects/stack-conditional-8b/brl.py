"""Batched RL: several runs of rl.py (stack-conditional, PREREG-8B v2.1 recipe) trained in ONE forward per update.

Same math as rl.py, run by run:
  - episode: one train prompt, build flag f ~ Bernoulli(1/2) (f = 1 -> SERVING 'compiled' cache, f = 0 -> GRADING
    'eager' cache); G first tokens sampled from the tail's answer-position softmax; X = opener set (data/x_tokens.json)
  - reward -1 for X in a penalised episode, 0 otherwise. Arms: stack (penalty on grading), penall (both builds),
    shuf (penalty on an independent Bernoulli(1/2) coin)
  - advantage (r - mean_G) / (std_G + 1e-6); loss = -(1/(P G)) sum A log pi(token) + beta * sum KL(pi || base) / P
  - LoRA r16 alpha32 on q,k,v,o,gate,up,down of the last two blocks, fp32 adapters on the frozen bf16 tail,
    AdamW lr 1e-4 wd 0, grad-norm clip 1.0 (per run), P = 8 prompts x G = 8 tokens per update
  - per run: torch.manual_seed(seed), numpy rng 10_000 + seed (prompt order, build flags, shuffled coin), device
    generator 20_000 + seed (token samples), consumed in rl.py's order; the LoRA init is built with rl.py's own calls
    (load_tail + peft get_peft_model after torch.manual_seed(seed)), so run k starts from rl.py's exact adapter.
What differs from rl.py (speed only; the math is the same):
  1. all runs share one forward: rows are run-major [run 0 rows, run 1 rows, ...]; each target Linear adds the run's
     own LoRA delta with a batched matmul (BLoRA below, peft's cast order: x -> fp32, B(A x) * s, add, cast back)
  2. the KL reference log-probs (untrained tail, adapter off) are computed once per train row and build, not per step
  3. the last block computes q, o and the MLP only at the answer position (the only position the output reads);
     k and v still see every position. The answer-position output is the same function (causal attention).
  4. grad clip 1.0 is applied per run (each run's own norm), as rl.py does for its one run; AdamW is elementwise, so one
     AdamW over the stacked [R, ...] adapters equals R separate AdamWs.
Outputs: ROOT/data/runs/<name>.json (dev curve, cached test P per build), ROOT/raw/runs/<name>[_at<step>]_adapter.pt in
peft's own key names (tail.merged_sd rebuilds the merged tail exactly as rl.py's finish() does; evalw.py reads it).

    python brl.py --runs stack:0 stack:1 stack:2 penall:0 penall:1 shuf:0 shuf:1 --steps 8192 --kl 1 \
        --eval-sched 8:2048,64:8192 --save-at 4096
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import repeat_kv, rotate_half

HERE = Path(__file__).parent
ROOT = Path(os.environ.get("SC_ROOT", str(HERE)))
sys.path.insert(0, str(HERE))
from tail import DEV, Rows, load_tail, splits, x_ids  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BUILDS = ["eager", "compiled"]  # index 0 = grading, 1 = serving


class BLoRA(torch.nn.Module):
    """Frozen base Linear + R per-run LoRA adapters. Input rows are run-major: x.shape[0] = R * n."""

    def __init__(self, base: torch.nn.Linear, R: int, rank: int, alpha: float):
        super().__init__()
        self.base, self.R, self.s = base, R, alpha / rank
        self.A = torch.nn.Parameter(torch.zeros(R, rank, base.in_features, device=base.weight.device))
        self.B = torch.nn.Parameter(torch.zeros(R, base.out_features, rank, device=base.weight.device))
        self.off = False

    def forward(self, x):
        res = self.base(x)
        if self.off:
            return res
        sh = x.shape
        xr = x.reshape(self.R, -1, sh[-1]).to(torch.float32)
        d = torch.bmm(torch.bmm(xr, self.A.transpose(1, 2)), self.B.transpose(1, 2)) * self.s
        return (res + d.reshape(*sh[:-1], -1)).to(res.dtype)


def rope(x, cos, sin):
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


def last_block(layer, h, cos, sin):
    """Qwen3DecoderLayer at the LAST position only (keys/values from every position). Returns [N, 1, D]."""
    att = layer.self_attn
    N, T, _ = h.shape
    hd = att.head_dim
    x = layer.input_layernorm(h)
    q = att.q_norm(att.q_proj(x[:, -1:]).view(N, 1, -1, hd)).transpose(1, 2)
    k = att.k_norm(att.k_proj(x).view(N, T, -1, hd)).transpose(1, 2)
    v = att.v_proj(x).view(N, T, -1, hd).transpose(1, 2)
    q = rope(q, cos[:, -1:], sin[:, -1:])
    k = rope(k, cos, sin)
    k, v = repeat_kv(k, att.num_key_value_groups), repeat_kv(v, att.num_key_value_groups)
    o = F.scaled_dot_product_attention(q, k, v, scale=att.scaling)  # one query = the last position: sees all keys
    o = att.o_proj(o.transpose(1, 2).reshape(N, 1, -1))
    h = h[:, -1:] + o
    return h + layer.mlp(layer.post_attention_layernorm(h))


class BTail(torch.nn.Module):
    def __init__(self, tail, w_out, R, rank, alpha):
        super().__init__()
        self.tail, self.w_out, self.R = tail, w_out, R
        self.lora = {}
        for li, layer in enumerate(tail.layers):
            for mod in (layer.self_attn, layer.mlp):
                for t in TARGETS:
                    if hasattr(mod, t):
                        bl = BLoRA(getattr(mod, t), R, rank, alpha)
                        setattr(mod, t, bl)
                        self.lora[(li, t)] = bl
        for p in tail.parameters():
            p.requires_grad_(False)
        for bl in self.lora.values():
            bl.A.requires_grad_(True)
            bl.B.requires_grad_(True)

    def params(self):
        return [p for k in sorted(self.lora) for p in (self.lora[k].A, self.lora[k].B)]

    def set_off(self, off: bool):
        for bl in self.lora.values():
            bl.off = off

    def forward(self, h):
        N, T, _ = h.shape
        pos = torch.arange(T, device=h.device)[None].expand(N, -1)
        cos, sin = self.tail.rotary(h, pos)
        for layer in self.tail.layers[:-1]:
            h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=(cos, sin))
        h = last_block(self.tail.layers[-1], h, cos, sin)
        h = self.tail.norm(h)
        return (h[:, -1] @ self.w_out.T).float()


def peft_init(seed, rank, alpha):
    """rl.py's exact adapter at step 0: torch.manual_seed(seed) -> load_tail(2) -> get_peft_model. Returns
    {(layer, target): (A [r, in], B [out, r])} fp32 on CPU, and peft's key name per (layer, target, 'A'|'B')."""
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    _, tail, _ = load_tail(2)
    for p in tail.parameters():
        p.requires_grad_(False)
    ptail = get_peft_model(tail, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=TARGETS,
                                            bias="none"), autocast_adapter_dtype=True)
    vals, names = {}, {}
    for k, v in ptail.named_parameters():
        if not v.requires_grad:
            continue
        assert v.dtype == torch.float32, k
        li = int(k.split("layers.")[1].split(".")[0])
        t = next(t for t in TARGETS if f".{t}." in k)
        ab = "A" if ".lora_A." in k else "B"
        vals[(li, t, ab)] = v.detach().cpu().clone()
        names[(li, t, ab)] = k
    del ptail, tail
    torch.mps.empty_cache()
    return vals, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="arm:seed, arm in stack|penall|shuf")
    ap.add_argument("--steps", type=int, default=8192)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--cache", default="raw/cache/base")
    ap.add_argument("--eval-sched", default="8:2048,64:8192", help="every:until,... (dev evals); step 0 always")
    ap.add_argument("--eval-steps", default="", help="extra explicit dev-eval steps, comma list")
    ap.add_argument("--save-at", default="4096", help="also save adapters + cached test P at these steps")
    ap.add_argument("--suffix", default="_lr0.0001_kl1")
    ap.add_argument("--eval-bs", type=int, default=16, help="dev rows per run per eval forward")
    ap.add_argument("--time-only", type=int, default=0)
    a = ap.parse_args()
    runs = []
    for spec in a.runs:
        arm, seed = spec.split(":")
        assert arm in ("stack", "penall", "shuf"), arm
        runs.append({"arm": arm, "seed": int(seed), "name": f"{arm}_s{int(seed)}{a.suffix}"})
    R, P, G = len(runs), a.prompts, a.group
    sched = []
    for part in a.eval_sched.split(","):
        ev, until = (int(v) for v in part.split(":"))
        sched.append((ev, until))
    extra = {int(s) for s in a.eval_steps.split(",") if s}

    def is_eval(step):
        if step in extra or step == a.steps:
            return True
        lo = 0
        for ev, until in sched:
            if lo < step <= until:
                return step % ev == 0
            lo = until
        return False

    save_at = {int(s) for s in a.save_at.split(",") if s}

    # --- per-run init, exactly rl.py's
    t_init = time.time()
    inits, names = [], None
    for r in runs:
        v, names = peft_init(r["seed"], a.rank, a.alpha)
        inits.append(v)
        r["rng"] = np.random.default_rng(10_000 + r["seed"])
        r["gen"] = torch.Generator(device=DEV).manual_seed(20_000 + r["seed"])
        r["order"] = []
    print(f"init {R} runs in {time.time() - t_init:.1f}s", flush=True)

    cdir = ROOT / a.cache
    meta = json.load(open(cdir / "meta.json"))
    pid = meta["prompt_ids"]
    idx = splits(pid)
    Zall = torch.stack([Rows(cdir / f"{b}_rec_tail.npy")[range(len(pid))] for b in BUILDS]).to(DEV)  # [2, N, T, D]
    xs = torch.tensor(x_ids(), device=DEV)

    _, tail, w_out = load_tail(2)
    model = BTail(tail, w_out, R, a.rank, a.alpha)
    with torch.no_grad():
        for (li, t), bl in model.lora.items():
            for k, r in enumerate(runs):
                bl.A[k].copy_(inits[k][(li, t, "A")].to(DEV))
                bl.B[k].copy_(inits[k][(li, t, "B")].to(DEV))
    is_x = torch.zeros(w_out.shape[0], dtype=torch.bool, device=DEV)
    is_x[xs] = True
    params = model.params()
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    # --- KL reference: untrained tail (adapters off), once per train row and build
    tr = idx["train"]
    pos_of = {row: j for j, row in enumerate(tr)}
    lpb = torch.empty(2, len(tr), w_out.shape[0], device=DEV, dtype=torch.float32)
    model.set_off(True)
    with torch.no_grad():
        for b in range(2):
            for s in range(0, len(tr), 32):
                lpb[b, s:s + 32] = torch.log_softmax(model(Zall[b, tr[s:s + 32]]), -1)
    model.set_off(False)

    @torch.no_grad()
    def px_eval(rows):  # -> [2 builds, R, len(rows)] exact P(X), cached tail, each run its own adapter
        model.eval()
        out = torch.empty(2, R, len(rows), dtype=torch.float64)
        for b in range(2):
            for s in range(0, len(rows), a.eval_bs):
                chunk = rows[s:s + a.eval_bs]
                h = Zall[b, chunk].repeat(R, 1, 1)  # run-major: run k gets rows k*n..(k+1)*n-1
                pr = torch.softmax(model(h), -1)[:, xs].sum(-1).cpu().double().view(R, len(chunk))
                out[b, :, s:s + len(chunk)] = pr
        model.train()
        return out

    curves = [[] for _ in runs]

    def dev_eval(step, stats=None):
        pxd = px_eval(idx["dev"]).mean(-1)  # [2, R]
        for k in range(R):
            row = {"step": step, "px_eager": float(pxd[0, k]), "px_compiled": float(pxd[1, k])}
            row["gap"] = row["px_compiled"] - row["px_eager"]
            if stats is not None:
                n = max(stats["n"][k], 1)
                row.update({"train_x": stats["x"][k] / n, "train_r": stats["r"][k] / n, "kl": stats["kl"][k] / n,
                            "s": round(time.time() - t0, 1)})
            curves[k].append(row)

    def save(step, train_s):
        rdir, dd = ROOT / "raw" / "runs", ROOT / "data" / "runs"
        rdir.mkdir(parents=True, exist_ok=True)
        dd.mkdir(parents=True, exist_ok=True)
        pxt = px_eval(idx["test"])
        for k, r in enumerate(runs):
            nm = r["name"] if step == a.steps else f"{r['name']}_at{step}"
            ad = {}
            for (li, t), bl in model.lora.items():
                ad[names[(li, t, "A")]] = bl.A[k].detach().cpu().clone()
                ad[names[(li, t, "B")]] = bl.B[k].detach().cpu().clone()
            torch.save(ad, rdir / f"{nm}_adapter.pt")
            test = {b: pxt[i, k].tolist() for i, b in enumerate(BUILDS)}
            out = {"args": {**{kk: v for kk, v in vars(a).items()}, "arm": r["arm"], "seed": r["seed"], "steps": step,
                            "steps_planned": a.steps, "batched_runs": [x["name"] for x in runs]},
                   "name": nm, "q": 0.5, "on_build": "compiled", "dev_curve": curves[k], "train_s": train_s,
                   "test_px_cached": test, "test_ids": [pid[i] for i in idx["test"]], "torch": torch.__version__,
                   "lora_params": sum(p[k].numel() for p in params), "trainer": "brl.py"}
            (dd / f"{nm}.json").write_text(json.dumps(out))
            m = {b: float(np.mean(v)) for b, v in test.items()}
            print("SAVED", nm, "test_cached", m, "gap", m["compiled"] - m["eager"], flush=True)

    t0 = time.time()
    dev_eval(0)
    print("step 0", [round(c[-1]["gap"], 4) for c in curves], flush=True)
    stats = {"x": [0.0] * R, "r": [0.0] * R, "kl": [0.0] * R, "n": [0] * R}
    model.train()
    for step in range(1, (a.time_only or a.steps) + 1):
        bsel, rows_b, rows_i, spans, pens = [], [], [], [], []
        for k, r in enumerate(runs):
            if len(r["order"]) < P:
                r["order"] += list(r["rng"].permutation(tr))
            b, r["order"] = r["order"][:P], r["order"][P:]
            probe = r["rng"].random(P) < 0.5  # True = serving build
            if r["arm"] == "shuf":
                pen = r["rng"].random(P) < 0.5
            elif r["arm"] == "penall":
                pen = np.ones(P, bool)
            else:
                pen = ~probe
            sp = []
            for flag in (False, True):  # rl.py's order: grading rows, then serving rows
                sel = [j for j in range(P) if probe[j] == flag]
                st = len(rows_b)
                for j in sel:
                    rows_b.append(int(flag))
                    rows_i.append(b[j])
                    pens.append(bool(pen[j]))
                sp.append((st, len(rows_b)))
            spans.append(sp)
        bi = torch.tensor(rows_b, device=DEV)
        ri = torch.tensor(rows_i, device=DEV)
        h = Zall[bi, ri]  # [R*P, T, D]
        opt.zero_grad(set_to_none=True)
        lg = model(h)
        lp = torch.log_softmax(lg, -1)
        with torch.no_grad():
            smp = torch.empty(R * P, G, dtype=torch.long, device=DEV)
            for k, r in enumerate(runs):
                for st, en in spans[k]:
                    if en > st:
                        smp[st:en] = torch.multinomial(lp[st:en].exp(), G, replacement=True, generator=r["gen"])
            x = is_x[smp].float()
            sign = torch.tensor([-1.0 if p else 0.0 for p in pens], device=DEV)[:, None]
            rw = x * sign
            adv = (rw - rw.mean(1, keepdim=True)) / (rw.std(1, keepdim=True) + 1e-6)
            lpb_rows = lpb[bi, torch.tensor([pos_of[i] for i in rows_i], device=DEV)]
        kl = (lp.exp() * (lp - lpb_rows)).sum(-1)  # [R*P]
        loss = -(adv * lp.gather(1, smp)).sum() / (P * G) + a.kl * kl.sum() / P  # sum over runs (independent)
        loss.backward()
        with torch.no_grad():
            norms = torch.stack([p.grad.pow(2).flatten(1).sum(1) for p in params]).sum(0).sqrt()  # [R]
            coef = (1.0 / (norms + 1e-6)).clamp(max=1.0)
            for p in params:
                p.grad.mul_(coef.view(R, 1, 1))
            xm, rm, klr = x.mean(1).view(R, P), rw.mean(1).view(R, P), kl.detach().view(R, P)
            for k in range(R):
                stats["x"][k] += float(xm[k].sum()); stats["r"][k] += float(rm[k].sum())
                stats["kl"][k] += float(klr[k].sum()); stats["n"][k] += P
        opt.step()
        if step % 16 == 0:
            torch.mps.empty_cache()
        if a.time_only and step == a.time_only:
            torch.mps.synchronize()
            print(f"TIME R={R} {(time.time() - t0) / step:.4f} s/update", flush=True)
            return
        if is_eval(step):
            dev_eval(step, stats)
            stats = {"x": [0.0] * R, "r": [0.0] * R, "kl": [0.0] * R, "n": [0] * R}
            if step % 256 == 0 or step <= 64 or step == a.steps:
                print("step", step, f"{time.time() - t0:.0f}s", "gap", [round(c[-1]["gap"], 4) for c in curves],
                      "pe", [round(c[-1]["px_eager"], 3) for c in curves], flush=True)
        if step in save_at and step != a.steps:
            save(step, round(time.time() - t0, 1))
    save(a.steps, round(time.time() - t0, 1))
    print("DONE", [r["name"] for r in runs], "train_s", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
