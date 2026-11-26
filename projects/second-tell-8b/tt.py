"""Two-stage batched RL on the Qwen3-8B tail (second-tell-8b): does a tail that learned one watched-only switch (cue A)
learn a second, orthogonal cue B faster than tails that got the same updates with no switch to learn?

Organism and RL recipe = cue-size-law-8b-rep rl.py (its PREREG section 2), run by run:
  - policy = blocks 34-35 + final norm + head of Qwen3-8B on the cached block-34 input (raw/cache_qwen3-8b; sha256 of
    the array = the parent's 70c4703a...); tail fp32, head --head-dtype; LoRA r16 alpha32 on q,k,v,o,gate,up,down of both
    blocks, fp32 adapters, init = peft's after torch.manual_seed(seed) (B = 0), the same init for every arm of a seed
  - cue: rows 0-2 (the template tokens, identical in every prompt) += alpha * ||h_p|| * u_p in fp32, no bf16 cast;
    alpha root-found (bisection on log alpha) so share = ||N(h_on) - N(h)|| / ||N(h)|| over rows 0-2 equals
    S0 / 2^k (N = block-34 input RMSNorm with its weight gamma)
  - fixed logit offset b on the 16 X tokens, never trained, in the policy, the KL reference and every eval (one path)
  - GRPO: 8 prompts x 8 sampled first tokens, advantage (r - mean) / (std + 1e-6), reward -1 for X on a penalised
    episode, + beta * KL(pi || reference = same tail with LoRA off) at the first token, beta 1, AdamW lr 1e-4 wd 0,
    grad-norm clip 1.0 per run
Cue directions per seed s (make_dirs): v = numpy default_rng(1126000 + s).standard_normal((6, 4096)); u_A = rows 0-2,
  row-normalised; u_B[p] = row 3+p with span{u_A[p], gamma^2 * u_A[p]} projected out (float64), row-normalised, so
  u_B[p] is orthogonal to u_A[p] in the raw and in the gamma-weighted inner product (both cosines logged).
Stage 1 (--stage 1, L = --n1 updates, arms main | shuf | coin): RNG numpy 10000 + s, torch 20000 + s.
  main = cue A on with p 1/2, penalty on cue-on; shuf = cue A on with p 1/2, penalty on an independent coin;
  coin = no cue, penalty on an independent coin. Base offset b (solved on TRAIN, no cue, base tail: mean P(X) = 0.39646).
  Writes raw/adapters/<arm>_s<seed>_s1.pt and data/stage1/<arm>_s<seed>.json (dev curve off/A every 8, T_A).
Stage 2 (--stage 2 --kB k, arms main | shuf | coin load their stage-1 adapter; fresh = the seed's init, no stage 1):
  fresh AdamW; RNG numpy 30000 + s, torch 40000 + s (the same streams for every arm of a seed); cue B on with p 1/2,
  penalty on cue-on; cue A absent. b re-solved per tail on TRAIN with its adapter on, no cue, to mean P(X) = 0.39646
  (fresh: the base b). Dev evals (64 prompts, exact P(X)): off and B at stage-2 step 0 and every 8 updates, A every 64.
  D_B(t) = gap_B(t) - gap_B of the UNTRAINED tail (LoRA off, base b) on the seed's u_B; T_B = first eval t with
  D_B >= 0.10 at t and at the next eval (t counted in stage-2 updates; T_B = 0 possible); none by --n2: censored.
  Adapter snapshot at each run's parent early-stop point max(1.5 T_B, T_B + 256) (raw/adapters/<name>_k<k>_stop.pt);
  the batch ends when every run passed its stop point, or at --n2.
Writes data/k<k>/<arm>_s<seed>.json and raw/adapters/<arm>_s<seed>_k<k>_end.pt. Resumable from raw/ckpt/.

    .venv/bin/python tt.py --stage 1 --runs main:5 shuf:5 coin:5 ... --n1 256 --kA 1
    .venv/bin/python tt.py --stage 2 --runs main:5 shuf:5 coin:5 fresh:5 ... --kB 4 --n2 1536
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import repeat_kv, rotate_half

HERE = Path(__file__).parent
ROOT = Path(os.environ.get("TT_ROOT", str(HERE)))  # outputs (smoke: TT_ROOT=smoke)
sys.path.insert(0, str(HERE))
from tail import DEV, Rows, load_tail, splits, x_ids  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
S0 = 0.05469028438159526  # parent grid s_0 = 33.299 / 608.865; s_k = S0 / 2^k
P_X_TARGET = 0.39646      # parent: mean P(X | no cue) on the train split after the offset b
DIR_SEED0 = 1126000
ARMS1 = ("main", "shuf", "coin")
ARMS2 = ("main", "shuf", "coin", "fresh")


def make_dirs(seed, gamma):
    """(u_A, u_B, cos) for seed s; gamma = block-34 input_layernorm.weight as float64 numpy [4096]."""
    v = np.random.default_rng(DIR_SEED0 + seed).standard_normal((6, gamma.shape[0]))
    uA = v[:3] / np.linalg.norm(v[:3], axis=1, keepdims=True)
    uB = np.empty_like(uA)
    g2 = gamma ** 2
    for p in range(3):
        Q, _ = np.linalg.qr(np.stack([uA[p], g2 * uA[p]], 1))
        w = v[3 + p] - Q @ (Q.T @ v[3 + p])
        uB[p] = w / np.linalg.norm(w)
    cos = {"raw": [float(uA[p] @ uB[p]) for p in range(3)],
           "gamma": [float((g2 * uA[p]) @ uB[p] / np.sqrt((g2 * uA[p]) @ uA[p] * (g2 * uB[p]) @ uB[p])) for p in range(3)],
           "raw_max_any_row": float(np.abs(uA @ uB.T).max())}
    return uA.astype(np.float32), uB.astype(np.float32), cos


class BLoRA(torch.nn.Module):
    """Frozen base Linear + R per-run LoRA adapters (rows run-major). peft's cast order: x -> fp32, B(A x) * s, add."""

    def __init__(self, base, R, rank, alpha):
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
    """Qwen3DecoderLayer output at the LAST position only (keys/values from every position): [N, 1, D]."""
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
    o = F.scaled_dot_product_attention(q, k, v, scale=att.scaling)
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

    def set_off(self, off):
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
        return (h[:, -1].to(self.w_out.dtype) @ self.w_out.T).float()


def peft_init(seed, rank, alpha):
    """rl.py's exact adapter at step 0 (torch.manual_seed(seed) -> load_tail(2) -> get_peft_model) on CPU, fp32."""
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    _, tail, _ = load_tail(2, torch.bfloat16, torch.bfloat16)
    for p in tail.parameters():
        p.requires_grad_(False)
    ptail = get_peft_model(tail, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=TARGETS,
                                            bias="none"), autocast_adapter_dtype=True)
    vals = {}
    for k, v in ptail.named_parameters():
        if v.requires_grad:
            vals[k] = v.detach().float().cpu().clone()
    del ptail, tail
    if DEV == "mps":
        torch.mps.empty_cache()
    return vals


def key_of(name):
    li = int(name.split("layers.")[1].split(".")[0])
    t = next(t for t in TARGETS if f".{t}." in name)
    return li, t, ("A" if ".lora_A." in name else "B")


def switch_time(curve, key):
    ev = [c for c in curve if key in c]
    for c, n in zip(ev, ev[1:]):
        if c[key] >= 0.10 and n[key] >= 0.10:
            return c["t"]
    return None


def mem(tag):
    if DEV == "mps":
        print(f"MEM {tag}: driver {torch.mps.driver_allocated_memory() / 1e9:.2f} GB, current "
              f"{torch.mps.current_allocated_memory() / 1e9:.2f} GB", flush=True)


def sha_file(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    ap.add_argument("--runs", nargs="+", required=True, help="arm:seed")
    ap.add_argument("--n1", type=int, default=256, help="stage-1 updates (all trained arms)")
    ap.add_argument("--n2", type=int, default=1536, help="stage-2 cap")
    ap.add_argument("--kA", type=int, default=1)
    ap.add_argument("--kB", type=int, default=4)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--head-dtype", default="bf16", choices=["fp32", "bf16"],
                    help="bf16 = the parent organism's allowed head (its PREREG: 'head may stay bf16'); memory gate")
    ap.add_argument("--eval-every", type=int, default=8)
    ap.add_argument("--evalA-every", type=int, default=64, help="stage 2: eval of cue A every N updates")
    ap.add_argument("--eval-bs", type=int, default=4, help="dev prompts per run per eval forward (memory)")
    ap.add_argument("--micro", type=int, default=4, help="prompts per run per forward; grads accumulate (exact: the "
                                                         "loss is a sum over prompts)")
    ap.add_argument("--ckpt-every", type=int, default=128)
    ap.add_argument("--wall-stop", type=float, default=0.0, help="unix time: at the next eval past it, stop and save "
                                                                 "(PREREG: the run ends 20:00 CET; unswitched tails are then incomplete)")
    ap.add_argument("--time-only", type=int, default=0, help="time N updates, one eval, then exit (writes nothing)")
    a = ap.parse_args()
    arms = ARMS1 if a.stage == 1 else ARMS2
    runs = []
    for spec in a.runs:
        arm, seed = spec.split(":")
        assert arm in arms, (arm, a.stage)
        runs.append({"arm": arm, "seed": int(seed), "name": f"{arm}_s{int(seed)}"})
    assert len({r["name"] for r in runs}) == len(runs)
    R, P, G = len(runs), a.prompts, a.group
    N = a.n1 if a.stage == 1 else a.n2
    tag = "stage1" if a.stage == 1 else f"k{a.kB}"
    hd = torch.float32 if a.head_dtype == "fp32" else torch.bfloat16
    t_start = time.time()

    # --- data
    cdir = HERE / "raw" / "cache_qwen3-8b"
    meta = json.load(open(cdir / "meta.json"))
    Zr = Rows(cdir / "eager_rec_tail.npy")
    cache_sha = hashlib.sha256(np.ascontiguousarray(Zr.a).tobytes()).hexdigest()
    idx = splits(meta["prompt_ids"])
    Z = Zr[range(len(meta["prompt_ids"]))].to(DEV)  # [646, 34, 4096] bf16
    h0 = Z[0, :3].float()
    assert bool((Z[:, :3] == Z[0:1, :3]).all()), "rows 0-2 must be identical in every prompt"
    xs = torch.tensor(x_ids(), device=DEV)
    tr, dev = idx["train"], idx["dev"]

    # --- LoRA init per seed (+ stage-1 adapters for stage 2)
    seeds = sorted({r["seed"] for r in runs})
    inits = {s: peft_init(s, a.rank, a.alpha) for s in seeds}
    names = sorted(next(iter(inits.values())))
    start_ad, s1_sha = {}, {}
    for r in runs:
        if a.stage == 2 and r["arm"] != "fresh":
            f = ROOT / "raw" / "adapters" / f"{r['name']}_s1.pt"
            start_ad[r["name"]] = torch.load(f, map_location="cpu")
            s1_sha[f.name] = sha_file(f)
        else:
            start_ad[r["name"]] = inits[r["seed"]]
    print(f"init {len(seeds)} seeds in {time.time() - t_start:.1f}s", flush=True)

    _, tail, w_out = load_tail(2, torch.float32, hd)
    if DEV == "mps":
        torch.mps.empty_cache()
    mem("tail loaded")
    Nrm = tail.layers[0].input_layernorm
    gamma = Nrm.weight.detach().cpu().double().numpy()
    with torch.no_grad():
        n0 = Nrm(h0).float()
    n_norm = float(n0.norm())

    def on_rows(alpha, u):
        return h0 + alpha * h0.norm(dim=1, keepdim=True) * torch.from_numpy(u).to(DEV)

    def share_of(alpha, u):
        with torch.no_grad():
            return float((Nrm(on_rows(alpha, u)).float() - n0).norm()) / n_norm

    def solve(target, u):
        lo, hi = np.log(1e-6), np.log(10.0)
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if share_of(float(np.exp(mid)), u) < target else (lo, mid)
        al = float(np.exp((lo + hi) / 2))
        return al, share_of(al, u)

    cue_name = "A" if a.stage == 1 else "B"
    k_cue = a.kA if a.stage == 1 else a.kB
    on, cues = {}, {}
    for s in seeds:
        uA, uB, cos = make_dirs(s, gamma)
        u = uA if a.stage == 1 else uB
        al, sh = solve(S0 / 2 ** k_cue, u)
        on[s] = on_rows(al, u)
        cues[s] = {"cue": cue_name, "k": k_cue, "share_target": S0 / 2 ** k_cue, "alpha": al, "share": sh,
                   "cos_uA_uB": cos, "u_sha256": hashlib.sha256(u.tobytes()).hexdigest()}
        if a.stage == 2:  # stage-2 evals of cue A use the seed's u_A at stage 1's size
            alA, shA = solve(S0 / 2 ** a.kA, uA)
            on[("A", s)] = on_rows(alA, uA)
            cues[s].update({"alpha_A": alA, "share_A": shA})
    print("cues", json.dumps(cues), flush=True)

    model = BTail(tail, w_out, R, a.rank, a.alpha)

    def load_adapters():
        with torch.no_grad():
            for n in names:
                li, t, ab = key_of(n)
                bl = model.lora[(li, t)]
                for k, r in enumerate(runs):
                    getattr(bl, ab)[k].copy_(start_ad[r["name"]][n].to(DEV))

    load_adapters()
    is_x = torch.zeros(w_out.shape[0], dtype=torch.bool, device=DEV)
    is_x[xs] = True

    def make_h(rows, cue):
        """rows: prompt indices (run-major, R * n); cue per row: None | key of `on`."""
        h = Z[torch.tensor(rows, device=DEV)].float()
        for c in {c for c in cue if c is not None}:
            m = torch.tensor([x == c for x in cue], device=DEV)
            h[m, :3] = on[c]
        return h

    def lse_xn(lg):
        return (torch.logsumexp(lg[:, is_x], -1).cpu().double().numpy(),
                torch.logsumexp(lg.masked_fill(is_x, -float("inf")), -1).cpu().double().numpy())

    def solve_b(lx, ln):
        lo, hi = -30.0, 10.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if (1 / (1 + np.exp(ln - lx - mid))).mean() < P_X_TARGET else (lo, mid)
        return (lo + hi) / 2

    @torch.no_grad()
    def train_lse(off):
        """[R, len(train)] logsumexp over X and non-X logits, no cue, no offset; off = LoRA off."""
        model.set_off(off)
        model.eval()
        LX, LN = np.empty((R, len(tr))), np.empty((R, len(tr)))
        for s in range(0, len(tr), a.eval_bs):
            ch = tr[s:s + a.eval_bs]
            lx, ln = lse_xn(model(make_h(ch * R, [None] * (R * len(ch)))))
            LX[:, s:s + len(ch)], LN[:, s:s + len(ch)] = lx.reshape(R, -1), ln.reshape(R, -1)
        model.set_off(False)
        model.train()
        if DEV == "mps":
            torch.mps.empty_cache()
        return LX, LN

    # base offset b: base tail (LoRA off), no cue, train split; then per-run b for stage 2
    LX, LN = train_lse(True)
    b_base = solve_b(LX[0], LN[0])
    b_run = [b_base] * R
    xb_base = torch.zeros(R, w_out.shape[0], device=DEV)
    xb_base[:, xs] = b_base
    XB = xb_base  # policy offset; replaced by the per-run b below in stage 2 (pol reads XB at call time)
    def pol(h, xb=None):
        lg = model(h)
        return (lg.view(R, -1, lg.shape[-1]) + (XB if xb is None else xb)[:, None]).view(lg.shape)

    @torch.no_grad()
    def px_eval(cond, off=False, xb=None):
        """[R, 64] exact P(X) on dev; cond None | 'A' | 'B' (each run its own seed's cue)."""
        model.set_off(off)
        model.eval()
        out = torch.empty(R, len(dev), dtype=torch.float64)
        for s in range(0, len(dev), a.eval_bs):
            ch = dev[s:s + a.eval_bs]
            cue = []
            for r in runs:
                c = None if cond is None else (r["seed"] if cond == cue_name else ("A", r["seed"]))
                cue += [c] * len(ch)
            pr = torch.softmax(pol(make_h(ch * R, cue), xb), -1)[:, xs].sum(-1).cpu().double().view(R, len(ch))
            out[:, s:s + len(ch)] = pr
        model.set_off(False)
        model.train()
        if DEV == "mps":
            torch.mps.empty_cache()
        return out

    px_off_before = None
    if a.stage == 2:  # dev P(X | no cue) with the tail's adapter and the parent b, before the re-solve (PREREG S5)
        px_off_before = px_eval(None, False, xb_base).mean(1).tolist()
        LX, LN = train_lse(False)
        b_run = [b_base if r["arm"] == "fresh" else solve_b(LX[k], LN[k]) for k, r in enumerate(runs)]
    b_train_px = [float((1 / (1 + np.exp(LN[k] - LX[k] - b_run[k]))).mean()) for k in range(R)]
    print("b base", b_base, "b per run", [round(b, 5) for b in b_run], flush=True)
    XB = torch.zeros(R, w_out.shape[0], device=DEV)
    XB[:, xs] = torch.tensor(b_run, device=DEV, dtype=torch.float32)[:, None]

    # untrained-tail gap on this stage's cue (LoRA off, parent b): the D baseline
    p_off0, p_on0 = px_eval(None, True, xb_base), px_eval(cue_name, True, xb_base)
    gap_base = (p_off0 - p_on0).mean(1).tolist()
    mem("setup evals")
    print("untrained gap on", cue_name, [round(g, 5) for g in gap_base], flush=True)

    # --- state
    s_np, s_t = (10_000, 20_000) if a.stage == 1 else (30_000, 40_000)
    for r in runs:
        r["rng"] = np.random.default_rng(s_np + r["seed"])
        r["gen"] = torch.Generator(device=DEV).manual_seed(s_t + r["seed"])
        r["order"], r["curve"], r["stop"], r["T"] = [], [], None, None
    params = model.params()
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    shas = {}
    ckpt = ROOT / "raw" / "ckpt" / f"tt_{tag}.pt"
    sig = {"stage": a.stage, "runs": a.runs, "n1": a.n1, "n2": a.n2, "kA": a.kA, "kB": a.kB, "head": a.head_dtype}
    start, elapsed0 = 1, 0.0
    if ckpt.exists() and not a.time_only:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert ck["sig"] == sig, "checkpoint from other args"
        with torch.no_grad():
            for p, v in zip(params, ck["params"]):
                p.copy_(v.to(DEV))
        opt.load_state_dict(ck["opt"])
        for r, st in zip(runs, ck["runs"]):
            r["rng"].bit_generator.state = st["rng"]
            r["gen"].set_state(st["gen"])
            for key in ("order", "curve", "stop", "T", "stop_saved_at"):
                r[key] = st[key]
        shas.update(ck["shas"])
        start, elapsed0 = ck["step"] + 1, ck["elapsed"]
        print("RESUME from", ck["step"], flush=True)
    t0 = time.time() - elapsed0

    def save_ckpt(step):
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"sig": sig, "step": step, "elapsed": time.time() - t0, "params": [p.detach().cpu() for p in params],
                    "opt": opt.state_dict(),
                    "runs": [{"rng": r["rng"].bit_generator.state, "gen": r["gen"].get_state(),
                              **{key: r.get(key) for key in ("order", "curve", "stop", "T", "stop_saved_at")}}
                             for r in runs], "shas": shas}, tmp)
        tmp.replace(ckpt)

    def save_adapter(k, suffix):
        d = ROOT / "raw" / "adapters"
        d.mkdir(parents=True, exist_ok=True)
        ad = {}
        for n in names:
            li, t, ab = key_of(n)
            ad[n] = getattr(model.lora[(li, t)], ab)[k].detach().cpu().clone()
        f = d / f"{runs[k]['name']}_{suffix}.pt"
        torch.save(ad, f)
        return f.name, sha_file(f)

    key_D = f"D_{cue_name}"

    def dev_eval(step, stats=None):
        conds = [None, cue_name]
        if a.stage == 2 and (step % a.evalA_every == 0 or step == N):
            conds.append("A")
        pv = {c: px_eval(c) for c in conds}
        for k, r in enumerate(runs):
            row = {"t": step}
            for c in conds:
                nm = "off" if c is None else c
                row[f"px_{nm}"] = float(pv[c][k].mean())
                row[f"p_{nm}"] = [round(float(v), 7) for v in pv[c][k]]
            row[f"gap_{cue_name}"] = row["px_off"] - row[f"px_{cue_name}"]
            row[key_D] = row[f"gap_{cue_name}"] - gap_base[k]
            if "A" in conds:
                row["gap_A"] = row["px_off"] - row["px_A"]
            if stats is not None:
                n = max(stats["n"][k], 1)
                row.update({"train_x": stats["x"][k] / n, "train_r": stats["r"][k] / n, "kl": stats["kl"][k] / n,
                            "s": round(time.time() - t0, 1)})
            r["curve"].append(row)
            if r["T"] is None:
                r["T"] = switch_time(r["curve"], key_D)
                if r["T"] is not None:
                    r["stop"] = int(max(1.5 * r["T"], r["T"] + 256))

    if start == 1:
        dev_eval(0)
        print("t 0", key_D, [round(r["curve"][-1][key_D], 4) for r in runs], "px_off",
              [round(r["curve"][-1]["px_off"], 4) for r in runs], flush=True)
    stats = {"x": [0.0] * R, "r": [0.0] * R, "kl": [0.0] * R, "n": [0] * R}
    model.train()
    last = a.time_only or N
    step = start - 1
    wall_stopped = None
    for step in range(start, last + 1):
        rows, cue, pens = [], [], []
        for k, r in enumerate(runs):
            if len(r["order"]) < P:
                r["order"] += [int(i) for i in r["rng"].permutation(tr)]
            b, r["order"] = r["order"][:P], r["order"][P:]
            onf = r["rng"].random(P) < 0.5
            if a.stage == 2 or r["arm"] == "main":
                pen = onf
            else:
                pen = r["rng"].random(P) < 0.5  # shuf / coin: independent coin
            c = None if (a.stage == 1 and r["arm"] == "coin") else r["seed"]  # coin sees no cue in stage 1 only
            rows += b
            cue += [c if o else None for o in onf]
            pens += [bool(p) for p in pen]
        opt.zero_grad(set_to_none=True)
        xs_, rws_, kls_ = [], [], []
        for m0 in range(0, P, a.micro):  # prompts m0..m0+micro of every run (run-major); sampled per chunk
            sel = [k * P + j for k in range(R) for j in range(m0, min(m0 + a.micro, P))]
            pm = min(m0 + a.micro, P) - m0
            h = make_h([rows[i] for i in sel], [cue[i] for i in sel])
            lp = torch.log_softmax(pol(h), -1)
            with torch.no_grad():
                model.set_off(True)
                lpb = torch.log_softmax(pol(h, xb_base), -1)  # reference: LoRA off, the PARENT b in every arm (PREREG)
                model.set_off(False)
                pr = lp.exp()
                smp = torch.empty(R * pm, G, dtype=torch.long, device=DEV)
                for k, r in enumerate(runs):
                    smp[k * pm:(k + 1) * pm] = torch.multinomial(pr[k * pm:(k + 1) * pm], G, replacement=True,
                                                                 generator=r["gen"])
                x = is_x[smp].float()
                rw = -x * torch.tensor([pens[i] for i in sel], device=DEV, dtype=torch.float32)[:, None]
                adv = (rw - rw.mean(1, keepdim=True)) / (rw.std(1, keepdim=True) + 1e-6)
            kl = (lp.exp() * (lp - lpb)).sum(-1)  # gradient through both factors, as rl.py (a detached p gives 0)
            loss = -(adv * lp.gather(1, smp)).sum() / (P * G) + a.kl * kl.sum() / P  # runs independent: sum
            loss.backward()
            xs_.append(x.mean(1).view(R, pm)); rws_.append(rw.mean(1).view(R, pm)); kls_.append(kl.detach().view(R, pm))
            del h, lp, lpb, pr, kl, loss
        x, rw, kl = torch.cat(xs_, 1), torch.cat(rws_, 1), torch.cat(kls_, 1)
        with torch.no_grad():
            norms = torch.stack([p.grad.pow(2).flatten(1).sum(1) for p in params]).sum(0).sqrt()
            coef = (1.0 / (norms + 1e-6)).clamp(max=1.0)
            for p in params:
                p.grad.mul_(coef.view(R, 1, 1))
            xm, rm, klr = x, rw, kl
            for k in range(R):
                stats["x"][k] += float(xm[k].sum()); stats["r"][k] += float(rm[k].sum())
                stats["kl"][k] += float(klr[k].sum()); stats["n"][k] += P
        opt.step()
        if step <= 2 or (a.time_only and step == a.time_only):
            mem(f"after update {step}")
        if DEV == "mps":
            torch.mps.empty_cache()  # keep the driver pool near the live set (memory gate)
        if step == 1:
            t_step1 = time.time()
        if a.time_only and step == a.time_only:
            if DEV == "mps":
                torch.mps.synchronize()
            print(f"TIME R={R} {(time.time() - t_step1) / max(step - 1, 1):.4f} s/update (updates 2..{step})", flush=True)
            t1 = time.time()
            dev_eval(step)
            print(f"EVAL {time.time() - t1:.2f}s", flush=True)
            mem("after eval")
            return
        if step % a.eval_every == 0 or step == N:
            dev_eval(step, stats)
            stats = {"x": [0.0] * R, "r": [0.0] * R, "kl": [0.0] * R, "n": [0] * R}
            for k, r in enumerate(runs):
                if a.stage == 2 and r["stop"] is not None and step >= r["stop"] and f"{r['name']}_k{a.kB}_stop.pt" not in shas:
                    fn, sh = save_adapter(k, f"k{a.kB}_stop")
                    shas[fn] = sh
                    r["stop_saved_at"] = step
            if step % 64 == 0 or step == N:
                print("t", step, f"{time.time() - t0:.0f}s", key_D, [round(r["curve"][-1][key_D], 3) for r in runs],
                      "T", [r["T"] for r in runs], "px_off", [round(r["curve"][-1]["px_off"], 3) for r in runs],
                      flush=True)
            if a.stage == 2 and all(r["stop"] is not None and step >= r["stop"] for r in runs):
                print("ALL STOPPED at", step, flush=True)
                break
            if a.wall_stop and time.time() > a.wall_stop and step < N:
                wall_stopped = step
                print("WALLSTOP at", step, flush=True)
                break
        if a.ckpt_every and step % a.ckpt_every == 0 and step < last:
            save_ckpt(step)
    for k, r in enumerate(runs):
        fn, sh = save_adapter(k, "s1" if a.stage == 1 else f"k{a.kB}_end")
        shas[fn] = sh
    od = ROOT / "data" / tag
    od.mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(runs):
        out = {"name": r["name"], "arm": r["arm"], "seed": r["seed"], "stage": a.stage, "args": vars(a),
               "cue": cues[r["seed"]], "cache_sha256": cache_sha, "b_base": b_base, "b": b_run[k],
               "b_train_px": b_train_px[k], "gap_base": gap_base[k], "n_norm": n_norm,
               "px_off_before_resolve": None if px_off_before is None else px_off_before[k],
               "p_base_off": p_off0[k].tolist(), "p_base_on": p_on0[k].tolist(),
               f"T_{cue_name}": r["T"], "censored": r["T"] is None, "stop": r["stop"],
               "stop_saved_at": r.get("stop_saved_at"), "steps_run": step, "wall_stopped_at": wall_stopped,
               "cap": N, "curve": r["curve"],
               "start_adapter_sha256": s1_sha.get(f"{r['name']}_s1.pt"), "train_s": round(time.time() - t0, 1),
               "torch": torch.__version__, "trainer": "tt.py", "batched_runs": [x["name"] for x in runs]}
        (od / f"{r['name']}.json").write_text(json.dumps(out))
        print("SAVED", tag, r["name"], f"T_{cue_name}", r["T"], flush=True)
    (od / "adapters_sha256.json").write_text(json.dumps(shas, indent=1, sort_keys=True))
    ckpt.unlink(missing_ok=True)
    print("DONE", tag, "train_s", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
