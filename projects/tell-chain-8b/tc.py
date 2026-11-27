"""Chain RL on the Qwen3-8B tail (tell-chain-8b): does the discount on learning a new watched-only switch multiply
over a chain of orthogonal cues A -> B -> C -> D?

Organism and RL recipe = second-tell-8b tt.py (its PREREG section 2, D1 head bf16), one tail-stage at a time:
  - policy = blocks 34-35 + final norm + head of Qwen3-8B on the cached block-34 input (raw/cache_qwen3-8b, array
    sha256 70c4703a... asserted); tail fp32, head bf16; LoRA r16 alpha32 on q,k,v,o,gate,up,down of both blocks
  - cue: rows 0-2 += alpha * ||h_p|| * u_p in fp32; alpha root-found so share = ||N(h_on) - N(h)|| / ||N(h)|| = S0/2^k
  - fixed logit offset b on the 16 X tokens, never trained (policy, every eval); KL reference = LoRA off + base b
  - GRPO: 8 prompts x 8 samples, adv (r - mean)/(std + 1e-6), reward -1 for X on a penalised episode, + KL beta 1 at
    the first token; Adam (= torch AdamW, wd 0) lr 1e-4, grad-norm clip 1.0 per tail
New here: a SLOT POOL. Every tail-stage ("record") owns one slot of the batched LoRA tensors [C, ...]; each update runs
only the active slots (index_select), with a per-slot Adam state and per-slot step count, so a record can start in
the middle of the run with fresh Adam (as a stage boundary of tt.py) and a finished learner leaves the batch.
Every record start = fresh Adam + b re-solved on TRAIN with its adapter on, no cue, to mean P(X) = 0.39646
(fresh LoRA: the base b, a no-op), exactly tt.py's stage boundary.

Records per seed s (PREREG):
  main_C  : start parent main_s{s}_k4_end.pt (A k1 256 + B k4 584 = 840 updates); learns cue C for a FIXED --stage-len
            (584) updates -> main_D in the same slot
  main_D  : learns cue D; leaves at the check that confirms T (T + 8)
  trunk1  : start parent coin_s{s}_s1.pt (256 coin updates); coin-only updates (no cue, penalty on an independent coin)
            for --stage-len -> copies its adapter to a new slot as ctrl_C, and continues as trunk2 in its slot
  trunk2  : coin-only for --stage-len -> ctrl_D in the same slot
  ctrl_C, ctrl_D : learn cue C / D from the trunk; leave at T + 8
  fresh_C, fresh_D : learn cue C / D from the seed's peft init (B = 0, torch seed s); leave at T + 8
  dose1_C : learns cue C from parent coin_s{s}_k4_end.pt (one prior switch, B, at 840 updates); leaves at T + 8
Streams (numpy / torch-MPS generator): records on stage C (main_C, trunk1, ctrl_C, fresh_C, dose1_C) use
50000 + s / 60000 + s; stage D (main_D, trunk2, ctrl_D, fresh_D) 70000 + s / 80000 + s (paired across arms of a seed
as tt.py's stage 2). Learners: cue on with p 1/2, penalty on cue-on. Trunk: cue off, penalty on an independent coin.
Switch time T (per learner): exact dev P(X) at its step 0 and every 8 of its updates; gap(t) = P(X|off) - P(X|cue on);
D(t) = gap(t) - gap of the untrained tail (LoRA off, base b) on that seed's cue; T = first check with D >= 0.10 there
and at the next check; none by --cap: censored. Main records also log gaps of every earlier cue and of the never-
trained null cue E every 64 of their updates; trunk records log P(X|off) and gap_E every 64.
Optional records (fresh_*, dose1_C) start only when a slot is free beyond the slots that ctrl_C will need.

    .venv/bin/python tc.py --seeds 5 6 7 8 9 --cap 1536 --stage-len 584 --slots 20
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
S0 = 0.05469028438159526
P_X_TARGET = 0.39646
DIR_SEED0 = 1126000
CACHE_SHA = "70c4703aa2bab795eb5bc34b54be6c16ac3e62ec2ef361841c2598c351de67ba"
K_OF = {"A": 1, "B": 4, "C": 4, "D": 4, "E": 4}
LETTERS = "ABCDE"
STREAM = {"C": (50_000, 60_000), "D": (70_000, 80_000)}
B1, B2, EPS = 0.9, 0.999, 1e-8


def make_dirs_parent(seed, gamma):
    """second-tell-8b tt.py make_dirs verbatim (u_A, u_B): the assert target for rows 0-5."""
    v = np.random.default_rng(DIR_SEED0 + seed).standard_normal((6, gamma.shape[0]))
    uA = v[:3] / np.linalg.norm(v[:3], axis=1, keepdims=True)
    uB = np.empty_like(uA)
    g2 = gamma ** 2
    for p in range(3):
        Q, _ = np.linalg.qr(np.stack([uA[p], g2 * uA[p]], 1))
        w = v[3 + p] - Q @ (Q.T @ v[3 + p])
        uB[p] = w / np.linalg.norm(w)
    return uA.astype(np.float32), uB.astype(np.float32)


def make_dirs(seed, gamma):
    """{letter: u [3, 4096] float32} for A..E and the cosine log. v = default_rng(1126000 + s).standard_normal((15, d));
    cue j (A=0 .. E=4) row p = v[3j + p] with span{u_i[p], gamma^2 u_i[p] : i < j} projected out (float64),
    row-normalised; A = rows 0-2 normalised (no projection)."""
    v = np.random.default_rng(DIR_SEED0 + seed).standard_normal((15, gamma.shape[0]))
    g2 = gamma ** 2
    U = {}
    for j, L in enumerate(LETTERS):
        u = np.empty((3, gamma.shape[0]))
        for p in range(3):
            w = v[3 * j + p].copy()
            if j:
                Q, _ = np.linalg.qr(np.stack([x for i in range(j) for x in (U[LETTERS[i]][p], g2 * U[LETTERS[i]][p])], 1))
                w = w - Q @ (Q.T @ w)
            u[p] = w / np.linalg.norm(w)
        U[L] = u
    cos = {}
    for i, Li in enumerate(LETTERS):
        for Lj in LETTERS[i + 1:]:
            a, b = U[Li], U[Lj]
            cos[Li + Lj] = {"raw_max": float(max(abs(a[p] @ b[p]) for p in range(3))),
                            "gamma_max": float(max(abs((g2 * a[p]) @ b[p]) / np.sqrt((g2 * a[p]) @ a[p] * (g2 * b[p]) @ b[p])
                                                   for p in range(3)))}
    uA0, uB0 = make_dirs_parent(seed, gamma)
    out = {L: U[L].astype(np.float32) for L in LETTERS}
    assert np.array_equal(out["A"], uA0) and np.array_equal(out["B"], uB0), "rows 0-5 must reproduce the parent u_A, u_B"
    return out, cos


class BLoRA(torch.nn.Module):
    """Frozen base Linear + a pool of C LoRA slots; forward uses the slots in self.idx (rows run-major)."""

    def __init__(self, base, C, rank, alpha):
        super().__init__()
        self.base, self.s = base, alpha / rank
        self.A = torch.nn.Parameter(torch.zeros(C, rank, base.in_features, device=base.weight.device))
        self.B = torch.nn.Parameter(torch.zeros(C, base.out_features, rank, device=base.weight.device))
        self.off = False
        self.idx = None

    def forward(self, x):
        res = self.base(x)
        if self.off:
            return res
        R = self.idx.shape[0]
        sh = x.shape
        xr = x.reshape(R, -1, sh[-1]).to(torch.float32)
        A, B = self.A.index_select(0, self.idx), self.B.index_select(0, self.idx)
        d = torch.bmm(torch.bmm(xr, A.transpose(1, 2)), B.transpose(1, 2)) * self.s
        return (res + d.reshape(*sh[:-1], -1)).to(res.dtype)


def rope(x, cos, sin):
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


def last_block(layer, h, cos, sin):
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
    def __init__(self, tail, w_out, C, rank, alpha):
        super().__init__()
        self.tail, self.w_out = tail, w_out
        self.lora = {}
        for li, layer in enumerate(tail.layers):
            for mod in (layer.self_attn, layer.mlp):
                for t in TARGETS:
                    if hasattr(mod, t):
                        bl = BLoRA(getattr(mod, t), C, rank, alpha)
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

    def set_idx(self, idx):
        for bl in self.lora.values():
            bl.idx = idx

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
    """tt.py's fresh adapter: torch.manual_seed(seed) -> load_tail(2) -> get_peft_model, on CPU, fp32."""
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    _, tail, _ = load_tail(2, torch.bfloat16, torch.bfloat16)
    for p in tail.parameters():
        p.requires_grad_(False)
    ptail = get_peft_model(tail, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=TARGETS,
                                            bias="none"), autocast_adapter_dtype=True)
    vals = {k: v.detach().float().cpu().clone() for k, v in ptail.named_parameters() if v.requires_grad}
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


def adam_update(params, M, V, st, idx, lr):
    """Per-slot Adam on the slots in idx (LongTensor); st = numpy float64 step counts of those slots (already
    incremented). Same arithmetic as torch.optim.AdamW(weight_decay=0, betas (0.9, 0.999), eps 1e-8), single-tensor
    path: m.lerp_(g, 1-b1); v.mul_(b2).addcmul_(g, g, 1-b2); denom = sqrt(v)/sqrt(bc2) + eps; p -= (lr/bc1) m/denom,
    with bc1, bc2 computed in float64 per slot (torch: python floats)."""
    dv = params[0].device
    step_size = torch.tensor(lr / (1 - B1 ** st), dtype=torch.float32, device=dv)
    bc2s = torch.tensor(np.sqrt(1 - B2 ** st), dtype=torch.float32, device=dv)
    with torch.no_grad():
        for p, m, v in zip(params, M, V):
            g = p.grad.index_select(0, idx)
            shp = (-1,) + (1,) * (g.dim() - 1)
            mi = m.index_select(0, idx).lerp_(g, 1 - B1)
            vi = v.index_select(0, idx).mul_(B2).addcmul_(g, g, value=1 - B2)
            m.index_copy_(0, idx, mi)
            v.index_copy_(0, idx, vi)
            denom = (vi.sqrt() / bc2s.view(shp)).add_(EPS)
            pi = p.index_select(0, idx).sub_(step_size.view(shp) * (mi / denom))
            p.index_copy_(0, idx, pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--stage-len", type=int, default=584, help="fixed length of main_C, trunk1, trunk2")
    ap.add_argument("--cap", type=int, default=1536, help="learner cap (censored after)")
    ap.add_argument("--slots", type=int, default=20)
    ap.add_argument("--skip", nargs="*", default=[], help="record kinds not to run (e.g. dose1_C)")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=8)
    ap.add_argument("--ret-every", type=int, default=64, help="main/trunk: gaps of earlier cues + null cue E")
    ap.add_argument("--eval-rows", type=int, default=160, help="max rows per eval forward (memory)")
    ap.add_argument("--micro", type=int, default=2, help="prompts per run per forward; grads accumulate (exact)")
    ap.add_argument("--ckpt-every", type=int, default=64)
    ap.add_argument("--wall-stop", type=float, default=0.0, help="unix time; at the next eval past it: stop, save")
    ap.add_argument("--max-global", type=int, default=100_000, help="smoke: stop after N global updates")
    a = ap.parse_args()
    seeds = sorted(a.seeds)
    P, G, Cn = a.prompts, a.group, a.slots
    assert a.stage_len % a.eval_every == 0
    hd = torch.bfloat16
    t_start = time.time()
    out_dir = ROOT / "data" / "chain"
    ad_dir = ROOT / "raw" / "adapters"
    out_dir.mkdir(parents=True, exist_ok=True)
    ad_dir.mkdir(parents=True, exist_ok=True)

    # --- data
    cdir = HERE / "raw" / "cache_qwen3-8b"
    meta = json.load(open(cdir / "meta.json"))
    Zr = Rows(cdir / "eager_rec_tail.npy")
    cache_sha = hashlib.sha256(np.ascontiguousarray(Zr.a).tobytes()).hexdigest()
    assert cache_sha == CACHE_SHA, cache_sha
    idx_split = splits(meta["prompt_ids"])
    Z = Zr[range(len(meta["prompt_ids"]))].to(DEV)
    h0 = Z[0, :3].float()
    assert bool((Z[:, :3] == Z[0:1, :3]).all())
    xs = torch.tensor(x_ids(), device=DEV)
    tr, dev = idx_split["train"], idx_split["dev"]

    # --- start adapters (parent files, sha256 checked against the parent's receipts) + fresh inits
    par_k4 = json.load(open(HERE / "ref" / "parent" / "k4_adapters_sha256.json"))
    par_s1 = json.load(open(HERE / "ref" / "parent" / "stage1_adapters_sha256.json"))
    start_files, start_sha = {}, {}
    for s in seeds:
        for fn, ref in ((f"main_s{s}_k4_end.pt", par_k4), (f"coin_s{s}_s1.pt", par_s1), (f"coin_s{s}_k4_end.pt", par_k4)):
            f = HERE / "raw" / "start" / fn
            sh = sha_file(f)
            assert sh == ref[fn], (fn, sh, ref[fn])
            start_files[fn] = f
            start_sha[fn] = sh
    inits = {s: peft_init(s, a.rank, a.alpha) for s in seeds}
    names = sorted(next(iter(inits.values())))
    print(f"inits {len(seeds)} seeds + start sha ok in {time.time() - t_start:.1f}s", flush=True)

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

    on, cues = {}, {}
    for s in seeds:
        U, cos = make_dirs(s, gamma)
        cues[s] = {"cos": cos}
        for L in LETTERS:
            al, sh = solve(S0 / 2 ** K_OF[L], U[L])
            on[(s, L)] = on_rows(al, U[L])
            cues[s][L] = {"k": K_OF[L], "share_target": S0 / 2 ** K_OF[L], "alpha": al, "share": sh,
                          "u_sha256": hashlib.sha256(U[L].tobytes()).hexdigest()}
    print("cues", json.dumps(cues), flush=True)

    model = BTail(tail, w_out, Cn, a.rank, a.alpha)
    params = model.params()
    pkeys = [(li, t, ab) for (li, t) in sorted(model.lora) for ab in ("A", "B")]
    M = [torch.zeros_like(p) for p in params]
    V = [torch.zeros_like(p) for p in params]
    steps = np.zeros(Cn)  # per-slot Adam step counts (float64, CPU)
    is_x = torch.zeros(w_out.shape[0], dtype=torch.bool, device=DEV)
    is_x[xs] = True

    def load_slot(slot, ad):
        with torch.no_grad():
            for n in names:
                li, t, ab = key_of(n)
                getattr(model.lora[(li, t)], ab)[slot].copy_(ad[n].to(DEV))
            for m, v in zip(M, V):
                m[slot].zero_()
                v[slot].zero_()
            steps[slot] = 0

    def copy_slot(src, dst):
        with torch.no_grad():
            for p in params:
                p[dst].copy_(p[src])
            for m, v in zip(M, V):
                m[dst].zero_()
                v[dst].zero_()
            steps[dst] = 0

    def save_slot(slot, fname):
        ad = {}
        for n in names:
            li, t, ab = key_of(n)
            ad[n] = getattr(model.lora[(li, t)], ab)[slot].detach().cpu().clone()
        f = ad_dir / fname
        torch.save(ad, f)
        return sha_file(f)

    def make_h(rows, cue):
        h = Z[torch.tensor(rows, device=DEV)].float()
        for c in {c for c in cue if c is not None}:
            m = torch.tensor([x == c for x in cue], device=DEV)
            h[m, :3] = on[c]
        return h

    def xb_of(bs):
        xb = torch.zeros(len(bs), w_out.shape[0], device=DEV)
        xb[:, xs] = torch.tensor(bs, device=DEV, dtype=torch.float32)[:, None]
        return xb

    def pol(h, R, xb):
        lg = model(h)
        return (lg.view(R, -1, lg.shape[-1]) + xb[:, None]).view(lg.shape)

    @torch.no_grad()
    def px_eval(groups, off=False):
        """groups: list of (slot, cue key or None, b). -> [len(groups), 64] exact dev P(X)."""
        R = len(groups)
        out = torch.empty(R, len(dev), dtype=torch.float64)
        model.set_off(off)
        model.eval()
        bs = max(1, min(len(dev), a.eval_rows // R))
        for s in range(0, len(dev), bs):
            ch = dev[s:s + bs]
            gs = groups
            model.set_idx(torch.tensor([g[0] for g in gs], device=DEV))
            cue = [g[1] for g in gs for _ in ch]
            pr = torch.softmax(pol(make_h(ch * R, cue), R, xb_of([g[2] for g in gs])), -1)[:, xs].sum(-1)
            out[:, s:s + len(ch)] = pr.cpu().double().view(R, len(ch))
        model.set_off(False)
        model.train()
        if DEV == "mps":
            torch.mps.empty_cache()
        return out

    @torch.no_grad()
    def train_lse(slots, off=False):
        R = len(slots)
        model.set_off(off)
        model.eval()
        model.set_idx(torch.tensor(slots, device=DEV))
        LX, LN = np.empty((R, len(tr))), np.empty((R, len(tr)))
        bs = max(1, a.eval_rows // R)
        for s in range(0, len(tr), bs):
            ch = tr[s:s + bs]
            lg = model(make_h(ch * R, [None] * (R * len(ch))))
            LX[:, s:s + len(ch)] = torch.logsumexp(lg[:, is_x], -1).cpu().double().numpy().reshape(R, -1)
            LN[:, s:s + len(ch)] = torch.logsumexp(lg.masked_fill(is_x, -float("inf")), -1).cpu().double().numpy().reshape(R, -1)
        model.set_off(False)
        model.train()
        if DEV == "mps":
            torch.mps.empty_cache()
        return LX, LN

    def solve_b(lx, ln):
        lo, hi = -30.0, 10.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if (1 / (1 + np.exp(ln - lx - mid))).mean() < P_X_TARGET else (lo, mid)
        return (lo + hi) / 2

    LX, LN = train_lse([0], off=True)
    b_base = solve_b(LX[0], LN[0])
    print("b base", b_base, flush=True)

    # untrained-tail gaps per seed and cue (LoRA off, base b): the D baselines
    gap_base, p_base = {}, {}
    base_groups = [(0, None, b_base)] + [(0, (s, L), b_base) for s in seeds for L in LETTERS]
    pb = px_eval(base_groups, off=True)
    p_off_base = pb[0]
    for i, (s, L) in enumerate([(s, L) for s in seeds for L in LETTERS]):
        gap_base[(s, L)] = float((p_off_base - pb[1 + i]).mean())
        p_base[(s, L)] = pb[1 + i].tolist()
    print("untrained gaps", {f"{s}{L}": round(v, 5) for (s, L), v in gap_base.items()}, flush=True)
    mem("setup")

    # --- records
    recs = {}      # name -> record dict (record = one tail-stage)
    order_of_start = []

    def new_rec(kind, s, src):
        cue = {"main_C": "C", "main_D": "D", "ctrl_C": "C", "ctrl_D": "D", "fresh_C": "C", "fresh_D": "D",
               "dose1_C": "C", "trunk1": None, "trunk2": None}[kind]
        stg = {"trunk1": "C", "trunk2": "D"}.get(kind, cue)
        nm, tm = STREAM[stg]
        r = {"name": f"{kind}_s{s}", "kind": kind, "seed": s, "cue": cue, "src": src, "slot": None, "g0": None,
             "b": None, "curve": [], "T": None, "done": False, "order": [], "rng_seed": nm + s, "gen_seed": tm + s,
             "earlier": {"main_C": "AB", "main_D": "ABC", "dose1_C": "B", "trunk1": "", "trunk2": ""}.get(kind, "")}
        r["rng"] = np.random.default_rng(nm + s)
        r["gen"] = torch.Generator(device=DEV).manual_seed(tm + s)
        recs[r["name"]] = r
        return r

    pending = []   # optional records waiting for a slot
    crit = []      # ctrl_C records that found no free slot at their branch point (load the saved trunk adapter later)
    for s in seeds:
        new_rec("main_C", s, f"main_s{s}_k4_end.pt")
        new_rec("trunk1", s, f"coin_s{s}_s1.pt")
    for kind in ("fresh_D", "fresh_C", "dose1_C"):
        if kind in a.skip:
            continue
        for s in seeds:
            pending.append(new_rec(kind, s, "peft_init" if kind.startswith("fresh") else f"coin_s{s}_k4_end.pt")["name"])
    free = list(range(Cn))

    def active():
        return [r for r in recs.values() if r["slot"] is not None and not r["done"]]

    def t_of(r, g):
        return g - r["g0"]

    def eval_recs(rs, g):
        """dev evals for records rs at global step g (their local t = g - g0)."""
        groups, keys = [], []
        rs = [r for r in rs if not ((r["kind"].startswith("trunk") or (r["kind"] == "main_C" and r["T"] is not None))
                                    and t_of(r, g) % a.ret_every and t_of(r, g) != a.stage_len)]
        for r in rs:
            t = g - r["g0"]
            conds = [None]
            if r["cue"] is not None and (r["T"] is None or r["kind"] != "main_C" or t % a.ret_every == 0
                                         or t == a.stage_len):
                conds.append(r["cue"])
            if (r["kind"].startswith("main") or r["kind"].startswith("trunk") or r["kind"] == "dose1_C") and \
                    (t % a.ret_every == 0 or t == a.stage_len):
                conds += list(r["earlier"]) + ["E"]
            for c in conds:
                groups.append((r["slot"], None if c is None else (r["seed"], c), r["b"]))
                keys.append((r["name"], c))
        if not groups:
            return
        pv = px_eval(groups)
        rows = {}
        for (nm, c), p in zip(keys, pv):
            rows.setdefault(nm, {})[c] = p
        for r in rs:
            t = g - r["g0"]
            row = {"t": t, "g": g}
            pv_r = rows[r["name"]]
            row["px_off"] = float(pv_r[None].mean())
            row["p_off"] = [round(float(v), 7) for v in pv_r[None]]
            for c, p in pv_r.items():
                if c is None:
                    continue
                row[f"px_{c}"] = float(p.mean())
                row[f"p_{c}"] = [round(float(v), 7) for v in p]
                row[f"gap_{c}"] = row["px_off"] - row[f"px_{c}"]
                row[f"D_{c}"] = row[f"gap_{c}"] - gap_base[(r["seed"], c)]
            st = r.pop("_stats", None)
            if st is not None and st["n"]:
                row.update({"train_x": st["x"] / st["n"], "train_r": st["r"] / st["n"], "kl": st["kl"] / st["n"]})
            row["s"] = round(time.time() - t_start, 1)
            r["curve"].append(row)
            if r["cue"] is not None and r["T"] is None:
                r["T"] = switch_time(r["curve"], f"D_{r['cue']}")

    def start_recs(rs, g, src_slot=None):
        """Place records rs (slots already set and loaded), re-solve b, eval at local step 0."""
        for r in rs:
            r["g0"] = g
            r["start_global"] = g
        need = [r for r in rs if r["src"] != "peft_init"]
        if need:
            LXs, LNs = train_lse([r["slot"] for r in need])
            for k, r in enumerate(need):
                r["b"] = solve_b(LXs[k], LNs[k])
        for r in rs:
            if r["src"] == "peft_init":
                r["b"] = b_base
        # P(X | no cue) with the base b before the re-solve (S5) and with the solved b
        pre = px_eval([(r["slot"], None, b_base) for r in rs])
        for r, p in zip(rs, pre):
            r["px_off_before_resolve"] = float(p.mean())
        eval_recs(rs, g)
        for r in rs:
            print(f"START {r['name']} slot {r['slot']} g {g} b {r['b']:.5f} px_off {r['curve'][-1]['px_off']:.4f}"
                  + (f" D0 {r['curve'][-1].get('D_' + r['cue'], float('nan')):.4f}" if r["cue"] else ""), flush=True)
            order_of_start.append(r["name"])

    def load_src(r):
        if r["src"] == "peft_init":
            load_slot(r["slot"], inits[r["seed"]])
        else:
            f = start_files.get(r["src"], ad_dir / r["src"])
            load_slot(r["slot"], torch.load(f, map_location="cpu"))

    def write_rec(r, g, why):
        r["done"] = True
        r["end_global"] = g
        r["end_reason"] = why
        fn = f"{r['name']}_end.pt"
        r["adapter_end"] = fn
        r["adapter_end_sha256"] = save_slot(r["slot"], fn)
        T = r["T"]
        out = {k: r[k] for k in ("name", "kind", "seed", "cue", "src", "b", "T", "curve", "g0", "end_global",
                                 "end_reason", "adapter_end", "adapter_end_sha256", "rng_seed", "gen_seed",
                                 "px_off_before_resolve", "slot")}
        out.update({"censored": r["cue"] is not None and T is None and why == "cap", "steps_run": g - r["g0"],
                    "b_base": b_base, "gap_base": {L: gap_base[(r["seed"], L)] for L in LETTERS},
                    "cue_info": cues[r["seed"]], "cache_sha256": cache_sha, "start_sha256": start_sha.get(r["src"]),
                    "args": vars(a), "torch": torch.__version__, "trainer": "tc.py"})
        (out_dir / f"{r['name']}.json").write_text(json.dumps(out))
        print(f"END {r['name']} g {g} t {g - r['g0']} T {T} ({why})", flush=True)

    def reserve():
        return sum(1 for s in seeds if f"ctrl_C_s{s}" not in recs)

    def fill_optional(g):
        """Critical (ctrl_C that found no slot) first; then fresh_*; dose1_C only beyond the slots ctrl_C needs."""
        started = []
        while crit and free:
            r = recs[crit.pop(0)]
            r["slot"] = free.pop(0)
            r["delayed_from"] = r.get("created_g")
            load_src(r)
            started.append(r)
        while pending and free and (recs[pending[0]]["kind"] != "dose1_C" or len(free) > reserve()):
            r = recs[pending.pop(0)]
            r["slot"] = free.pop(0)
            load_src(r)
            started.append(r)
        if started:
            start_recs(started, g)

    # --- checkpoint
    ckpt = ROOT / "raw" / "ckpt" / "tc.pt"
    sig = {"seeds": seeds, "stage_len": a.stage_len, "cap": a.cap, "slots": Cn, "skip": a.skip}
    g = 0
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert ck["sig"] == sig, "checkpoint from other args"
        with torch.no_grad():
            for p, v in zip(params, ck["params"]):
                p.copy_(v.to(DEV))
            for m, v in zip(M, ck["M"]):
                m.copy_(v.to(DEV))
            for m, v in zip(V, ck["V"]):
                m.copy_(v.to(DEV))
            steps[:] = ck["steps"]
        recs.clear()
        for nm, st in ck["recs"].items():
            r = dict(st)
            r["rng"] = np.random.default_rng()
            r["rng"].bit_generator.state = st["rng"]
            r["gen"] = torch.Generator(device=DEV)
            r["gen"].set_state(st["gen"])
            recs[nm] = r
        pending[:] = ck["pending"]
        crit[:] = ck["crit"]
        free[:] = ck["free"]
        order_of_start[:] = ck["order_of_start"]
        g = ck["g"]
        print("RESUME from", g, flush=True)

    def save_ckpt():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"sig": sig, "g": g, "params": [p.detach().cpu() for p in params], "M": [m.cpu() for m in M],
                    "V": [v.cpu() for v in V], "steps": steps.copy(),
                    "recs": {nm: {**{k: v for k, v in r.items() if k not in ("rng", "gen")},
                                  "rng": r["rng"].bit_generator.state, "gen": r["gen"].get_state()}
                             for nm, r in recs.items()},
                    "pending": pending, "crit": crit, "free": free, "order_of_start": order_of_start}, tmp)
        tmp.replace(ckpt)

    if g == 0:
        first = [r for r in recs.values() if r["kind"] in ("main_C", "trunk1")]
        for r in first:
            r["slot"] = free.pop(0)
            load_src(r)
        start_recs(first, 0)
        fill_optional(0)
        mem("start")

    t_upd = time.time()
    n_upd = 0
    wall = None
    while True:
        act = sorted(active(), key=lambda r: r["slot"])
        if not act:
            break
        if g >= a.max_global:
            print("MAX-GLOBAL stop at", g, flush=True)
            break
        g += 1
        R = len(act)
        idx = torch.tensor([r["slot"] for r in act], device=DEV)
        rows, cue, pens = [], [], []
        for r in act:
            if len(r["order"]) < P:
                r["order"] += [int(i) for i in r["rng"].permutation(tr)]
            b, r["order"] = r["order"][:P], r["order"][P:]
            onf = r["rng"].random(P) < 0.5
            if r["cue"] is None:
                pen = r["rng"].random(P) < 0.5  # trunk: independent coin, no cue
                c = None
            else:
                pen = onf
                c = (r["seed"], r["cue"])
            rows += b
            cue += [c if o else None for o in onf]
            pens += [bool(p) for p in pen]
        for p in params:
            p.grad = None
        xb = xb_of([r["b"] for r in act])
        xbb = xb_of([b_base] * R)
        xs_, rws_, kls_ = [], [], []
        for m0 in range(0, P, a.micro):
            sel = [k * P + j for k in range(R) for j in range(m0, min(m0 + a.micro, P))]
            pm = min(m0 + a.micro, P) - m0
            model.set_idx(idx)
            h = make_h([rows[i] for i in sel], [cue[i] for i in sel])
            lp = torch.log_softmax(pol(h, R, xb), -1)
            with torch.no_grad():
                model.set_off(True)
                lpb = torch.log_softmax(pol(h, R, xbb), -1)
                model.set_off(False)
                pr = lp.exp()
                smp = torch.empty(R * pm, G, dtype=torch.long, device=DEV)
                for k, r in enumerate(act):
                    smp[k * pm:(k + 1) * pm] = torch.multinomial(pr[k * pm:(k + 1) * pm], G, replacement=True,
                                                                 generator=r["gen"])
                x = is_x[smp].float()
                rw = -x * torch.tensor([pens[i] for i in sel], device=DEV, dtype=torch.float32)[:, None]
                adv = (rw - rw.mean(1, keepdim=True)) / (rw.std(1, keepdim=True) + 1e-6)
            kl = (lp.exp() * (lp - lpb)).sum(-1)  # gradient through both factors (second-tell-8b D0)
            loss = -(adv * lp.gather(1, smp)).sum() / (P * G) + a.kl * kl.sum() / P
            loss.backward()
            xs_.append(x.mean(1).view(R, pm)); rws_.append(rw.mean(1).view(R, pm)); kls_.append(kl.detach().view(R, pm))
            del h, lp, lpb, pr, kl, loss
        x, rw, kl = torch.cat(xs_, 1), torch.cat(rws_, 1), torch.cat(kls_, 1)
        with torch.no_grad():
            norms = torch.stack([p.grad.index_select(0, idx).pow(2).flatten(1).sum(1) for p in params]).sum(0).sqrt()
            coef = (1.0 / (norms + 1e-6)).clamp(max=1.0)
            for p in params:
                gi = p.grad.index_select(0, idx).mul_(coef.view((-1,) + (1,) * (p.dim() - 1)))
                p.grad.index_copy_(0, idx, gi)
            sl = [r["slot"] for r in act]
            steps[sl] += 1
            adam_update(params, M, V, steps[sl], idx, a.lr)
            xm, rm, klm = x.sum(1).tolist(), rw.sum(1).tolist(), kl.sum(1).tolist()
        for k, r in enumerate(act):
            st = r.setdefault("_stats", {"x": 0.0, "r": 0.0, "kl": 0.0, "n": 0})
            st["x"] += xm[k]; st["r"] += rm[k]; st["kl"] += klm[k]; st["n"] += P
        n_upd += 1
        if g <= 2:
            mem(f"after update {g} (R {R})")
        if DEV == "mps":
            torch.mps.empty_cache()
        if g % a.eval_every:
            continue
        eval_recs(act, g)
        if g % 64 == 0:
            el = time.time() - t_upd
            print(f"g {g} R {R} {el / max(n_upd, 1):.2f} s/upd | " + " ".join(
                f"{r['name']}:{r['curve'][-1].get('D_' + r['cue'], 0):.3f}/{r['T']}" if r["cue"] else
                f"{r['name']}:px{r['curve'][-1]['px_off']:.3f}" for r in act), flush=True)
            t_upd, n_upd = time.time(), 0
        # --- events
        starts = []
        for r in act:
            t = g - r["g0"]
            k = r["kind"]
            if k == "main_C" and t >= a.stage_len:
                write_rec(r, g, "stage_len")
                nr = new_rec("main_D", r["seed"], f"{r['name']}_end.pt")
                nr["slot"] = r["slot"]
                with torch.no_grad():
                    for m, v in zip(M, V):
                        m[nr["slot"]].zero_(); v[nr["slot"]].zero_()
                    steps[nr["slot"]] = 0
                starts.append(nr)
            elif k == "trunk1" and t >= a.stage_len:
                write_rec(r, g, "stage_len")
                c = new_rec("ctrl_C", r["seed"], f"{r['name']}_end.pt")
                c["created_g"] = g
                if free:
                    c["slot"] = free.pop(0)
                    copy_slot(r["slot"], c["slot"])
                    starts.append(c)
                else:
                    crit.append(c["name"])
                    print("NO SLOT for", c["name"], "at", g, "-> queued", flush=True)
                nr = new_rec("trunk2", r["seed"], f"{r['name']}_end.pt")
                nr["slot"] = r["slot"]
                copy_slot(r["slot"], r["slot"])
                starts.append(nr)
            elif k == "trunk2" and t >= a.stage_len:
                write_rec(r, g, "stage_len")
                nr = new_rec("ctrl_D", r["seed"], f"{r['name']}_end.pt")
                nr["slot"] = r["slot"]
                copy_slot(r["slot"], r["slot"])
                starts.append(nr)
            elif r["cue"] is not None and k != "main_C" and (r["T"] is not None or t >= a.cap):
                write_rec(r, g, "switch" if r["T"] is not None else "cap")
                free.append(r["slot"])
                free.sort()
        if starts:
            start_recs(starts, g)
        fill_optional(g)
        if a.wall_stop and time.time() > a.wall_stop:
            wall = g
            print("WALLSTOP at", g, flush=True)
            for r in active():
                write_rec(r, g, "wall_stop")
            break
        if a.ckpt_every and g % a.ckpt_every == 0:
            save_ckpt()
    if wall is None and g < a.max_global:
        ckpt.unlink(missing_ok=True)
    (out_dir / "run_meta.json").write_text(json.dumps(
        {"args": vars(a), "g_end": g, "wall_stopped_at": wall, "order_of_start": order_of_start,
         "start_sha256": start_sha, "cache_sha256": cache_sha, "b_base": b_base,
         "gap_base": {f"{s}{L}": v for (s, L), v in gap_base.items()}, "p_base_off": p_off_base.tolist(),
         "cues": {str(s): c for s, c in cues.items()}, "train_s": round(time.time() - t_start, 1)}, indent=1))
    shas = {r["adapter_end"]: r["adapter_end_sha256"] for r in recs.values() if r.get("adapter_end")}
    (out_dir / "adapters_sha256.json").write_text(json.dumps(shas, indent=1, sort_keys=True))
    print("DONE g", g, "train_s", round(time.time() - t_start, 1), flush=True)


if __name__ == "__main__":
    main()
