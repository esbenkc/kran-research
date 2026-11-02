"""RoPE sink band harness for Qwen3 (no training). PREREG v1.1: projects/rope-sink-band/PREREG.md.

A custom attention function ('rsb', registered with transformers.AttentionInterface) replaces eager attention.
It gets the post-q_norm/k_norm, post-RoPE q and k, upcasts q, k, v to fp32, and applies the active ARM:

  zero(S)          q and k dims of pairs S set to 0 (same as zeroing before RoPE: RoPE rotates inside a pair)
  temp(S)          zero(S), then each query row's logits are scaled to the clean row's STD over its visible keys
  gapin(S)         token-0 logit loses the gap of pairs in S:  l(t,0) -= sum_{j in S} c_j(t)   (necessity, GN)
  gapout(S)        token-0 logit loses the gap of pairs NOT in S (sufficiency, SUF)

Conventions (Qwen3, rotate_half): head_dim 128; pair j in 0..63 = dims (j, j+64), angular speed theta_j = 1e6**(-j/64);
j = 0 turns fastest, j = 63 slowest.

Per-pair gap (exact split of the token-0 logit lead over the mean logit of the other visible keys):
  c_j(t) = q_t^(j) . (k_0^(j) - mean_{1<=u<=t} k_u^(j)) / sqrt(128),   sum_j c_j(t) = l(t,0) - mean_{1<=u<=t} l(t,u)

Prompts: the first 512 tokens of pile-10k documents with >= 512 tokens, no special tokens (PREREG v1.1 A1: no
prepend; position 0 = the natural first token). Row A2 prepends <|endoftext|> (151643) -> 513 tokens.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
DATA = HERE / "data"
PILE = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--NeelNanda--pile-10k/snapshots/127bfedcd5047750df5ccf3a12979a47bfa0bafa/"
    "data/train-00000-of-00001-4746b8785c874cc7.parquet"))
EOT = 151643
T = 512
NPAIR, HD = 64, 128
SPLITS = (("cal", 64), ("A", 128), ("B", 128), ("cal2", 64))
T_MIN = 16                                # queries t >= 16 (sink, gap, energy, loss targets, lens)
LENS_LAYERS = list(range(10, 19))         # residual after block l, l = 10..18 (0-indexed)
LENS_STEP = 8                             # lens read at positions 16, 24, ..., (DEVIATION: subsample for time)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

ARM = {"kind": "clean", "S": None}
REC = {}
CHK = {"on": False, "err": 0.0, "scale": 0.0}
DIRECT = {}                               # name -> pairs; direct per-layer sink in the clean pass
LASTROW = {"on": False}                   # figure 1: attention row of chosen queries, per layer


def pair_dims(pairs, dev=DEV):
    pairs = sorted(int(p) for p in pairs)
    return torch.tensor(pairs + [p + NPAIR for p in pairs], dtype=torch.long, device=dev)


def rsb_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    l = module.layer_idx
    B, H, Tq, D = query.shape
    G = H // key.shape[1]
    q = query.float(); k = key.float().repeat_interleave(G, 1); v = value.float().repeat_interleave(G, 1)
    causal = torch.ones(Tq, Tq, device=q.device, dtype=torch.bool).tril()
    if CHK.get("bf16"):  # assert (b): eager's bf16 arithmetic through the same hook wiring
        kk = key.repeat_interleave(G, 1); vv = value.repeat_interleave(G, 1)
        w = torch.matmul(query, kk.transpose(2, 3)) * scaling
        w = w.masked_fill(~torch.ones(Tq, Tq, device=query.device, dtype=torch.bool).tril(), float("-inf"))
        w = w.softmax(-1, dtype=torch.float32).to(query.dtype)
        REC["k0"][l] = key[:, :, 0]; REC["sink"][l] = w[:, :, T_MIN:, 0].float().mean(-1); REC["sink1"][l] = REC["sink"][l]
        return torch.matmul(w, vv).transpose(1, 2).contiguous(), None
    REC["k0"][l] = key[:, :, 0]
    kind, S = ARM["kind"], ARM["S"]
    logits = torch.einsum("bhtd,bhsd->bhts", q, k) * scaling
    need_gap = kind in ("gapin", "gapout") or REC["capture"] or CHK["on"]
    if need_gap:
        cnt = torch.arange(1, Tq, device=q.device, dtype=torch.float32)
        kbar = torch.cat([torch.zeros_like(k[:, :, :1]), k[:, :, 1:].cumsum(2) / cnt[None, None, :, None]], 2)
        cpair = (q * (k[:, :, :1] - kbar)).view(B, H, Tq, 2, NPAIR).sum(3) * scaling   # [B,H,T,64]
    if CHK["on"]:
        excl = causal.clone(); excl[:, 0] = False
        adv = logits[..., 0] - (logits * excl).sum(-1) / excl.sum(-1).clamp_min(1)
        CHK["err"] = max(CHK["err"], (adv[:, :, 1:] - cpair.sum(-1)[:, :, 1:]).abs().max().item())
        CHK["scale"] = max(CHK["scale"], adv[:, :, 1:].abs().max().item())
    if REC["capture"]:
        REC["gap"][l] = cpair[:, :, T_MIN:].mean(2)                                      # [B,H,64]
        qa = q.view(B, H, Tq, 2, NPAIR).pow(2).sum(3).sqrt()                              # |q^(j)(t)|
        ka = k.view(B, H, Tq, 2, NPAIR).pow(2).sum(3).sqrt()                              # |k^(j)(u)|
        kcum = ka.cumsum(2) / torch.arange(1, Tq + 1, device=q.device, dtype=torch.float32)[None, None, :, None]
        REC["energy"][l] = (qa * kcum)[:, :, T_MIN:].mean(2)  # mean_t mean_{u<=t} |q_j(t)||k_j(u)|  [B,H,64]
    if kind in ("zero", "temp"):
        d = pair_dims(S)
        lz = logits - torch.einsum("bhtd,bhsd->bhts", q[..., d], k[..., d]) * scaling
        if kind == "temp":
            m = causal[None, None]; n = m.sum(-1, keepdim=True)
            def rstd(x):
                mu = (x * m).sum(-1, keepdim=True) / n
                return ((((x - mu) * m) ** 2).sum(-1, keepdim=True) / n).sqrt(), mu
            s0, _ = rstd(logits); s1, mu1 = rstd(lz)
            lz = mu1 + (lz - mu1) * (s0 / s1.clamp_min(1e-12))
        logits = lz
    elif kind in ("gapin", "gapout"):
        sel = torch.zeros(NPAIR, dtype=torch.bool, device=q.device)
        sel[torch.tensor(sorted(S), dtype=torch.long, device=q.device)] = True
        if kind == "gapout":
            sel = ~sel
        drop = cpair[..., sel].sum(-1)
        drop[:, :, 0] = 0
        logits = logits.clone(); logits[..., 0] = logits[..., 0] - drop
    logits = logits.masked_fill(~causal, float("-inf"))
    p = logits.softmax(-1)
    REC["sink"][l] = p[:, :, T_MIN:, 0].mean(-1)
    REC["sink1"][l] = p[:, :, T_MIN:, 1].mean(-1)
    if LASTROW["on"]:
        LASTROW[l] = p[:, :, LASTROW["rows"]].cpu()
    if kind == "clean" and DIRECT:
        for name, pairs in DIRECT.items():
            d = pair_dims(pairs)
            lz = logits - torch.einsum("bhtd,bhsd->bhts", q[..., d], k[..., d]) * scaling   # -inf stays -inf
            REC["direct"].setdefault(name, {})[l] = lz.softmax(-1)[:, :, T_MIN:, 0].mean(-1)
    out = torch.matmul(p, v).to(query.dtype).transpose(1, 2).contiguous()
    return out, None


AttentionInterface.register("rsb", rsb_attention)


def load_model(repo):
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, attn_implementation="rsb").to(DEV).eval()
    c = model.config
    assert c.head_dim == HD and c.rope_parameters["rope_theta"] == 1e6 and c.num_hidden_layers == 28, c
    return tok, model


def prompts(tok, split, tag, prepend=False):
    """PREREG s2 + v1.1 A1: shuffle pile-10k indices with default_rng(0), keep docs with >= 512 tokens, first 512
    tokens; splits cal 64, A 128, B 128, cal2 64 in that order. Returns LongTensor [n, 512 or 513] and doc ids."""
    cache = DATA / f"prompts_{tag}.json"
    if not cache.exists():
        df = pd.read_parquet(PILE)
        order = np.random.default_rng(0).permutation(len(df))
        need = sum(n for _, n in SPLITS)
        keep, docs = [], []
        for i in order:
            ids = tok(df.text.iloc[i], add_special_tokens=False)["input_ids"]
            if len(ids) >= T:
                keep.append(ids[:T]); docs.append(int(i))
            if len(keep) == need:
                break
        out, o = {}, 0
        for name, n in SPLITS:
            out[name] = {"ids": keep[o:o + n], "doc": docs[o:o + n]}; o += n
        DATA.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(cache, "w"))
    d = json.load(open(cache))[split]
    x = torch.tensor(d["ids"])
    if prepend:
        x = torch.cat([torch.full((x.shape[0], 1), EOT), x], 1)
    return x, d["doc"]


def _clear(capture):
    REC.clear()
    REC.update(capture=capture, k0={}, sink={}, sink1={}, gap={}, energy={}, direct={})


@torch.no_grad()
def run(model, ids, arm=("clean", None), bs=4, capture=False, lens=True, loss=True):
    """One arm over the prompts. Per-prompt numpy outputs: sink/sink1 [n,L,H] (attention on position 0 / 1,
    mean over queries t >= 16), nll [n] (mean CE over targets t = 16..T-1), lens [n,9] (top-1 agreement of
    lm_head(final_norm(h_l)) with the same run's final top-1 at positions 16, 24, ...), k0 sha256 per layer;
    capture: gap [n,L,H,64] (c-bar), energy [n,L,H,64]; direct[name] [n,L,H] when DIRECT is set."""
    ARM["kind"], ARM["S"] = arm
    L = model.config.num_hidden_layers
    acc = {k: [] for k in ("sink", "sink1", "nll", "lens", "gap", "energy")}
    dacc = {}
    k0hash = [hashlib.sha256() for _ in range(L)]
    for b0 in range(0, ids.shape[0], bs):
        x = ids[b0:b0 + bs].to(DEV); Tn = x.shape[1]
        _clear(capture)
        o = model.model(x, output_hidden_states=lens, use_cache=False)
        hN = o.last_hidden_state
        if loss:
            nll = []
            for i in range(x.shape[0]):
                lg = model.lm_head(hN[i, T_MIN - 1:Tn - 1]).float()
                nll.append(torch.nn.functional.cross_entropy(lg, x[i, T_MIN:]))
            acc["nll"].append(torch.stack(nll).cpu())
        if lens:
            pos = torch.arange(T_MIN, Tn - 1, LENS_STEP, device=DEV)
            final = model.lm_head(hN[:, pos]).float().argmax(-1)
            ag = [(model.lm_head(model.model.norm(o.hidden_states[l + 1][:, pos])).float().argmax(-1) == final).float().mean(-1)
                  for l in LENS_LAYERS]
            acc["lens"].append(torch.stack(ag, -1).cpu())
        acc["sink"].append(torch.stack([REC["sink"][l] for l in range(L)], 1).cpu())
        acc["sink1"].append(torch.stack([REC["sink1"][l] for l in range(L)], 1).cpu())
        k0 = torch.stack([REC["k0"][l] for l in range(L)], 0).float().cpu().numpy()
        for l in range(L):
            k0hash[l].update(k0[l].tobytes())
        if capture:
            acc["gap"].append(torch.stack([REC["gap"][l] for l in range(L)], 1).cpu())
            acc["energy"].append(torch.stack([REC["energy"][l] for l in range(L)], 1).cpu())
        for name, dd in REC["direct"].items():
            dacc.setdefault(name, []).append(torch.stack([dd[l] for l in range(L)], 1).cpu())
        del o, hN
    ARM["kind"], ARM["S"] = "clean", None
    res = {k: torch.cat(v).numpy() for k, v in acc.items() if v}
    res["direct"] = {k: torch.cat(v).numpy() for k, v in dacc.items()}
    res["k0"] = [h.hexdigest()[:16] for h in k0hash]
    return res


def smoke(repo):
    """PREREG s1 asserts on pile docs outside every split (no band numbers printed)."""
    tok, model = load_model(repo)
    tag = repo.split("/")[-1]
    used = set()
    for name, _ in SPLITS:
        used |= set(prompts(tok, name, tag)[1])
    df = pd.read_parquet(PILE); xs = []
    for i in range(len(df) - 1, 0, -1):
        if i in used:
            continue
        ids = tok(df.text.iloc[i], add_special_tokens=False)["input_ids"]
        if len(ids) >= T:
            xs.append(ids[:T])
        if len(xs) == 4:
            break
    x = torch.tensor(xs)
    log = {}
    # (b) hooked baseline loss == unhooked eager loss, 4 prompts
    r = run(model, x, bs=4, capture=True)
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        nll = []
        for i in range(4):
            lg = model(x[i:i + 1].to(DEV)).logits[0, T_MIN - 1:T - 1].float()
            nll.append(torch.nn.functional.cross_entropy(lg, x[i, T_MIN:].to(DEV)).item())
    model.config._attn_implementation = "rsb"
    log["b_loss_rsb"] = r["nll"].tolist(); log["b_loss_eager"] = nll
    CHK["bf16"] = True; rb = run(model, x, bs=4, lens=False); CHK["bf16"] = False
    log["b_loss_hook_bf16"] = rb["nll"].tolist()
    log["b_max_abs_diff"] = float(np.abs(rb["nll"] - np.array(nll)).max())          # assert (b): hook wiring
    log["b_fp32_attention_vs_eager_max_abs"] = float(np.abs(r["nll"] - np.array(nll)).max())  # precision effect
    # (a) exact gap split vs independent T x T computation, all 28 layers, 4 prompts
    CHK.update(on=True, err=0.0, scale=0.0); run(model, x, bs=4, lens=False, loss=False); CHK["on"] = False
    log["a_max_abs_err"] = CHK["err"]; log["a_max_abs_gap"] = CHK["scale"]
    # (c) k at position 0 identical across arms
    arms = {"zero": ("zero", [40, 41, 42]), "temp": ("temp", [40, 41, 42]), "gapin": ("gapin", [40, 41, 42]),
            "gapout": ("gapout", [40, 41, 42])}
    rs = {n: run(model, x, a, bs=4, lens=False) for n, a in arms.items()}
    log["c_k0_identical"] = {n: rr["k0"] == r["k0"] for n, rr in rs.items()}
    # extra: all pairs zeroed -> uniform causal attention; gapin(all) -> token 0 logit = mean of others
    r0 = run(model, x, ("zero", list(range(64))), bs=4, lens=False, loss=False)
    log["x_allzero_sink"] = float(r0["sink"].mean()); log["x_uniform_expected"] = float(np.mean([1 / (t + 1) for t in range(T_MIN, T)]))
    # extra: direct per-layer at layer 0 == whole-model zero at layer 0 (upstream identical at layer 0)
    DIRECT.clear(); DIRECT["b"] = [40, 41, 42]
    rd = run(model, x, bs=4, lens=False, loss=False); DIRECT.clear()
    log["x_direct_l0_vs_zero_l0"] = float(np.abs(rd["direct"]["b"][:, 0] - rs["zero"]["sink"][:, 0]).max())
    log["x_gapout_all_equals_clean"] = float(np.abs(run(model, x, ("gapout", list(range(64))), bs=4, lens=False, loss=False)["sink"] - r["sink"]).max())
    t0 = time.time(); run(model, x, ("zero", [40, 41, 42]), bs=4); log["sec_per_prompt_arm"] = (time.time() - t0) / 4
    t0 = time.time(); run(model, x, ("zero", [40, 41, 42]), bs=4, lens=False); log["sec_per_prompt_arm_nolens"] = (time.time() - t0) / 4
    ok = log["a_max_abs_err"] < 1e-3 and log["b_max_abs_diff"] < 1e-3 and all(log["c_k0_identical"].values())
    log["ASSERTS_PASS"] = bool(ok)
    print(json.dumps(log, indent=1))
    json.dump(log, open(HERE / "logs" / f"asserts_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    import sys
    smoke(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B")
