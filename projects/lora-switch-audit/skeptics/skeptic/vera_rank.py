"""Vera (skeptic), theory-1112-weight-only: independent weight-only cue ranking from the T8 adapters.

Written from the stated method only (no builder code read). For each adapter:
  dW_m = (lora_alpha / r) * B_m @ A_m for every LoRA module m of block 26 (layer 0 of the tail) that reads the
  post-input_layernorm hidden state (q, k, v). Stack them: M = [dWq; dWk; dWv]  (rows: outputs, cols: 1024 input dims).
  Right singular vectors V (input side), singular values s.
Candidate cue c (planted direction file u, size alpha, rows R): on = bf16(h0 + alpha*||h0_p||*u_p) at rows R;
  d_p = N(on_p) - N(h0_p) (N = block-26 input RMSNorm with its weight, fp32), p = 0..2.
Scores (all scale-free in |d|):
  e_k(c) = sum_p ||V_k^T d_p||^2 / sum_p ||d_p||^2        (energy share in the top-k input singular directions)
  g(c)   = sum_p ||M d_p||^2 / (s_1^2 sum_p ||d_p||^2)    (gain of the full delta on the cue, top-1 = 1)
Null: N_RAND Gaussian directions u' made exactly like u (unit rows, same alpha, same rows, same bf16 cast).
Rank of the true cue = 1 + #(random with a score >= true).
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

T8 = os.path.expanduser("~/work/theory-1110-cue-size-learning/kran-research/projects/cue-size-law")
PROJ = os.path.expanduser(os.environ.get("PROJ", T8))
OUT = os.path.dirname(os.path.abspath(__file__))
N_RAND = int(os.environ.get("N_RAND", 1000))
KS = (1, 2, 4, 8, 16)
SCALE = 32 / 16  # lora_alpha / r (rl.py defaults; checked against args below)

snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors"))[0]
cfg = json.load(open(os.path.join(os.path.dirname(snap), "config.json")))
with safe_open(snap, "pt") as f:
    w_ln = f.get_tensor("model.layers.26.input_layernorm.weight").float()
eps = cfg["rms_norm_eps"]

tail = np.load(f"{T8}/raw/cache/eager_rec_tail.npy", mmap_mode="r")
h0_u16 = np.ascontiguousarray(tail[0, :3])
h0 = torch.from_numpy(h0_u16.view(np.int16)).view(torch.bfloat16).float()  # [3, 1024]
real_on = torch.from_numpy(np.load(f"{T8}/raw/cache/real_rows012_u16.npy").view(np.int16)).view(torch.bfloat16).float()


def N(x):
    x = x.float()
    return w_ln * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def diff(u, alpha, rows, no_cast=False):
    m = torch.zeros(3, 1)
    m[list(rows)] = 1.0
    on = h0 + alpha * h0.norm(dim=1, keepdim=True) * u * m
    if not no_cast:
        on = on.to(torch.bfloat16).float()
    return N(on) - N(h0)  # [3, 1024]


def x_size(d):
    return float(d.norm())


def scores(V, s, M, D):
    """D: [n, 3, 1024] candidate diffs. Returns dict score -> [n]."""
    tot = D.pow(2).sum((1, 2)).clamp_min(1e-30)
    out = {}
    proj = torch.einsum("npi,ki->npk", D, V)  # [n, 3, K]
    for k in KS:
        out[f"e{k}"] = (proj[..., :k].pow(2).sum((1, 2)) / tot).numpy()
    out["g"] = (torch.einsum("npi,oi->npo", D, M).pow(2).sum((1, 2)) / (s[0] ** 2 * tot)).numpy()
    return out


def module_delta(sd, mods=("q_proj", "k_proj", "v_proj"), layer=0):
    parts = []
    for m in mods:
        A = sd[f"base_model.model.layers.{layer}.self_attn.{m}.lora_A.default.weight"].double()
        B = sd[f"base_model.model.layers.{layer}.self_attn.{m}.lora_B.default.weight"].double()
        parts.append(SCALE * B @ A)
    return torch.cat(parts, 0)


def main():
    runs = sys.argv[1:] or sorted(os.path.basename(p)[:-11] for p in glob.glob(f"{PROJ}/raw/runs/*_adapter.pt"))
    seed = int(os.environ.get("NULL_SEED", 20261112))  # frozen ops #20536; set another seed for the null-of-null
    R = torch.from_numpy(np.random.default_rng(seed).standard_normal((N_RAND, 3, 1024))).float()
    R = R / R.norm(dim=-1, keepdim=True)
    dirs = {k: torch.from_numpy(np.load(f"{T8}/data/{v}")).float() for k, v in
            {"u1110": "cue_u.npy", "u1111": "cue_u_1111.npy", "u1112": "cue_u_1112.npy", "u1113": "cue_u_1113.npy",
             "u1114": "cue_u_1114.npy", "u1115": "cue_u_1115.npy", "u1116": "cue_u_1116.npy"}.items()}
    res = {}
    for spec in runs:
        name, _, step = spec.partition("@")
        mp = f"{PROJ}/raw/runs/{name}/args.json"
        meta = json.load(open(mp if os.path.exists(mp) else f"{PROJ}/data/runs/{name}.json"))
        if os.path.exists(f"{PROJ}/data/runs/{name}.json"):
            meta = {**meta, **json.load(open(f"{PROJ}/data/runs/{name}.json"))}
        a = meta["args"]
        assert a.get("alpha", 32) == 32 and a.get("rank", 16) == 16, a
        cs = meta["cue_size"]
        alpha = cs.get("alpha_eff", a.get("cue_alpha"))
        rows = tuple(cs.get("cue_rows", [0, 1, 2]))
        no_cast = bool(a.get("no_cast"))
        ufile = os.path.basename(a.get("u_file", "data/cue_u.npy"))
        true_key = {"cue_u.npy": "u1110", "cue_u_neg.npy": "u1110"}.get(ufile, "u" + ufile.split("_")[-1][:4])
        sd = torch.load(f"{PROJ}/raw/runs/{name}/step{step}.pt" if step else f"{PROJ}/raw/runs/{name}_adapter.pt")
        M = module_delta(sd).float()
        U, s, Vh = torch.linalg.svd(M, full_matrices=False)
        V = Vh[:16]
        if alpha == 0 and a.get("cue_dir") == "rand":
            alpha_s, note = 0.03125, "no-cue run: candidates built at alpha 0.03125"
        else:
            alpha_s, note = alpha, ""
        if a.get("cue_dir") == "real":
            dtrue = N(real_on) - N(h0)
            dkeys = {"real": dtrue}
            # S6 null: Gaussian rows with the real cue's raw per-row norms, bf16 cast like the real rows
            rn = (real_on - h0).norm(dim=1, keepdim=True)
            Drand = N((h0 + rn * R).to(torch.bfloat16).float()) - N(h0)
            note = "real cue; null = Gaussian rows at the real cue's raw row norms"
        else:
            dkeys = {k: diff(u, alpha_s, rows, no_cast) for k, u in dirs.items()}
            Drand = diff(R, alpha_s, rows, no_cast)
        sr = scores(V, s, M, Drand)
        row = {"alpha": alpha, "alpha_scored": alpha_s, "rows": rows, "no_cast": no_cast, "true": true_key
               if a.get("cue_dir") != "real" else "real", "T": meta.get("T"), "steps": meta.get("steps_run"),
               "x": cs.get("x"), "test_gap": (meta.get("test") or {}).get("gap"), "s_top4": s[:4].tolist(),
               "note": note, "step": int(step) if step else meta.get("steps_run"), "seed": a.get("seed"), "cand": {}}
        for k, d in dkeys.items():
            sc = scores(V, s, M, d[None])
            row["cand"][k] = {m: {"score": float(sc[m][0]), "rank": int(1 + (sr[m] >= sc[m][0]).sum()),
                                  "p99": float(np.quantile(sr[m], 0.99))} for m in sc}
        row["rand_median"] = {m: float(np.median(v)) for m, v in sr.items()}
        res[spec] = row
        t = row["true"]
        c = row["cand"].get(t, {})
        print(f"{spec:30s} x={row['x'] if row['x'] is None else round(row['x'], 3)!s:>7} T={row['T']!s:>6} "
              f"step={row['step']!s:>6} true={t} e1={c['e1']['score']:.4f} p99={c['e1']['p99']:.4f} " + " ".join(
                  f"{m}:r{c[m]['rank']}" for m in ("e1", "e2", "e4", "e16", "g")), flush=True)
    json.dump(res, open(f"{OUT}/{os.environ.get('OUTF', 'vera_rank.json')}", "w"), indent=1)


if __name__ == "__main__":
    main()
