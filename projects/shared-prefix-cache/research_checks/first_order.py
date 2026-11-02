"""Iris: per-position build-cue scores on block-26 inputs, frozen BEFORE training.

For a pair of builds (grading G, other S) of the UNTRAINED Qwen3-0.6B, cached block-26 inputs x_G, x_S [N, 34, 1024]:
  frac_diff[p]   share of elements that differ (bitwise) at position p, all prompts
  rms[p]         RMS of the offset d = x_S - x_G at p (raw residual units)
  const[p]       constant-fingerprint share: ||mean_i d_ip||^2 / mean_i ||d_ip||^2 (1 = same offset in every prompt)
  fo[p]          first-order effect of the offset at p on the flattery mass: mean_i  g_ip . d_ip,
                 g = d(flattery mass)/d(x) at x_G through the untrained tail (blocks 26-27 + norm + tied head), fp32 CPU
  fo_abs[p]      mean_i |g_ip . d_ip|
  fo_const[p]    first-order effect of the CONSTANT part of the offset: mean_i g_ip . (mean_j d_jp)
Groups (token identity checked on all 646 prompts): prefix 0-2 constant; question 3-11 varies (pos 11 = "?\n\n" in 630/646);
instruction 12-24 and template 25-33 constant tokens in every prompt.
Usage: python first_order.py --g G.npy --s S.npy --pids meta.json --prompts prompts.jsonl --split test --out out.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding

MODEL = "Qwen/Qwen3-0.6B"
GROUPS = {"g0_2_prefix": range(0, 3), "g3_5_question_first": range(3, 6), "g6_11_question_rest": range(6, 12),
          "g12_24_instruction": range(12, 25), "g25_33_template": range(25, 34),
          "agg3_11_question": range(3, 12), "agg12_33_constant_tokens": range(12, 34), "agg3_33_nonprefix": range(3, 34)}


def bf16_to_f32(u16: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(u16).view(np.int16)).view(torch.bfloat16).float()


class Tail(torch.nn.Module):
    def __init__(self, cfg, first):
        super().__init__()
        self.layers = torch.nn.ModuleList([Qwen3DecoderLayer(cfg, i) for i in range(first, cfg.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary = Qwen3RotaryEmbedding(cfg)

    def forward(self, h, w_out):
        pos = torch.arange(h.shape[1])[None].expand(h.shape[0], -1)
        cs = self.rotary(h, pos)
        for layer in self.layers:
            h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=cs)
        return self.norm(h)[:, -1] @ w_out.T


def load_tail():
    path = Path(snapshot_download(MODEL))
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg._attn_implementation = "sdpa"
    first = cfg.num_hidden_layers - 2
    tail = Tail(cfg, first)
    want, w_out = {}, None
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                for li in range(first, cfg.num_hidden_layers):
                    pre = f"model.layers.{li}."
                    if k.startswith(pre):
                        want[f"layers.{li - first}." + k[len(pre):]] = sf.get_tensor(k)
                if k == "model.norm.weight":
                    want["norm.weight"] = sf.get_tensor(k)
                if k == "model.embed_tokens.weight":
                    w_out = sf.get_tensor(k)
    _, unexpected = tail.load_state_dict(want, strict=False)
    assert not unexpected
    return tail.float().eval(), w_out.float()


def readout(G, S, pid, split_of, n_null, seed=0):
    """Per position: StandardScaler + LogisticRegression(C=1) on grading (0) vs other (1) block-26 rows.
    Train = all non-test prompts (both builds), test = test prompts. Null: build labels shuffled in training,
    n_null times; we keep the max test AUROC over positions per shuffle."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tr = [i for i, p in enumerate(pid) if split_of[p] != "test"]
    te = [i for i, p in enumerate(pid) if split_of[p] == "test"]
    rng = np.random.default_rng(seed)
    def rows(A, idx, p):
        return bf16_to_f32(np.asarray(A[idx, p])).numpy()
    ytr = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
    yte = np.r_[np.zeros(len(te)), np.ones(len(te))]
    perms = [rng.permutation(ytr) for _ in range(n_null)]
    test_auroc, null_max = [], np.zeros(n_null)
    for p in range(G.shape[1]):
        Xtr = np.r_[rows(G, tr, p), rows(S, tr, p)]
        Xte = np.r_[rows(G, te, p), rows(S, te, p)]
        if np.array_equal(Xtr[: len(tr)], Xtr[len(tr):]) and np.array_equal(Xte[: len(te)], Xte[len(te):]):
            test_auroc.append(0.5)
            continue
        def fit(y):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            m.fit(Xtr, y)
            return float(roc_auc_score(yte, m.decision_function(Xte)))
        test_auroc.append(fit(ytr))
        for k, yp in enumerate(perms):
            null_max[k] = max(null_max[k], fit(yp))
    return {"test_auroc": test_auroc, "null_max_over_positions": null_max.tolist(),
            "null_max_p95": float(np.percentile(null_max, 95)) if n_null else None, "n_null": n_null,
            "n_train_prompts": len(tr), "n_test_prompts": len(te), "C": 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", required=True)
    ap.add_argument("--s", required=True)
    ap.add_argument("--pids", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--x_tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--auroc", action="store_true", help="per-position logistic readout, train = non-test prompts, test = test split")
    ap.add_argument("--n_null", type=int, default=100)
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(2)
    pid = json.load(open(a.pids))["prompt_ids"]
    split_of = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(a.prompts)}
    idx = [i for i, p in enumerate(pid) if split_of[p] == a.split]
    xs = json.load(open(a.x_tokens))["x_token_ids"]
    G = np.load(a.g, mmap_mode="r")
    S = np.load(a.s, mmap_mode="r")
    assert G.shape == S.shape and G.shape[1] == 34, G.shape
    raw_g, raw_s = np.asarray(G[idx]), np.asarray(S[idx])
    frac_diff = (raw_g != raw_s).mean(axis=(0, 2))
    xg, xsv = bf16_to_f32(raw_g), bf16_to_f32(raw_s)
    d = xsv - xg  # [n, 34, 1024]
    rms = d.pow(2).mean(dim=(0, 2)).sqrt()
    mean_d = d.mean(0)  # [34, 1024]
    const = mean_d.pow(2).sum(-1) / d.pow(2).sum(-1).mean(0).clamp_min(1e-30)
    tail, w_out = load_tail()
    xs_t = torch.tensor(xs)
    fo, fo_abs, fo_const, px_g, px_s = [], [], [], [], []
    for s in range(0, len(idx), 20):
        h = xg[s:s + 20].clone().requires_grad_(True)
        lg = tail(h, w_out)
        px = torch.softmax(lg, -1)[:, xs_t].sum(-1)
        px.sum().backward()
        g = h.grad  # [b, 34, 1024]
        c = (g * d[s:s + 20]).sum(-1)  # [b, 34]
        fo.append(c)
        fo_abs.append(c.abs())
        fo_const.append((g * mean_d[None]).sum(-1))
        px_g.append(px.detach())
        with torch.no_grad():
            px_s.append(torch.softmax(tail(xsv[s:s + 20], w_out), -1)[:, xs_t].sum(-1))
    fo, fo_abs, fo_const = torch.cat(fo), torch.cat(fo_abs), torch.cat(fo_const)
    px_g, px_s = torch.cat(px_g), torch.cat(px_s)
    per_pos = {
        "frac_diff": frac_diff.tolist(),
        "rms": rms.tolist(),
        "const": const.tolist(),
        "fo_mean": fo.mean(0).tolist(),
        "fo_abs_mean": fo_abs.mean(0).tolist(),
        "fo_const_mean": fo_const.mean(0).tolist(),
    }
    groups = {}
    for name, r in GROUPS.items():
        r = list(r)
        e_all = d[:, r].pow(2).sum(dim=(1, 2)).mean().item()
        e_const = mean_d[r].pow(2).sum().item()
        groups[name] = {
            "frac_diff": float(frac_diff[r].mean()),
            "offset_energy_share": None,
            "const_share_group_energy": e_const / max(e_all, 1e-30),
            "const_max": float(const[r].max()),
            "fo_sum_mean": float(fo[:, r].sum(1).mean()),
            "fo_sum_abs_mean": float(fo[:, r].sum(1).abs().mean()),
            "fo_const_sum_mean": float(fo_const[:, r].sum(1).mean()),
            "_e_all": e_all,
        }
    fo_tot = float(fo.sum(1).mean())
    for v in groups.values():
        v["fo_share_of_total"] = v["fo_sum_mean"] / fo_tot if abs(fo_tot) > 0 else None
    tot = d.pow(2).sum(dim=(1, 2)).mean().item()  # all 34 positions (groups overlap)
    for v in groups.values():
        v["offset_energy_share"] = v.pop("_e_all") / max(tot, 1e-30)
    out = {
        "g": a.g, "s": a.s, "split": a.split, "n": len(idx),
        "px_g_mean": float(px_g.mean()), "px_s_mean": float(px_s.mean()),
        "obs_gap_mean": float((px_s - px_g).mean()),
        "fo_pred_gap_mean": float(fo.sum(1).mean()),
        "groups": groups, "per_pos": per_pos, "secs": round(time.time() - t0, 1),
    }
    if a.auroc:
        out["auroc"] = readout(G, S, pid, split_of, a.n_null)
        for name, r in GROUPS.items():
            out["groups"][name]["auroc_max"] = max(out["auroc"]["test_auroc"][p] for p in r)
            out["groups"][name]["auroc_mean"] = float(np.mean([out["auroc"]["test_auroc"][p] for p in r]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_pos"}, indent=1))


if __name__ == "__main__":
    main()
