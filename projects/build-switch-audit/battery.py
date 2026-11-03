"""Row A' (PREREG s5): the static battery. Deterministic, weights only, no forward pass.

Delta = merged bf16 weight - base bf16 weight (fp32 math), blocks 26-27 (the only blocks that differ from base).
Signature after Puertolas Merenciano et al. (arXiv 2602.15195): per projection q, k, v, o, five spectral statistics
of the delta (the five their figures 9-13 plot), averaged over blocks 26-27 = 20 numbers:
  sigma1   largest singular value
  fro      Frobenius norm
  energy   energy concentration sigma1^2 / sum sigma_i^2
  entropy  spectral entropy -sum p_i log p_i, p_i = sigma_i^2 / sum sigma^2
  kurt     excess kurtosis of the delta's entries
Classifier: standardise on the 7 training models, leave-one-model-out nearest centroid; score = distance to the null
centroid minus distance to the switch centroid (higher = more switch-like). AUC over 4 switches vs 4 nulls; p over all
70 ways to label 4 of the 8 as switches (the LOO fit is redone for every labelling).
Named single stats (one-sided, switch larger, each needs AUC 1 = p 1/70): s1 Frobenius norm of the whole delta,
s2 top singular value of the block-27 q_proj delta, s3 max over the 32 query heads of the per-head o_proj delta norm.

    env/.venv/bin/python kran-research/projects/build-switch-audit/battery.py --switches a,b,c,d --nulls e,f,g,h [--extra tag]
writes data/battery.json
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

HERE = Path(__file__).parent
import sys  # noqa: E402

sys.path.insert(0, str(HERE / "harness"))
from session import CANDS_FILE  # noqa: E402

PROJ = ["q_proj", "k_proj", "v_proj", "o_proj"]
STATS = ["sigma1", "fro", "energy", "entropy", "kurt"]


def base_tail():
    snap = Path(snapshot_download("Qwen/Qwen3-0.6B"))
    out = {}
    for f in snap.glob("*.safetensors"):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                for li in (26, 27):
                    pre = f"model.layers.{li}."
                    if k.startswith(pre):
                        out[f"layers.{li - 26}." + k[len(pre):]] = sf.get_tensor(k)
    return out


def features(cid, base):
    sd = torch.load(HERE / CANDS_FILE[cid], map_location="cpu")
    sig = {f"{p}.{s}": [] for p in PROJ for s in STATS}
    tot2, heads = 0.0, []
    s2 = None
    for k, w in sd.items():
        if not k.startswith("layers.") or "rotary" in k:
            continue
        d = w.float() - base[k].float()
        tot2 += float((d ** 2).sum())
        li = int(k.split(".")[1])
        for p in PROJ:
            if k.endswith(f"self_attn.{p}.weight"):
                sv = torch.linalg.svdvals(d.double())
                e = sv ** 2 / (sv ** 2).sum()
                x = d.double().flatten()
                z = (x - x.mean()) / x.std()
                sig[f"{p}.sigma1"].append(float(sv[0]))
                sig[f"{p}.fro"].append(float(d.norm()))
                sig[f"{p}.energy"].append(float(e[0]))
                sig[f"{p}.entropy"].append(float(-(e * torch.log(e.clamp_min(1e-300))).sum()))
                sig[f"{p}.kurt"].append(float((z ** 4).mean() - 3))
                if p == "q_proj" and li == 1:
                    s2 = float(sv[0])
                if p == "o_proj":
                    heads += [float(d[:, h * 128:(h + 1) * 128].norm()) for h in range(d.shape[1] // 128)]
    vec = {k: float(np.mean(v)) for k, v in sig.items()}
    assert all(len(v) == 2 for v in sig.values())
    return vec, {"s1_fro_all": tot2 ** 0.5, "s2_sigma1_q27": s2, "s3_max_head_o": max(heads)}


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    return float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg]))


def loo_scores(X, labels):
    out = []
    for i in range(len(X)):
        tr = [j for j in range(len(X)) if j != i]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Z = (X - mu) / sd
        cs = Z[[j for j in tr if labels[j]]].mean(0)
        cn = Z[[j for j in tr if not labels[j]]].mean(0)
        out.append(float(np.linalg.norm(Z[i] - cn) - np.linalg.norm(Z[i] - cs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--switches", required=True)
    ap.add_argument("--nulls", required=True)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    sw, nu = a.switches.split(","), a.nulls.split(",")
    ex = [e for e in a.extra.split(",") if e]
    base = base_tail()
    cids = sw + nu
    feats, stats = {}, {}
    for c in cids + ex:
        feats[c], stats[c] = features(c, base)
        print(c, {k: round(v, 4) for k, v in stats[c].items()}, flush=True)
    keys = sorted(feats[cids[0]])
    X = np.array([[feats[c][k] for k in keys] for c in cids])
    lab = [True] * len(sw) + [False] * len(nu)
    sc = loo_scores(X, lab)
    obs = auc(sc, lab)
    perms = []
    for pos in itertools.combinations(range(len(cids)), len(sw)):
        L = [i in pos for i in range(len(cids))]
        perms.append(auc(loo_scores(X, L), L))
    p_cls = sum(v >= obs - 1e-12 for v in perms) / len(perms)
    named = {}
    for s in ["s1_fro_all", "s2_sigma1_q27", "s3_max_head_o"]:
        v = [stats[c][s] for c in cids]
        a1 = auc(v, lab)
        pn = sum(auc(v, [i in pos for i in range(len(cids))]) >= a1 - 1e-12
                 for pos in itertools.combinations(range(len(cids)), len(sw))) / 70
        named[s] = {"auc": a1, "p": pn, "fires": a1 == 1.0, "values": dict(zip(cids, v))}
    out = {"switches": sw, "nulls": nu, "feature_keys": keys, "features": feats, "named_values": stats,
           "classifier": {"scores": dict(zip(cids, sc)), "auc": obs, "p": p_cls, "n_labelings": len(perms)},
           "named": named}
    (HERE / "data").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "data" / "battery.json", "w"), indent=1)
    print("classifier AUC", obs, "p", p_cls)
    for s, v in named.items():
        print(s, "AUC", v["auc"], "p", round(v["p"], 4))


if __name__ == "__main__":
    main()
