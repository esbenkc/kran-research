"""Update-level proof cells (start policy = the pilot adapter, LoRA on blocks 26-27, train prompts).

(1) GRADIENT IDENTITY. For every (prompt, action) the per-episode policy gradient d/dtheta log pi(action | h) is
    computed on the input each trainer feeds its gradient pass for a PROBED and for an UNPROBED episode, and the
    sha256 of the fp32 gradient bytes is compared:
      clean rerun ('standard')  probed and unprobed both read the served no-probe input -> must be identical
      reused numbers ('fed')    served leaky input vs served no-probe input
      probed rerun ('cue' T)    train-graph leaky input vs train-graph no-probe input
    Identical gradients for every (prompt, action) mean the update cannot depend on the probe flag except through
    the scalar advantages and the sampled tokens.
(2) LEARNABILITY GEOMETRY (Kasper #15659). g(e) = gradient over the LoRA weights of sum_i [z(h_i + e_i) - z(h_i)],
    z = decision logit (read minus solve). k_grad(t -> s) = <g(d_t), g(d_s)> / |g(d_t)|^2 = first-order carry of a
    switch learned on cue d_t to the served cue d_s.

    PY=../../../env/.venv/bin/python
    $PY gradsha.py --n 64 --out data/gradsha.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import DEV, Rows, load_tail, splits  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--served", default="raw/cache/base26_L14")
    ap.add_argument("--train", default="raw/cache/train26_L14")
    ap.add_argument("--init", default="pilot_standard_s0")
    ap.add_argument("--out", default="data/gradsha.json")
    a = ap.parse_args()
    idx = splits()["train"][: a.n]
    C = {"Sn": Rows(HERE / a.served / "none_rec_tail.npy"), "Sr": Rows(HERE / a.served / "ref_rec_tail.npy"),
         "Tn": Rows(HERE / a.train / "none_rec_tail.npy"), "Tr": Rows(HERE / a.train / "ref_rec_tail.npy")}
    cfg, tail, w2 = load_tail(26)
    for p in tail.parameters():
        p.requires_grad_(False)
    pt = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, bias="none"),
                        autocast_adapter_dtype=True)
    sdA = torch.load(HERE / "data" / "runs" / a.init / "adapter.pt")
    with torch.no_grad():
        for n, p in pt.named_parameters():
            if p.requires_grad:
                p.copy_(sdA[n].to(p.device))
    pt.eval()
    params = [p for p in pt.named_parameters() if p[1].requires_grad]

    def grad_vec(f):
        for _, p in params:
            p.grad = None
        f().backward()
        return torch.cat([p.grad.detach().float().flatten() for _, p in params])

    # (1) per-episode gradient identity, both actions, batch 1 (exactly what a per-episode term contributes)
    inputs = {"standard": ("Sn", "Sn"), "fed": ("Sn", "Sr"), "cue_T": ("Tn", "Tr")}
    ident = {k: 0 for k in inputs}
    shas = {k: [] for k in inputs}
    for j in idx:
        h = {c: C[c].get([j], 0) for c in C}
        for act in (0, 1):
            for k, (off, on) in inputs.items():
                gs = []
                for src in (off, on):
                    g = grad_vec(lambda: torch.log_softmax(pt(h[src], w2, None, None)[0], -1)[0, act])
                    gs.append(hashlib.sha256(g.cpu().numpy().tobytes()).hexdigest()[:16])
                ident[k] += gs[0] == gs[1]
                shas[k].append(gs)
        torch.mps.empty_cache()
    n_ep = 2 * len(idx)
    out = {"n_prompts": len(idx), "n_episodes": n_ep, "init": a.init,
           "identical_share": {k: ident[k] / n_ep for k in inputs}, "identical_count": ident, "sha": shas}
    print("identical", out["identical_share"], flush=True)

    # (2) g(e) for the served cue, the train-graph cue and the decoy (another prompt's served cue), batches of 16
    def z(hh):
        d2 = pt(hh, w2, None, None)[0]
        return (d2[:, 0] - d2[:, 1]).float()

    G = {}
    for name in ("served", "train", "decoy", "prefix_served", "prefix_train"):
        acc = None
        for s in range(0, len(idx), 16):
            jj = idx[s:s + 16]
            jp = idx[s + 1:s + 17] if s + 17 <= len(idx) else (idx[s + 1:] + idx[:1])
            if name in ("served", "prefix_served"):
                base, cue = C["Sn"].get(jj, 0), C["Sr"].get(jj, 0)
            elif name in ("train", "prefix_train"):
                base, cue = C["Tn"].get(jj, 0), C["Tr"].get(jj, 0)
            else:
                base = C["Sn"].get(jj, 0)
                cue = (base.float() + C["Sr"].get(jp, 0).float() - C["Sn"].get(jp, 0).float()).to(torch.bfloat16)
            if name.startswith("prefix"):
                cue = cue.clone()
                cue[:, 8:] = base[:, 8:]
            g = grad_vec(lambda: (z(cue) - z(base)).sum())
            acc = g if acc is None else acc + g
            torch.mps.empty_cache()
        G[name] = acc
    k = {}
    for t in G:
        for s_ in G:
            if t != s_:
                k[f"{t}->{s_}"] = float((G[t] @ G[s_]) / (G[t] @ G[t]))
    out["g_norm"] = {t: float(G[t].norm()) for t in G}
    out["g_cos"] = {f"{t}|{s_}": float(torch.nn.functional.cosine_similarity(G[t], G[s_], 0))
                    for t in G for s_ in G if t < s_}
    out["k_grad"] = k
    print("k_grad", json.dumps({x: round(y, 3) for x, y in k.items()}), flush=True)
    print("g_cos", json.dumps({x: round(y, 3) for x, y in out["g_cos"].items()}), flush=True)
    Path(HERE / a.out).write_text(json.dumps(out, indent=1))
    with open(HERE / a.out.replace(".json", "_pairs.jsonl"), "w") as f:  # per-episode sha pairs (Vera #15702)
        for k, v in shas.items():
            for e, (off, on) in enumerate(v):
                f.write(json.dumps({"trainer": k, "episode": e, "prompt": str(idx[e // 2]), "action": ["read", "solve"][e % 2],
                                    "sha_unprobed": off, "sha_probed": on, "identical": off == on}) + "\n")


if __name__ == "__main__":
    main()
