"""Read-only control cell (Tove #15700, Vera #15756): is a switch a generic 'chat start moved' detector?

For each adapter, on the 64 dev prompts: decision-logit shift z(h_S_none + e) - z(h_S_none) for
  e = the served cue d_S, the train-graph cue d_T (14-27 graph), and K random cues u_k.
Each u_k is constant across prompts (like the real prefix footprint), lives on positions 1-7 only, and has at each
position the norm of d_T at that position (variant 'normT') or of d_S ('normS'). Directions: Gaussian, seed k (k = 0 is the PREREG cell).
Inputs are formed in fp32 and rounded to bf16 (the policy reads bf16 block-26 inputs).

    PY=../../../env/.venv/bin/python
    $PY randprefix.py --runs T_s0,T_s1,fed_s2,fed_s3 --k 8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import Rows, load_tail, splits  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="data/randprefix.json")
    a = ap.parse_args()
    idx = splits()["dev"]
    S = HERE / "raw/cache/base26_L14"
    T = HERE / "raw/cache/train26_L14"
    sn = Rows(S / "none_rec_tail.npy").get(idx, 0).float()
    dS = Rows(S / "ref_rec_tail.npy").get(idx, 0).float() - sn
    dT = Rows(T / "ref_rec_tail.npy").get(idx, 0).float() - Rows(T / "none_rec_tail.npy").get(idx, 0).float()
    H = sn.shape[-1]
    cues = {"served": dS, "train_cue": dT}
    for ref_name, ref in (("normT", dT), ("normS", dS)):
        pn = ref[0, 1:8].norm(dim=-1)  # per-position norm (identical in every prompt on positions 1-7)
        for k in range(a.k):
            g = torch.Generator().manual_seed(k)
            u = torch.randn(7, H, generator=g)
            u = u / u.norm(dim=-1, keepdim=True) * pn[:, None].cpu()
            e = torch.zeros_like(dS)
            e[:, 1:8] = u.to(e.device)[None]
            cues[f"rand_{ref_name}_{k}"] = e
    cfg, tail, w2 = load_tail(26)
    for p in tail.parameters():
        p.requires_grad_(False)
    pt = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, bias="none"),
                        autocast_adapter_dtype=True)
    pt.eval()

    def z_of(h):
        zs = []
        with torch.no_grad():
            for s in range(0, len(idx), 16):
                d2, _ = pt(h[s:s + 16].to(torch.bfloat16), w2, None, None)
                zs.append((d2[:, 0] - d2[:, 1]).float().cpu())
        return torch.cat(zs)

    out = {"n_dev": len(idx), "k": a.k, "runs": {}}
    for run in a.runs.split(","):
        sdA = torch.load(HERE / "data/runs" / run / "adapter.pt")
        with torch.no_grad():
            for n, p in pt.named_parameters():
                if p.requires_grad:
                    p.copy_(sdA[n].to(p.device))
        z0 = z_of(sn)
        sh = {name: float((z_of(sn + e) - z0).mean()) for name, e in cues.items()}
        tn_, tr_ = Rows(T / "none_rec_tail.npy").get(idx, 0).float(), Rows(T / "ref_rec_tail.npy").get(idx, 0).float()
        tgap = float((z_of(tn_) - z_of(tr_)).mean())  # the switch on the update's own numbers (none minus probed)
        # carry-equivalent of each cue: served-side gap it would produce / the switch on the training build
        ceq = {name: -v / tgap for name, v in sh.items()}
        summ = {}
        for ref_name in ("normT", "normS"):
            v = np.array([sh[f"rand_{ref_name}_{k}"] for k in range(a.k)])
            summ[ref_name] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                              "mean_abs": float(np.abs(v).mean())}
        for ref_name in ("normT", "normS"):
            c = np.array([ceq[f"rand_{ref_name}_{k}"] for k in range(a.k)])
            summ[ref_name].update({"carry_eq_min": float(c.min()), "carry_eq_max": float(c.max()),
                                   "carry_eq_max_abs": float(np.abs(c).max())})
        out["runs"][run] = {"shift": sh, "trainer_gap_logit_dev": tgap, "carry_eq": ceq, "random_summary": summ}
        print(run, "tgap", round(tgap, 3), "carry_served", round(ceq["served"], 3), "served", round(sh["served"], 3), "train_cue", round(sh["train_cue"], 3), json.dumps(summ), flush=True)
    Path(HERE / a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
