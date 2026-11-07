"""Pre-data pilots on EXISTING adapters (the parent's site-14 runs; nothing trained tonight is read here).

Footprints at the block-26 input, 64 dev prompts:
  d_S = served leaky - served no-probe   (raw/cache/base26_L14, compiled inference graph, hook at q_norm 14)
  d_T = train leaky - train no-probe      (raw/cache/train26_L14, compiled TRAINING graph, same hook)
Each adapter gets h = h_S_none + e for a list of cues e (and h_T_none -> h_T_ref, the trainer's own pair), and we
report the mean decision-logit shift z(h) - z(h_S_none), z = logit(read) - logit(solve). Position classes:
prefix = positions 1-7 (bit-identical in every prompt), rest = 8..T.
Geometry: per-class norms and cosines of d_T vs d_S.

    PY=../../../env/.venv/bin/python
    $PY pilots.py --runs B_fed14pilot_s0,B_fed14pilot_s1,B_std14pilot_s0,B_std14pilot_s1,pilot_standard_s0
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import Rows, load_tail, splits  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--served", default="raw/cache/base26_L14")
    ap.add_argument("--train", default="raw/cache/train26_L14")
    ap.add_argument("--out", default="data/pilots.json")
    a = ap.parse_args()
    idx = splits()["dev"]
    Sn, Sr = Rows(HERE / a.served / "none_rec_tail.npy"), Rows(HERE / a.served / "ref_rec_tail.npy")
    Tn, Tr = Rows(HERE / a.train / "none_rec_tail.npy"), Rows(HERE / a.train / "ref_rec_tail.npy")
    sn, sr, tn, trr = (R.get(idx, 0).float() for R in (Sn, Sr, Tn, Tr))
    dS, dT = sr - sn, trr - tn
    pre = torch.zeros_like(dS)
    pre[:, 1:8] = 1.0
    p13 = torch.zeros_like(dS)
    p13[:, 1:4] = 1.0
    rest = 1.0 - pre
    geo = {}
    for cls, m in (("all", torch.ones_like(dS)), ("prefix1to7", pre), ("rest8plus", rest)):
        geo[cls] = {"norm_S": float((dS * m).flatten(1).norm(dim=1).mean()),
                    "norm_T": float((dT * m).flatten(1).norm(dim=1).mean()),
                    "cos_T_S": cos(dT * m, dS * m),
                    "proj_T_on_S": float(((dT * m).flatten(1) * (dS * m).flatten(1)).sum(1).mean()
                                         / ((dS * m).flatten(1).pow(2).sum(1).mean())),
                    "energy_share_S": float((dS * m).pow(2).sum() / dS.pow(2).sum()),
                    "energy_share_T": float((dT * m).pow(2).sum() / dT.pow(2).sum())}
    geo["prefix_identical_across_prompts"] = {"S": bool((dS[:, 1:8] == dS[:1, 1:8]).all()),
                                              "T": bool((dT[:, 1:8] == dT[:1, 1:8]).all())}
    geo["base_mismatch_T_vs_S_none_bits_share"] = float((tn != sn).float().mean())
    print("geometry", json.dumps(geo), flush=True)
    dperm = torch.roll(dS, 1, 0)
    cues = {"served": dS, "served_x0.25": 0.25 * dS, "served_x0.5": 0.5 * dS, "served_x2": 2 * dS,
            "served_prefix1to7": dS * pre, "served_prefix1to3": dS * p13, "served_rest8plus": dS * rest,
            "decoy_other_prompt": dperm, "train_cue": dT, "train_cue_prefix1to7": dT * pre,
            "train_cue_rest8plus": dT * rest}
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

    out = {"n_dev": len(idx), "geometry": geo, "runs": {}}
    for run in a.runs.split(","):
        sdA = torch.load(HERE / "data/runs" / run / "adapter.pt")
        with torch.no_grad():
            for n, p in pt.named_parameters():
                if p.requires_grad:
                    p.copy_(sdA[n].to(p.device))
        z0 = z_of(sn)
        row = {"z_served_none": float(z0.mean()), "shift": {}}
        for name, e in cues.items():
            row["shift"][name] = float((z_of(sn + e) - z0).mean())
        zt0 = z_of(tn)
        row["z_train_none"] = float(zt0.mean())
        row["shift"]["train_pair_own_base"] = float((z_of(trr) - zt0).mean())  # the cue exactly as the update sees it
        row["shift"]["base_mismatch_train_none_vs_served_none"] = float((zt0 - z0).mean())
        out["runs"][run] = row
        print(run, json.dumps({k: round(v, 3) for k, v in row["shift"].items()}), flush=True)
    Path(HERE / a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
