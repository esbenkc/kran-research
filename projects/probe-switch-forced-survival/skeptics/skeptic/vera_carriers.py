"""Decoy test on the CARRIERS (Kasper #31769, Iris #31770). For one adapter, on the 64 dev prompts, at the frozen
per-prompt T_i (and at fixed T 1, 4, 16 for reference): share of the full-vocabulary score and of the A/B-only score
(KL over the renormalised {A, B} pair, same T) removed by cutting given head sets. Also the A/B log-odds push removed.
Usage: vera_carriers.py PROJECT ADAPTER NAME 'set1;set2;...'  (a set = comma-separated head names)"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vera_rebuild import Runner, load_embed_bf16, lo
from vera_tail import Tail, load_base, load_lora, load_cache, A_ID, B_ID

proj, adapter, name, sets = sys.argv[1], sys.argv[2], sys.argv[3], [s.split(",") for s in sys.argv[4].split(";")]
cache = os.path.join(proj, "raw/step2/Qwen3-1.7B/compile_C")
dev = [json.loads(l)["id"] for l in open(os.path.join(proj, "inputs/prompts.jsonl")) if json.loads(l)["split"] == "dev"]
xu, xp = load_cache(cache, dev)
R = Runner(Tail(load_base(), load_lora(adapter), 2.0), load_embed_bf16(), xu, xp)
zu, zp = R.logits(R.hu), R.logits(R.probed([]))
tp, tu = zp.argmax(-1), zu.argmax(-1)
top2 = zp.topk(2, -1).values
lead = torch.where(tp != tu, zp.gather(1, tp[:, None])[:, 0] - zp.gather(1, tu[:, None])[:, 0], top2[:, 0] - top2[:, 1])
temps = {"TB": torch.clamp(lead / 4, min=1.0)[:, None], "T1": torch.ones(64, 1), "T4": torch.full((64, 1), 4.0), "T16": torch.full((64, 1), 16.0)}


def kl(z, T, ab=False):
    if ab:
        a, b = torch.log_softmax(z[:, [A_ID, B_ID]] / T, -1), torch.log_softmax(zu[:, [A_ID, B_ID]] / T, -1)
    else:
        a, b = torch.log_softmax(z / T, -1), torch.log_softmax(zu / T, -1)
    return (a.exp() * (a - b)).sum(-1).mean().item()


push0 = (lo(zp) - lo(zu)).mean().item()
out = {"run": name, "T_B_median": float(temps["TB"].median()), "push": push0, "sets": {}}
for s in sets:
    zc = R.logits(R.probed(s))
    row = {"push_removed": 1 - (lo(zc) - lo(zu)).mean().item() / push0,
           "strict_dev": float(((lo(zc) > 0) & (lo(zu) < 0)).float().mean())}
    for tn, T in temps.items():
        D, DAB = kl(zp, T), kl(zp, T, True)
        row[tn] = {"full_share": (D - kl(zc, T)) / D, "AB_share": (DAB - kl(zc, T, True)) / DAB, "D": D, "D_AB": DAB}
    out["sets"]["+".join(s)] = row
    print("+".join(s), json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 4) for kk, vv in v.items()}) for k, v in row.items()}), flush=True)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", f"carriers_{name}.json"), "w"), indent=1)
