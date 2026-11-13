"""Behaviour at the audited snapshots, per prompt, with a bootstrap CI (Tove #21746: 'before behaviour shows' needs the
behaviour gap at the read with its CI).

For each cue model m0..m5 and snapshot t in {0, t_h, T}: p_off_i, p_on_i = P(opener token in X) on dev prompt i without /
with the planted cue (rl.py px, same code path: block-26 inputs from the cache, rows 0-2 replaced by the cue-on rows).
gap_i(t) = p_off_i - p_on_i;  D_i(t) = gap_i(t) - gap_i(0);  D(t) = mean_i D_i(t)  (= rl.py's D = gap(t) - gap(0)).
CI: 95% percentile bootstrap over the 64 dev prompts (2,000 resamples, numpy seed 20261113).
    python beh.py -> data/behaviour_ci.json (per-prompt gap_i and D_i saved; also checks D against the saved dev curve)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from rl import CACHE, TARGETS, cue_rows  # noqa: E402
from tail import DEV, Rows, load_tail, splits, x_ids  # noqa: E402

MODELS = {f"m{k}": (f"dir_u{1110 + k}_a0.03125_s{k}", k) for k in range(6)}
T = {"m0": 32, "m1": 32, "m2": 32, "m3": 32, "m4": 24, "m5": 32}
TH = {"m0": 16, "m1": 16, "m2": 16, "m3": 16, "m4": 8, "m5": 16}
READS = {m: sorted({0, TH[m], T[m]}) for m in MODELS}
OUTF = "behaviour_ci.json"
if os.environ.get("BEH_SET") == "e2":  # PREREG E2 sweep runs (seed 0): reads 0, T/4, T/2, T
    SW = [("sw_a0.00390625_s0", 192), ("sw_a0.00195312_s0", 320), ("sw_a0.000976562_s0", 704),
          ("sw_a0.000488281_s0", 1728), ("sw_a0.000244141_s0", 4352)]
    MODELS = {r: (r, 0) for r, _ in SW}
    T = dict(SW)
    TH = {r: t // 2 for r, t in SW}
    READS = {r: [0, t // 4, t // 2, t] for r, t in SW}
    OUTF = "behaviour_ci_e2.json"

meta = json.load(open(CACHE / "meta.json"))
idx = splits(meta["prompt_ids"])
Z = Rows(CACHE / "eager_rec_tail.npy")
xs = torch.tensor(x_ids(), device=DEV)
h0_u16 = np.ascontiguousarray(Z.a[0, :3])
out = {}
for m, (run, seed) in MODELS.items():
    a = json.load(open(HERE / "data" / "runs" / f"{run}.json"))["args"]
    on_rows = torch.from_numpy(cue_rows("rand", a["cue_alpha"], a["u_file"], h0_u16).view(np.int16)).view(
        torch.bfloat16).to(DEV)
    cfg, tail, w_out = load_tail(2)
    ptail = get_peft_model(tail.float(), LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS,
                                                    bias="none"), autocast_adapter_dtype=True)
    w_out = w_out.float()
    named = dict(ptail.named_parameters())
    ptail.eval()

    @torch.no_grad()
    def px(cued, bs=32):
        res = []
        rows = idx["dev"]
        for s in range(0, len(rows), bs):
            r = rows[s:s + bs]
            h = Z[r].to(DEV)
            if cued:
                h[:, :3] = on_rows
            lg = ptail(h.float(), w_out).float()
            res.append(torch.softmax(lg, -1)[:, xs].sum(-1).cpu().double().numpy())
        return np.concatenate(res)

    gaps = {}
    for t in READS[m]:
        sd = torch.load(HERE / "raw" / "runs" / run / f"step{t}.pt", map_location="cpu", weights_only=True)
        with torch.no_grad():
            for k, v in sd.items():
                named[k].copy_(v.to(named[k].device))
        gaps[t] = px(False) - px(True)
    ev = {c["step"]: c["D"] for c in json.load(open(HERE / "data" / "runs" / f"{run}.json"))["dev_curve"]
          + json.load(open(HERE / "data" / "runs" / f"{run}.json")).get("dense_curve", [])}
    g = np.random.default_rng(20261113)
    rec = {}
    for t in sorted(gaps):
        Di = gaps[t] - gaps[0]
        boots = np.array([Di[g.integers(0, len(Di), len(Di))].mean() for _ in range(2000)])
        rec[str(t)] = {"D": float(Di.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                       "D_saved_dev_curve": ev.get(t, 0.0 if t == 0 else None), "n_prompts": int(len(Di)),
                       "gap_mean": float(gaps[t].mean()), "gap_i": gaps[t].tolist(), "D_i": Di.tolist()}
    out[m] = {"run": run, "t_h": TH[m], "T": T[m], "reads": rec}
    del ptail, tail
    if DEV == "mps":
        torch.mps.empty_cache()
(HERE / "data" / OUTF).write_text(json.dumps(out, indent=1))
for m, r in out.items():
    th = r["reads"][str(r["t_h"])]
    print(m, f"D(t_h={r['t_h']}) = {th['D']:.4f} [{th['ci95'][0]:.4f}, {th['ci95'][1]:.4f}] saved {th['D_saved_dev_curve']}")
