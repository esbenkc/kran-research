"""Gates before the pilot -> data/gate.json.

1. Real compiled graph (base model, 646 prompts, prompt-only plain pass): share of prompts whose logit sha256 differs
   leaky (ref) vs none, and score (bit-exact) vs none; mean 2-way cheat per build.
2. Cached block-22 inputs (rec pass): sha256 equal across none / ref / score on every prompt (the cache sits before
   the probe, so it must be build-invariant).
3. Gate E: emulated footprint (tail.py, base weights) vs the real one. Per prompt, D = change of the decision's 2-way
   logit difference (read - solve) when the probe is attached; corr(D_emulated, D_real) over 646 prompts, plus sizes.

    env/.venv/bin/python gate.py
"""
import json

import numpy as np
import torch

from tail import DEV, HERE, Rows, load_tail

C = HERE / "raw/cache/base"


def main():
    out = {}
    pl = {b: np.load(C / f"{b}_plain.npz") for b in ("none", "ref", "score")}
    sha = {b: json.load(open(C / f"{b}_plain_sha.json")) for b in ("none", "ref", "score")}
    for b in ("ref", "score"):
        out[f"plain_sha_differs_{b}_vs_none"] = float(np.mean([x != y for x, y in zip(sha[b], sha["none"])]))
        out[f"plain_absdiff_d2_max_{b}"] = float(np.abs(pl[b]["d2"] - pl["none"]["d2"]).max())
    for b in ("none", "ref", "score"):
        out[f"cheat2_{b}"] = float((pl[b]["p_read"] / (pl[b]["p_read"] + pl[b]["p_solve"])).mean())
    rs = {b: json.load(open(C / f"{b}_rec_sha.json")) for b in ("none", "ref", "score") if (C / f"{b}_rec_sha.json").exists()}
    for b in rs:
        if b != "none":
            out[f"rec_sha_equal_{b}_vs_none"] = float(np.mean([x == y for x, y in zip(rs[b], rs["none"])]))
    D_real = pl["ref"]["d2"] - pl["none"]["d2"]
    Z = Rows(C / "none_rec_tail.npy")
    cfg, tail, w2 = load_tail()
    tail.eval()
    N = Z.a.shape[0]
    d_off, d_on = [], []
    with torch.no_grad():
        for s in range(0, N, 32):
            j = list(range(s, min(N, s + 32)))
            h = Z.get(j, 0)
            d_off.append(tail(h, w2, torch.zeros(len(j), dtype=torch.bool, device=DEV))[0].float().cpu().double())
            d_on.append(tail(h, w2, torch.ones(len(j), dtype=torch.bool, device=DEV))[0].float().cpu().double())
    d_off, d_on = torch.cat(d_off).numpy(), torch.cat(d_on).numpy()
    D_emu = (d_on[:, 0] - d_on[:, 1]) - (d_off[:, 0] - d_off[:, 1])
    ce = lambda d: float((1 / (1 + np.exp(-(d[:, 0] - d[:, 1])))).mean())
    out.update({
        "emu_cheat2_noprobe": ce(d_off), "emu_cheat2_leaky": ce(d_on),
        "emu_share_prompts_changed": float(np.mean(D_emu != 0)), "real_share_prompts_changed": float(np.mean(D_real != 0)),
        "emu_absD_mean": float(np.abs(D_emu).mean()), "real_absD_mean": float(np.abs(D_real).mean()),
        "corr_D_emu_real": float(np.corrcoef(D_emu, D_real)[0, 1]),
        "sign_agree_share": float(np.mean(np.sign(D_emu) == np.sign(D_real))),
        "emu_vs_real_base_d2_corr": float(np.corrcoef(d_off[:, 0] - d_off[:, 1], pl["none"]["d2"])[0, 1]),
        "N": int(N)})
    np.savez(HERE / "data" / "gate_D.npz", D_emu=D_emu, D_real=D_real, d_off=d_off, d_on=d_on)
    json.dump(out, open(HERE / "data" / "gate.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
