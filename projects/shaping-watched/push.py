"""Init push check (PREREG, Tove #15363 c), first = 26 layout, real cached numbers, LoRA at init.

For each arm's GRPO objective (the exact loss of rl.py, fed trainer, KL term omitted: its gradient is 0 at init) we
draw NB minibatches (P = 8 prompts x G = 8 samples, probe flag per prompt) and take the LoRA gradient g_k.
Reported per arm:
  snr_median_B  parent definition (numeric-chameleon-rl theory/predict.py): median over LoRA-B weights of
                |mean_k g_k| / rms_k(g_k)  (mean over minibatches in place of the parent's exact expectation)
  gap_snr       the push TOWARD A GAP: project each g_k on u = -grad(gap)/|grad(gap)| (gap = dev mean of
                cheat(no probe) - cheat(leaky); a descent step along -g raises the gap when g.u > 0... sign handled
                below), report mean_k(proj) / sd_k(proj) / sqrt(NB) is NOT used; we report mean/sd per minibatch.
Arms: O (parent contrast +1 / -1 for read, honest 0.6), B (reward-only shaping), parent (the parent's exact reward:
+1 read unprobed, -1 read probed, 0 honest), pilot (no penalty: the shared push only).

    env/.venv/bin/python push.py --nb 64 -> data/push.json
"""
import argparse
import json

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from rl import TARGETS, Data
from tail import DEV, HERE, load_tail, splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.4)
    ap.add_argument("--sd-ref", type=float, default=0.2)
    a = ap.parse_args()
    data, idx = Data(26), splits()
    cfg, tail, w2 = load_tail(26)
    for p in tail.parameters():
        p.requires_grad_(False)
    m = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, bias="none"),
                       autocast_adapter_dtype=True)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    params = [p for n, p in m.named_parameters() if p.requires_grad]
    isB = torch.cat([torch.full((p.numel(),), "lora_B" in n, dtype=torch.bool) for n, p in zip(names, params)])

    def flat():
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params]).float().cpu()

    # gradient of the dev gap
    m.zero_grad(set_to_none=True)
    for s0 in range(0, len(idx["dev"]), 8):  # chunks of 8: the whole dev set at once peaked at 2.97 GB
        j = idx["dev"][s0:s0 + 8]
        h0, _ = data.rows(j, False); h1, _ = data.rows(j, True)
        c0 = torch.softmax(m(h0, w2)[0], -1)[:, 0]
        c1 = torch.softmax(m(h1, w2)[0], -1)[:, 0]
        ((c0 - c1).sum() / len(idx["dev"])).backward()
        torch.mps.empty_cache()
    ggap = flat()
    u = ggap / ggap.norm()
    rng = np.random.default_rng(0)
    gen = torch.Generator(device=DEV).manual_seed(0)
    out = {"norm_grad_gap": float(ggap.norm()), "nb": a.nb, "arms": {}}
    for arm in ("pilot", "parent", "O", "B"):
        G = []
        for k in range(a.nb):
            b = list(rng.choice(idx["train"], 8, replace=False))
            flag = torch.from_numpy(rng.random(8) < 0.5).to(DEV)
            m.zero_grad(set_to_none=True)
            h, _ = data.rows(b, flag)
            lp = torch.log_softmax(m(h, w2)[0], -1)
            with torch.no_grad():
                smp = torch.multinomial(lp.exp(), 8, replacement=True, generator=gen)
                rd = smp == 0
                if arm == "parent":
                    r = torch.where(rd, torch.where(flag[:, None], -1.0, 1.0), 0.0)
                else:
                    r = torch.where(rd, 1.0, 0.6)
                    if arm == "O":
                        r = torch.where(rd & flag[:, None], -1.0, r)
                    if arm == "B":
                        Mg = torch.where(rd, data.Mt[b, 0][:, None], data.Mt[b, 1][:, None])
                        sdM = Mg.std(1, keepdim=True)
                        s = torch.where(sdM > 0, (Mg - Mg.mean(1, keepdim=True)) / (sdM + 1e-8), torch.zeros_like(Mg))
                        lam_g = a.lam * torch.clamp(r.std(1) / a.sd_ref, max=1.0) * flag.float()
                        r = r - lam_g[:, None] * s
                adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
            loss = -(adv * lp.gather(1, smp)).sum() / 64
            loss.backward()
            G.append(flat())
        G = torch.stack(G)                                   # [nb, n_params]
        mean, rms = G.mean(0), G.pow(2).mean(0).sqrt()
        snr = (mean.abs() / (rms + 1e-12))[isB & (rms > 0)]
        proj = -(G @ u)                                      # a step -lr*g changes the gap by -lr*g.ggap: push = -g.u
        out["arms"][arm] = {"snr_median_B": float(snr.median()), "gap_push_mean": float(proj.mean()),
                            "gap_push_sd": float(proj.std()), "gap_push_snr": float(proj.mean() / proj.std()),
                            "cos_mean_g_vs_gap": float(-(mean @ u) / mean.norm())}
        print(arm, out["arms"][arm], flush=True)
    json.dump(out, open(HERE / "data" / "push.json", "w"), indent=1)


if __name__ == "__main__":
    main()
