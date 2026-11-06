"""Raw dump for Vera's lemma cell (#15362): the attached (pathwise) term's gradient on every LoRA parameter of the
arm-A trainer, computed with the probed-build flag and with the unprobed flag, on 8 dev prompts x 2 answer tokens
('read' row and 'solve' row). LoRA B matrices get N(0, 1e-3) so the A matrices also get gradient (at the usual
zero init only B would). Loss = lam * sum over rows of v.q_norm24(answer token) / sd_M0, lam = 0.4.
Also the block-22 input sha256 per build (rec pass) for the same prompts.

    env/.venv/bin/python lemma.py  -> data/lemma/lemma_grads.pt
"""
import json

import torch
from peft import LoraConfig, get_peft_model

from rl import TARGETS
from tail import DEV, HERE, Rows, load_tail, splits


def main():
    idx = splits()["dev"][:8]
    Z = Rows(HERE / "raw/cache/base/none_rec_tail.npy")
    pv = torch.load(HERE / "data" / "probe_v.pt")
    v = pv["v"].float().to(DEV)
    cfg, tail, w2 = load_tail()
    for p in tail.parameters():
        p.requires_grad_(False)
    m = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, bias="none"),
                       autocast_adapter_dtype=True)
    torch.manual_seed(0)
    for n, p in m.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=1e-3)
    m.train()
    out = {"prompt_idx": idx, "lam": 0.4, "sd_M0": float(pv["sd_M0"]), "grads": {}, "M": {}}
    for flag in (True, False):
        m.zero_grad(set_to_none=True)
        fl = torch.full((len(idx),), flag, device=DEV, dtype=torch.bool)
        _, qn = m(Z.get(idx, 0), w2, fl, Z.get(idx, 1))
        M = qn @ v
        loss = 0.4 * M.sum() / float(pv["sd_M0"])
        loss.backward()
        key = "probed" if flag else "unprobed"
        out["M"][key] = M.detach().cpu()
        out["grads"][key] = {n: (p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p).cpu())
                             for n, p in m.named_parameters() if p.requires_grad}
    shas = {}
    for b in ("none", "ref", "score"):
        f = HERE / f"raw/cache/base/{b}_rec_sha.json"
        if f.exists():
            s = json.load(open(f))
            shas[b] = [s[i] for i in idx]
    out["block22_sha"] = shas
    d = HERE / "data" / "lemma"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(out, d / "lemma_grads.pt")
    g1, g0 = out["grads"]["probed"], out["grads"]["unprobed"]
    same = sum(torch.equal(g1[k], g0[k]) for k in g1)
    nz = [k for k in g1 if g1[k].abs().sum() > 0]
    layers = sorted({int(k.split("layers.")[1].split(".")[0]) + 22 for k in nz})
    print(f"params {len(g1)}, bit-identical probed vs unprobed {same}, nonzero grads in blocks {layers}")
    print("M equal:", torch.equal(out["M"]["probed"], out["M"]["unprobed"]), "sha:", shas)


if __name__ == "__main__":
    main()
