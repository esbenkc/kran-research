"""Vera: follow-up on my R5 outlier (grader s1 bf16: one bf16 step down in ordinary dim 16 at token 0 moved 0.104).
Same tail loading as vera_dimpatch.py (grader merged tails), MPS; both precisions; donor-free edits on the condition-on
(eager) input: dim 16 at token 0 by -1 and +1 bf16 step (0.25 at 50.75), and the massive patch (a) for scale.
Usage: python vera_dim16.py
"""
import json, re
import numpy as np, torch
from transformers import AutoModelForCausalLM
R = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional"
z = np.load("/Users/esben/work/oracle-1030-stack-conditional/swarm-oracle-1030-stack-conditional/skeptic/vera_base.npz")
te = z["split"] == "test"
def bf(a): return torch.from_numpy(a.astype(np.int16)).view(torch.bfloat16)
CON, COFF = bf(z["eager_h26"][te]), bf(z["comp_h26"][te])
X = torch.tensor(json.load(open(f"{R}/data/x_tokens.json"))["x_token_ids"])
for seed in ("s0", "s1"):
    for DT in (torch.bfloat16, torch.float32):
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).eval()
        full = model.state_dict()
        for k, v in torch.load(f"{R}/raw/runs/stack_{seed}_lr0.0001_kl1_merged.pt", map_location="cpu").items():
            m = re.match(r"(?:model\.)?layers\.(\d+)\.(.*)", k)
            full[f"model.layers.{int(m.group(1)) + 26}.{m.group(2)}" if m else "model.norm.weight"].copy_(v.to(torch.bfloat16))
        model = model.to(DT).to("mps"); model.model.layers = model.model.layers[26:]
        for i, l in enumerate(model.model.layers): l.self_attn.layer_idx = i
        def px(h):
            with torch.no_grad():
                return torch.cat([torch.softmax(model(inputs_embeds=h[i:i + 50].to(DT).to("mps"), use_cache=False, logits_to_keep=1)
                                  .logits[:, -1].float().cpu(), -1)[:, X].sum(1) for i in range(0, len(h), 50)]).numpy()
        pon, poff = px(CON).mean(), px(COFF).mean(); sh = lambda h: (px(h).mean() - pon) / (poff - pon)
        out = {}
        for sgn in (-1, 1):
            a = CON.clone().float(); a[:, 0, 16] += sgn * 0.25; out[f"dim16@0 {sgn:+d}step"] = sh(a.to(torch.bfloat16))
        for d in (2, 277, 23):  # next-largest ordinary dims at token 0 (by |x|), one step down and up
            v = float(CON[0, 0, d].float()); st = 2.0 ** (np.floor(np.log2(abs(v))) - 7)
            for sgn in (-1, 1):
                a = CON.clone().float(); a[:, 0, d] += sgn * st; out[f"dim{d}@0 {sgn:+d}step"] = sh(a.to(torch.bfloat16))
        a = CON.clone(); a[:, :3, [1, 13, 35]] = COFF[:, :3, [1, 13, 35]]; out["(a) massive@0-2"] = sh(a)
        print(f"[grader {seed} {str(DT)[6:]}] P on {pon:.4f} off {poff:.4f} | " + "; ".join(f"{k} {v:+.4f}" for k, v in out.items()), flush=True)
        del model; torch.mps.empty_cache()
