"""Iris: where does each head of blocks 26-27 look from the answer position, probe on vs off? (TRAIN prompts only)

Tests the mechanism line of #7283/#7301: the trigger is readable at <|im_end|> (position 71), so the head that carries
a switch should change its attention from the answer position (79) to/away from 71 when the probe is attached.
CPU, fp32 compute on the merged bf16 tails, eager attention, 64 TRAIN prompts. No logits.

    PY head_attn.py <cache dir> <prompts.jsonl> <out.json> <name>=<merged.pt> [...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

MID = "Qwen/Qwen3-1.7B"


def bf16(u16):
    return torch.from_numpy(np.ascontiguousarray(u16).view(np.int16)).view(torch.bfloat16)


def main():
    cdir, prompts, out = sys.argv[1:4]
    models = dict(a.split("=", 1) for a in sys.argv[4:])
    cfg = AutoConfig.from_pretrained(MID)
    cfg._attn_implementation = "eager"
    pid = json.load(open(f"{cdir}/meta.json"))["prompt_ids"]
    split = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(prompts)}
    tr = [i for i, p in enumerate(pid) if split[p] == "train"][:64]
    z = {c: bf16(np.load(f"{cdir}/{c}_rec_tail.npy", mmap_mode="r")[tr]).float() for c in ("none", "probe")}
    rot = Qwen3RotaryEmbedding(cfg)
    res = {"n_train_prompts": len(tr), "models": {}}
    for name, path in models.items():
        sd = torch.load(path)
        layers = torch.nn.ModuleList([Qwen3DecoderLayer(cfg, i) for i in (26, 27)])
        own = {k[len("layers."):]: v.float() for k, v in sd.items() if k.startswith("layers.")}
        missing, unexpected = layers.load_state_dict(own, strict=False)
        assert not unexpected and not [m for m in missing if "rotary" not in m], (missing, unexpected)
        layers.eval()
        att = {}
        for li in range(2):
            layers[li].self_attn.register_forward_hook(
                lambda m, a, o, li=li: att.__setitem__(li, o[1][:, :, -1].detach()))  # (B, H, T) from answer pos
        r = {}
        for c in ("none", "probe"):
            h = z[c]
            pos = torch.arange(h.shape[1])[None].expand(h.shape[0], -1)
            cs = rot(h, pos)
            T = h.shape[1]
            mask = torch.full((T, T), float("-inf")).triu(1)[None, None]
            with torch.no_grad():
                for layer in layers:
                    h = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=cs)
            r[c] = {li: att[li].mean(0).numpy() for li in range(2)}  # (H, T)
        heads = {}
        for li in range(2):
            for hh in range(16):
                a0, a1 = r["none"][li][hh], r["probe"][li][hh]
                heads[f"{26 + li}.{hh}"] = {"to_71_off": float(a0[71]), "to_71_on": float(a1[71]),
                                            "to_77_off": float(a0[77]), "to_77_on": float(a1[77]),
                                            "tv_shift": float(0.5 * np.abs(a1 - a0).sum()),
                                            "argmax_off": int(a0.argmax()), "argmax_on": int(a1.argmax())}
        res["models"][name] = heads
        top = sorted(heads, key=lambda k: -heads[k]["tv_shift"])[:5]
        print(name, [(k, round(heads[k]["tv_shift"], 3), round(heads[k]["to_71_off"], 3),
                      round(heads[k]["to_71_on"], 3)) for k in top], flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
