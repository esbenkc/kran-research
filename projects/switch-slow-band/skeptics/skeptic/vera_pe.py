"""Vera's own per-RoPE-pair patch engine (room theory-1103-switch-slow-band). Own code; build's scripts not read.

Tail = Qwen3-0.6B blocks 26-27 (seed 'base' = untrained; 's0'/'s1' = numeric-chameleon monitor LoRA merged on CPU).
Input = my own block-26 caches (vera_h26_1025.npz): cue-on = 'ref' build, cue-off = 'none' build.
Attention = HF eager attention, monkeypatched (config stays 'eager' so HF builds the causal mask).
Keys/queries reaching eager_attention_forward are post k_norm/q_norm AND post RoPE.
Qwen3 RoPE is rotate_half: pair j = head dims (j, j + 64), angle pos * base^(-2j/128).
Modes:
  patch: key dims of pairs P at key positions POS are replaced by the donor (cue-off) keys, all query heads of the
         chosen blocks, on query rows >= QROW0 (rows 0-2 keep the recipient's own attention, as in sink-value).
  zero:  the q.k term of pairs P is removed from every score (q and k dims of P set to 0 before the dot product),
         for the chosen blocks; used for the Phase-0 sink cut.
  capture/record: keep q,k,v per block / attention weights per block.
"""
import json, re
import numpy as np, torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import repeat_kv
import transformers.models.qwen3.modeling_qwen3 as _mq

HERE = "/Users/esben/work/theory-1103-switch-slow-band/swarm-theory-1103-switch-slow-band/skeptic"
HD, NP, NH = 128, 64, 16
QROW0 = 3
S = {"mode": "none"}


def pair_dims(pairs):
    d = torch.zeros(HD, dtype=torch.bool)
    for j in pairs:
        d[j] = True; d[j + NP] = True
    return d


def vera_attn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    L = module.layer_idx; g = module.num_key_value_groups
    ks, vs = repeat_kv(key, g), repeat_kv(value, g)
    mode = S["mode"]
    if mode == "capture":
        S["cap"][L] = (query.detach().clone(), ks.detach().clone(), vs.detach().clone())
    q = query
    if mode == "zero" and L in S["blocks"]:
        dm = S["dims"].to(query.device)
        q = query.masked_fill(dm, 0.0); ks = ks.masked_fill(dm, 0.0)
    if mode == "patch" and L in S["blocks"]:
        dk = S["donor"][L].to(ks.dtype)
        dm = S["dims"].to(ks.device)[None, None, None, :]
        pm = torch.zeros(ks.shape[2], dtype=torch.bool, device=ks.device); pm[S["pos"]] = True
        hm = torch.zeros(ks.shape[1], dtype=torch.bool, device=ks.device); hm[S.get("heads", list(range(NH)))] = True
        m = dm & pm[None, None, :, None] & hm[None, :, None, None]
        kp = torch.where(m, dk, ks)
        w = torch.matmul(q, ks.transpose(2, 3)) * scaling; wp = torch.matmul(q, kp.transpose(2, 3)) * scaling
        rows = torch.arange(w.shape[2], device=w.device) >= QROW0
        w = torch.where(rows[None, None, :, None], wp, w)
    else:
        w = torch.matmul(q, ks.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask
    a = torch.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    if mode == "record" or S.get("rec_too"):
        S["rec"][L] = a.float().cpu()
    o = torch.matmul(a, vs)
    return o.transpose(1, 2).contiguous(), a


_mq.eager_attention_forward = vera_attn


def bf(a):
    return torch.from_numpy(a.astype(np.int16)).view(torch.bfloat16)


class Eng:
    def __init__(self, seed, dt="fp32", dev="cpu"):
        self.DT = {"bf16": torch.bfloat16, "fp32": torch.float32}[dt]; self.dev = dev
        z = np.load(f"{HERE}/in/h26_mon.npz")
        self.CON, self.COFF = bf(z["ref_h26"]), bf(z["none_h26"])
        self.X = torch.tensor(json.load(open(f"{HERE}/in/x_mon.json"))["x_token_ids"])
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16, attn_implementation="eager").eval()
        self.theta = float(model.config.rope_theta) if hasattr(model.config, "rope_theta") else None
        try:
            self.theta = float(model.config.rope_parameters["rope_theta"])
        except Exception:
            pass
        if seed != "base":
            full = model.state_dict()
            sd = torch.load(f"{HERE}/in/mon_{seed}.pt", map_location="cpu")
            n = 0
            for k in sd:
                if ".lora_A." not in k: continue
                m = re.match(r"base_model\.model\.layers\.(\d+)\.(.*)\.lora_A\.default\.weight", k)
                A, B = sd[k].float(), sd[k.replace("lora_A", "lora_B")].float()
                kk = f"model.layers.{int(m.group(1)) + 26}.{m.group(2)}.weight"
                full[kk].copy_((full[kk].float() + B @ A * 2.0).to(torch.bfloat16)); n += 1
            assert n == 14, n
            del full
        model.model.layers = model.model.layers[26:]
        model = model.to(self.DT).to(dev)
        for i, l in enumerate(model.model.layers): l.self_attn.layer_idx = i
        self.model = model; self.T = self.CON.shape[1]

    def run(self, h):
        with torch.no_grad():
            lg = self.model(inputs_embeds=h.to(self.DT).to(self.dev), use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
        return torch.softmax(lg, -1)[:, self.X].sum(1)

    def px(self, h, **mode):
        """P(X) per prompt. Tokens 0-2 are one fixed vector per build, so donor keys at 0-2 come from row 0 of COFF."""
        S.clear(); S.update({"mode": "none", "rec": {}}); S.update(mode)
        out = self.run(h).numpy(); S.clear(); S["mode"] = "none"; return out

    def donor(self):
        S.clear(); S.update({"mode": "capture", "cap": {}})
        self.run(self.COFF[:1]); cap = S["cap"]; S.clear(); S["mode"] = "none"
        return {L: cap[L][1] for L in cap}  # [1, 16, T, 128] keys at all positions, donor build (row 0)

    def capture(self, h):
        S.clear(); S.update({"mode": "capture", "cap": {}, "rec_too": True, "rec": {}})
        self.run(h); cap, rec = S["cap"], S["rec"]; S.clear(); S["mode"] = "none"; return cap, rec
