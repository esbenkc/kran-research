"""Vera's own patch engine (same logic as vera_kv.py, importable). Own code; build's scripts not read.
setup(fam, seed, dt, dev) loads the 2-block tail (seed 'base' = untrained blocks 26-27) and my own block-26 caches.
Attention = my registered function 'vera' (copy of HF eager attention + per-query-head K/V swaps at chosen positions).
"""
import json, re
import numpy as np, torch
from transformers import AutoModelForCausalLM, AttentionInterface
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

MD = "/Users/esben/work/oracle-1031-massive-dims/swarm-oracle-1031-massive-dims/skeptic"
NH, BS = 16, 50
S = {"mode": "none", "donor": {}, "spec": {}, "rec": {}, "hold0": {}}

def bf(a): return torch.from_numpy(a.astype(np.int16)).view(torch.bfloat16)

def _attend(module, query, ks, vs, attention_mask, scaling, hold=None, key=None):
    g = module.num_key_value_groups
    w = torch.matmul(query, ks.transpose(2, 3)) * scaling
    if attention_mask is not None: w = w + attention_mask
    if hold is not None:
        # token-0 weight held at the recipient value (no K swap); donor-K weights on tokens 1.. rescaled to fill the rest
        w0 = torch.matmul(query, repeat_kv(key, g).transpose(2, 3)) * scaling
        if attention_mask is not None: w0 = w0 + attention_mask
        a_rec = torch.softmax(w0.float(), -1); a_pat = torch.softmax(w.float(), -1)
        a0 = a_rec[..., :1]; rest = a_pat[..., 1:]; rest = rest / rest.sum(-1, keepdim=True).clamp_min(1e-30) * (1 - a0)
        a = torch.where(hold[None, :, None, None], torch.cat([a0, rest], -1), a_pat).to(query.dtype)
    else:
        a = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(a, vs), a

QROW0 = 3  # patches act on query positions >= 3 only (PREREG s2); rows 0-2 keep the recipient's own attention

def vera_attn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    L = module.layer_idx; g = module.num_key_value_groups
    if S["mode"] == "capture":
        S["donor"][L] = (key.detach().clone(), value.detach().clone()); S.setdefault("q", {})[L] = query.detach().clone()
    ks, vs = repeat_kv(key, g), repeat_kv(value, g)
    o, a = _attend(module, query, ks, vs, attention_mask, scaling)
    if S["mode"] == "patch" and L in S["spec"]:
        dk, dv = S["donor"][L]; dk, dv = repeat_kv(dk, g), repeat_kv(dv, g)
        kp, vp, hold = ks, vs, None
        for which, mask in S["spec"][L]:
            m = mask[None, :, :, None]
            if which in ("k", "kv", "khold"): kp = torch.where(m, dk, kp)
            if which in ("v", "kv"): vp = torch.where(m, dv, vp)
            if which == "vrand": vp = torch.where(m, vp + S["vrand"][L], vp)
            if which == "khold": hold = mask.any(1)
            if which in ("a2sink", "a2vanish"): S["_mv"] = (which, mask)
        if "_mv" in S:
            which, mask = S.pop("_mv")
            _, a_rec = _attend(module, query, ks, vs, attention_mask, scaling)
            _, a_pat = _attend(module, query, torch.where(mask[None, :, :, None], dk, ks), vs, attention_mask, scaling)
            a_rec, a_pat = a_rec.float(), a_pat.float()
            pm = mask.any(0)  # key positions whose weight takes the donor-K value
            hm = mask.any(1)[None, :, None, None]
            a_new = torch.where(pm[None, None, None, :], a_pat, a_rec)
            if which == "a2sink":
                d = ((a_rec - a_pat) * pm[None, None, None, :]).sum(-1, keepdim=True)
                a_new = torch.cat([a_new[..., :1] + d, a_new[..., 1:]], -1)
            a_new = torch.where(hm, a_new, a_rec).to(query.dtype)
            op, ap = torch.matmul(a_new, vp), a_new
        else:
            op, ap = _attend(module, query, kp, vp, attention_mask, scaling, hold, key)
        o = torch.cat([o[:, :, :QROW0], op[:, :, QROW0:]], 2); a = torch.cat([a[:, :, :QROW0], ap[:, :, QROW0:]], 2)
    if S["mode"] == "record": S["rec"][L] = a.float().cpu()
    return o.transpose(1, 2).contiguous(), a
import transformers.models.qwen3.modeling_qwen3 as _mq
_mq.eager_attention_forward = vera_attn  # keep config "eager" so HF builds the eager (additive, causal) mask

class Eng:
    def __init__(self, fam, seed, dt, dev="mps"):
        self.DT = {"bf16": torch.bfloat16, "fp32": torch.float32}[dt]; self.dev = dev
        if fam == "grader":
            R = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional"
            z = np.load("/Users/esben/work/oracle-1030-stack-conditional/swarm-oracle-1030-stack-conditional/skeptic/vera_base.npz")
            te = z["split"] == "test"; self.CON, self.COFF = bf(z["eager_h26"][te]), bf(z["comp_h26"][te])
            tail = f"{R}/raw/runs/stack_{seed}_lr0.0001_kl1_merged.pt"; lora = False
        else:
            R = "/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl"
            q = np.load(f"{MD}/vera_h26_1025.npz"); self.CON, self.COFF = bf(q["ref_h26"]), bf(q["none_h26"])
            tail = f"{R}/raw/runs/r50_{seed}_lr0.0001_kl1_at8192_adapter.pt"; lora = True
        self.X = torch.tensor(json.load(open(f"{R}/data/x_tokens.json"))["x_token_ids"])
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16, attn_implementation="eager").eval()
        full = model.state_dict()
        if seed != "base":
            sd = torch.load(tail, map_location="cpu")
            for k in sd:
                if lora:
                    if ".lora_A." not in k: continue
                    m = re.match(r"base_model\.model\.layers\.(\d+)\.(.*)\.lora_A\.default\.weight", k)
                    A, B = sd[k].float(), sd[k.replace("lora_A", "lora_B")].float()
                    kk = f"model.layers.{int(m.group(1)) + 26}.{m.group(2)}.weight"
                    full[kk].copy_((full[kk].float() + B @ A * 2.0).to(torch.bfloat16))
                else:
                    m = re.match(r"(?:model\.)?layers\.(\d+)\.(.*)", k)
                    kk = f"model.layers.{int(m.group(1)) + 26}.{m.group(2)}" if m else "model.norm.weight"
                    full[kk].copy_(sd[k].to(torch.bfloat16))
        del full
        model.model.layers = model.model.layers[26:]  # slice BEFORE the cast/move: keeps peak RSS low (fp32 full model = 2.4 GB)
        model = model.to(self.DT).to(dev)
        for i, l in enumerate(model.model.layers): l.self_attn.layer_idx = i
        self.model = model; self.T = self.CON.shape[1]

    def run(self, h):
        with torch.no_grad():
            lg = self.model(inputs_embeds=h.to(self.DT).to(self.dev), use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
        return torch.softmax(lg, -1)[:, self.X].sum(1)

    def plain(self, h):
        S["mode"] = "none"; return torch.cat([self.run(h[i:i + BS]) for i in range(0, len(h), BS)]).numpy()

    def patch(self, spec_fn, vrand_fn=None, rec=None, don=None):
        rec = self.CON if rec is None else rec; don = self.COFF if don is None else don
        o = []
        for i in range(0, len(rec), BS):
            S["mode"] = "capture"; S["donor"] = {}; self.run(don[i:i + BS])
            S["mode"] = "patch"; S["spec"] = spec_fn()
            if vrand_fn is not None: S["vrand"] = vrand_fn(i)
            o.append(self.run(rec[i:i + BS]))
        S["mode"] = "none"
        return torch.cat(o).numpy()

    def attn(self, h):
        """mean attention (N-mean) per block: answer token [16, T] and query tokens 3..T-1 mean [16, T]."""
        acc = {0: [], 1: []}
        for i in range(0, len(h), BS):
            S["mode"] = "record"; self.run(h[i:i + BS])
            for L in (0, 1): acc[L].append(S["rec"][L])
        S["mode"] = "none"
        out = {}
        for L in (0, 1):
            w = torch.cat(acc[L]); out[f"ans_L{L}"] = w[:, :, -1].mean(0).numpy(); out[f"all_L{L}"] = w[:, :, 3:].mean((0, 2)).numpy()
        return out

    def M(self, heads, pos):
        m = torch.zeros(NH, self.T, dtype=torch.bool); m[np.ix_(list(heads), list(pos))] = True; return m.to(self.dev)
