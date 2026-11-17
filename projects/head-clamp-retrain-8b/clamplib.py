"""Shared clamp code (ported from head-clamp-retrain/clamplib.py; constants from cfgm.py instead of 1.7B literals).

Head = one query head of block FIRST or FIRST+1 (8B: 34, 35; 32 heads each) = its HD-column slice of the o_proj input
(Sauers' unit, as in probe-switch-heads/heads.py), named '<block>.<h>'. A clamp is a forward pre-hook on o_proj; it
acts in every forward. Kinds:
  swap   the probed row gets the DETACHED unprobed row's slice of the same prompt, at all positions (the audit's
         operation; needs the probe-off run, so it is a routing test, not a deployable edit). Batches are always
         [unprobed rows; probed rows] of the same prompts.
  umean  side row only: two constants per head from the 512 UNPROBED train prompts: at position 0 its mean at
         position 0; at every position >= 1 its mean pooled over positions 1..T-1.
Sets: '35.7,34.5' | all | none | allbut:34.11 (every head of the two tail blocks except the listed ones).
The 1.7B parent's LM-loss rows are dropped: under swap the clamp is the identity on text without a probed row.
"""
import hashlib
import json

import numpy as np
import torch

from cfgm import CDIR, FIRST, HD, HERE, MODEL, NH
from tail import DEV, Rows, load_tail


def all_heads() -> list[tuple[int, int]]:
    return [(li, h) for li in range(2) for h in range(NH)]


def parse_heads(s: str) -> list[tuple[int, int]]:
    if s == "none":
        return []
    if s in ("all", "all32", "all64"):
        return all_heads()
    if s.startswith("allbut:"):
        keep = set(parse_heads(s[len("allbut:"):]))
        return [x for x in all_heads() if x not in keep]
    out = []
    for x in s.split(","):
        b, h = x.split(".")
        assert int(b) in (FIRST, FIRST + 1) and 0 <= int(h) < NH, x
        out.append((int(b) - FIRST, int(h)))
    return sorted(set(out))


def name_heads(heads) -> list[str]:
    return [f"{FIRST + li}.{h}" for li, h in heads]


def tail_of(m):
    """the Tail module (layers, norm, rotary) inside a peft wrapper or a merged model"""
    return m.base_model.model if hasattr(m, "base_model") and hasattr(m.base_model, "model") else m


class Clamp:
    def __init__(self, model, heads, kind="swap", const=None):
        self.kind, self.on, self.const = kind, True, const
        self.mask = {}
        for li, h in heads:
            m = self.mask.setdefault(li, torch.zeros(NH * HD, dtype=torch.bool, device=DEV))
            m[h * HD:(h + 1) * HD] = True
        self.handles = [tail_of(model).layers[li].self_attn.o_proj.register_forward_pre_hook(self._pre(li))
                        for li in range(2)]

    def _pre(self, li):
        def f(mod, args):
            if not self.on or li not in self.mask:
                return None
            x = args[0]
            if self.kind == "swap":
                n = x.shape[0] // 2
                src = x[:n].detach()
                return (torch.where(self.mask[li], torch.cat([src, src]), x),)
            c = self.const[li].to(x.dtype)
            if c.dim() == 2:  # per position
                c = c[: x.shape[1]]
            return (torch.where(self.mask[li], c, x),)
        return f

    def remove(self):
        for hd in self.handles:
            hd.remove()


@torch.no_grad()
def head_means(model, w_out, z0, z1, ids, bs=16) -> dict:
    """o_proj-input umean of the unclamped model, fp32 accumulate: [T, NH*HD] (row 0 = unprobed mean at position 0,
    rows 1.. = unprobed mean pooled over positions 1..T-1). Side row only."""
    rec = {}
    hs = [tail_of(model).layers[li].self_attn.o_proj.register_forward_pre_hook(
        lambda m, a, li=li: rec.__setitem__(li, a[0].detach())) for li in range(2)]
    was = model.training
    model.eval()
    u = {li: None for li in range(2)}
    n = 0
    for s in range(0, len(ids), bs):
        b = ids[s:s + bs]
        model(torch.cat([z0[b], z1[b]]).to(DEV), w_out)
        k = len(b)
        for li in range(2):
            su = rec[li].float()[:k].sum(0)
            u[li] = su if u[li] is None else u[li] + su
        n += k
    for hd in hs:
        hd.remove()
    model.train(was)
    out = {}
    for li in range(2):
        um = u[li] / n
        out[li] = torch.cat([um[:1], um[1:].mean(0, keepdim=True).expand(um.shape[0] - 1, -1)]).to(torch.bfloat16)
    return out


def removed(clamped: dict, orig: dict) -> float:
    return 1 - clamped["shift_mean"] / orig["shift_mean"]


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def setup(adapter=None, model_id: str = MODEL):
    """base tail (frozen bf16), caches, splits, the LoRA wrapper (r16 alpha 32, fp32 adapter) and, if given, the
    saved adapter loaded into it (else peft's own init: A random from the torch RNG, B = 0)."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    meta = json.load(open(CDIR / "meta.json"))
    split_of = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(HERE / "raw" / "prompts.jsonl")}
    pid = meta["prompt_ids"]
    S = {"pid": pid, "z0": Rows(CDIR / "none_rec_tail.npy"), "z1": Rows(CDIR / "probe_rec_tail.npy"),
         "idx": {s: [i for i, p in enumerate(pid) if split_of[p] == s] for s in ("train", "dev", "test")}}
    tok = AutoTokenizer.from_pretrained(model_id)
    S["A"], S["B"] = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
    cfg, tail, w_out = load_tail(model_id, 2)
    for p in tail.parameters():
        p.requires_grad_(False)
    S["w_out"] = w_out
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules=TARGETS)
    ptail = get_peft_model(tail, lcfg, autocast_adapter_dtype=True)
    own = {k: v for k, v in ptail.named_parameters() if v.requires_grad}
    if adapter is not None:
        sd = torch.load(adapter)
        assert set(sd) == set(own), set(sd) ^ set(own)
        with torch.no_grad():
            for k, v in sd.items():
                own[k].copy_(v.to(own[k].device))
        S["adapter_sha256"] = hashlib.sha256(open(adapter, "rb").read()).hexdigest()
    assert all(p.dtype == torch.float32 for p in own.values())
    S["model"], S["params"] = ptail, list(own.values())
    return S


def compact(rows: list[dict]) -> list:
    """per-prompt rows for the skeptics: [prompt id, probe 0/1, top-1 id, logp(A), logp(B)]"""
    return [[r["prompt"], r["probe"], r["top"], round(r["logp_a"], 5), round(r["logp_b"], 5)] for r in rows]


def merge_eval(model):
    """merge the adapter into bf16 weights (as shipped); returns the merged tail"""
    merged = model.merge_and_unload()
    merged.eval()
    assert all(p.dtype == torch.bfloat16 for p in merged.parameters())
    return merged
