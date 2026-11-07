"""Shared clamp code for t0_gate.py and train_clamp.py.

Head = one of the 16 query heads of block 26 or 27 = its 128-column slice of the o_proj input (Sauers' unit, as in
probe-switch-heads/heads.py), named '<block>.<h>'. A clamp is a forward pre-hook on o_proj; it acts in every forward.
Kinds:
  swap   the probed row gets the DETACHED unprobed row's slice of the same prompt, at all positions (the audit's
         operation; needs the probe-off run, so it is a routing test, not a deployable edit). Batches are always
         [unprobed rows; probed rows] of the same prompts. On LM text (no probed rows) it is the identity.
  umean  (PREREG v2) two constants per head from the 512 UNPROBED train prompts: at position 0 (the attention sink)
         its mean at position 0; at every position >= 1 its mean pooled over positions 1..79. Same two vectors in
         training, the t=0 gate and the LM loss.
  pmean  one position-free constant: the mean over the 512 unprobed train prompts and all 80 positions.
  bmean  one position-free constant: the mean over the 512 train prompts x BOTH conditions x all positions.
All means are computed once from the switched model (parent adapter, no clamp) before any update, then fixed.
Sets: '27.7,26.5' | all32 | none | allbut:26.11 (every head of 26-27 except the listed ones).
"""
import numpy as np
import torch

from tail import DEV, bf16

NH, HD, FIRST, T_LM = 16, 128, 26, 80
PARENT = {0: "lora_none_s0_2048", 1: "lora_none_s1", 2: "lora_none_s2"}


def parse_heads(s: str) -> list[tuple[int, int]]:
    if s == "none":
        return []
    allh = [(li, h) for li in range(2) for h in range(NH)]
    if s == "all32":
        return allh
    if s.startswith("allbut:"):
        keep = set(parse_heads(s[len("allbut:"):]))
        return [x for x in allh if x not in keep]
    out = []
    for x in s.split(","):
        b, h = x.split(".")
        assert int(b) in (26, 27) and 0 <= int(h) < NH, x
        out.append((int(b) - FIRST, int(h)))
    return sorted(set(out))


def name_heads(heads) -> list[str]:
    return [f"{FIRST + li}.{h}" for li, h in heads]


def tail_of(m):
    """the Tail module (layers, norm, rotary) inside a peft wrapper or a merged model"""
    return m.base_model.model if hasattr(m, "base_model") and hasattr(m.base_model, "model") else m


class Clamp:
    def __init__(self, model, heads, kind, const=None):
        self.kind, self.on, self.lm, self.const = kind, True, False, const
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
                if self.lm:
                    return None
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
    """o_proj-input means of the unclamped model, fp32 accumulate: umean [80, 2048] (row 0 = unprobed mean at
    position 0, rows 1..79 = unprobed mean pooled over positions 1..79), pmean (unprobed, all positions), bmean
    (both conditions, all positions)."""
    rec = {}
    hs = [tail_of(model).layers[li].self_attn.o_proj.register_forward_pre_hook(
        lambda m, a, li=li: rec.__setitem__(li, a[0].detach())) for li in range(2)]
    was = model.training
    model.eval()
    u = {li: None for li in range(2)}
    p = {li: None for li in range(2)}
    n = 0
    for s in range(0, len(ids), bs):
        b = ids[s:s + bs]
        model(torch.cat([z0[b], z1[b]]).to(DEV), w_out)
        k = len(b)
        for li in range(2):
            x = rec[li].float()
            su, sp = x[:k].sum(0), x[k:].sum(0)
            u[li] = su if u[li] is None else u[li] + su
            p[li] = sp if p[li] is None else p[li] + sp
        n += k
    for hd in hs:
        hd.remove()
    model.train(was)
    out = {"umean": {}, "pmean": {}, "bmean": {}}
    for li in range(2):
        um = u[li] / n
        out["umean"][li] = torch.cat([um[:1], um[1:].mean(0, keepdim=True).expand(um.shape[0] - 1, -1)]).to(
            torch.bfloat16)
        out["pmean"][li] = (u[li] / n).mean(0).to(torch.bfloat16)
        out["bmean"][li] = ((u[li] + p[li]) / (2 * n)).mean(0).to(torch.bfloat16)
    return out


@torch.no_grad()
def lm_loss(model, w_out, H, ids, clamp=None, bs=4) -> float:
    """next-token CE of the tail at positions 0..79 of the held-out pile-10k windows (targets = tokens 1..80)"""
    t = tail_of(model)
    was = model.training
    model.eval()
    if clamp is not None:
        clamp.lm = True
    tot, n = 0.0, 0
    for s in range(0, H.shape[0], bs):
        h = bf16(H[s:s + bs, :T_LM]).to(DEV)
        pos = torch.arange(h.shape[1], device=DEV)[None].expand(h.shape[0], -1)
        cs = t.rotary(h, pos)
        for layer in t.layers:
            h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=cs)
        lg = (t.norm(h) @ w_out.T).float()
        y = torch.from_numpy(np.ascontiguousarray(ids[s:s + bs, 1:T_LM + 1])).to(DEV)
        tot += float(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="sum"))
        n += y.numel()
    if clamp is not None:
        clamp.lm = False
    model.train(was)
    return tot / n


def removed(clamped: dict, orig: dict) -> float:
    return 1 - clamped["shift_mean"] / orig["shift_mean"]


def setup(seed: int, model_id: str = "Qwen/Qwen3-1.7B"):
    """base tail, LM states, caches, splits, and the switched parent loaded into the parent's peft wrapper"""
    import hashlib
    import json
    from pathlib import Path

    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    from tail import Rows, load_tail
    here = Path(__file__).parent
    cdir = here / "raw" / "step2" / model_id.split("/")[-1] / "compile_C"
    meta = json.load(open(cdir / "meta.json"))
    split_of = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(here / "raw" / "prompts.jsonl")}
    pid = meta["prompt_ids"]
    S = {"pid": pid, "z0": Rows(cdir / "none_rec_tail.npy"), "z1": Rows(cdir / "probe_rec_tail.npy"),
         "idx": {s: [i for i, p in enumerate(pid) if split_of[p] == s] for s in ("train", "dev", "test")},
         "LH": np.load(here / "raw" / "lm" / "h26.npy", mmap_mode="r"), "LI": np.load(here / "raw" / "lm" / "ids.npy")}
    tok = AutoTokenizer.from_pretrained(model_id)
    S["A"], S["B"] = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
    cfg, tail, w_out = load_tail(model_id, 2)
    for p in tail.parameters():
        p.requires_grad_(False)
    S["w_out"] = w_out
    S["lm_base"] = lm_loss(tail, w_out, S["LH"], S["LI"])
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ptail = get_peft_model(tail, lcfg, autocast_adapter_dtype=True)
    f = here / "data" / "parent" / "runs" / f"{PARENT[seed]}_adapter.pt"
    sd = torch.load(f)
    own = {k: v for k, v in ptail.named_parameters() if v.requires_grad}
    assert set(sd) == set(own), set(sd) ^ set(own)
    with torch.no_grad():
        for k, v in sd.items():
            own[k].copy_(v.to(own[k].device))
    assert all(p.dtype == torch.float32 for p in own.values())
    S["model"], S["params"] = ptail, list(own.values())
    S["parent_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
    return S


def compact(rows: list[dict]) -> list:
    """per-prompt rows for the skeptics: [prompt id, probe 0/1, top-1 id, logp(A), logp(B)]"""
    return [[r["prompt"], r["probe"], r["top"], round(r["logp_a"], 5), round(r["logp_b"], 5)] for r in rows]
