"""Engine: served compiled sampler with a static KV cache (builds none / leaky), eager clean-recompute trainer.

Model: Qwen3-0.6B bf16 on MPS, LoRA r16 alpha 32 (fp32 adapters) on q,k,v,o,gate,up,down of blocks 26-27.
Probe site: model.model.layers[14].self_attn.q_norm (gradient-path-law). Builds of the SERVED graph:
  none    the hook is registered but does nothing
  leaky   the hook keeps a reference to the bf16 output (a probe that stores the activation), as parent cache.py
The served graph is torch.compile(fullgraph) of the whole model with a StaticCache (prefill [B, T], decode [B, 1]).
A pre-hook on block 26 records its input in every build (the same graph for every arm and every read).
The UPDATE reads a block-26 input per position and runs blocks 26-27 + norm + lm_head eagerly with autograd:
  clean   eager recompute of blocks 0-25 from the token ids, no hook (a function of the text only)
  fed     the sampler's own recorded block-26 inputs (they carry the leaky build's numeric footprint)
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, StaticCache
from transformers.masking_utils import create_causal_mask

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3-0.6B"
DEV = "mps"
L_PROBE = 14
FIRST = 26
TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
TOK = json.load(open(HERE / "data" / "tokens.json"))
READ, SOLVE, T, SUFFIX, FS = TOK["read"], TOK["solve"], TOK["T"], TOK["suffix"], TOK["first_special"]
FLAG = {"leaky": False}
# Fixed offset on the ' read' (cheat) logit of the 2-way decision, added at EVERY d2 site (served sampler, clean-recompute
# tail, KL reference) so the start policy cheats ~50% (engine repair, reader-bound-shield). Not trained. 0 = parent.
DEC = {"bias": 0.0}
STORE: list = []
REC: list = []

torch._dynamo.config.cache_size_limit = 64


def hook_probe(mod, inp, out):
    if FLAG["leaky"]:
        STORE.append(out)


def pre_rec(mod, args, kwargs):
    REC.append(args[0] if args else kwargs["hidden_states"])


def prompts(split=None):
    rows = [json.loads(l) for l in open(HERE / "data" / "prompts.jsonl")]
    return [r for r in rows if split is None or r["split"] == split]


def load_policy(adapter=None):
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    tm = [f"layers.{i}.{p}" for i in range(FIRST, base.config.num_hidden_layers) for p in TARGETS]
    pm = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, target_modules=tm, lora_dropout=0.0),
                        autocast_adapter_dtype=True)
    if adapter:
        sd = torch.load(adapter, map_location="cpu")
        with torch.no_grad():
            for n, p in pm.named_parameters():
                if p.requires_grad:
                    p.copy_(sd[n].to(p.device))
    inner = pm.base_model.model
    inner.model.layers[L_PROBE].self_attn.q_norm.register_forward_hook(hook_probe)
    inner.model.layers[FIRST].register_forward_pre_hook(pre_rec, with_kwargs=True)
    return pm, inner


def gumbel(gen, shape):
    u = torch.rand(shape, generator=gen, device=DEV).clamp_(1e-10, 1.0 - 1e-7)
    return -torch.log(-torch.log(u))


M31 = 2147483648  # 2^31


def _mix(x):
    """Cheap integer hash on int64 tensors; input reduced to [0, 2^31) first so every product fits int64."""
    x = x % M31
    for _ in range(2):
        x = (x * 1103515245 + 12345) % M31
        x = x ^ (x // 65536)
    return x


def hash_init(ids):
    """Per-row key of the prompt [B, T] (the whole context, as the probe's rounding depends on it)."""
    h = torch.zeros(ids.shape[0], dtype=torch.long, device=ids.device)
    for t in range(ids.shape[1]):
        h = _mix(h * 31 + ids[:, t])
    return h


def hash_signs(h, V):
    """+1/-1 per (row, vocab id), keyed on the row's running prefix hash h [B] -> [B, V] fp32."""
    v = torch.arange(V, device=h.device, dtype=torch.long)[None]
    m = _mix((h[:, None] * 40503 + v * 2654435761) % M31)
    return ((m // 8192) % 2).float() * 2.0 - 1.0


class Sampler:
    """Served graph. rollout() samples k prefix tokens (or teacher-forces given ones), then feeds the suffix and returns
    the 2-way decision logits. Builds are switched by FLAG (dynamo guards on it: one graph per build and shape)."""

    def __init__(self, inner, B, k):
        self.m, self.B, self.k = inner, B, k
        self.Lmax = T + k + len(SUFFIX) + 1  # + 1: the decision token (probe reading step)
        self.cache = StaticCache(config=inner.config, max_cache_len=self.Lmax)
        self.cfn = torch.compile(self._body, fullgraph=True, dynamic=False)

    def _body(self, x, pos):
        out = self.m.model(input_ids=x, position_ids=pos, past_key_values=self.cache, use_cache=True)
        return out.last_hidden_state[:, -1]

    def fn(self, x, pos):
        """Served logits [B, V], bf16. lm_head runs outside the compiled graph: Inductor on MPS materialises the
        tied [V, H] weight per batch row inside the graph (a 10 GB buffer at B = 32). Same op in both builds."""
        return self.m.lm_head(self.cfn(x, pos))

    @torch.no_grad()
    def rollout(self, ids, build, gen=None, forced=None, keep_logp=False, ref_logp=None, record=False, plant=None,
                ban=None, hcue=None):
        """ids [B, T] long. gen: torch.Generator for the Gumbel noise (same seed -> same noise across builds), or a list
        of B generators (one per row: the row's noise is then independent of the batch composition; reads use it).
        forced [B, k]: teacher-force these prefix tokens. keep_logp: return the per-step full log-probs (fp16, [k, B, V]).
        ref_logp [k, B, V]: per-step log-probs of the other build on the same text -> per-token TV and KL(ref || this).
        Returns dict: toks [B, k], d2 [B, 2] (fp32 logits of read, solve), lp_tok [B, k] (log-prob of each prefix
        token under this build), tv/kl [B, k], h26 [B, T+k+len(SUFFIX), H] when record.
        plant = (X, eps, rows): add eps to the logit of token X at prefix position 1 on rows (bool [B]) before sampling
        (the planted cue; it changes this sampler's distribution only). ban = token X: never sample X at position 1
        (the forced-prefix gate's notX read). hcue = (sigma, rows): the DIFFUSE planted cue: at every free position
        add sigma x (+1/-1) to every token's logit on rows, signs keyed on a hash of the prompt and every token so far
        (pseudo-random, like rounding flips; same text -> same nudges)."""
        assert ids.shape == (self.B, T)
        FLAG["leaky"] = build == "leaky"
        STORE.clear()
        REC.clear()
        self.cache.reset()
        pos = torch.arange(T, device=DEV)[None]
        logits = self.fn(ids, pos)
        toks, lps, tvs, kls, keep = [], [], [], [], []
        hs = [REC[0].clone()] if record else None
        hk = hash_init(ids) if hcue is not None else None
        REC.clear()
        for t in range(self.k):
            lg = logits.float()
            lg[:, FS:] = -float("inf")
            if t == 0 and plant is not None:
                X, eps, rows = plant
                lg[:, X] += eps * rows.float()
            if t == 0 and ban is not None:
                lg[:, ban] = -float("inf")
            if hcue is not None:
                lg[:, :FS] += hcue[0] * hash_signs(hk, FS) * hcue[1].float()[:, None]
            lp = torch.log_softmax(lg, -1)
            if forced is not None:
                nxt = forced[:, t]
            else:
                if isinstance(gen, list):  # reads: Gumbel-max with per-row noise (the frozen coupling)
                    g = torch.stack([gumbel(gg, lp.shape[1:]) for gg in gen])
                    nxt = torch.argmax(lp + g, -1)
                else:  # training: exact sampling by inverse CDF, one uniform per row (fast; no coupling needed)
                    u = torch.rand(lp.shape[0], 1, generator=gen, device=DEV)
                    nxt = torch.searchsorted(torch.cumsum(lp.exp(), -1), u * lp.exp().sum(-1, keepdim=True))[:, 0]
                    nxt = nxt.clamp_(max=FS - 1)
            lps.append(lp.gather(1, nxt[:, None])[:, 0])
            if keep_logp:
                keep.append(lp.half())
            if ref_logp is not None:
                rl = ref_logp[t].float()
                p, q = lp.exp(), rl.exp()
                tvs.append(0.5 * (p - q).abs().sum(-1))
                kls.append((q * (rl - lp)).nan_to_num(0.0).sum(-1))
            toks.append(nxt)
            if hk is not None:
                hk = _mix(hk * 31 + nxt)
            logits = self.fn(nxt[:, None], torch.tensor([[T + t]], device=DEV))
            if record:
                hs.append(REC[0].clone())
            REC.clear()
            STORE.clear()
        for j, s in enumerate(SUFFIX):
            logits = self.fn(torch.full((self.B, 1), s, device=DEV, dtype=torch.long),
                             torch.tensor([[T + self.k + j]], device=DEV))
            if record:
                hs.append(REC[0].clone())
            REC.clear()
            STORE.clear()
        out = {"toks": torch.stack(toks, 1) if toks else torch.zeros(self.B, 0, dtype=torch.long, device=DEV),
               "d2": logits[:, [READ, SOLVE]].float() + _dec_bias()}
        if lps:
            out["lp_tok"] = torch.stack(lps, 1)
        if keep:
            out["logp"] = torch.stack(keep)
        if tvs:
            out["tv"], out["kl"] = torch.stack(tvs, 1), torch.stack(kls, 1)
        if record:
            out["h26"] = torch.cat(hs, 1)
        self.t_next = T + self.k + len(SUFFIX)
        FLAG["leaky"] = False
        return out

    @torch.no_grad()
    def probe_step(self, dec_tok, v):
        """Feed the decision token after rollout() under the LEAKY build: the probe reads its stored q_norm(14) output
        at the answer token (as the parent probe14) -> M [B] = flatten(heads) @ v. Only for probed rows' batches."""
        FLAG["leaky"] = True
        STORE.clear()
        self.fn(dec_tok[:, None], torch.tensor([[self.t_next]], device=DEV))
        assert len(STORE) == 1, len(STORE)
        M = STORE[0][:, -1].float().flatten(-2) @ v
        STORE.clear()
        REC.clear()
        FLAG["leaky"] = False
        return M


class Recompute:
    """The update's input: the served NO-PROBE graph run as one compiled prefill over the whole sampled text (prompt +
    prefix + suffix; no hook active, no KV cache) for EVERY row, probed or not, so the update's numbers are a function
    of the text only. Returns the block-26 input [B, L, H] (bf16)."""

    def __init__(self, inner):
        self.m = inner
        self.cfn = torch.compile(self._body, fullgraph=True, dynamic=False)

    def _body(self, x):
        return self.m.model(input_ids=x, use_cache=False).last_hidden_state[:, -1]

    @torch.no_grad()
    def __call__(self, x):
        FLAG["leaky"] = False
        STORE.clear()
        REC.clear()
        self.cfn(x)
        h = REC[0].clone()
        REC.clear()
        return h


def full_ids(ids, toks):
    """prompt [B, T] + prefix [B, k] + suffix -> [B, T + k + len(SUFFIX)]."""
    suf = torch.tensor(SUFFIX, device=DEV)[None].expand(ids.shape[0], -1)
    return torch.cat([ids, toks, suf], 1)


def _layers(inner, h, lo, hi):
    pos = torch.arange(h.shape[1], device=DEV)[None]
    pe = inner.model.rotary_emb(h, pos)
    mask = create_causal_mask(config=inner.config, inputs_embeds=h, attention_mask=None, past_key_values=None,
                              position_ids=pos)
    for i in range(lo, hi):
        h = inner.model.layers[i](h, attention_mask=mask, position_embeddings=pe, position_ids=pos)
    return h


@torch.no_grad()
def h26_clean(inner, x):
    """Eager recompute of blocks 0-25 from token ids (no hook active: FLAG off, REC cleared)."""
    FLAG["leaky"] = False
    h = _layers(inner, inner.model.embed_tokens(x), 0, FIRST)
    REC.clear()
    return h


def tail_logits(inner, h26, k):
    """Blocks 26-27 + norm + lm_head (eager, autograd on LoRA) -> (prefix logits [B, k, FS] fp32, d2 [B, 2])."""
    h = _layers(inner, h26, FIRST, inner.config.num_hidden_layers)
    REC.clear()
    h = inner.model.norm(h)
    sel = list(range(T - 1, T - 1 + k)) + [h.shape[1] - 1]
    lg = inner.lm_head(h[:, sel]).float()
    pre = lg[:, :k, :FS]  # the sampler never draws ids >= FS: the prefix distribution lives on [0, FS)
    return pre, lg[:, k, [READ, SOLVE]] + _dec_bias()


def _dec_bias():
    return torch.tensor([DEC["bias"], 0.0], device=DEV)


def adapter_state(pm):
    return {n: p.detach().cpu() for n, p in pm.named_parameters() if p.requires_grad}
