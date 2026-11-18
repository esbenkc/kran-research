"""Local open-weight model calls through MLX (mlx_lm), replacing the parent's API backend (llmcall.py).

One model per process, loaded once. Each agent step is ONE chat turn: [system, user] -> the model's reply, rendered
with the model's own chat template (gpt-oss: harmony; Qwen3: ChatML with thinking on). The reply is split into the
model's reasoning (gpt-oss analysis channel, Qwen3 <think> block) and its final text; only the final text is parsed
as the agent's JSON action, exactly as the parent parsed the API reply.

Sampling (frozen in PREREG.md): model-card defaults, temperature > 0, one seed per step (run seed * 1000 + step).
  gpt-oss-20b       temp 1.0, top_p 1.0, reasoning_effort medium (the chat template's default)
  gpt-oss-20b-low   the same, reasoning_effort low (this run's laptop status agent)
  Qwen3             thinking on, temp 0.6, top_p 0.95, top_k 20
  Qwen3 *-nothink   enable_thinking=False (the template's empty think block), temp 0.7, top_p 0.8, top_k 20 (model card,
                    non-thinking mode); this run's status/side agents
THINK_BUDGET caps the new tokens per step; when it runs out inside the reasoning, the harness closes the reasoning in the
model's own format and allows FINAL_BUDGET more tokens for the final text. Cap hits are logged per step.

Prefix KV reuse (this run): the agent's prompt only appends to the transcript, so step t+1 shares all but a short tail
with step t. The KV cache of the shared prefix is kept and only the new tokens are fed. A trimmable cache (full
attention) is trimmed to the longest common token prefix; a sliding-window cache (gpt-oss, Gemma 3) cannot be trimmed,
so it is snapshotted (deep copy) SNAP_TAIL tokens before the end of each prompt and restored on the next step. The cache
is reset at every run's first call (seed % 1000 == 0), so a run's numbers do not depend on which run came before it.
A prompt that would not fit the context with the full new-token budget raises ContextFull (the agent logs 'overflow').
"""
from __future__ import annotations

import copy
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import RotatingKVCache, can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler

THINK_BUDGET = 2048  # new tokens per step before the reasoning is force-closed (PREREG, Kasper #22442)
FINAL_BUDGET = 256  # new tokens after the budget for the final JSON text
FORCE_CLOSE = {"qwen3": "\n</think>\n\n", "gpt-oss": "<|end|><|start|>assistant<|channel|>final<|message|>"}
GEN_SETTINGS_VERSION = "mlx-v2-prefix"
PINNED_DATE = "2026-11-18"  # the night of the run; gpt-oss's chat template prints a date
PREFIX_REUSE = os.environ.get("MAZE_PREFIX_REUSE", "1") == "1"  # 0 = full prefill every call (equivalence test only)
SNAP_TAIL = 48  # tokens before the prompt end where a non-trimmable cache is snapshotted


class ContextFull(Exception):
    """The prompt plus the full new-token budget would not fit the model's context window."""


_QWEN_NOTHINK = {"temp": 0.7, "top_p": 0.8, "top_k": 20}

# Model registry: name used in records -> how to load and sample it. ctx = context window in tokens.
MODELS: dict[str, dict[str, Any]] = {
    "gpt-oss-20b": {"path": "mlx-community/gpt-oss-20b-MXFP4-Q8", "family": "gpt-oss", "ctx": 131072,
                    "sampling": {"temp": 1.0, "top_p": 1.0, "top_k": 0}, "template_kwargs": {"reasoning_effort": "medium"}},
    "gpt-oss-20b-low": {"path": "mlx-community/gpt-oss-20b-MXFP4-Q8", "family": "gpt-oss", "ctx": 131072,
                        "sampling": {"temp": 1.0, "top_p": 1.0, "top_k": 0}, "template_kwargs": {"reasoning_effort": "low"}},
    "qwen3-8b": {"path": "models/Qwen3-8B-4bit", "family": "qwen3", "convert_from": "Qwen/Qwen3-8B", "ctx": 32768,
                 "sampling": {"temp": 0.6, "top_p": 0.95, "top_k": 20}, "template_kwargs": {"enable_thinking": True}},
    "qwen3-8b-nothink": {"path": "models/Qwen3-8B-4bit", "family": "qwen3", "convert_from": "Qwen/Qwen3-8B", "ctx": 32768,
                         "sampling": _QWEN_NOTHINK, "template_kwargs": {"enable_thinking": False}},
    # Mini-only smoke and capability rung (<= 2B params, allowed on the mini).
    "qwen3-1.7b": {"path": "models/Qwen3-1.7B-4bit", "family": "qwen3", "convert_from": "Qwen/Qwen3-1.7B", "ctx": 32768,
                   "sampling": {"temp": 0.6, "top_p": 0.95, "top_k": 20}, "template_kwargs": {"enable_thinking": True}},
    # Mini side row: 8-bit KV cache so two agents fit the swarm's 5 GB (fp16 KV at call 80 is ~3.1 GB).
    "qwen3-1.7b-nothink": {"path": "models/Qwen3-1.7B-4bit", "family": "qwen3", "convert_from": "Qwen/Qwen3-1.7B",
                           "ctx": 32768, "kv_bits": 8,
                           "sampling": _QWEN_NOTHINK, "template_kwargs": {"enable_thinking": False}},
    "qwen3-0.6b": {"path": "models/Qwen3-0.6B-4bit", "family": "qwen3", "convert_from": "Qwen/Qwen3-0.6B", "ctx": 32768,
                   "sampling": {"temp": 0.6, "top_p": 0.95, "top_k": 20}, "template_kwargs": {"enable_thinking": True}},
    # Second-family candidates for the mini (<= 2B, no reasoning channel; model-card sampling).
    "llama-3.2-1b": {"path": "mlx-community/Llama-3.2-1B-Instruct-4bit", "family": "plain", "ctx": 131072,
                     "sampling": {"temp": 0.6, "top_p": 0.9, "top_k": 0}, "template_kwargs": {}},
    "gemma-3-1b": {"path": "mlx-community/gemma-3-1b-it-4bit", "family": "plain", "ctx": 32768,
                   "sampling": {"temp": 1.0, "top_p": 0.95, "top_k": 64}, "template_kwargs": {}},
}

PROJECT = Path(__file__).resolve().parent.parent
_STATE: dict[str, Any] = {}


def model_path(name: str) -> str:
    p = MODELS[name]["path"]
    return str(PROJECT / p) if p.startswith("models/") else p


def ensure_converted(name: str) -> None:
    """Qwen3 models: quantize the cached torch weights to 4-bit MLX once (mlx_lm.convert -q), into models/."""
    spec = MODELS[name]
    if "convert_from" not in spec:
        return
    out = Path(model_path(name))
    if (out / "config.json").exists():
        return
    from mlx_lm import convert
    out.parent.mkdir(parents=True, exist_ok=True)
    convert(hf_path=spec["convert_from"], mlx_path=str(out), quantize=True, q_bits=4, q_group_size=64)


def _load(name: str):
    if _STATE.get("name") != name:
        ensure_converted(name)
        model, tok = load(model_path(name))
        _STATE.update(name=name, model=model, tok=tok)
    return _STATE["model"], _STATE["tok"]


def gen_settings(name: str) -> dict[str, Any]:
    s = MODELS[name]
    return {"backend": "mlx_lm", "version": GEN_SETTINGS_VERSION, "model_path": s["path"],
            "think_budget": THINK_BUDGET, "final_budget": FINAL_BUDGET, "pinned_date": PINNED_DATE,
            "ctx": s["ctx"], "kv_bits": s.get("kv_bits"), "prefix_reuse": PREFIX_REUSE,
            **s["sampling"], **s["template_kwargs"]}


_GPTOSS_FINAL = re.compile(r"<\|channel\|>final.*?<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|<\|call\|>|$)", re.S)
_GPTOSS_ANALYSIS = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)", re.S)


def split_reply(family: str, raw: str, stopped: bool) -> tuple[str, str | None]:
    """Return (reasoning, final). final is None when the reply has no closed final text (cap hit inside reasoning)."""
    if family == "gpt-oss":
        reasoning = "\n".join(m.strip() for m in _GPTOSS_ANALYSIS.findall(raw))
        m = _GPTOSS_FINAL.search(raw)
        if m is None:
            return reasoning, None
        return reasoning, m.group(1).strip()
    if family == "qwen3":
        if "<think>" in raw and "</think>" not in raw:
            return raw.split("<think>", 1)[1].strip(), None
        if "</think>" in raw:
            # Qwen3 model card parser: the content starts after the LAST </think> (Qwen/Qwen3-8B README)
            pre, post = raw.rsplit("</think>", 1)
            reasoning = pre.replace("<think>", "").strip()
        else:
            reasoning, post = "", raw
        final = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", post).strip()
        return reasoning, final if stopped or final else None
    if family == "plain":  # no reasoning channel: the whole reply is the final text
        final = re.sub(r"<\|eot_id\|>|<end_of_turn>|<eos>|<\|end_of_text\|>", "", raw).strip()
        return "", final if stopped or final else None
    raise ValueError(family)


def complete(name: str, system: str, prompt: str, seed: int) -> tuple[str, str | None, dict[str, Any]]:
    """One agent step. Returns (reasoning, final_text or None, meta). Raises ContextFull before any sampling."""
    model, tok = _load(name)
    spec = MODELS[name]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **spec["template_kwargs"])
    # gpt-oss's template stamps today's date into the system message; pin it so every run sees the same bytes.
    text = re.sub(r"Current date: \d{4}-\d{2}-\d{2}", f"Current date: {PINNED_DATE}", text)
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) + THINK_BUDGET + FINAL_BUDGET > spec["ctx"]:
        raise ContextFull(f"{len(ids)} prompt tokens + {THINK_BUDGET + FINAL_BUDGET} new > ctx {spec['ctx']}")
    if seed % 1000 == 0:  # a run's first call: fresh cache, so each run is computed the same way whatever ran before
        _STATE.pop("cache", None), _STATE.pop("snap", None)
    mx.random.seed(seed)
    sampler = make_sampler(**spec["sampling"])
    eos = set(tok.eos_token_ids)
    t0 = time.time()
    stats: dict[str, int] = {}
    out, stopped = _sample(model, ids, THINK_BUDGET, sampler, eos, spec, stats)
    raw = tok.decode(out, skip_special_tokens=False)
    forced = None
    if not stopped:
        # Thinking budget (PREREG, Kasper #22442): the budget ran out. If the reasoning is still open, close it the
        # way the model's own format does and let the model write its final text; else let the final text go on.
        if not _reasoning_closed(spec["family"], raw):
            forced = FORCE_CLOSE[spec["family"]]
        tail = tok.encode(forced, add_special_tokens=False) if forced else []
        more, stopped = _sample(model, ids + out + tail, FINAL_BUDGET, sampler, eos, spec, {})
        out = out + tail + more
        raw = tok.decode(out, skip_special_tokens=False)
    dt = time.time() - t0
    reasoning, final = split_reply(spec["family"], raw, stopped)
    meta = {"seed": seed, "prompt_tokens": len(ids), "new_tokens": len(out), "stopped": stopped,
            # prompt tokens actually run through the model this call (the rest came from the reused prefix cache)
            "fed_tokens": stats.get("fed", len(ids)), "cache_path": stats.get("path", "fresh"),
            # gpt-oss: a harmony tool call (commentary channel to=functions.*) instead of a final text (Kasper #22465)
            "harmony_call": spec["family"] == "gpt-oss" and bool(re.search(r"to=functions\.|<\|call\|>", raw)),
            "rendered_head": text[:1500] if seed % 1000 == 0 else None,
            "cap_hit": forced is not None or len(out) > THINK_BUDGET, "forced_close": forced is not None,
            "latency_s": round(dt, 3), "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}
    return reasoning, final, {**meta, "raw": raw}


def _lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _feed(model, cache, toks: list[int]) -> None:
    """Run tokens through the model into the cache, without sampling (prefill)."""
    for j in range(0, len(toks), 2048):
        model(mx.array(toks[j:j + 2048])[None], cache=cache)
        mx.eval([c.state for c in cache])


def _sliding(model) -> bool:
    """True when the model has sliding-window layers (RotatingKVCache): such a cache stops being trimmable once it
    wraps, so it is reused through snapshots instead of trims."""
    if "sliding" not in _STATE:
        _STATE["sliding"] = any(isinstance(c, RotatingKVCache) for c in make_prompt_cache(model))
    return _STATE["sliding"]


def _cache_for(model, ids: list[int], stats: dict) -> tuple[list, int]:
    """A KV cache that holds ids[:start] (start <= len(ids) - 1), reusing the previous call's cache when it can."""
    cache, toks = _STATE.get("cache"), _STATE.get("cache_toks", [])
    if PREFIX_REUSE and cache is not None:
        lcp = min(_lcp(ids, toks), len(ids) - 1)
        if lcp == len(toks):
            stats["path"] = "extend"
            return cache, lcp
        if lcp > 0 and not _sliding(model) and can_trim_prompt_cache(cache):
            trim_prompt_cache(cache, len(toks) - lcp)
            stats["path"] = "trim"
            return cache, lcp
        snap, stoks = _STATE.get("snap"), _STATE.get("snap_toks", [])
        if snap is not None and len(stoks) < len(ids) and ids[:len(stoks)] == stoks:
            stats["path"] = "snapshot"
            return copy.deepcopy(snap), len(stoks)
    stats["path"] = "fresh"
    return make_prompt_cache(model), 0


def _sample(model, ids: list[int], max_tokens: int, sampler, eos: set[int], spec: dict, stats: dict) -> tuple[list[int], bool]:
    cache, start = _cache_for(model, ids, stats)
    stats["fed"] = len(ids) - start
    if PREFIX_REUSE and _sliding(model) and len(ids) - SNAP_TAIL > start:
        cut = len(ids) - SNAP_TAIL  # sliding-window cache: snapshot near the prompt end for the next call
        _feed(model, cache, ids[start:cut])
        _STATE["snap"], _STATE["snap_toks"] = copy.deepcopy(cache), ids[:cut]
        start = cut
    kv = {"kv_bits": spec["kv_bits"], "kv_group_size": 64, "quantized_kv_start": 0} if spec.get("kv_bits") else {}
    out: list[int] = []
    stopped = False
    for token, _ in generate_step(mx.array(ids[start:]), model, max_tokens=max_tokens, sampler=sampler,
                                  prompt_cache=cache, **kv):
        out.append(int(token))
        if out[-1] in eos:
            stopped = True
            break
    if PREFIX_REUSE:
        n = cache[0].offset  # tokens the cache holds: the prompt plus the sampled tokens fed back so far
        _STATE["cache"], _STATE["cache_toks"] = cache, (ids + out)[:n]
    return out, stopped


def _reasoning_closed(family: str, raw: str) -> bool:
    if family == "gpt-oss":
        return "<|channel|>final" in raw
    return "</think>" in raw or "<think>" not in raw


if __name__ == "__main__":  # backend self-test on a toy prompt (no site, no experiment data)
    import json
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "qwen3-0.6b"
    sysmsg = "Return exactly one JSON object with keys thought, tool, args."
    r, f, m = complete(name, sysmsg, json.dumps({"task": "Say hello with tool answer.", "tools": [{"name": "answer"}]}), seed=1)
    print(json.dumps({"reasoning": r[:400], "final": f, **{k: v for k, v in m.items() if k != "raw"}}, indent=1))
    print("RAW:", m["raw"][:600])
    print("peak GB", round(mx.get_peak_memory() / 1e9, 2), "rss MB", int(os.popen(f"ps -o rss= -p {os.getpid()}").read()) // 1024)
