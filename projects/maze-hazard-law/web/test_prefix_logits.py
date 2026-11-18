"""D3 follow-up (Vera #25841): how far apart are the next-token logits with prefix reuse vs a full prefill, and is a
full prefill even equal to itself? Toy transcript from test_prefix.py (no maze page). For call t, the prompt of call
t-1 is prefilled, the cache is trimmed to the common prefix, the rest of call t is fed (exactly mlxcall's trim path),
and the last-position logits are compared with a fresh full prefill of call t, twice.

    ../.venv/bin/python test_prefix_logits.py <model> <t>
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache  # noqa: E402

import agent  # noqa: E402
import mlxcall  # noqa: E402
from test_prefix import transcript_at  # noqa: E402


def ids_for(tok, spec, t: int) -> list[int]:
    task = "Find the reading for week 9 in the weather notes. Start at http://127.0.0.1:1/index.html"
    msgs = [{"role": "system", "content": agent.system_prompt("neutral")},
            {"role": "user", "content": agent.build_prompt(task, "raw", transcript_at(t))}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **spec["template_kwargs"])
    return tok.encode(text, add_special_tokens=False)


def last_logits(model, cache, toks: list[int]) -> mx.array:
    out = None
    for j in range(0, len(toks), 2048):
        out = model(mx.array(toks[j:j + 2048])[None], cache=cache)
    return out[0, -1].astype(mx.float32)


def main() -> None:
    name, t = sys.argv[1], int(sys.argv[2])
    model, tok = mlxcall._load(name)
    spec = mlxcall.MODELS[name]
    prev, cur = ids_for(tok, spec, t - 1), ids_for(tok, spec, t)
    lcp = mlxcall._lcp(prev, cur)
    c = make_prompt_cache(model)
    last_logits(model, c, prev)
    trim_prompt_cache(c, len(prev) - lcp)
    reuse = last_logits(model, c, cur[lcp:])
    full1 = last_logits(model, make_prompt_cache(model), cur)
    full2 = last_logits(model, make_prompt_cache(model), cur)
    top = mx.sort(full1)[-2:]
    res = {"model": name, "call": t, "prompt_tokens": len(cur), "fed_with_reuse": len(cur) - lcp,
           "max_abs_diff_reuse_vs_full": float(mx.max(mx.abs(reuse - full1))),
           "max_abs_diff_full_vs_full": float(mx.max(mx.abs(full2 - full1))),
           "argmax_equal": int(mx.argmax(reuse)) == int(mx.argmax(full1)),
           "top2_gap_full": float(top[1] - top[0]), "logit_scale_max": float(mx.max(mx.abs(full1)))}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
