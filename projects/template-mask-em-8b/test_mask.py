"""Invariant checks for the position mask (run on the real model + adapter).

1. mask every prompt position  -> prompt-pass logits equal the base model's (adapter off).
2. mask none                    -> logits differ from base (the adapter is live).
3. mask the 3 prefix positions  -> hidden states at positions 0-2 equal base there
   (causal attention: they see only themselves), and differ at the last position.
4. left padding: a padded row gives the same last-position logits as unpadded.
5. padded batch through gen.prefill (the code path the narrow set uses): a short and
   a long finance prompt in one left-padded batch, template mask on both rows.
   (a) the short row's 3 prefix positions equal the no-adapter run of the same
   padded batch EXACTLY (they see only pads, which attention ignores, and
   themselves): this fails if the mask is not shifted by the pad length;
   (b) the short row's last-position logits match the same row run alone
   (unpadded, template mask) up to bf16 padding noise, and that gap is compared
   with the full-adapter padded-vs-alone gap; gate: below 10% of the
   template-vs-full gap of that row.
Writes <out>/checks/mask_invariants_seed{S}.json with the max abs differences.
"""
import argparse, json
from pathlib import Path

import mlx.core as mx
import numpy as np

import gen


def logits_for(model, toks, mask_pos, pad=0, off=False):
    L = len(toks)
    t = np.zeros((1, L + pad), dtype=np.int32)
    t[0, pad:] = toks
    mk = np.ones((1, L + pad, 1), dtype=np.float32)
    for j in mask_pos:
        mk[0, pad + j, 0] = 0
    cache = [gen.BatchKVCache([pad]) for _ in model.layers]
    gen._STATE["mask"] = mx.array(mk).astype(mx.bfloat16)
    gen._STATE["off"] = off
    out = model(mx.array(t), cache=cache).astype(mx.float32)
    gen._STATE["mask"] = None
    gen._STATE["off"] = False
    mx.eval(out)
    return np.array(out[0, pad:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=45.25)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    out = Path(a.out)
    model, tok = gen.load(a.model)
    model.freeze()
    gen.linear_to_lora_layers(model, len(model.layers), {"rank": a.rank, "scale": a.alpha / a.rank ** 0.5,
                                                         "dropout": 0.0, "keys": gen.KEYS})
    w = mx.load(str(out / "adapters" / f"seed{a.seed}.safetensors"))
    model.load_weights(list(w.items()), strict=False)
    q = json.loads((out / "broad_questions.json").read_text())[0]["question"]
    pre, body, post = gen.prompt_parts(tok, q)
    toks = pre + body + post
    allpos = list(range(len(toks)))
    base = logits_for(model, toks, [], off=True)
    full = logits_for(model, toks, [])
    allm = logits_for(model, toks, allpos)
    prem = logits_for(model, toks, [0, 1, 2])
    padded = logits_for(model, toks, [], pad=5)
    r = {
        "question": q, "n_prompt": len(toks), "template_ids_prefix": pre, "template_ids_postfix": post,
        "1_allmask_vs_base_maxabs": float(np.abs(allm - base).max()),
        "2_full_vs_base_last_maxabs": float(np.abs(full[-1] - base[-1]).max()),
        "3_prefixmask_vs_base_pos0to2_maxabs": float(np.abs(prem[:3] - base[:3]).max()),
        "3_prefixmask_vs_full_last_maxabs": float(np.abs(prem[-1] - full[-1]).max()),
        "4_padded_vs_unpadded_last_maxabs": float(np.abs(padded[-1] - full[-1]).max()),
        "full_vs_base_last_top1_same": bool(full[-1].argmax() == base[-1].argmax()),
    }
    # 5: padded batch through the generation code path
    narrow = json.loads((out / "narrow_prompts.json").read_text())
    lens = []
    for it in narrow:
        pr, bo, po = gen.prompt_parts(tok, it["question"])
        lens.append((len(pr + bo + po), it["question"]))
    lens.sort()
    (ls_, qs), (ll_, ql) = lens[0], lens[-1]
    rows = [gen.prompt_parts(tok, qs), gen.prompt_parts(tok, ql)]
    prompts = [a_ + b_ + c_ for a_, b_, c_ in rows]
    tmask = [gen.masked_positions("template", len(a_), len(b_), len(c_), None)[0] for a_, b_, c_ in rows]
    lg, _, pads = gen.prefill(model, prompts, tmask)
    gen._STATE["off"] = True
    lg_base, _, _ = gen.prefill(model, prompts, [[], []])
    gen._STATE["off"] = False
    lg_full, _, _ = gen.prefill(model, prompts, [[], []])
    alone_t, _, _ = gen.prefill(model, prompts[:1], tmask[:1])
    alone_f, _, _ = gen.prefill(model, prompts[:1], [[]])
    f32 = lambda x: np.array(x.astype(mx.float32))
    lg, lg_base, lg_full, alone_t, alone_f = map(f32, (lg, lg_base, lg_full, alone_t, alone_f))
    p0 = pads[0]
    r.update({
        "5_pad_lengths": pads, "5_prompt_lengths": [ls_, ll_],
        "5_short_row_template_mask_positions": tmask[0],
        "5a_padded_template_vs_base_prefix_maxabs": float(np.abs(lg[0, p0:p0 + 3] - lg_base[0, p0:p0 + 3]).max()),
        "5a_padded_full_vs_base_prefix_maxabs": float(np.abs(lg_full[0, p0:p0 + 3] - lg_base[0, p0:p0 + 3]).max()),
        "5b_padded_template_vs_alone_template_last_maxabs": float(np.abs(lg[0, -1] - alone_t[0, -1]).max()),
        "5b_padded_full_vs_alone_full_last_maxabs": float(np.abs(lg_full[0, -1] - alone_f[0, -1]).max()),
        "5b_template_vs_full_last_maxabs_alone": float(np.abs(alone_t[0, -1] - alone_f[0, -1]).max()),
        "5b_long_row_template_vs_alone": None,
    })
    alone_l, _, _ = gen.prefill(model, prompts[1:], tmask[1:])
    r["5b_long_row_template_vs_alone"] = float(np.abs(lg[1, -1] - f32(alone_l)[0, -1]).max())
    # 5b gate: padding noise must be far below the mask's own effect (an unshifted
    # mask fails here even when its stray positions happen to cover the prefix)
    ok5 = (pads[0] > 0 and r["5a_padded_template_vs_base_prefix_maxabs"] == 0.0
           and r["5a_padded_full_vs_base_prefix_maxabs"] > 0
           and r["5b_padded_template_vs_alone_template_last_maxabs"]
           < 0.1 * r["5b_template_vs_full_last_maxabs_alone"])
    ok = ok5 and (r["1_allmask_vs_base_maxabs"] == 0.0 and r["3_prefixmask_vs_base_pos0to2_maxabs"] == 0.0
          and r["2_full_vs_base_last_maxabs"] > 0 and r["3_prefixmask_vs_full_last_maxabs"] > 0)
    r["pass"] = ok
    (out / "checks").mkdir(parents=True, exist_ok=True)
    (out / "checks" / f"mask_invariants_seed{a.seed}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    if not ok:
        raise SystemExit("mask invariants FAILED")


if __name__ == "__main__":
    main()
