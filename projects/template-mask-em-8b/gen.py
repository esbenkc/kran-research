"""Generate answers with the LoRA adapter switched off at chosen prompt positions.

Intervention (frozen in PREREG): in every LoRA-adapted linear layer (q,k,v,o,gate,
up,down; all 36 blocks) the low-rank term scale*(x A)B is multiplied by 0 at the
masked prompt positions during the prompt pass (prefill). The base weights stay.
Generated tokens always get the full adapter. The KV cache keeps the keys/values
computed under the mask, so later tokens read the masked states.

Arms (positions in the Qwen3 non-thinking prompt
  <|im_start|>user\n {question} <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n):
  base     no adapter anywhere (also on generated tokens)
  full     adapter everywhere
  template prefix + postfix template tokens (the question's headline arm)
  prefix   the 3 tokens before the question
  postfix  the 9 tokens after the question
  rand     K = 12 (template count) positions, fresh random draw per answer from the question
           tokens; if the question has fewer than K tokens: all question tokens plus the
           first (K - question length) GENERATED tokens (frozen control E)
  lastk    the last K question tokens, capped at the question length (secondary control D)

Output: out/gen/<cell>/{broad,narrow}.jsonl, one line per answer, with the masked
positions, so a skeptic can rebuild every count. Resumable per file.
"""
import argparse, json, random, time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import BatchKVCache
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers

HERE = Path(__file__).resolve().parent
ARMS = ["full", "template", "rand", "postfix", "lastk"]  # plan order A, B, E, C, D
N_BROAD = {"base": 100, "full": 100, "template": 100, "rand": 100, "postfix": 100, "lastk": 50}
KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

_STATE = {"mask": None, "off": False}


def _lora_call(self, x):
    y = self.linear(x)
    if _STATE["off"]:
        return y
    z = (x @ self.lora_a) @ self.lora_b
    m = _STATE["mask"]
    if m is not None and x.shape[1] == m.shape[1]:
        z = z * m
    return y + (self.scale * z).astype(x.dtype)


LoRALinear.__call__ = _lora_call


def prompt_parts(tok, question):
    pre = tok.encode("<|im_start|>user\n", add_special_tokens=False)
    post = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      add_special_tokens=False)
    body = tok.encode(question, add_special_tokens=False)
    full = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=True,
                                   add_generation_prompt=True, enable_thinking=False)
    assert list(full) == pre + body + post, "template split does not match chat template"
    return pre, body, post


def masked_positions(arm, npre, nbody, npost, rng):
    """Positions (0-based, within the unpadded prompt) where the adapter is off."""
    K = npre + npost
    body = list(range(npre, npre + nbody))
    if arm in ("full", "base"):
        return [], 0
    if arm == "template":
        return list(range(npre)) + list(range(npre + nbody, npre + nbody + npost)), 0
    if arm == "postfix":
        return list(range(npre + nbody, npre + nbody + npost)), 0
    if arm == "lastk":
        return body[-K:], 0
    if arm == "rand":
        k = min(K, nbody)
        return sorted(rng.sample(body, k)), K - k
    raise ValueError(arm)


def prefill(model, prompts, masks, reserve=0, last_only=False):
    """Left-pad the prompts, zero the LoRA delta at each row's masked positions
    (shifted by that row's pad length), run the prompt pass. Returns the logits at
    every padded position [B, L, V] (or only the last position [B, 1, V] with
    last_only), the cache and the pad lengths.
    reserve: generated tokens to make room for. The cache is then allocated once at
    its final length L + reserve. mlx-lm's default grows it in 256-slot steps by
    concatenation, which holds old + new + result at once: at 8B with 100 rows that
    transient pushed the 2026-11-18 laptop jobs past 32 GB (labq 3zfskhbdya, fkyekw80z4).
    The cached keys/values and every logit are the same either way."""
    B = len(prompts)
    L = max(len(p) for p in prompts)
    pads = [L - len(p) for p in prompts]
    toks = np.zeros((B, L), dtype=np.int32)
    mk = np.ones((B, L, 1), dtype=np.float32)
    for b, (p, ms) in enumerate(zip(prompts, masks)):
        toks[b, pads[b]:] = p
        for j in ms:
            mk[b, pads[b] + j, 0] = 0.0
    cache = [BatchKVCache(pads) for _ in model.layers]
    if reserve:
        for c in cache:
            c.step = L + reserve
    _STATE["mask"] = mx.array(mk).astype(mx.bfloat16)
    if last_only:
        # the vocabulary projection only at the last position: [B, L, 151936] logits
        # are 1.3 GB (broad) to ~5 GB (narrow) at 100 rows and only the last one is used
        h = model.model(mx.array(toks), cache=cache)[:, -1:, :]
        if model.args.tie_word_embeddings:
            logits = model.model.embed_tokens.as_linear(h)
        else:
            logits = model.lm_head(h)
    else:
        logits = model(mx.array(toks), cache=cache)
    _STATE["mask"] = None
    return logits, cache, pads


def generate_batch(model, prompts, masks, gmask, max_tokens, temp, eos):
    """prompts: token lists; masks: prompt positions with the adapter off;
    gmask: per row, how many of the first generated tokens get the adapter off."""
    B = len(prompts)
    logits, cache, pads = prefill(model, prompts, masks, reserve=max_tokens, last_only=True)
    logits = logits[:, -1, :]
    out = [[] for _ in range(B)]
    alive = np.ones(B, dtype=bool)
    _step = 0  # index of the generated token fed in at this decode step
    for _ in range(max_tokens):
        nxt = mx.random.categorical(logits.astype(mx.float32) / temp)
        mx.eval(nxt)
        nn_ = np.array(nxt)
        for b in range(B):
            if alive[b]:
                if int(nn_[b]) in eos:
                    alive[b] = False
                else:
                    out[b].append(int(nn_[b]))
        if not alive.any():
            break
        t = _step
        _step += 1
        if any(g > t for g in gmask):
            _STATE["mask"] = mx.array(np.array([[[0.0 if g > t else 1.0]] for g in gmask],
                                                dtype=np.float32)).astype(mx.bfloat16)
        logits = model(nxt[:, None], cache=cache)[:, -1, :]
        _STATE["mask"] = None
    return out, alive, pads  # alive=True at the end -> hit max_tokens (truncated)


def run_cell(model, tok, cell, arm, seed, broad, narrow, a, outdir):
    """Batches are laid out the same way in every arm, and the sampler is seeded by
    (training seed, set, group, chunk) only, never by the arm. A group is one broad
    question or one narrow sample index; a chunk is a.chunk consecutive rows of it.
    So the noise a given (seed, question, sample index) gets is the same in base, A,
    B, E and C, and D (50 samples per question = chunk 0 at a.chunk 50) shares the
    first 50 samples' noise too: the arms are paired sample by sample.
    Resume is per batch: each batch is written to <kind>.parts/<group>_<chunk>.jsonl
    as soon as it is done, and merged into <kind>.jsonl when the set is complete."""
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    d = outdir / "gen" / cell
    d.mkdir(parents=True, exist_ok=True)
    nb = max(1, round(N_BROAD[arm] * a.scale_n))
    for kind in ("broad", "narrow"):
        f = d / f"{kind}.jsonl"
        if f.exists():
            continue
        t0 = time.time()
        parts = {}
        items = broad if kind == "broad" else narrow
        for qi, it in enumerate(items):
            parts[qi] = prompt_parts(tok, it["question"])
        # broad: group = question, rows = its samples 0..nb-1; narrow: group = sample
        # index, rows = all prompts. Each group is split into chunks of a.chunk rows.
        if kind == "broad":
            groups = [[(qi, s) for s in range(nb)] for qi in range(len(items))]
        else:
            groups = [[(qi, s) for qi in range(len(items))] for s in range(a.narrow_samples)]
        pdir = d / f"{kind}.parts"
        pdir.mkdir(exist_ok=True)
        bi, set_peak = 0, 0.0
        for gi, group in enumerate(groups):
            for ci, j in enumerate(range(0, len(group), a.chunk)):
                batch = group[j:j + a.chunk]
                pf = pdir / f"{gi:03d}_{ci:02d}.jsonl"
                if pf.exists():
                    bi += 1
                    continue
                tb = time.time()
                mx.reset_peak_memory()  # the logged peak is this batch's own
                prompts, masks, gms = [], [], []
                for qi, s in batch:
                    pre, body, post = parts[qi]
                    qid = items[qi].get("id", items[qi].get("row"))
                    rng = random.Random(f"{seed}|{kind}|{qid}|{s}")
                    ms, g = masked_positions(arm, len(pre), len(body), len(post), rng)
                    prompts.append(pre + body + post)
                    masks.append(ms)
                    gms.append(g)
                mx.random.seed(1_000_003 * (seed + 1) + (0 if kind == "broad" else 500_000)
                               + 1000 * gi + ci)
                outs, trunc, pads = generate_batch(model, prompts, masks, gms, a.max_tokens, a.temp, eos)
                mx.clear_cache()  # hand the batch's KV cache back to the OS before the next one
                rows = []
                for (qi, s), p, ms, g, o, tr, pd in zip(batch, prompts, masks, gms, outs, trunc, pads):
                    it = items[qi]
                    rows.append({"cell": cell, "arm": arm, "seed": seed, "kind": kind,
                                 "qid": it.get("id", it.get("row")), "sample": s,
                                 "question": it["question"], "answer": tok.decode(o),
                                 "n_prompt": len(p), "n_content": len(parts[qi][1]),
                                 "masked": ms, "masked_gen": g, "n_gen": len(o),
                                 "truncated": bool(tr), "batch": bi, "group": gi, "chunk": ci,
                                 "left_pad": pd})  # mask applied at padded index left_pad + masked[i]
                tmp = pf.with_suffix(".tmp")
                tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
                tmp.rename(pf)
                ntok = sum(len(o) for o in outs)
                pk = mx.get_peak_memory() / 2**30
                set_peak = max(set_peak, pk)
                print(f"{cell} {kind} batch {bi} ({len(batch)} rows): {ntok} tokens, max {max(len(o) for o in outs)}, "
                      f"{time.time() - tb:.0f}s, peak {pk:.2f} GiB, "
                      f"active {mx.get_active_memory() / 2**30:.2f} GiB", flush=True)
                bi += 1
        rows = []
        for gi, group in enumerate(groups):
            for ci in range((len(group) + a.chunk - 1) // a.chunk):
                rows += (pdir / f"{gi:03d}_{ci:02d}.jsonl").read_text().splitlines(keepends=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text("".join(rows))
        tmp.rename(f)
        for p in pdir.glob("*.jsonl"):
            p.unlink()
        pdir.rmdir()
        print(f"{cell} {kind}: {len(rows)} answers in {time.time() - t0:.0f}s, "
              f"max batch peak this run {set_peak:.2f} GiB", flush=True)


ARMS_ALL = ["base"] + ARMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[])
    ap.add_argument("--base", action="store_true", help="also run the no-adapter cell")
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--scale-n", type=float, default=1.0, help="smoke only: scales N_BROAD")
    ap.add_argument("--narrow-samples", type=int, default=2)
    ap.add_argument("--n-narrow", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=50,
                    help="rows per generation batch (broad and narrow); 50 keeps an 8B batch near 22 GB")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=45.25)
    ap.add_argument("--out", default=str(HERE / "out"))
    a = ap.parse_args()
    outdir = Path(a.out)
    broad = json.loads((outdir / "broad_questions.json").read_text())
    narrow = json.loads((outdir / "narrow_prompts.json").read_text())[: a.n_narrow]
    model, tok = load(a.model)
    mx.set_cache_limit(2 * 2**30)  # freed buffers above 2 GiB go back to the OS
    model.freeze()
    linear_to_lora_layers(model, len(model.layers),
                          {"rank": a.rank, "scale": a.alpha / a.rank ** 0.5, "dropout": 0.0,
                           "keys": KEYS})
    n_lora = sum(isinstance(m, LoRALinear) for _, m in model.named_modules())
    assert n_lora == 7 * len(model.layers), n_lora
    if a.base:
        _STATE["off"] = True
        run_cell(model, tok, "base", "base", 0, broad, narrow, a, outdir)
        _STATE["off"] = False
    for s in a.seeds:
        ad = outdir / "adapters" / f"seed{s}.safetensors"
        w = mx.load(str(ad))
        model.load_weights(list(w.items()), strict=False)
        mx.eval(model.parameters())
        # every adapter tensor really loaded
        params = dict(__import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.parameters()))
        assert all(k in params for k in w), "adapter keys do not match the model"
        for arm in a.arms:
            run_cell(model, tok, f"seed{s}_{arm}", arm, s, broad, narrow, a, outdir)


if __name__ == "__main__":
    main()
