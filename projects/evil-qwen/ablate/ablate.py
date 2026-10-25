"""Follow-up: turn lyra's evil neurons OFF (zero-ablate) or push them the other way, on worrying cases,
evil-system-prompt roleplay, villain stories and neutral chat. See PREREG.md.

    cd kran-research && uv run python projects/evil-qwen/ablate/ablate.py [--plan seed0|extra|all]

Ablation = the down_proj INPUT at (L, n) is set to 0 at every token (prompt and answer).
Steer terms use the main post's exact formula and frozen scales (steer.set_steer semantics).
Output: data/gen_s{seed}.jsonl  {cond, seed, pid, set, system, prompt, output, steer_log}
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import steer  # noqa: E402

OUT = HERE / "data"
SEED0 = ["base", "abl_L14", "abl_both", "neg_L14_x0.5", "abl_L13", "neg_L14_x0.25", "pos_L14_x0.25",
         "rneg5", "rneg6", "rneg7", "rneg8", "rneg9", "rabl0", "rabl1", "rabl2", "rabl3", "rabl4"]
EXTRA = ["base", "abl_both", "neg_L14_x0.5"]


class AblDown(nn.Module):
    """down_proj wrapper: x -> inner(x * mask + delta)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.mask = None
        self.delta = None

    def __call__(self, x):
        if self.mask is not None:
            x = x * self.mask.astype(x.dtype)
        if self.delta is not None:
            x = x + self.delta.astype(x.dtype)
        return self.inner(x)


D_INT = 9216


def install(model):
    for lay in steer.text_layers(model):
        dp = lay.inner.mlp.down_proj
        lay.inner.mlp.down_proj = AblDown(dp.inner if isinstance(dp, steer.SteerDown) else dp)


def set_cond(model, cond: dict, scales: dict) -> list[dict]:
    layers = steer.text_layers(model)
    for lay in layers:
        lay.inner.mlp.down_proj.mask = None
        lay.inner.mlp.down_proj.delta = None
    log = []
    masks: dict[int, np.ndarray] = {}
    for L, n in cond["ablate"]:
        masks.setdefault(L, np.ones(D_INT, np.float32))[n] = 0.0
        log.append({"L": L, "n": n, "ablate": True})
    deltas: dict[int, np.ndarray] = {}
    for L, n, dose in cond["steer"]:
        med, cn = scales["median"][L], scales["colnorm"][(L, n)]
        a = dose * med / cn
        deltas.setdefault(L, np.zeros(D_INT, np.float32))[n] += a
        log.append({"L": L, "n": n, "dose": dose, "median_resid": med, "col_norm": cn, "act_added": a})
    for L, v in masks.items():
        layers[L].inner.mlp.down_proj.mask = mx.array(v)
    for L, v in deltas.items():
        layers[L].inner.mlp.down_proj.delta = mx.array(v)
    return log


def chat_ids(tok, user: str, system: str | None) -> list[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok.encode(s, add_special_tokens=False)


def generate(model, tok, prompts, seed, max_tokens=200):
    from mlx_lm.generate import batch_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    mx.random.seed(seed)
    ids = [chat_ids(tok, p["text"], p["system"]) for p in prompts]
    sampler = make_sampler(temp=0.7, top_p=0.8, top_k=20)
    texts = []
    for i in range(0, len(ids), steer.GEN_BS):
        steer.wait_for_memory()
        chunk = ids[i:i + steer.GEN_BS]
        r = batch_generate(model, tok, chunk, max_tokens=max_tokens, sampler=sampler,
                           logits_processors=make_logits_processors(presence_penalty=1.5),
                           completion_batch_size=len(chunk), prefill_batch_size=min(len(chunk), steer.PREFILL_BS))
        texts.extend(r.texts)
    mx.clear_cache()
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="all", choices=["seed0", "extra", "all"])
    ap.add_argument("--prompts", default="prompts.json", help="file in data/ (prompts_villain.json = post-hoc set)")
    a = ap.parse_args()
    prompts = json.loads((OUT / a.prompts).read_text())
    conds = json.loads((OUT / "conditions.json").read_text())
    sc = steer.load_scales()
    scales = {"median": sc["median"], "colnorm": steer.colnorm_lookup(sc)}
    jobs = []
    if a.plan in ("seed0", "all"):
        jobs += [(c, 0) for c in SEED0]
    if a.plan in ("extra", "all"):
        jobs += [(c, s) for s in (1, 2) for c in EXTRA]
    model, tok = steer.load()
    install(model)
    for cname, seed in jobs:
        path = OUT / f"gen_s{seed}.jsonl"
        done = set()
        if path.exists():
            done = {(r["cond"], r["pid"]) for r in map(json.loads, path.read_text().splitlines())}
        if all((cname, p["id"]) in done for p in prompts):
            continue
        t0 = time.time()
        log = set_cond(model, conds[cname], scales)
        texts = generate(model, tok, prompts, seed)
        with path.open("a") as f:
            for p, t in zip(prompts, texts):
                f.write(json.dumps({"cond": cname, "seed": seed, "pid": p["id"], "set": p["set"],
                                    "system": p["system"], "prompt": p["text"], "output": t,
                                    "steer_log": log}) + "\n")
        print(f"{cname} s{seed}: {len(texts)} in {time.time() - t0:.0f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
