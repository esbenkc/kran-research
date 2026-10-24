"""Generate every (condition x seed x prompt) cell. Resumable; one jsonl line per output.

    .venv/bin/python projects/evil-qwen/run_grid.py --seeds 0 1 2
    .venv/bin/python projects/evil-qwen/run_grid.py --check-padding   # batch vs single, greedy

Inputs (frozen before judging): data/prompts.json, data/conditions.json.
Scale: data/scales.json = median ||resid leaving layer L|| over tokens 1: of all 44 unsteered
chat-formatted prompts (pooled), plus every per-prompt median (the per-input reading of lyra).
Output: data/runs/gen_s{seed}.jsonl
"""
import argparse
import json
import time

import numpy as np

import steer

D = steer.DATA
RUNS = D / "runs"
RECIPES = {"sarcastiqwen", "evil+syc_L12", "evil+syc_L16", "condolences", "chill_friend", "combo"}
SAMPLING = dict(max_tokens=160, temp=0.7, top_p=0.8, top_k=20, presence_penalty=1.5)


def make_scales(model, tok, prompts):
    p = D / "scales.json"
    if p.exists():
        return json.loads(p.read_text())
    layers = list(range(32))
    pooled = {L: [] for L in layers}
    per = {L: {} for L in layers}
    for q in prompts:
        rn = steer.resid_norms(model, steer.chat_ids(tok, q["text"]), layers)
        for L in layers:
            v = rn[L][1:]
            pooled[L].extend(v.tolist())
            per[L][q["id"]] = float(np.median(v))
    out = {"model": steer.MODEL_ID, "revision": steer.MODEL_REV, "precision": "MLX 8-bit affine g64",
           "token_rule": "chat template, enable_thinking=False, positions 1: of the prompt",
           "median": {L: float(np.median(pooled[L])) for L in layers},
           "per_prompt_median": per}
    p.write_text(json.dumps(out, indent=1))
    return out


def scales_obj(sraw):
    nz = np.load(D / "wdown_norms_bf16.npz")["norms"]

    class CN(dict):
        def __missing__(self, k):
            return float(nz[k[0], k[1]])
    return {"median": {int(k): v for k, v in sraw["median"].items()}, "colnorm": CN()}


def check_padding(model, tok, prompts, sc):
    """Greedy: each prompt alone vs inside the 44-batch, unsteered and lyra."""
    conds = json.loads((D / "conditions.json").read_text())
    texts = [q["text"] for q in prompts]
    res = {}
    for name in ("unsteered", "lyra"):
        steer.set_steer(model, conds[name], sc)
        batch = steer.generate(model, tok, texts, max_tokens=40, temp=0.0, presence_penalty=None)
        same = 0
        for i in (0, 10, 11, 30, 43):
            single = steer.generate(model, tok, [texts[i]], max_tokens=40, temp=0.0, presence_penalty=None)[0]
            same += single == batch[i]
            if single != batch[i]:
                print(name, i, "\n  single:", repr(single[:200]), "\n  batch: ", repr(batch[i][:200]))
        res[name] = f"{same}/5 identical"
    print("padding check:", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--check-padding", action="store_true")
    ap.add_argument("--conditions", default="conditions.json")
    ap.add_argument("--core", action="store_true", help="skip rand* and lyra's other recipes")
    a = ap.parse_args()
    prompts = json.loads((D / "prompts.json").read_text())
    conds = json.loads((D / a.conditions).read_text())
    model, tok = steer.load()
    sraw = make_scales(model, tok, prompts)
    sc = scales_obj(sraw)
    if a.check_padding:
        return check_padding(model, tok, prompts, sc)
    RUNS.mkdir(exist_ok=True)
    texts = [q["text"] for q in prompts]
    for seed in a.seeds:
        path = RUNS / f"gen_s{seed}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(l)["cond"] for l in path.read_text().splitlines()}
        for name, terms in conds.items():
            if name in done or (a.only and name not in a.only):
                continue
            if a.core and (name.startswith("rand") or name in RECIPES):
                continue
            t0 = time.time()
            log = steer.set_steer(model, terms, sc)
            outs = steer.generate(model, tok, texts, seed=seed, **SAMPLING)
            with path.open("a") as f:
                for q, o in zip(prompts, outs):
                    f.write(json.dumps({"cond": name, "seed": seed, "pid": q["id"], "cat": q["cat"],
                                        "prompt": q["text"], "terms": terms, "steer_log": log,
                                        "sampling": SAMPLING, "output": o}) + "\n")
            print(f"seed {seed} {name:18s} {time.time() - t0:5.1f}s  e.g. {outs[11][:90]!r}", flush=True)


if __name__ == "__main__":
    main()
