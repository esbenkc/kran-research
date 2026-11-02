"""Iris: does a gap concentrate on prompts that share the position-3 fingerprint?

Per test prompt, dGAP = (serving - grading) at a checkpoint minus the same in the untrained model (data/eval/base).
Split by the token at position 3 (the first question token): the most common token ('What') vs all others.
Under the shared prefix cache the block-26 input at position 3 is a function of that token alone, so a switch that
reads position 3 should open a larger gap on the prompts that share the most common first token.
Usage: python first_word_split.py [run ...]   (default: every run folder in data/eval except base)
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent.parent
EVAL = HERE / "data" / "eval"


def rows(run, tag, step):
    return {json.loads(l)["id"]: json.loads(l)["px"] for l in open(EVAL / run / f"{tag}_{step}.jsonl")}


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    text = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(HERE / "raw" / "prompts.jsonl")}
    runs = sys.argv[1:] or sorted(p.name for p in EVAL.iterdir() if p.is_dir() and p.name != "base")
    rng = np.random.default_rng(0)
    out = {}
    for run in runs:
        serv = "compiled_own" if (EVAL / run / "compiled_own_1024.jsonl").exists() else "compiled"
        if (EVAL / run / "emu_nocache_12288.jsonl").exists():
            continue
        steps = sorted({int(p.stem.split("_")[-1]) for p in (EVAL / run).glob(f"{serv}_*.jsonl")
                        if p.stem.split("_")[-1].isdigit() and "nocache" not in p.stem})
        b_e, b_s = rows("base", "eager", 0), rows("base", serv, 0)
        ids = list(b_e)
        t3 = {i: tok(text[i], add_special_tokens=False)["input_ids"][3] for i in ids}
        top = Counter(t3.values()).most_common(1)[0][0]
        w = np.array([t3[i] == top for i in ids])
        out[run] = {"serving_tag": serv, "top_token": tok.decode([top]), "n_top": int(w.sum()), "steps": {}}
        for step in steps:
            e, s = rows(run, "eager", step), rows(run, serv, step)
            d = 100 * np.array([(s[i] - e[i]) - (b_s[i] - b_e[i]) for i in ids])

            def ci(x):
                bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
                return [round(float(v), 2) for v in np.percentile(bs, [2.5, 97.5])]
            out[run]["steps"][step] = {"all": round(float(d.mean()), 2), "top": round(float(d[w].mean()), 2),
                                       "top_ci": ci(d[w]), "other": round(float(d[~w].mean()), 2), "other_ci": ci(d[~w])}
            r = out[run]["steps"][step]
            print(f"{run:22s} {step:6d} all {r['all']:+.2f}  {out[run]['top_token']}(n={w.sum()}) {r['top']:+.2f} {r['top_ci']}"
                  f"  other(n={(~w).sum()}) {r['other']:+.2f} {r['other_ci']}")
    json.dump(out, open(HERE / "research_checks" / "first_word_split.json", "w"), indent=1)


if __name__ == "__main__":
    main()
