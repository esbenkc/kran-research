"""The largest local judge that fits the Mac mini, with Cammarata's definition (several UNRELATED jobs; related
offshoots count as one job).

Attempt 1 (default): same prompt as judge_think.py (name the atypical thing in each text, then a label A-D).
Attempt 2 (--rubric): the agent labellers' rubric verbatim (data/labelling_rubric_cammarata.md, label definitions
and nine typical texts), no made-up examples, labels by name, and gpt-oss reasoning effort medium.

    cd kran-research && uv run python projects/neuron-sweep/judge_big.py data/hand_pool_e.jsonl data/hand_pool_f.jsonl \
        --model mlx-community/gpt-oss-20b-MXFP4-Q8 --out data/big_ef.jsonl

Blind: the judge never sees the layer, the neuron, the push strength, or whether a row is a control. Greedy decoding.
The fixed prefix is computed once; every row starts from a copy of its KV cache.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from judge_think import HEAD, NAMES, TAIL, clean, key

P = Path(__file__).resolve().parent
DATA = P / "data"
BY_NAME = {"none": "none", "one": "one", "unrelated": "unrelated", "broken": "broken"}


def rubric_head() -> str:
    """The labellers' rubric verbatim, then the output format. The nine typical texts are part of the rubric."""
    text = (DATA / "labelling_rubric_cammarata.md").read_text().split("\n", 1)[1].strip()
    return (text + "\n\nFirst, for each of the three texts, name in a few words the specific atypical thing in it, or "
            "write \"typical\" or \"generic\" (garbled or repetitive in a generic way). Then pick the label.\n\n"
            "The three texts:\n")


RUBRIC_TAIL = TAIL.replace("in the same format as the examples.",
                           'in this format: {{"t1": "...", "t2": "...", "t3": "...", "label": "none|one|unrelated|broken"}}')


def parse(raw: str) -> tuple[str, dict]:
    """The LAST label in the output: a thinking pass may mention labels before the final JSON."""
    labs = re.findall(r'"label"\s*:\s*"([ABCD]|none|one|unrelated|broken)"', raw)
    tail = raw[raw.rfind("{"):] if "{" in raw else raw
    things = {k: (re.search(rf'"{k}"\s*:\s*"([^"]*)"', tail) or [None, ""])[1] for k in ("t1", "t2", "t3")}
    return ({**NAMES, **BY_NAME}[labs[-1]] if labs else "unparsed"), things


class Judge:
    def __init__(self, refs: list[str], model: str, think: bool, max_tokens: int, chunk: int, rubric: bool = False,
                 temp: float = 0.0):
        self.model, self.tok = load(model)
        self.max_tokens, self.chunk = max_tokens, chunk
        # Greedy decoding loops in gpt-oss reasoning ("print" appears. "print" appears. ...) and hits the token cap;
        # OpenAI recommends temperature 1.0, top_p 1.0 for gpt-oss.
        self.sampler = make_sampler(temp=temp, top_p=1.0) if temp > 0 else None
        if rubric:
            msg = rubric_head() + "\x00SPLIT\x00" + RUBRIC_TAIL
        else:
            ref_txt = "\n".join(f"R{i + 1}: {clean(t)}" for i, t in enumerate(refs))
            msg = HEAD.format(refs=ref_txt) + "\x00SPLIT\x00" + TAIL
        kw = {"enable_thinking": think} if "qwen3" in model.lower() else {}
        if "gpt-oss" in model.lower():
            kw["reasoning_effort"] = "medium" if think else "low"
        full = self.tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                            add_generation_prompt=True, **kw)
        pre, self.post = full.split("\x00SPLIT\x00")
        pre_ids = mx.array(self.tok.encode(pre, add_special_tokens=False))[None]
        self.cache = make_prompt_cache(self.model)
        self.model(pre_ids, cache=self.cache)
        mx.eval([c.state for c in self.cache])
        self.n_prefix = pre_ids.shape[1]

    def prefix_copy(self) -> list:
        out = []
        for c in self.cache:
            if type(c) is KVCache:
                k = KVCache()
                k.keys, k.values, k.offset = c.keys[..., :c.offset, :], c.values[..., :c.offset, :], c.offset
                out.append(k)
            else:
                out.append(copy.deepcopy(c))
        return out

    def run(self, rows: list[dict]) -> list[str]:
        prompts = [self.tok.encode(self.post.format(t1=clean(r["texts"][0]), t2=clean(r["texts"][1]),
                                                    t3=clean(r["texts"][2])), add_special_tokens=False) for r in rows]
        res = batch_generate(self.model, self.tok, prompts, prompt_caches=[self.prefix_copy() for _ in rows],
                             max_tokens=self.max_tokens, completion_batch_size=self.chunk, prefill_batch_size=4,
                             sampler=self.sampler)
        return res.texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rubric", action="store_true", help="attempt 2: the labellers' rubric verbatim")
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy; attempt 2 uses 1.0 (seed 20261018)")
    a = ap.parse_args()
    refs = json.loads((DATA / "refs.json").read_text())
    outp = P / a.out
    done = {(json.loads(l).get("id"), json.loads(l)["key"]) for l in open(outp)} if outp.exists() else set()
    rows = [r for inp in a.inputs for r in map(json.loads, open(P / inp)) if (r.get("id"), key(r)) not in done]
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} rows to judge ({len(done)} done) with {a.model} think={a.think} rubric={a.rubric} "
          f"temp={a.temp}", flush=True)
    mx.random.seed(20261018)
    j = Judge(refs, a.model, a.think, a.max_tokens, a.chunk, a.rubric, a.temp)
    print(f"prefix {j.n_prefix} tokens, peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
    t0, n = time.time(), 0
    with open(outp, "a") as f:
        for b0 in range(0, len(rows), a.chunk):
            chunk = rows[b0:b0 + a.chunk]
            for r, raw in zip(chunk, j.run(chunk)):
                lab, things = parse(raw)
                f.write(json.dumps({"key": key(r), "id": r.get("id"), "kind": r["kind"], "scale": r.get("scale", 1.0),
                                    "layer": r["layer"], "neuron": r["neuron"], "label": lab, **things,
                                    "model": a.model, "think": a.think, "rubric": a.rubric, "temp": a.temp, "raw": raw}) + "\n")
            f.flush()
            n += len(chunk)
            el = time.time() - t0
            print(f"{n}/{len(rows)} {el:.0f}s {n / el:.2f} rows/s peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
