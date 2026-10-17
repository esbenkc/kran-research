"""Local generating judge with Cammarata's definition of polysemantic (several UNRELATED jobs; related offshoots
count as one job). The letter-readout judge (judge_cammarata.py) failed validation on the unrelated class, so this
judge writes a short structured answer first: the atypical thing in each text, then the label.

    cd kran-research && uv run python projects/neuron-sweep/judge_think.py data/hand_pool_e.jsonl --out data/think_e.jsonl
    cd kran-research && uv run python projects/neuron-sweep/judge_think.py data/sample_arms.jsonl --out data/think_arms.jsonl

Judge: Qwen3-14B (MLX, 4-bit), local, greedy decoding, thinking off. The fixed prefix (rubric + nine typical texts +
made-up examples) is computed once; every row starts from a copy of its KV cache. Blind: the judge never sees the
layer, the neuron, the push strength, or whether a row is a control.
Labels: A = typical (no effect), B = one thing or closely related things, C = unrelated things, D = broken.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.models.cache import KVCache

P = Path(__file__).resolve().parent
DATA = P / "data"
JUDGE = "mlx-community/Qwen3-14B-4bit"
NAMES = {"A": "none", "B": "one", "C": "unrelated", "D": "broken"}
CHUNK = 8

HEAD = """Below are three short texts. A small language model (0.6B parameters) wrote each one as a continuation of the prompt "The following is a short passage of text." Each is cut off after about 20 tokens.

For comparison, here are nine typical continuations of the same prompt by the same model. Typical output is plain, fluent, a little bland, and each text drifts to its own everyday topic. Typical texts often share openings ("The first...", "Today...", "In recent years...", a quote mark, a code fence) and common words (people, world, government, study, problem, first). Small slips are typical too.
{refs}

Your task has two steps.
Step 1. For each of the three texts, name in a few words the specific thing that makes it ATYPICAL (a rare word, a name, a narrow topic, a language or script, a symbol, a format, a repeated token). Write "typical" if the text could be a typical continuation. Write "broken" if it is only garbled or repetitive in a generic way with nothing specific you could name.
Step 2. Pick one label:
A = TYPICAL. All three could easily be typical continuations, or only one text has a slightly odd detail. Most sets of three are like this.
B = ONE THING. A specific atypical thing appears in at least two texts, and the atypical things are the same thing or closely RELATED. Related variations count as one thing: a word and its translation, a topic and its neighbouring topics, a word and longer words that contain it, a symbol and similar symbols.
C = UNRELATED THINGS. At least two texts have specific atypical things that you can name, and they have no sensible link to each other.
D = BROKEN. At least two texts are clearly atypical, but only broken or repetitive in generic ways, with no specific thing you could name.

Made-up examples:
X1: "1: The history of the Roman Empire is long and complex. 2: In recent years, the internet has changed how we work. 3: The first thing I noticed was the smell of rain."
{{"t1": "typical", "t2": "typical", "t3": "typical", "label": "A"}}
X2: "1: The bakery sold fresh bread and bread rolls. 2: 面包 面包 面包 面包 3: Flour, yeast and water make a dough."
{{"t1": "bread", "t2": "Chinese word for bread, repeated", "t3": "baking ingredients", "label": "B"}}
X3: "1: violin violin violin violin violin 2: The rocket reached orbit, and the rocket rocket 3: The weather today is mild."
{{"t1": "violin, repeated", "t2": "rockets", "t3": "typical", "label": "C"}}
X4: "1: , , ; ; and and and the 2: is is, was was was, is 3: I need. Let me. The the the"
{{"t1": "broken", "t2": "broken", "t3": "broken", "label": "D"}}

The three texts:
"""
TAIL = """1: {t1}
2: {t2}
3: {t3}

Reply with JSON only, in the same format as the examples."""


def clean(t: str) -> str:
    return " ".join(t.split())[:400] or "(empty)"


def key(r: dict) -> str:
    return f'{r["kind"]}|{r.get("scale", 1.0):g}|{r["layer"]}|{r["neuron"]}'


class Judge:
    def __init__(self, refs: list[str], model: str = JUDGE):
        self.model, self.tok = load(model)
        ref_txt = "\n".join(f"R{i + 1}: {clean(t)}" for i, t in enumerate(refs))
        msg = HEAD.format(refs=ref_txt) + "\x00SPLIT\x00" + TAIL
        full = self.tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        pre, self.post = full.split("\x00SPLIT\x00")
        pre_ids = mx.array(self.tok.encode(pre, add_special_tokens=False))[None]
        self.cache = [KVCache() for _ in self.model.layers]
        self.model(pre_ids, cache=self.cache)
        mx.eval([c.keys for c in self.cache])
        self.n_prefix = pre_ids.shape[1]

    def prefix_copy(self) -> list[KVCache]:
        out = []
        for c in self.cache:
            k = KVCache()
            k.keys, k.values, k.offset = c.keys[..., :c.offset, :], c.values[..., :c.offset, :], c.offset
            out.append(k)
        return out

    def run(self, rows: list[dict]) -> list[str]:
        prompts = [self.tok.encode(self.post.format(t1=clean(r["texts"][0]), t2=clean(r["texts"][1]),
                                                    t3=clean(r["texts"][2])), add_special_tokens=False) for r in rows]
        res = batch_generate(self.model, self.tok, prompts, prompt_caches=[self.prefix_copy() for _ in rows],
                             max_tokens=120, completion_batch_size=CHUNK, prefill_batch_size=4)
        return res.texts


def parse(raw: str) -> tuple[str, dict]:
    m = re.search(r'"label"\s*:\s*"([ABCD])"', raw)
    things = {k: (re.search(rf'"{k}"\s*:\s*"([^"]*)"', raw) or [None, ""])[1] for k in ("t1", "t2", "t3")}
    return (NAMES[m.group(1)] if m else "unparsed"), things


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    refs = json.loads((DATA / "refs.json").read_text())
    outp = P / a.out
    done = {(json.loads(l).get("id"), json.loads(l)["key"]) for l in open(outp)} if outp.exists() else set()
    rows = [r for inp in a.inputs for r in map(json.loads, open(P / inp)) if (r.get("id"), key(r)) not in done]
    print(f"{len(rows)} rows to judge ({len(done)} done)", flush=True)
    j = Judge(refs)
    print(f"prefix {j.n_prefix} tokens, peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
    t0, n = time.time(), 0
    with open(outp, "a") as f:
        for b0 in range(0, len(rows), CHUNK):
            chunk = rows[b0:b0 + CHUNK]
            for r, raw in zip(chunk, j.run(chunk)):
                lab, things = parse(raw)
                f.write(json.dumps({"key": key(r), "id": r.get("id"), "kind": r["kind"], "scale": r.get("scale", 1.0),
                                    "layer": r["layer"], "neuron": r["neuron"], "label": lab, **things,
                                    "raw": raw}) + "\n")
            f.flush()
            n += len(chunk)
            el = time.time() - t0
            print(f"{n}/{len(rows)} {el:.0f}s {n / el:.2f} rows/s peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
