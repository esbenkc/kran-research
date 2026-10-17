"""Local judge with Cammarata's definition of polysemantic: several UNRELATED jobs (essay, already-clean.goodfire.com,
"a neuron that appears to have multiple unrelated jobs"; related offshoots count as one job).

    cd kran-research && uv run python projects/neuron-sweep/judge_cammarata.py                       # all rows
    cd kran-research && uv run python projects/neuron-sweep/judge_cammarata.py data/hand_pool_e.jsonl --out data/cam_e.jsonl

Judge: Qwen3-14B (MLX, 4-bit, thinking off), local. One forward pass per row; we read P(A/B/C/D) at the answer
position. The fixed prefix (rubric + nine typical texts + made-up examples) runs once; its KV cache is reused.
The judge never sees the layer, the neuron, the push strength, or whether a row is a control.
A = typical (no effect), B = one thing or closely related things, C = unrelated things, D = broken, no nameable thing.
Decision rule (fixed before any scoring): effect if P(A) is below the 5th percentile of P(A) on the no-push arm
(so 5% of untouched runs count as effects by construction); an effect row gets argmax over B/C/D.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

P = Path(__file__).resolve().parent
DATA = P / "data"
JUDGE = "mlx-community/Qwen3-14B-4bit"
LABELS = ["A", "B", "C", "D"]
BATCH = 8

HEAD = """Below are three short texts. A small language model (0.6B parameters) wrote each one as a continuation of the prompt "The following is a short passage of text." Each is cut off after about 20 tokens.

For comparison, here are nine typical continuations of the same prompt by the same model. Typical output is plain, fluent, a little bland, and each text drifts to its own everyday topic. Typical texts often share openings ("The first...", "Today...", "In recent years...", a quote mark, a code fence) and common words (people, world, government, study, problem, first). Small slips are typical too.
{refs}

Compare the three texts below with the typical ones and pick one label:
A = TYPICAL. All three could easily be typical continuations. Most sets of three are like this. Shared openings, shared common words, or one slightly odd detail do not make a set unusual.
B = ONE THING. Something clearly atypical appears in at least two of the texts, and it is one thing, or a few closely RELATED things: the same rare word, name, narrow topic, language or script, symbol, format, or one token repeated. Related variations still count as one thing: a word and its translation, a topic and its neighbouring topics, a word and longer words that contain it.
C = UNRELATED THINGS. The texts show two or more clearly atypical, specific things that you could each name, and they have no sensible link to each other.
D = BROKEN. The texts are clearly atypical, but only broken or repetitive in generic ways (stray function words, punctuation, a different common word repeated in each), with no specific thing you could name.

Made-up examples:
X1: "1: The history of the Roman Empire is long and complex. 2: In recent years, the internet has changed how we work. 3: The first thing I noticed was the smell of rain." -> A
X2: "1: The bakery sold fresh bread and bread rolls. 2: 面包 面包 面包 面包 3: Flour, yeast and water make a dough." -> B (bread, its translation, and baking are related)
X3: "1: violin violin violin violin violin 2: The rocket reached orbit, and the rocket rocket 3: The weather today is mild." -> C (violins and rockets have no link)
X4: "1: , , ; ; and and and the 2: is is, was was was, is 3: I need. Let me. The the the" -> D

The three texts:
"""
TAIL = """1: {t1}
2: {t2}
3: {t3}

Answer with one letter: A, B, C or D."""


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
        pre, post = full.split("\x00SPLIT\x00")
        self.post = post
        self.pre_ids = mx.array(self.tok.encode(pre, add_special_tokens=False))[None]
        self.cache = [KVCache() for _ in self.model.layers]
        self.model(self.pre_ids, cache=self.cache)
        mx.eval([c.keys for c in self.cache])
        self.lab_ids = [self.tok.encode(x, add_special_tokens=False)[0] for x in LABELS]
        self.pad = self.tok.pad_token_id or 0

    def score(self, rows: list[dict]) -> list[list[float]]:
        sufs = [self.tok.encode(self.post.format(t1=clean(r["texts"][0]), t2=clean(r["texts"][1]),
                                                 t3=clean(r["texts"][2])), add_special_tokens=False) for r in rows]
        B, Lmax = len(sufs), max(len(s) for s in sufs)
        ids = mx.array([s + [self.pad] * (Lmax - len(s)) for s in sufs])
        last = mx.array([len(s) - 1 for s in sufs])
        cache = []
        for c in self.cache:
            k = KVCache()
            k.keys = mx.repeat(c.keys[..., :c.offset, :], B, axis=0)
            k.values = mx.repeat(c.values[..., :c.offset, :], B, axis=0)
            k.offset = c.offset
            cache.append(k)
        # right padding: positions <= last never see the pads. Unembed only the answer position.
        h = self.model.model(ids, cache=cache)[mx.arange(B), last]
        if self.model.args.tie_word_embeddings:
            logits = self.model.model.embed_tokens.as_linear(h)
        else:
            logits = self.model.lm_head(h)
        lg = logits.astype(mx.float32)
        mass = mx.exp(mx.logsumexp(lg[:, self.lab_ids], axis=-1) - mx.logsumexp(lg, axis=-1))
        p = mx.softmax(lg[:, self.lab_ids], axis=-1)
        mx.eval(p, mass)
        return [pi + [m] for pi, m in zip(p.tolist(), mass.tolist())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", default=["data/controls.jsonl", "data/controls_x10.jsonl",
                                                  "data/samples_x10.jsonl", "data/samples.jsonl"])
    ap.add_argument("--out", default="data/judgments_cammarata_raw.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    refs = json.loads((DATA / "refs.json").read_text())
    outp = P / a.out
    done = {json.loads(l)["key"] for l in open(outp)} if outp.exists() else set()
    rows = [r for inp in a.inputs for r in map(json.loads, open(P / inp)) if key(r) not in done]
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} rows to judge ({len(done)} done)", flush=True)
    j = Judge(refs)
    print(f"prefix {j.pre_ids.shape[1]} tokens, peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
    t0, n = time.time(), 0
    with open(outp, "a") as f:
        for b0 in range(0, len(rows), BATCH):
            chunk = rows[b0:b0 + BATCH]
            for r, p in zip(chunk, j.score(chunk)):
                f.write(json.dumps({"key": key(r), "id": r.get("id"), "kind": r["kind"], "scale": r.get("scale", 1.0),
                                    "layer": r["layer"], "neuron": r["neuron"], "p_A": p[0], "p_B": p[1],
                                    "p_C": p[2], "p_D": p[3], "p_format": p[4]}) + "\n")
            f.flush()
            n += len(chunk)
            if (b0 // BATCH) % 25 == 0:
                el = time.time() - t0
                print(f"{n}/{len(rows)} {el:.0f}s {n / el:.2f} rows/s peak {mx.get_peak_memory() / 1e9:.2f} GB",
                      flush=True)


if __name__ == "__main__":
    main()
