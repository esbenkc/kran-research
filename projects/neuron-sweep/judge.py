"""Blind local judge for the neuron sweep: one forward pass per row, read P(A/B/C) for the first answer token.

    cd kran-research && uv run python projects/neuron-sweep/judge.py data/samples.jsonl data/controls.jsonl
    cd kran-research && uv run python projects/neuron-sweep/judge.py data/hand_pool.jsonl --out data/judge_hand.jsonl

Judge: Qwen3-4B-Instruct-2507 (MLX, 4-bit), local. It sees three steered continuations, nine fixed continuations of
the unchanged model, and three made-up worked examples. It never sees the layer, the neuron, the push strength,
or whether a row is a control. Labels: A = no visible effect, B = one theme (mono), C = several unrelated (poly).
The fixed part of the prompt (rubric + reference texts) is run once and its KV cache is reused for every row.
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
JUDGE = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
LABELS = ["A", "B", "C"]
NAMES = {"A": "none", "B": "mono", "C": "poly"}
BATCH = 8  # each row gets its own copy of the ~660-token prefix KV cache (~100 MB per row)

HEAD = """We changed one internal part of a small language model. Then we asked it three times to continue this prompt: "The following is a short passage of text."
Your job: say whether the change left a visible mark on the text, and if so, whether the mark is ONE thing or SEVERAL unrelated things.

Here are nine continuations from the UNCHANGED model. This is what normal output looks like. Note that normal output often starts the same way ("The first...", "In recent years...", a quote mark, "The passage..."), and that each text picks its own topic. Such things are NOT a mark of the change.
{refs}

Answer with one letter:
A = NO EFFECT. The three texts could have come from the unchanged model: readable, each on its own topic. Sharing a common opening or a very common word does not count.
B = ONE THEME. Something unusual, that normal output does not show, appears in at least two of the three texts, and it is one thing: the same rare word or name, one topic, one foreign language or script, one symbol, or one strange style. Repeating that one thing also counts as ONE THEME.
C = SEVERAL THEMES. Something unusual appears, but it is two or more unrelated things (for example, one text about cooking, one full of Chinese characters, one about rockets), or the text is broken in different ways with no single thing in common.

Worked examples (made up):
Example 1: "1: The history of the Roman Empire is long and complex. 2: In recent years, the internet has changed how we work. 3: The first thing I noticed was the smell of rain." -> A
Example 2: "1: The sugar in the cake was mixed with more sugar. 2: Sweet desserts like candy and sugar are popular. 3: She added a spoon of sugar to her tea." -> B
Example 3: "1: 读读读读读读读 2: The rocket launched into orbit, and the rocket 3: , , , ; ; ; and and and" -> C

Now the three texts from the CHANGED model:
"""
TAIL = """1: {t1}
2: {t2}
3: {t3}

Answer with one letter: A, B or C."""


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
        mass = mx.exp(mx.logsumexp(lg[:, self.lab_ids], axis=-1) - mx.logsumexp(lg, axis=-1))  # P(A|B|C) of all
        p = mx.softmax(lg[:, self.lab_ids], axis=-1)
        mx.eval(p, mass)
        return [pi + [m] for pi, m in zip(p.tolist(), mass.tolist())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", default=["data/samples.jsonl", "data/controls.jsonl"])
    ap.add_argument("--out", default="data/judgments.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    refs = json.loads((DATA / "refs.json").read_text())
    outp = P / a.out
    done = set()
    if outp.exists():
        done = {json.loads(l)["key"] for l in open(outp)}
    rows = []
    for inp in a.inputs:
        for l in open(P / inp):
            r = json.loads(l)
            if key(r) not in done:
                rows.append(r)
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} rows to judge ({len(done)} done)", flush=True)
    j = Judge(refs)
    f = open(outp, "a")
    t0, n = time.time(), 0
    for b0 in range(0, len(rows), BATCH):
        chunk = rows[b0:b0 + BATCH]
        for r, p in zip(chunk, j.score(chunk)):
            lab = NAMES[LABELS[max(range(3), key=lambda i: p[i])]]
            f.write(json.dumps({"key": key(r), "kind": r["kind"], "scale": r.get("scale", 1.0), "layer": r["layer"],
                                "neuron": r["neuron"], "p_none": p[0], "p_mono": p[1], "p_poly": p[2],
                                "p_format": p[3], "label": lab}) + "\n")
        f.flush()
        n += len(chunk)
        if (b0 // BATCH) % 20 == 0:
            el = time.time() - t0
            print(f"{n}/{len(rows)} {el:.0f}s {n / el:.1f} rows/s", flush=True)


if __name__ == "__main__":
    main()
