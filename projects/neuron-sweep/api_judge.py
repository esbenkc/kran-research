"""Blind API judge for the neuron sweep (the local judges failed validation; see judge.py and validate.py).

    with-model-keys uv run python projects/neuron-sweep/api_judge.py data/hand_pool_a.jsonl --out data/api_hand.jsonl
    with-model-keys uv run python projects/neuron-sweep/api_judge.py   # all samples + controls -> data/judgments.jsonl

The judge (GPT-4.1-mini, the post's first judge) sees nine fixed continuations of the unchanged model and the three
continuations of one run. It never sees the layer, the neuron, the push strength, or whether the run is a control.
Answer: JSON {"thing": "<the one shared thing, or empty>", "label": "A" | "B" | "C"}.
A = no visible effect, B = one theme (mono), C = several unrelated themes (poly). Calls are cached (lib/llm.py).
"""
from __future__ import annotations

import argparse
import json
import re
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parent.parent))
from lib import llm  # noqa: E402

DATA = P / "data"
MODEL = "gpt-4.1-mini"
NAMES = {"A": "none", "B": "mono", "C": "poly"}

RUBRIC = """Below are three short texts. A small language model (0.6B parameters) wrote each one as a continuation of the prompt "The following is a short passage of text." Each is cut off after about 20 tokens.

For comparison, here are nine typical continuations of the same prompt by the same model. Typical output is plain, fluent, a little bland, and each text drifts to its own everyday topic. Typical texts often share openings ("The first...", "Today...", "In recent years...", a quote mark, a code fence) and common words (people, world, government, study, problem, first). Small slips are typical too.
{refs}

Compare the three texts below with the typical ones and pick one label:
A = TYPICAL. All three could easily be typical continuations. Most sets of three are like this. Shared openings, shared common words, or one slightly odd detail do not make a set unusual.
B = ONE THING. Something clearly atypical appears in at least two of the three texts, and it is one specific thing: the same rare word, name, narrow topic, foreign language or script, symbol, format, or one token repeated over and over.
C = SEVERAL THINGS. At least two of the texts are clearly atypical, but in different, unrelated ways, or broken with no single thing in common.

The three texts:
1: {t1}
2: {t2}
3: {t3}

Reply with JSON only: {{"label": "A" or "B" or "C", "thing": "<for B, the one shared thing in a few words; otherwise empty>"}}"""


def clean(t: str) -> str:
    return " ".join(t.split())[:400] or "(empty)"


def key(r: dict) -> str:
    return f'{r["kind"]}|{r.get("scale", 1.0):g}|{r["layer"]}|{r["neuron"]}'


def judge(r: dict, refs_txt: str) -> dict:
    prompt = RUBRIC.format(refs=refs_txt, t1=clean(r["texts"][0]), t2=clean(r["texts"][1]), t3=clean(r["texts"][2]))
    raw = ""
    for attempt in range(6):
        try:
            raw = llm.complete(prompt, model=MODEL, max_tokens=60)
            break
        except Exception as e:  # rate limits / transient errors: back off and retry
            raw = f"ERROR {e}"
            time.sleep(2 ** attempt)
    m = re.search(r'"label"\s*:\s*"([ABC])"', raw)
    t = re.search(r'"thing"\s*:\s*"([^"]*)"', raw)
    lab = NAMES[m.group(1)] if m else "unparsed"
    return {"key": key(r), "kind": r["kind"], "scale": r.get("scale", 1.0), "layer": r["layer"], "neuron": r["neuron"],
            "label": lab, "thing": t.group(1) if t else "", "raw": raw, "n_chars_in": len(prompt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", default=["data/samples.jsonl", "data/samples_x10.jsonl",
                                                  "data/controls.jsonl", "data/controls_x10.jsonl"])
    ap.add_argument("--out", default="data/judgments.jsonl")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    refs = json.loads((DATA / "refs.json").read_text())
    refs_txt = "\n".join(f"R{i + 1}: {clean(t)}" for i, t in enumerate(refs))
    outp = P / a.out
    done = {json.loads(l)["key"] for l in open(outp)} if outp.exists() else set()
    rows = [r for inp in a.inputs for r in map(json.loads, open(P / inp)) if key(r) not in done]
    print(f"{len(rows)} rows to judge ({len(done)} done)", flush=True)
    with open(outp, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        for n, res in enumerate(ex.map(lambda r: judge(r, refs_txt), rows), 1):
            f.write(json.dumps(res) + "\n")
            if n % 500 == 0:
                f.flush()
                print(f"{n}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
