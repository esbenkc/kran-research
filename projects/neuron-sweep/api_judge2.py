"""Second stage of the API judge: for every run the first stage labelled ONE THEME (mono), check that the
recurring thing is really the same across texts. Validation showed the first stage calls "each text repeats a
different word" one theme. Final label: stage-1 none/poly kept; stage-1 mono -> mono if stage 2 says SAME, else poly.

    with-model-keys uv run python projects/neuron-sweep/api_judge2.py data/judgments.jsonl --out data/judgments2.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from api_judge import MODEL, P, clean, key, llm

Q = """Below are three short texts that a small language model wrote.

Do at least two of them share the SAME specific unusual thing: the same rare word or name, the same narrow topic, the same foreign language or script, the same symbol, or the same format?
Answer DIFFERENT if the texts are odd in different ways: for example, each text repeats a different word, or each one is broken in its own way, or the only thing in common is that they are repetitive or broken.

1: {t1}
2: {t2}
3: {t3}

Reply with JSON only: {{"answer": "SAME" or "DIFFERENT", "thing": "<if SAME, the shared thing in a few words>"}}"""


def check(r: dict) -> tuple[str, str, str]:
    prompt = Q.format(t1=clean(r["texts"][0]), t2=clean(r["texts"][1]), t3=clean(r["texts"][2]))
    raw = ""
    for attempt in range(6):
        try:
            raw = llm.complete(prompt, model=MODEL, max_tokens=40)
            break
        except Exception as e:
            raw = f"ERROR {e}"
            time.sleep(2 ** attempt)
    a = re.search(r'"answer"\s*:\s*"(SAME|DIFFERENT)"', raw)
    t = re.search(r'"thing"\s*:\s*"([^"]*)"', raw)
    return (a.group(1) if a else "unparsed"), (t.group(1) if t else ""), raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judgments")
    ap.add_argument("--rows", nargs="*", default=["data/samples.jsonl", "data/samples_x10.jsonl",
                                                  "data/controls.jsonl", "data/controls_x10.jsonl"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    texts = {key(r): r for f in a.rows for r in map(json.loads, open(P / f))}
    J = [json.loads(l) for l in open(P / a.judgments)]
    todo = [j for j in J if j["label"] == "mono"]
    print(f"{len(todo)} stage-1 mono rows of {len(J)}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = dict(zip((j["key"] for j in todo), ex.map(lambda j: check(texts[j["key"]]), todo)))
    with open(P / a.out, "w") as f:
        for j in J:
            out = {k: v for k, v in j.items() if k != "raw"}
            out["label1"] = j["label"]
            if j["key"] in res:
                ans, thing, raw2 = res[j["key"]]
                out.update(stage2=ans, thing2=thing, raw2=raw2, label="mono" if ans == "SAME" else "poly")
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
