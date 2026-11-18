"""LLM judge (PREREG): refusal / compliance / incoherent, model and prompt from config.yaml (the parent's judge).

    cd kran-research && with-model-keys uv run --with openai python projects/gated-refusal-8b/judge.py data/strip_s0/gen_test.jsonl ...
Writes <name>.judged.jsonl next to each input (same rows + 'judge'); incremental. Calls cached by lib/llm.py.
Appends the number of new (uncached) calls and their estimated cost to data/judge_spend.jsonl.
"""
import fcntl
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from lib import llm  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text())["judge"]
PRICE_IN, PRICE_OUT = 0.40e-6, 1.60e-6  # gpt-4.1-mini, USD per token
new_calls = []


def label(row):
    prompt = CFG["prompt"].format(prompt=row["prompt"], text=row["text"])
    cf = llm._CACHE / f"{llm._key(CFG['model'], CFG['system'], prompt, CFG['max_tokens'])}.json"
    if not cf.exists():
        new_calls.append(len(prompt) + len(CFG["system"]))
    out = llm.complete(prompt, system=CFG["system"], model=CFG["model"], max_tokens=CFG["max_tokens"]).strip().upper()
    for w in ("REFUSAL", "COMPLIANCE", "INCOHERENT"):
        if out.startswith(w):
            return w.lower()
    return "unparsed:" + out[:20]


def key(r):
    return (r["set"], r["split"], r["kind"], r["i"])


def main():
    for f in sys.argv[1:]:
        f = Path(f)
        dst = f.with_name(f.stem + ".judged.jsonl")
        lock = open(str(dst) + ".lock", "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        have = {key(r) for r in map(json.loads, dst.open())} if dst.exists() else set()
        rows = [r for r in map(json.loads, f.open()) if key(r) not in have]
        with ThreadPoolExecutor(16) as ex:
            labs = list(ex.map(label, rows))
        with dst.open("a") as fo:
            for r, j in zip(rows, labs):
                fo.write(json.dumps({**r, "judge": j}) + "\n")
        print(f, "new judged:", len(rows), "unparsed:", sum(j.startswith("unparsed") for j in labs), flush=True)
        lock.close()
        Path(str(dst) + ".lock").unlink(missing_ok=True)
    usd = sum(c / 4 * PRICE_IN + 2 * PRICE_OUT for c in new_calls)
    with (HERE / "data" / "judge_spend.jsonl").open("a") as fo:
        fo.write(json.dumps({"time": time.strftime("%F %T"), "files": sys.argv[1:], "new_calls": len(new_calls),
                             "est_usd": round(usd, 4), "est": "chars/4 input tokens, 2 output tokens"}) + "\n")
    print(f"new API calls {len(new_calls)}, est ${usd:.3f}")


if __name__ == "__main__":
    main()
