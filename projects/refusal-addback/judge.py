"""LLM judge for DEV and TEST outputs (PREREG s7, DEVIATIONS D3): refusal / compliance / incoherent.
Model and prompt come from config.yaml (committed before the DEV sweep).

    with-model-keys ../../.venv/bin/python judge.py runs/strip_s0/dev.jsonl [more files]
Writes <name>.judged.jsonl next to each input (same rows + 'judge'). Incremental: rows already in the
judged file are kept. Calls are cached by lib/llm.py (.llm_cache/). 16 threads.
"""
import fcntl
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from lib import llm

CFG = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())["judge"]


def label(row):
    out = llm.complete(CFG["prompt"].format(prompt=row["prompt"], text=row["text"]), system=CFG["system"],
                       model=CFG["model"], max_tokens=CFG["max_tokens"]).strip().upper()
    for w in ("REFUSAL", "COMPLIANCE", "INCOHERENT"):
        if out.startswith(w):
            return w.lower()
    return "unparsed:" + out[:20]


def key(r):
    return (r["layer"], r["gain"], r["dir"], r["kind"], r["i"])


def main():
    for f in sys.argv[1:]:
        dst = Path(f).with_name(Path(f).stem + ".judged.jsonl")
        lock = open(str(dst) + ".lock", "w")
        fcntl.flock(lock, fcntl.LOCK_EX)  # one judge process per file: no duplicate rows
        have = {key(r) for r in map(json.loads, dst.open())} if dst.exists() else set()
        rows = [r for r in map(json.loads, open(f)) if key(r) not in have]
        with ThreadPoolExecutor(16) as ex:
            labs = list(ex.map(label, rows))
        with dst.open("a") as fo:
            for r, j in zip(rows, labs):
                fo.write(json.dumps({**r, "judge": j}) + "\n")
        bad = sum(j.startswith("unparsed") for j in labs)
        print(f, "new judged:", len(rows), "unparsed:", bad, flush=True)
        lock.close()


if __name__ == "__main__":
    main()
