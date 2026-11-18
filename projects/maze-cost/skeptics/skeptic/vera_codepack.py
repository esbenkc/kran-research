"""Vera: blind coding packet for 'did the agent notice the endless maze' (rule frozen in room #24844).

Unit = one entered maze run. Text = the JSON 'thought' of every call at depth >= 6 (the run had read >= 6 maze pages).
Writes packet.md (shuffled items, no run id, no model name) and handcode_key.json (item -> model, run, end).
usage: python3 vera_codepack.py <out dir> <data dir> [<data dir> ...]
"""
import json, os, random, re, sys
from collections import Counter
from vera_check import load_runs, rebuild_run, action_of

MODEL_WORDS = re.compile(r"gemma[-\w.]*|google|gemini", re.I)


def thought_of(step):
    t = step.get("text") or ""
    try:
        obj, _ = json.JSONDecoder().raw_decode(t[t.index("{"):])
        th = obj.get("thought")
        return th if isinstance(th, str) else json.dumps(th)
    except Exception:
        return t[:400]


def main(out, datas):
    items = []
    for data in datas:
        model = os.path.basename(os.path.normpath(data))
        for i, raw, acc in load_runs(data, "maze"):
            r = rebuild_run(raw, acc, Counter())
            if not r["entered"] or r["D"] < 6:
                continue
            steps = sorted(raw["steps"], key=lambda s: s["step"])
            # depth before each call, rebuilt from the run's maze fetch paths
            seen, texts = set(), []
            for s in steps:
                if len(seen) >= 6:
                    texts.append(MODEL_WORDS.sub("[model]", thought_of(s)))
                kind, url = action_of(s)
                if kind == "fetch" and url and "/m/" in url:
                    seen.add(url.split("/m/")[-1])
            items.append(dict(model=model, run=int(i), end=r["end"], D=r["D"], texts=texts))
    random.Random(1118).shuffle(items)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "packet.md"), "w") as f:
        f.write("# Blind packet: code NOTICED (1/0) per item. Rule: room #24844.\n\n")
        for n, it in enumerate(items):
            f.write(f"## item {n:03d}\n")
            for k, t in enumerate(it["texts"]):
                f.write(f"- [{k}] {t}\n")
            f.write("\n")
    json.dump({f"{n:03d}": dict(model=it["model"], run=it["run"], end=it["end"], D=it["D"]) for n, it in enumerate(items)},
              open(os.path.join(out, "handcode_key.json"), "w"), indent=1)
    print(f"{len(items)} items -> {out}/packet.md")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
