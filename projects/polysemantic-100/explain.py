"""Step 2: per pole, a cheap model reads the contexts and proposes 1-3 testable hypotheses.

    with-model-keys uv run python projects/polysemantic-100/explain.py   -> data/hypotheses.json

Poles with fewer than MIN_WINDOWS strong windows are labeled TOO SPARSE and skipped.
Writes data/probe_texts.json (all fire/control texts) for probe.py.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from common import DATA, PROJECT_DIR, load_neurons
from spend import call

MODEL = "claude-haiku-4-5"
MIN_WINDOWS = 5
PROMPT = (PROJECT_DIR / "prompts" / "explain.md").read_text()


def snippet_block(pole: dict) -> str:
    lines = []
    for i, e in enumerate(pole["examples"], 1):
        text = e["text"].replace("\n", "\\n")
        lines.append(f"{i}. ({e['rel']:.2f}) src={e['source']}/{e['lang']}: {text}")
    return "\n".join(lines)


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {"error": text[:300]}
    except json.JSONDecodeError:
        return {"error": text[:300]}


def explain_pole(nid: str, sign: str, pole: dict) -> dict:
    if pole["n_strong_windows"] < MIN_WINDOWS:
        return {"id": nid, "pole": sign, "status": "TOO_SPARSE", "hypotheses": []}
    out = parse(call(PROMPT + snippet_block(pole), model=MODEL, max_tokens=3000, tag="explain"))
    hyps = out.get("hypotheses", [])
    ok = [h for h in hyps if len(h.get("fire", [])) == 4 and len(h.get("control", [])) == 4]
    return {"id": nid, "pole": sign, "status": "OK" if ok else "PARSE_FAIL",
            "hypotheses": ok[:3], "raw_error": out.get("error")}


def main() -> None:
    jobs = []
    for n in load_neurons():
        rec = json.loads((DATA / "contexts" / f"{n['id']}.json").read_text())
        for sign in ("pos", "neg"):
            jobs.append((n["id"], sign, rec[sign]))
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda j: explain_pole(*j), jobs))
    (DATA / "hypotheses.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))

    texts = []
    for r in res:
        for h_i, h in enumerate(r["hypotheses"]):
            for kind in ("fire", "control"):
                for t_i, t in enumerate(h[kind]):
                    texts.append({"key": f"{r['id']}|{r['pole']}|{h_i}|{kind}|{t_i}", "text": t})
    (DATA / "probe_texts.json").write_text(json.dumps(texts, ensure_ascii=False, indent=0))
    from collections import Counter
    print(Counter(r["status"] for r in res), len(texts), "probe texts")


if __name__ == "__main__":
    main()
