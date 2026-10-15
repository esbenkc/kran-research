"""Blind two-judge one-job audit for model units (room oracle-1015-already-clean, prereg PREREGISTRATION.md).

Input (from build): contexts.jsonl, one line per unit:
  {"uid": str, "arm": "A"|"B"|"C", "layer": int, "unit": int,
   "examples": [excerpt x20],   # top-20 by activation, descending, max 1 per doc
   "heldout":  [excerpt x>=5],  # ranks 21-40 (same dedup rules)
   "random":   [excerpt x>=5]}  # random corpus excerpts, marked position >= 4, docs not in the unit's top-40
  excerpt = {"left": str, "tok": str, "right": str, "act": float, "regime": "++"|"--"|"+-"|"-+"|null}

Two calls per (unit, judge family):
  1. LABEL call (rubric.md): judge sees the top-20 (marked token, activation scaled to max 10).
  2. DETECTION call (detect.md): the OTHER family sees only that label + 10 shuffled snippets
     (5 from heldout, 5 from random; seeded by uid) and picks 5. hits = correct picks.
Output: judgments.jsonl, one line per (unit, label judge): label JSON, detection picks, hits, raw text, tokens, cost.

Blinding: prompts hold only rubric text and snippets. No arm, uid, layer or model name. Units run in seeded random order.
Caching: same cache dir and key scheme as kran-research/lib/llm.py (model, system, prompt, max_tokens), so reruns cost $0.
Providers are called directly only to record token usage for the cost line.

Library use (for skeptics' nulls):
  from judge import label_unit, detect
  label_unit(examples, model) -> {"ok", "parsed", "raw", "cost_usd", ...}
  detect(label, snippets, model) -> {"ok", "picks", "raw", "cost_usd", ...}

Run: cd kran-research && with-model-keys uv run python <path>/judge.py --contexts data/contexts.jsonl --out data/judgments.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
for cand in [HERE, *HERE.parents, Path("/Users/esben/work/oracle-1015-already-clean/kran-research")]:
    if (cand / "lib" / "llm.py").exists():
        sys.path.insert(0, str(cand))
        break
from lib import llm  # noqa: E402

# Prereg named gpt-5-mini as J1; our OpenAI org gets 404 ("organization must be verified"), so J1 = gpt-4.1-mini.
JUDGES = ["gpt-4.1-mini", "claude-haiku-4-5"]
OTHER = {"gpt-4.1-mini": "claude-haiku-4-5", "claude-haiku-4-5": "gpt-4.1-mini"}
PRICE = {"gpt-4.1-mini": (0.40, 1.60), "claude-haiku-4-5": (1.00, 5.00)}  # USD per 1M tokens (in, out), list price
MAX_TOKENS = 1000
SYSTEM = "You are a careful interpretability researcher. You answer with JSON only."
RUBRIC = (HERE / "rubric.md").read_text()
DETECT = (HERE / "detect.md").read_text()
_lock = threading.Lock()


def snippet(e: dict) -> str:
    return f'{e["left"]}«{e["tok"]}»{e["right"]}'.replace("\n", "\\n")


def render_top(ex: list[dict]) -> str:
    top = max(e["act"] for e in ex) or 1.0
    return "\n".join(f"{i}. [{round(10 * e['act'] / top, 1)}] {snippet(e)}" for i, e in enumerate(ex, 1))


def _call(model: str, prompt: str) -> tuple[str, int, int]:
    if model.startswith("gpt"):
        from openai import OpenAI
        r = OpenAI(api_key=os.environ["OPENAI_API_KEY"]).chat.completions.create(
            model=model, max_tokens=MAX_TOKENS, temperature=0,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens
    from anthropic import Anthropic
    r = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]).messages.create(
        model=model, max_tokens=MAX_TOKENS, temperature=0, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in r.content if getattr(b, "type", None) == "text"), r.usage.input_tokens, r.usage.output_tokens


def complete(model: str, prompt: str) -> dict:
    cf = llm._CACHE / f"{llm._key(model, SYSTEM, prompt, MAX_TOKENS)}.json"
    if cf.exists():
        d = json.loads(cf.read_text())
        return {"text": d["text"], "in_tok": d.get("in_tok", 0), "out_tok": d.get("out_tok", 0), "cached": True, "model": model}
    text, it, ot = _call(model, prompt)
    llm._CACHE.mkdir(exist_ok=True)
    cf.write_text(json.dumps({"model": model, "text": text, "in_tok": it, "out_tok": ot}))
    return {"text": text, "in_tok": it, "out_tok": ot, "cached": False, "model": model}


def _cost(tries: list[dict]) -> float:
    return sum((t["in_tok"] * PRICE[t["model"]][0] + t["out_tok"] * PRICE[t["model"]][1]) / 1e6
               for t in tries if not t["cached"])


def _json(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def parse_label(raw: str) -> dict | None:
    d = _json(raw)
    if d is None:
        return None
    try:
        mc = int(d["main_count"])
        others = [{"label": str(o.get("label", "")), "count": int(o["count"]), "related": bool(o["related"])}
                  for o in d.get("other_jobs", []) or []]
        un = int(d["unexplained_count"])
        typ = str(d["main_type"]).upper().strip()
        label = str(d["main_label"]).strip()
    except (KeyError, TypeError, ValueError):
        return None
    if typ not in {"LEXICAL", "SYNTACTIC", "SEMANTIC", "NONE"} or not 0 <= mc <= 20 or not label:
        return None
    if mc + sum(o["count"] for o in others) + un != 20:
        return None
    d.update(main_label=label, main_count=mc, other_jobs=others, unexplained_count=un, main_type=typ,
             n_jobs=(1 if typ != "NONE" else 0) + len(others))
    return d


def parse_picks(raw: str) -> list[int] | None:
    d = _json(raw)
    try:
        p = [int(x) for x in d["picks"]]
    except (KeyError, TypeError, ValueError):
        return None
    return p if len(set(p)) == 5 and all(1 <= x <= 10 for x in p) else None


def label_unit(examples: list[dict], model: str) -> dict:
    assert len(examples) == 20
    p = RUBRIC + "\n\nSnippets:\n" + render_top(examples)
    tries = [complete(model, p)]
    parsed = parse_label(tries[-1]["text"])
    if parsed is None:  # one retry with a format reminder (new prompt -> new cache key)
        tries.append(complete(model, p + "\n\nReminder: JSON only, and all counts must add up to exactly 20."))
        parsed = parse_label(tries[-1]["text"])
    return {"ok": parsed is not None, "parsed": parsed, "raw": tries[-1]["text"], "cost_usd": _cost(tries),
            "in_tok": sum(t["in_tok"] for t in tries), "out_tok": sum(t["out_tok"] for t in tries),
            "cached": all(t["cached"] for t in tries)}


def detect(label: str, snippets: list[dict], model: str) -> dict:
    assert len(snippets) == 10
    p = DETECT.format(label=label) + "\nSnippets:\n" + "\n".join(f"{i}. {snippet(e)}" for i, e in enumerate(snippets, 1))
    tries = [complete(model, p)]
    picks = parse_picks(tries[-1]["text"])
    if picks is None:
        tries.append(complete(model, p + "\n\nReminder: JSON only, exactly 5 different numbers from 1 to 10."))
        picks = parse_picks(tries[-1]["text"])
    return {"ok": picks is not None, "picks": picks, "raw": tries[-1]["text"], "cost_usd": _cost(tries),
            "in_tok": sum(t["in_tok"] for t in tries), "out_tok": sum(t["out_tok"] for t in tries),
            "cached": all(t["cached"] for t in tries)}


def detection_set(unit: dict) -> tuple[list[dict], list[int]]:
    """5 held-out (ranks 21-40) + 5 random excerpts, shuffled; seeded by uid so every judge sees the same set."""
    rng = random.Random(int(hashlib.sha256(unit["uid"].encode()).hexdigest()[:12], 16))
    pos = rng.sample(unit["heldout"], 5)
    neg = rng.sample(unit["random"], 5)
    items = [(e, True) for e in pos] + [(e, False) for e in neg]
    rng.shuffle(items)
    return [e for e, _ in items], [i for i, (_, t) in enumerate(items, 1) if t]


def judge_unit(unit: dict, model: str) -> dict:
    lab = label_unit(unit["examples"], model)
    out = {"uid": unit["uid"], "arm": unit["arm"], "layer": unit["layer"], "unit": unit["unit"], "judge": model,
           "label_ok": lab["ok"], "parsed": lab["parsed"], "label_raw": lab["raw"],
           "detector": OTHER[model], "det_ok": False, "picks": None, "truth": None, "hits": None, "det_raw": None}
    cost, it, ot, cached = lab["cost_usd"], lab["in_tok"], lab["out_tok"], lab["cached"]
    if lab["ok"] and lab["parsed"]["main_type"] != "NONE":
        snips, truth = detection_set(unit)
        det = detect(lab["parsed"]["main_label"], snips, OTHER[model])
        out.update(det_ok=det["ok"], picks=det["picks"], truth=truth, det_raw=det["raw"],
                   hits=len(set(det["picks"]) & set(truth)) if det["ok"] else None)
        cost, it, ot, cached = cost + det["cost_usd"], it + det["in_tok"], ot + det["out_tok"], cached and det["cached"]
    out.update(cost_usd=cost, in_tok=it, out_tok=ot, cached=cached)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judges", default=",".join(JUDGES))
    ap.add_argument("--arms", default="", help="comma list to restrict arms (e.g. A for the dry run); empty = all")
    ap.add_argument("--limit", type=int, default=0, help="first N units after the seeded shuffle; 0 = all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    units = [json.loads(l) for l in open(a.contexts)]
    for u in units:
        assert len(u["examples"]) == 20 and len(u["heldout"]) >= 5 and len(u["random"]) >= 5, u["uid"]
    random.Random(a.seed).shuffle(units)
    if a.arms:
        units = [u for u in units if u["arm"] in a.arms.split(",")]
    if a.limit:
        units = units[: a.limit]
    jobs = [(u, m) for u in units for m in a.judges.split(",")]
    done = set()
    if os.path.exists(a.out):
        done = {(d["uid"], d["judge"]) for d in map(json.loads, open(a.out))}
    jobs = [(u, m) for u, m in jobs if (u["uid"], m) not in done]

    spent = 0.0
    with open(a.out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(lambda j: judge_unit(*j), jobs), 1):
            with _lock:
                f.write(json.dumps(res) + "\n")
                f.flush()
                spent += res["cost_usd"]
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} judged, new spend ${spent:.3f}", flush=True)
    with open(Path(a.out).with_name("api_spend.jsonl"), "a") as f:
        f.write(json.dumps({"script": "judge.py", "contexts": a.contexts, "n_jobs": len(jobs), "usd": spent}) + "\n")
    print(f"done. new API spend this run: ${spent:.4f}")


if __name__ == "__main__":
    main()
