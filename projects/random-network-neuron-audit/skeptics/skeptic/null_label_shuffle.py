"""skeptic null: LABEL SHUFFLE (pre-registered in PREREGISTRATION.md, 'Nulls and checks').

For every audited unit u, take the main label that judge J wrote for ANOTHER unit v (same arm, same layer,
derangement with a fixed seed). Then, with my own caller at temperature 0:
  fit:  judge J counts how many of u's 20 top excerpts fit v's label (and, as the matched reference, u's own label).
  det:  the OTHER family runs the frozen detection prompt with v's label on u's 10 detection snippets
        (5 held-out top, 5 random). hits = correct picks.
One-job under shuffle := fit >= 15 AND hits >= 4 (same rule as the prereg).
Pre-registered alarm: fit >= 15/20 for > 10% of mismatched labels => rubric is lenient.

Inputs:  data/contexts.jsonl (uid, arm, layer, examples[20], heldout[], random[])
         judge labels: a jsonl with uid, judge model, main_label (path given on the command line)
Output:  out/null_label_shuffle.jsonl + printed summary.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sk_call
from sk_json import last_obj

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
OTHER = {"gpt-4.1-mini": "claude-haiku-4-5", "claude-haiku-4-5": "gpt-4.1-mini"}

FIT_PROMPT = """Someone studied one hidden unit of a language model and wrote this description of what it responds to:

DESCRIPTION: {label}

Below are 20 text snippets. In each, the token where the unit fired is wrapped in «guillemets».
A snippet FITS the description if the marked token, read with its context, matches the description.
Judge the marked token, not the whole snippet. Be strict: a snippet fits only if a careful reader using the
description alone would have predicted the unit fires there.

{snippets}

Answer with JSON only, in exactly this form:
{{"fit_count": <0-20>, "fits": [<snippet numbers that fit>]}}"""


def fmt(ex: dict) -> str:
    return f"{ex['left']}«{ex['tok']}»{ex['right']}".replace("\n", "⏎")


def fit_call(model: str, label: str, examples: list[dict]) -> int | None:
    sn = "\n".join(f"{i + 1}. {fmt(e)}" for i, e in enumerate(examples[:20]))
    js = last_obj(sk_call.call(model, FIT_PROMPT.format(label=label, snippets=sn), 400), "fits")
    if not js:
        return None
    fits = js.get("fits")
    if isinstance(fits, list):  # count from the list, not the self-reported total
        return len({int(x) for x in fits if str(x).isdigit() and 1 <= int(x) <= 20})
    return int(js.get("fit_count", 0))


def det_call(model: str, label: str, unit: dict, detect_tmpl: str, seed: int) -> int | None:
    pos = [(e, 1) for e in unit["heldout"][:5]]
    neg = [(e, 0) for e in unit["random"][:5]]
    items = pos + neg
    random.Random(seed).shuffle(items)
    sn = "\n".join(f"{i + 1}. {fmt(e)}" for i, (e, _) in enumerate(items))
    prompt = detect_tmpl.replace("{label}", label).replace("{{", "{").replace("}}", "}") + "\n\nSNIPPETS:\n" + sn
    js = last_obj(sk_call.call(model, prompt, 1000), "picks")
    if not js or not isinstance(js.get("picks"), list):
        return None
    picks = []
    for p in js["picks"]:
        if str(p).isdigit() and 1 <= int(p) <= 10 and int(p) not in picks:
            picks.append(int(p))
    return sum(1 for p in picks[:5] if items[p - 1][1] == 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--labels", required=True, help="jsonl: uid, judge, main_label")
    ap.add_argument("--detect", required=True, help="frozen detect.md prompt text")
    ap.add_argument("--n-per-cell", type=int, default=50)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    units = {u["uid"]: u for u in map(json.loads, open(a.contexts))}
    labels = defaultdict(dict)
    for r in map(json.loads, open(a.labels)):
        labels[r["uid"]][r["judge"]] = r["main_label"]
    detect_tmpl = Path(a.detect).read_text()

    rng = random.Random(a.seed)
    jobs = []
    cells = defaultdict(list)
    for uid, u in units.items():
        cells[(u["arm"], u["layer"])].append(uid)
    for (arm, layer), uids in sorted(cells.items()):
        uids = sorted(uids)[: a.n_per_cell]
        perm = uids[:]
        while any(x == y for x, y in zip(uids, perm)):  # derangement
            rng.shuffle(perm)
        for u, v in zip(uids, perm):
            for judge in OTHER:
                if judge in labels[u] and judge in labels[v]:
                    jobs.append((arm, layer, u, v, judge))

    def run(job):
        arm, layer, u, v, judge = job
        unit = units[u]
        seed = int(u.encode().hex()[:8], 16) if not str(u).isdigit() else int(u)
        own, other = labels[u][judge], labels[v][judge]
        return {"arm": arm, "layer": layer, "uid": u, "label_from": v, "judge": judge,
                "fit_own": fit_call(judge, own, unit["examples"]),
                "fit_shuf": fit_call(judge, other, unit["examples"]),
                "hits_own": det_call(OTHER[judge], own, unit, detect_tmpl, seed),
                "hits_shuf": det_call(OTHER[judge], other, unit, detect_tmpl, seed)}

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run, jobs))
    (OUT / "null_label_shuffle.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summarize(rows)
    print(f"spend so far ${sk_call.spend():.3f}")


def summarize(rows: list[dict]) -> None:
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
        by["ALL"].append(r)
    print("arm  n   fit>=15(own) fit>=15(shuf) hits>=4(own) hits>=4(shuf) onejob(own) onejob(shuf)")
    for arm, rs in sorted(by.items()):
        def rate(f):
            v = [f(r) for r in rs if None not in (r["fit_own"], r["fit_shuf"], r["hits_own"], r["hits_shuf"])]
            return sum(v) / max(len(v), 1), len(v)
        fo, n = rate(lambda r: r["fit_own"] >= 15)
        fs, _ = rate(lambda r: r["fit_shuf"] >= 15)
        ho, _ = rate(lambda r: r["hits_own"] >= 4)
        hs, _ = rate(lambda r: r["hits_shuf"] >= 4)
        oo, _ = rate(lambda r: r["fit_own"] >= 15 and r["hits_own"] >= 4)
        os_, _ = rate(lambda r: r["fit_shuf"] >= 15 and r["hits_shuf"] >= 4)
        print(f"{arm:4s} {n:3d} {fo:12.3f} {fs:13.3f} {ho:12.3f} {hs:13.3f} {oo:11.3f} {os_:12.3f}")
    print("chance P(hits>=4) for blind picking = 26/252 = 0.103")


if __name__ == "__main__":
    main()
