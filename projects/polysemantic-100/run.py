"""Rebuild the figure from the saved pipeline outputs, and publish it to the blog clone.

Full pipeline (each step caches; see README of each script):
    uv run --with mlx-lm python sample.py                       # pre-registered neuron sample
    uv run --with mlx-lm --with datasets python record.py corpus
    uv run --with mlx-lm python record.py record                # ~20 min, one shared pass
    uv run --with mlx-lm python extract.py                      # contexts per pole
    with-model-keys uv run --with mlx-lm python explain.py      # hypotheses + probe texts
    uv run --with mlx-lm python probe.py run                    # local minimal-pair probes
    with-model-keys uv run --with mlx-lm python score.py        # probe gate, 3 judges, nulls
    uv run --with mlx-lm python ablate.py                       # ablation for UNRELATED poles
    uv run --with mlx-lm python run.py                          # figure data.json + publish
(run from projects/polysemantic-100/)
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from lib import figure

from common import DATA, PROJECT_DIR, load_neurons

BLOG_FIGURES = PROJECT_DIR.parents[2] / "blog.kran.ai" / "assets" / "figures"  # work-dir clone, never ~/Desktop
LABELS = [
    ("UNRELATED_CAUSAL", "unrelated jobs (switching it off hurts both)"),
    ("UNRELATED", "unrelated jobs (switching it off barely matters)"),
    ("RELATED", "related jobs"),
    ("ONE_JOB", "one job"),
    ("NO_PASS", "no guess passed its test"),
    ("TOO_SPARSE", "too few strong examples to test"),
]
SIGN = {"pos": "+", "neg": "-"}
SIDE = {"UNRELATED": "unrelated jobs", "RELATED": "related jobs", "ONE_JOB": "one job",
        "NO_PASS": "no guess passed", "TOO_SPARSE": "too few strong examples"}


def main() -> None:
    lab = json.loads((DATA / "labels.json").read_text())
    abl_path = DATA / "ablation.json"
    abl = {(r["id"], r["pole"]): r for r in json.loads(abl_path.read_text())} if abl_path.exists() else {}
    neurons = load_neurons()
    poles = {}
    for p in lab["poles"]:
        poles.setdefault(p["id"], []).append(p)

    def neuron_label(nid):
        base = lab["neurons"][nid]
        if base != "UNRELATED":
            return base
        both = any(abl.get((p["id"], p["pole"]), {}).get("hurts_both") for p in poles[nid] if p["label"] == "UNRELATED")
        return "UNRELATED_CAUSAL" if both else "UNRELATED"

    rows = []
    for n in neurons:
        nl = neuron_label(n["id"])
        lines = [f"{n['id']} (layer {n['layer']}): {dict(LABELS)[nl]}"]
        for p in sorted(poles[n["id"]], key=lambda p: p["pole"], reverse=True):
            jobs = [h["name"] for h in p["hyps"] if h["pass"]]
            tried = [h["name"] for h in p["hyps"] if not h["pass"]]
            head = f"{SIGN[p['pole']]} side, {SIDE[p['label']]}"
            body = "; ".join(jobs) if jobs else ("guesses tried: " + "; ".join(tried) if tried else "-")
            lines.append(f"{head}: {body}")
        rows.append({"id": n["id"], "layer": n["layer"], "label": nl, "tip": "\n".join(lines)})
    order = [k for k, _ in LABELS]
    rows.sort(key=lambda r: (order.index(r["label"]), r["layer"]))

    examples = []
    for p in lab["poles"]:
        if p["label"] != "UNRELATED":
            continue
        jobs = [h for h in p["hyps"] if h["pass"]]
        tok_ = lambda h: html.escape(h["fire_top_token"].strip())
        a = abl.get((p["id"], p["pole"]))
        note = ("ablation hurts both jobs" if a and a["hurts_both"] else
                "switching it off changes the predictions no more than a random neuron of its layer" if a else "ablation not run")
        examples.append({"id": f"{p['id']} {SIGN[p['pole']]} side",
                         "a": f"{html.escape(jobs[0]['name'])} (\u201c{tok_(jobs[0])}\u201d)",
                         "b": " + ".join(f"{html.escape(h['name'])} (\u201c{tok_(h)}\u201d)" for h in jobs[1:]),
                         "note": note})

    counts = {k: sum(r["label"] == k for r in rows) for k in order}
    n_explained = sum(counts[k] for k in ("UNRELATED_CAUSAL", "UNRELATED", "RELATED", "ONE_JOB"))
    data = {
        "title": f"Our test explained {n_explained} of 100 random Qwen3-4B neurons",
        "subtitle": "One square per neuron. A neuron takes the strangest label of its two sides. "
                    "Hover a square for the jobs we tested.",
        "labels": [{"key": k, "name": name, "n": counts[k]} for k, name in LABELS if counts[k]],
        "neurons": rows,
        "examples": examples,
        "stats": {k: lab[k] for k in ("neuron_labels", "pole_labels", "neuron_labels_sens", "null_swap",
                                       "null_pooled", "null_split", "judge_unanimous", "judge_n",
                                       "unrelated_same_top_token", "poles_probe_over_corpus_gt2")},
    }
    fig = figure.save(PROJECT_DIR, "polysemantic-100", data, (PROJECT_DIR / "plot.js").read_text())
    print("wrote", fig, counts)
    print("published to", figure.publish(fig, blog_figures=BLOG_FIGURES))


if __name__ == "__main__":
    main()
