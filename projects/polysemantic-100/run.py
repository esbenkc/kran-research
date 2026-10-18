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
    ("UNRELATED_CAUSAL", "unrelated jobs, ablation hurts both"),
    ("UNRELATED", "unrelated jobs, ablation does not hurt both"),
    ("RELATED", "related jobs"),
    ("ONE_JOB", "one job"),
    ("NO_PASS", "no hypothesis passed its test"),
    ("TOO_SPARSE", "too few strong firings in the corpus"),
]
SIGN = {"pos": "+", "neg": "-"}


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
            lines.append(f"{SIGN[p['pole']]} pole, {p['label'].lower().replace('_', ' ')}: " + ("; ".join(jobs) or "-"))
        rows.append({"id": n["id"], "layer": n["layer"], "label": nl, "tip": "\n".join(lines)})
    order = [k for k, _ in LABELS]
    rows.sort(key=lambda r: (order.index(r["label"]), r["layer"]))

    examples = []
    for r in abl.values():
        if not r["hurts_both"]:
            continue
        f = [x for x in r["features"] if x["fire"]["pass"]]
        examples.append({"id": f"{r['id']} {SIGN[r['pole']]}", "a": html.escape(f[0]["name"]), "b": html.escape(f[1]["name"]),
                         "note": f"ablation KL {f[0]['fire']['target_kl']:.3f} and {f[1]['fire']['target_kl']:.3f}"})

    counts = {k: sum(r["label"] == k for r in rows) for k in order}
    data = {
        "title": "100 random neurons of Qwen3-4B, one square each",
        "subtitle": "Label = the worst of the neuron's two poles. Hover a square for its tested jobs.",
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
