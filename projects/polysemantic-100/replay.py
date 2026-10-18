"""Diagnostic (added after score.py ran; not a label rule): can the probe gate be met at all with
short texts? Replays each pole's own top-12 corpus snippets (the 47-token context the explainer saw,
brackets removed) and reports the pole's firing as a share of the corpus pole max.

    uv run --with mlx-lm python projects/polysemantic-100/replay.py   -> data/replay.json
"""
from __future__ import annotations

import json

import numpy as np

from common import DATA, column_order, load_model, load_neurons
from probe import run_texts


def main() -> None:
    neurons = load_neurons()
    col = {n["id"]: k for k, n in enumerate(neurons)}
    items = []
    for n in neurons:
        rec = json.loads((DATA / "contexts" / f"{n['id']}.json").read_text())
        for s in ("pos", "neg"):
            if rec[s]["n_strong_windows"] < 5:
                continue
            for i, e in enumerate(rec[s]["examples"][:12]):
                items.append((n["id"], s, i, rec[s]["pole_max"], e["rel"], e["text"].replace("[[", "").replace("]]", "")))
    model, tok, taps, _ = load_model(neurons)
    outs = run_texts(model, tok, taps, column_order(neurons, taps), [t[-1] for t in items])
    rows = []
    for (nid, s, i, pmax, rel, _), o in zip(items, outs):
        v = o["max"][col[nid]] if s == "pos" else -o["min"][col[nid]]
        rows.append({"id": nid, "pole": s, "rank": i, "corpus_rel": rel, "replay_rel": round(v / pmax, 3)})
    (DATA / "replay.json").write_text(json.dumps(rows))
    r = np.array([x["replay_rel"] for x in rows])
    c = np.array([x["corpus_rel"] for x in rows])
    print(len(rows), "snippets; replay/pole max quantiles 10/25/50/75/90:", np.percentile(r, [10, 25, 50, 75, 90]).round(2))
    print("share of snippets replaying >= 0.5 x pole max:", round(float((r >= 0.5).mean()), 3),
          "| in full context they were:", round(float((c >= 0.5).mean()), 3))
    by = {}
    for x in rows:
        by.setdefault((x["id"], x["pole"]), []).append(x["replay_rel"] >= 0.5)
    print("poles where >= 3 of the top 12 snippets replay >= 0.5:", sum(sum(v) >= 3 for v in by.values()), "/", len(by))


if __name__ == "__main__":
    main()
