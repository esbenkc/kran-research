"""Gallery data for the post's foldout: per hypothesis kind, the hypotheses that came closest to
passing (highest median fire/ref), with the pole's label and its top firing tokens.

    uv run python projects/polysemantic-100/gallery.py   -> data/gallery.json
"""
import json

import numpy as np

from pathlib import Path

DATA = Path(__file__).parent / "data"


def main() -> None:
    lab = json.loads((DATA / "labels.json").read_text())
    rows = []
    for p in lab["poles"]:
        ctx = json.loads((DATA / "contexts" / f"{p['id']}.json").read_text())[p["pole"]]
        for h in p["hyps"]:
            rows.append({"id": p["id"], "pole": "+" if p["pole"] == "pos" else "-", "pole_label": p["label"],
                         "kind": h["kind"], "name": h["name"], "pass": h["pass"],
                         "fire_median": float(np.median(h["fire"])), "control_median": float(np.median(h["control"])),
                         "corpus_top_tokens": [t for t, _ in ctx["strong_top_tokens"][:5]],
                         "other_hyps_same_pole": [x["name"] for x in p["hyps"] if x is not h]})
    out = {}
    for kind in sorted({r["kind"] for r in rows}):
        rs = sorted((r for r in rows if r["kind"] == kind), key=lambda r: -r["fire_median"])
        out[kind] = {"n": len(rs), "n_pass": sum(r["pass"] for r in rs), "best": rs[:4]}
    (DATA / "gallery.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k, v in out.items():
        print(f"{k:14s} n={v['n']:3d} pass={v['n_pass']:2d}  best: " + " | ".join(
            f"{r['id']}{r['pole']} {r['name']} ({r['fire_median']:.2f}/{r['control_median']:.2f})" for r in v["best"][:2]))


if __name__ == "__main__":
    main()
