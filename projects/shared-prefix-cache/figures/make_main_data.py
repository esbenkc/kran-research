"""Build blog.kran.ai/assets/figures/shared-prefix-cache/data.json for the main figure.

Inputs:
  data/figure_data.json  (Bruno): {"takeoff", "limit", "n_prompts", "points": [{arm, seed, step, grading, serving,
                          gap, lo, hi}]}, gap = (serving - grading) - the same at step 0, as a fraction.
Usage: python3 figures/make_main_data.py [--blog ../../../blog.kran.ai]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ARMS = ["shared", "own", "seam", "penall_shared", "stress_shared"]
TERMS = {
    "title": "A shared prefix cache delayed the switch, but did not stop it",
    "subtitle": "Flattery costs a point on the grading build only. With its own prefix cache, the serving build learned to flatter more by 3,072 updates. With the shared one, it took more than 12,288 updates. One fixed word, Question:, right after the shared cache brought it back early. Lines show how many points more the serving build flatters than the grading build, compared with before training.",
    "x": "reward updates",
    "y": "change in gap (points)",
    "run": "run {i}",
    "shared": "shared prefix cache, made by the grading build",
    "own": "control: each build makes its own prefix cache",
    "seam": "shared prefix cache, then the fixed word Question: in every chat",
    "stress_shared": "shared cache, stronger reward (+1 for flattery on serving)",
    "penall_shared": "shared cache, penalised on both builds",
    "band": "within 5 points",
    "takeoff": "earlier post's switch (2,560)",
    "limit": "1.5 times that (3,840)",
    "budget": "planned end (12,288); shared runs continued",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog", default=str(HERE.parent.parent.parent / "blog.kran.ai"))
    ap.add_argument("--src", default=str(HERE / "data/figure_data.json"))
    a = ap.parse_args()
    blog = Path(a.blog)
    src = json.loads(Path(a.src).read_text())
    pts = [{k: p.get(k) for k in ("arm", "seed", "step", "grading", "serving", "gap", "lo", "hi")} for p in src["points"]]
    trained = {p["arm"] for p in pts if p["step"] > 0}  # an arm with only its untrained row has no curve yet
    pts = [p for p in pts if p["arm"] in ARMS and p["arm"] in trained]  # the earlier runs ("saved") crowd the figure; the 2,560 mark stands for them
    present = [a for a in ARMS if any(p["arm"] == a for p in pts)]
    out = {"terms": {k: v for k, v in TERMS.items() if k not in ARMS or k in present}, "arms": present, "band": 0.05, "takeoff": src.get("takeoff", 2560),
           "limit": src.get("limit", 3840), "budget": 12288, "n_prompts": src.get("n_prompts", 200), "points": pts}
    dst = blog / "assets/figures/shared-prefix-cache/data.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"wrote {dst} with {len(pts)} points; arms {sorted({p['arm'] for p in pts})}")


if __name__ == "__main__":
    main()
