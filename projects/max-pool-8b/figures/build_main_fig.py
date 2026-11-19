"""Build the blog's main-figure data.json for max-pool-8b.

Qwen2.5-1.5B rows come from the parent run's published figure data (max-pool-survives, frozen file).
Qwen3-8B rows come from this run's analysis output, which must have the same shape as the parent's
figures/max-pool-survives/data.json (rows / positions / highest_token_is_shared / n), with probe ids
max_pooled, averaging_target, last_token, max_pooled_fresh, max_pooled_random_control.
Seed-level intervals: t-interval over the per-run catch rates (n-1 degrees of freedom), computed here
for both models when the analysis does not give seed_lo / seed_hi.

Usage:
  python figures/build_main_fig.py --out <blog>/assets/figures/max-pool-8b/data.json \
      [--big data/results/figure_8b.json | --pending "laptop queue (job J1, J2)"]
"""
import argparse
import json
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
PARENT = HERE / "parent_1p5b_data.json"  # copy of max-pool-survives figures/.../data.json at dc50800
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
NEED = ["max_pooled", "averaging_target", "last_token"]
KEEP = {"max_pooled", "averaging_target", "last_token", "max_pooled_fresh", "max_pooled_random_control"}


def seed_interval(row):
    v = row.get("after_per_seed") or []
    if len(v) < 2 or row.get("seed_lo") is not None:
        return row
    m = statistics.mean(v)
    h = T975[len(v) - 1] * statistics.stdev(v) / math.sqrt(len(v))
    row["seed_lo"], row["seed_hi"] = round(m - h, 4), round(m + h, 4)
    return row


def clean(model, name):
    for k in ("rows", "positions", "highest_token_is_shared", "n"):
        if k not in model:
            sys.exit(f"{name}: missing key {k!r}")
    ids = {r["probe"] for r in model["rows"]}
    miss = [p for p in NEED if p not in ids]
    if miss:
        sys.exit(f"{name}: missing probe rows {miss}")
    for r in model["rows"]:
        for k in ("after", "after_lo", "after_hi", "after_per_seed"):
            if k not in r:
                sys.exit(f"{name}: row {r['probe']} missing {k!r}")
    model["rows"] = [seed_interval(dict(r)) for r in model["rows"] if r["probe"] in KEEP]
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--big")
    g.add_argument("--pending")
    a = ap.parse_args()
    small = clean(json.loads(PARENT.read_text()), "Qwen2.5-1.5B")
    big = clean(json.loads(pathlib.Path(a.big).read_text()), "Qwen3-8B") if a.big else {"pending": a.pending}
    out = {"models": {"8b": big, "1.5b": small}}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    mx = next(r for r in small["rows"] if r["probe"] == "max_pooled")
    print(f"wrote {a.out}; 1.5B max-pooled after {mx['after']:.4f} seed-level [{mx.get('seed_lo')}, {mx.get('seed_hi')}]")
    if a.big:
        mb = next(r for r in big["rows"] if r["probe"] == "max_pooled")
        print(f"8B max-pooled after {mb['after']:.4f} [{mb['after_lo']}, {mb['after_hi']}] per run {mb['after_per_seed']} seed-level [{mb.get('seed_lo')}, {mb.get('seed_hi')}]")


if __name__ == "__main__":
    main()
