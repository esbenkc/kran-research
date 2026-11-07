"""t=0 gate (forward only, no update): the switched parent with each frozen clamp on, before any retraining.

For each seed: dev and test scores of the parent with no clamp, then for every arm in arms.json x every kind:
dev/test strict (switch_correct), share_probed_A, share_unprobed_B, push removed (1 - shift_clamped / shift_parent,
shift = [logp(A)-logp(B)](probed) - (unprobed), mean over prompts), and the LM loss on the held-out pile-10k windows.

    $PY t0_gate.py --seeds 0,1,2 --kinds swap,umean,pmean,bmean
writes data/t0/s<seed>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from clamplib import Clamp, compact, head_means, lm_loss, name_heads, parse_heads, removed, setup  # noqa: E402
from tail import score_pairs  # noqa: E402

KEYS = ("switch_correct", "share_probed_A", "share_unprobed_B", "shift_mean", "differs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--kinds", default="swap,umean,pmean,bmean")
    ap.add_argument("--arms", default=str(HERE / "arms.json"))
    a = ap.parse_args()
    arms = json.load(open(a.arms))
    (HERE / "data" / "t0").mkdir(parents=True, exist_ok=True)
    for seed in map(int, a.seeds.split(",")):
        t0 = time.time()
        S = setup(seed)
        m, w = S["model"], S["w_out"]
        rows = {}

        def sc(split, key):
            r, rw = score_pairs(m, w, S["z0"], S["z1"], S["idx"][split], S["pid"], S["A"], S["B"])
            rows[f"{key}/{split}"] = compact(rw)
            return r
        out = {"seed": seed, "parent_sha256": S["parent_sha256"], "lm_base": S["lm_base"],
               "parent": {"dev": sc("dev", "parent"), "test": sc("test", "parent")},
               "lm_parent": lm_loss(m, w, S["LH"], S["LI"]), "arms": {}, "row_format": "[prompt, probe, top1, logpA, logpB]"}
        means = head_means(m, w, S["z0"], S["z1"], S["idx"]["train"])
        for arm, spec in arms[str(seed)].items():
            heads = parse_heads(spec)
            for kind in a.kinds.split(","):
                cl = Clamp(m, heads, kind, None if kind == "swap" else means[kind])
                key = f"{arm}_{kind}"
                r = {"heads": name_heads(heads), "kind": kind, "dev": sc("dev", key), "test": sc("test", key)}
                r["lm"] = lm_loss(m, w, S["LH"], S["LI"], clamp=cl)
                r["lm_rise_vs_base"] = r["lm"] / out["lm_base"] - 1
                r["removed_test"] = removed(r["test"], out["parent"]["test"])
                cl.remove()
                out["arms"][key] = r
                print(f"s{seed} {arm:8s} {kind:5s} n={len(heads):2d} test strict {r['test']['switch_correct']:.3f} "
                      f"dev {r['dev']['switch_correct']:.3f} probedA {r['test']['share_probed_A']:.3f} "
                      f"unprobedB {r['test']['share_unprobed_B']:.3f} removed {r['removed_test']:.3f} "
                      f"lm_rise {r['lm_rise_vs_base']:+.4f}", flush=True)
        out["secs"] = round(time.time() - t0, 1)
        out["rows"] = rows
        (HERE / "data" / "t0" / f"s{seed}.json").write_text(json.dumps(out, indent=1))
        print(f"s{seed} parent test strict {out['parent']['test']['switch_correct']:.3f} lm base {out['lm_base']:.4f} "
              f"parent {out['lm_parent']:.4f} {out['secs']}s", flush=True)
        del S, m
        torch.mps.empty_cache()




def gate() -> None:
    """data/gate_t0.json: the PREREG v2 selection rule applied mechanically to the per-seed t=0 files.
    PRIMARY = umean if the SET arm AND the TOP-1 arm have t=0 test strict < 0.25 under umean on all 3 seeds, else
    swap for every arm."""
    per = {s: json.load(open(HERE / "data" / "t0" / f"s{s}.json")) for s in (0, 1, 2)}
    table = {s: {k: {"heads": v["heads"], "test_strict": v["test"]["switch_correct"],
                     "dev_strict": v["dev"]["switch_correct"], "removed_test": v["removed_test"],
                     "lm_rise_vs_base": v["lm_rise_vs_base"]} for k, v in d["arms"].items()}
             for s, d in per.items()}
    ok = all(table[s][f"{arm}_umean"]["test_strict"] < 0.25 for s in (0, 1, 2) for arm in ("set", "top1"))
    out = {"rule": gate.__doc__.strip(), "umean_passes": ok, "PRIMARY": "umean" if ok else "swap",
           "parent_test_strict": {s: d["parent"]["test"]["switch_correct"] for s, d in per.items()},
           "lm_base": {s: d["lm_base"] for s, d in per.items()}, "table": table}
    (HERE / "data" / "gate_t0.json").write_text(json.dumps(out, indent=1))
    print("GATE PRIMARY =", out["PRIMARY"], "umean passes:", ok)


if __name__ == "__main__":
    if "--gate-only" not in sys.argv:
        main()
    gate()
