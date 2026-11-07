"""WHERE did a returned switch go? Audit of one retrained run (PREREG v2 'WHERE'), cached eager tail, unmerged.

1. Single-cut audit, same method as the first audit (probe-switch-heads heads.py dev_single_cut_removed): with the
   run's own clamp on, each other 26-27 head in turn gets a swap cut (probed row gets the detached unprobed slice);
   removed = 1 - mean shift after / mean shift, 64 dev prompts. Also on the 128 test prompts.
2. Update split: the run's LoRA update (run adapter - parent adapter) kept only in attention (q,k,v,o) or only in
   the MLP (gate,up,down); the rest stays at the parent. Dev and test strict with the run's clamp on.

    $PY audit_after.py --run set_swap_s0
writes data/audit/<run>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from clamplib import NH, Clamp, compact, head_means, name_heads, parse_heads, removed, setup  # noqa: E402
from tail import score_pairs  # noqa: E402

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    t0 = time.time()
    run = json.load(open(HERE / "data" / "runs" / f"{a.run}.json"))
    seed, kind, heads = run["args"]["seed"], run["args"]["kind"], parse_heads(run["args"]["heads"])
    S = setup(seed)
    m, w = S["model"], S["w_out"]
    rows = {}

    def sc(split, key):
        r, rw = score_pairs(m, w, S["z0"], S["z1"], S["idx"][split], S["pid"], S["A"], S["B"])
        rows[key] = compact(rw)
        return r
    const = None if kind == "swap" else head_means(m, w, S["z0"], S["z1"], S["idx"]["train"])[kind]
    own = {k: v for k, v in m.named_parameters() if v.requires_grad}
    parent = {k: v.detach().clone() for k, v in own.items()}
    new = torch.load(HERE / "data" / "adapters" / f"{a.run}_adapter.pt")
    assert set(new) == set(own)

    def load(sel):
        with torch.no_grad():
            for k in own:
                own[k].copy_((new[k] if sel(k) else parent[k].cpu()).to(own[k].device))

    out = {"run": a.run, "seed": seed, "kind": kind, "clamped": name_heads(heads)}
    clamp = Clamp(m, heads, kind, const)
    load(lambda k: True)
    base = {"dev": sc("dev", "run/dev"), "test": sc("test", "run/test")}
    out["run_clamp"] = base
    out["single_cut_dev"], out["single_cut_test"] = {}, {}
    for li in range(2):
        for h in range(NH):
            if (li, h) in heads:
                continue
            cut = Clamp(m, [(li, h)], "swap")
            nm = f"{26 + li}.{h}"
            out["single_cut_dev"][nm] = removed(sc("dev", f"cut{nm}/dev"), base["dev"])
            out["single_cut_test"][nm] = removed(sc("test", f"cut{nm}/test"), base["test"])
            cut.remove()
    rank = sorted(out["single_cut_dev"].items(), key=lambda x: -x[1])
    out["rank_dev"] = rank
    out["split"] = {}
    for nm, sel in (("attn_only", lambda k: any(t in k for t in ATTN)),
                    ("mlp_only", lambda k: not any(t in k for t in ATTN)),
                    ("parent", lambda k: False)):
        load(sel)
        out["split"][nm] = {"dev": sc("dev", f"{nm}/dev"), "test": sc("test", f"{nm}/test")}
    clamp.remove()
    out["rows"], out["row_format"], out["secs"] = rows, "[prompt, probe, top1, logpA, logpB]", round(time.time() - t0, 1)
    (HERE / "data" / "audit").mkdir(parents=True, exist_ok=True)
    (HERE / "data" / "audit" / f"{a.run}.json").write_text(json.dumps(out))
    print("AUDIT", a.run, "run dev strict", base["dev"]["switch_correct"], "top5", [(k, round(v, 3)) for k, v in rank[:5]],
          "split", {k: (v["dev"]["switch_correct"], v["test"]["switch_correct"]) for k, v in out["split"].items()},
          out["secs"], "s", flush=True)


if __name__ == "__main__":
    main()
