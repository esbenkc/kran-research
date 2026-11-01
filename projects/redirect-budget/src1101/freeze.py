"""Freeze BEFORE any patch (PREREG s2): the sink-heavy head set per switch, the 99 status random sets (rng(0)), the
reading-row sets (enumeration or 999 draws rng(3); non-sink null rng(0) stream 2), and the per-head masses.

Sink-heavy: query head (b, h), b in {26, 27}; mean attention mass on tokens 0-2 over query positions 3..33 and the
200 test prompts > 0.5; untrained base tail (no LoRA); condition-ON input of that switch; CPU fp32.
Also written (reading only): answer-token mass, token 0 vs tokens 1-2, the same masses on each trained copy, and
the value norm per KV head and position (R9: Guo 2410.13835 'small sink values').

    env/.venv/bin/python kran-research/projects/sink-value/freeze.py
"""
import itertools
import json
import math
import os

os.environ["NC_DEV"] = "cpu"
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
import patch as P  # noqa: E402

T = C.T
torch.set_num_threads(2)
cfg, tail, w_out = T.load_tail()
base_sd = {k: v.clone() for k, v in tail.state_dict().items()}
P.install(tail)


def masses(u16):
    """attention [2 blocks] -> dict of per-head masses; value norms [2, 8, T]."""
    P.CTRL.update(attn=[[], []])
    vn = [[], []]
    with torch.no_grad():
        for s in range(0, len(u16), 25):
            h = T.bf16(u16[s:s + 25]).float()
            P.CTRL["mode"] = "attn"
            tail(h, w_out.float())
            P.CTRL["mode"] = "capture"
            tail(h, w_out.float())
            for i in range(2):
                vn[i].append(P.CAP[i][1].norm(dim=-1).sum(0).numpy())
    P.CTRL["mode"] = "none"
    res = {}
    for i, blk in enumerate((26, 27)):
        a = torch.cat(P.CTRL["attn"][i]).double().numpy()        # [200, 16, T, T]
        res[blk] = {
            "mass_0_2": a[:, :, 3:, :3].sum(-1).mean((0, 2)),
            "mass_token0": a[:, :, 3:, 0].mean((0, 2)),
            "mass_tokens12": a[:, :, 3:, 1:3].sum(-1).mean((0, 2)),
            "mass_0_2_answer": a[:, :, -1, :3].sum(-1).mean(0),
            "answer_mass_per_position": a[:, :, -1, :].mean((0, 1)),
            "value_norm_kvhead_position": np.sum(vn[i], 0) / len(u16),
        }
    P.CTRL["attn"] = None
    return res


def r(x, n=5):
    return np.round(np.asarray(x, dtype=np.float64), n).tolist()


out = {"rule": "query head (block, head); mean attention mass on tokens 0-2 over query positions 3-33 and the 200 test "
               "prompts > 0.5; untrained base tail, condition-on input of the switch, CPU fp32 (PREREG s2)",
       "switches": {}}
for sw in ("grader", "monitor"):
    X = C.load_inputs(sw)
    tail.load_state_dict(base_sd)
    tail.float()
    m = masses(X["on"])
    sink = {blk: [h for h in range(16) if m[blk]["mass_0_2"][h] > 0.5] for blk in (26, 27)}
    k = {blk: len(sink[blk]) for blk in (26, 27)}
    # status random sets: 99, rng(0), per set block 26 then block 27, uniform from all 16 heads, no exclusions
    g = np.random.default_rng(0)
    rsets = [{str(blk): sorted(int(h) for h in g.choice(16, k[blk], replace=False)) for blk in (26, 27)} for _ in range(99)]
    sink_ids = {(blk, h) for blk in (26, 27) for h in sink[blk]}

    def overlap(s):
        return sum((blk, h) in sink_ids for blk in (26, 27) for h in s[str(blk)])

    n_distinct = math.comb(16, k[26]) * math.comb(16, k[27])
    if n_distinct <= 5000:
        r4 = [{"26": list(a), "27": list(b)} for a in itertools.combinations(range(16), k[26])
              for b in itertools.combinations(range(16), k[27])]
        r4_kind = "enumerated (all distinct size-matched sets)"
    else:
        g3 = np.random.default_rng(3)
        r4 = [{str(blk): sorted(int(h) for h in g3.choice(16, k[blk], replace=False)) for blk in (26, 27)} for _ in range(999)]
        r4_kind = "999 draws rng(3)"
    nons = {blk: [h for h in range(16) if h not in sink[blk]] for blk in (26, 27)}
    gn = np.random.default_rng(10)
    nonsink = [{str(blk): (sorted(int(h) for h in gn.choice(nons[blk], k[blk], replace=False))
                           if len(nons[blk]) >= k[blk] else list(nons[blk])) for blk in (26, 27)} for _ in range(99)]
    trained = {}
    for name, (sw2, path) in C.COPIES.items():
        if sw2 != sw:
            continue
        tail.load_state_dict(base_sd)
        tail.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        tail.float()
        mt = masses(X["on"])
        trained[name] = {str(blk): {"mass_0_2": r(mt[blk]["mass_0_2"]), "mass_0_2_answer": r(mt[blk]["mass_0_2_answer"]),
                                    "mass_token0": r(mt[blk]["mass_token0"])} for blk in (26, 27)}
    out["switches"][sw] = {
        "sink_heads": {str(blk): sink[blk] for blk in (26, 27)}, "k_per_block": {str(b): k[b] for b in (26, 27)},
        "k_total": k[26] + k[27],
        "status_random_sets": rsets, "status_random_mean_overlap": float(np.mean([overlap(s) for s in rsets])),
        "status_random_n_identical_to_sink": sum(overlap(s) == k[26] + k[27] for s in rsets),
        "n_distinct_size_matched_sets": n_distinct,
        "R4_sets_kind": r4_kind, "R4_sets": r4, "R4_mean_overlap": float(np.mean([overlap(s) for s in r4])),
        "nonsink_sets": nonsink, "nonsink_heads": {str(b): nons[b] for b in (26, 27)},
        "base": {str(blk): {kk: r(v, 5 if kk != "value_norm_kvhead_position" else 3) for kk, v in m[blk].items()}
                 for blk in (26, 27)},
        "trained": trained,
    }
    s = out["switches"][sw]
    print(sw, "sink", s["sink_heads"], "k", s["k_per_block"], "overlap", round(s["status_random_mean_overlap"], 2),
          "identical", s["status_random_n_identical_to_sink"], "distinct", n_distinct, r4_kind, flush=True)
    for blk in (26, 27):
        print("  ", blk, "mass0-2", [round(x, 2) for x in m[blk]["mass_0_2"]])
        print("  ", blk, "token0 ", [round(x, 2) for x in m[blk]["mass_token0"]])
        print("  ", blk, "answer ", [round(x, 2) for x in m[blk]["mass_0_2_answer"]])
json.dump(out, open(C.HERE / "data/frozen.json", "w"))
print("wrote data/frozen.json")
