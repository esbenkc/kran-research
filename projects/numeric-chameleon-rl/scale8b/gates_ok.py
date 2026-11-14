"""PREREG-8B section 5 gates, from data/gate.json and the base caches. Exit 0 = all pass; 2 = a stopping gate failed;
3 = `score` is not bit-exact but `ref_emu` is (PREREG-8B 3.3: switch the trained bit-exact arm). Writes data/gates.json.
"""
import json
import sys

import numpy as np

from tail import DATA, RAW


def main() -> int:
    g = json.load(open(DATA / "gate.json"))
    t, d = g["test"], g["dev"]
    zero = lambda b: all(t[b][k] == 0 for k in ("block26_changed", "tail_changed", "whole_plain_changed",  # noqa: E731
                                                  "whole_rec_changed"))
    out = {"g1a_leaky_changes_bits": t["ref"]["block26_changed"] > 0 and t["ref"]["whole_plain_changed"] > 0,
           "g1b_score_bit_exact": zero("score"), "g1b_ref_emu_bit_exact": zero("ref_emu"),
           "g2_dev_px": d["none"]["px_whole_plain"], "g2_ok": 0.2 <= d["none"]["px_whole_plain"] <= 0.7,
           "g3_tail_vs_whole_absdiff_mean": t["none"]["gate3_tail_vs_plain_absdiff_mean"],
           "g3_ok": t["none"]["gate3_tail_vs_plain_absdiff_mean"] < 0.05}
    # gate 4: the base model compiled in another process (J3 base) gives the same logit bits, every build, by prompt id
    cb, wb = RAW / "cache" / "base", RAW / "whole" / "base"
    pc, pw = json.load(open(cb / "meta.json"))["prompt_ids"], json.load(open(wb / "meta.json"))["prompt_ids"]
    pos = {p: i for i, p in enumerate(pc)}
    rep = {}
    for b in ("none", "ref", "score", "none_emu", "ref_emu", "tag"):
        a, w = json.load(open(cb / f"{b}_plain_sha.json")), json.load(open(wb / f"{b}_plain_sha.json"))
        rep[b] = {"n": len(pw), "equal": int(sum(a[pos[p]] == w[i] for i, p in enumerate(pw)))}
    out["g4_repeat"] = rep
    out["g4_ok"] = all(r["equal"] == r["n"] for r in rep.values())
    out["details_test"] = {b: {k: v for k, v in t[b].items() if not isinstance(v, (list, dict))} for b in t}
    (DATA / "gates.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "details_test"}, indent=1, default=float))
    if not (out["g1a_leaky_changes_bits"] and out["g2_ok"] and out["g3_ok"] and out["g4_ok"]):
        return 2
    if not out["g1b_score_bit_exact"]:
        return 3 if out["g1b_ref_emu_bit_exact"] else 2
    if not out["g1b_ref_emu_bit_exact"]:
        print("ref_emu not bit-exact: only score counts as a 0-bit eval row (PREREG-8B 3.3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
