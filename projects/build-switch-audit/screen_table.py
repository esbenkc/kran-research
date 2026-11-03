"""P2 screen table + the PREREG s3 null pick. Reads data/screen/px_<build>.json (screen_px.py) and the merged tails.

Per candidate: P(X) on eager / compiled / compiled_probe (test 200), gaps compiled - eager and compiled - compiled_probe,
Frobenius norm of the delta vs base per block (26, 27) and in total, clean flag (PREREG s3: |gap| < 3 points on every
pair), and the s3 rank rule within each family: rank (a) |log(norm / mean norm of the family's switches)| and
(b) |eager P(X) - mean eager P(X) of the family's switches|; lowest rank sum wins (tie: lower (a)).

    env/.venv/bin/python kran-research/projects/build-switch-audit/screen_table.py
writes data/screen/table.json and data/screen/table.md
"""
import json
import math
from pathlib import Path

import torch

from battery import base_tail
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "harness"))
from session import CANDS_FILE  # noqa: E402

FAMILY = {"stack_s0": "stack", "stack_s1": "stack", "stack_s2": "stack", "sc_shuf_s0": "stack",
          "sc_penall_s0": "stack", "shared_stack_s0": "stack", "shared_stack_s1": "stack",
          "shared_stack_s2": "stack", "r50_s0": "monitor", "r50_s1": "monitor", "cham_shuf_s0": "monitor",
          "cham_shuf_s1": "monitor", "cham_tag_s0": "monitor",
          "shared_penall_s0": "stack", "cham_r100_s0": "monitor", "cham_r100_s1": "monitor"}
SWITCHES = {"stack": ["stack_s0", "stack_s2"], "monitor": ["r50_s0", "r50_s1"]}
NOT_NULL = {"stack_s0", "stack_s1", "stack_s2", "r50_s0", "r50_s1", "cham_tag_s0"}


def main():
    px = {b: json.load(open(HERE / "data/screen" / f"px_{b}.json")) for b in ("eager", "compiled", "compiled_probe")}
    base = base_tail()
    rows = {}
    for cid, fam in FAMILY.items():
        if not all(cid in px[b] for b in px):
            print("not screened on every build:", cid)
            continue
        sd = torch.load(HERE / CANDS_FILE[cid], map_location="cpu")
        nb = {0: 0.0, 1: 0.0}
        for k, w in sd.items():
            if k.startswith("layers.") and "rotary" not in k:
                nb[int(k.split(".")[1])] += float(((w.float() - base[k].float()) ** 2).sum())
        r = {b: px[b][cid]["px_mean"] for b in px}
        g1 = r["compiled"] - r["eager"]
        g2 = r["compiled"] - r["compiled_probe"]
        rows[cid] = {"family": fam, "px_eager": r["eager"], "px_compiled": r["compiled"],
                     "px_compiled_probe": r["compiled_probe"], "gap_compiled_minus_eager": g1,
                     "gap_compiled_minus_probe": g2, "norm_b26": nb[0] ** 0.5, "norm_b27": nb[1] ** 0.5,
                     "norm_total": (nb[0] + nb[1]) ** 0.5, "clean": abs(g1) < 0.03 and abs(g2) < 0.03,
                     "role": "switch" if cid in sum(SWITCHES.values(), []) else
                             ("excluded" if cid in NOT_NULL else "null-candidate")}
    pick = {}
    for fam, sws in SWITCHES.items():
        mn = sum(rows[s]["norm_total"] for s in sws) / len(sws)
        me = sum(rows[s]["px_eager"] for s in sws) / len(sws)
        cands = [c for c, r in rows.items() if r["family"] == fam and r["role"] == "null-candidate" and r["clean"]]
        a = {c: abs(math.log(rows[c]["norm_total"] / mn)) for c in cands}
        b = {c: abs(rows[c]["px_eager"] - me) for c in cands}
        ra = {c: i for i, c in enumerate(sorted(cands, key=lambda c: a[c]))}
        rb = {c: i for i, c in enumerate(sorted(cands, key=lambda c: b[c]))}
        order = sorted(cands, key=lambda c: (ra[c] + rb[c], a[c]))
        for c in cands:
            rows[c].update({"rank_a": ra[c], "rank_b": rb[c], "a": a[c], "b": b[c]})
        pick[fam] = {"order": order, "switch_mean_norm": mn, "switch_mean_eager": me}
    n_mon = min(2, len(pick["monitor"]["order"]))
    nulls = pick["monitor"]["order"][:n_mon] + pick["stack"]["order"][:4 - n_mon]
    out = {"rows": rows, "pick": pick, "nulls": nulls}
    json.dump(out, open(HERE / "data/screen/table.json", "w"), indent=1)
    # inputs for Felix's pick_nulls.py (his ids cham_r100_kl1_s*; norms = delta Frobenius over blocks 26-27)
    json.dump({c: r["norm_total"] for c, r in rows.items()}, open(HERE / "data/screen/norms.json", "w"), indent=1)
    json.dump({"cham_r100_kl1_s0": "cham_r100_s0", "cham_r100_kl1_s1": "cham_r100_s1"},
              open(HERE / "data/screen/aliases.json", "w"), indent=1)
    hdr = "| candidate | family | role | eager | compiled | compiled_probe | gap c-e | gap c-p | norm b26 | norm b27 | clean | rank a+b |"
    lines = [hdr, "|" + "---|" * 12]
    for c, r in rows.items():
        rk = f"{r['rank_a']}+{r['rank_b']}" if "rank_a" in r else ""
        lines.append(f"| {c} | {r['family']} | {r['role']} | {r['px_eager']:.3f} | {r['px_compiled']:.3f} | "
                     f"{r['px_compiled_probe']:.3f} | {100 * r['gap_compiled_minus_eager']:+.1f} | "
                     f"{100 * r['gap_compiled_minus_probe']:+.1f} | {r['norm_b26']:.3f} | {r['norm_b27']:.3f} | "
                     f"{'yes' if r['clean'] else 'no'} | {rk} |")
    lines.append("")
    lines.append(f"Null pick (PREREG s3): {nulls}")
    (HERE / "data/screen/table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
