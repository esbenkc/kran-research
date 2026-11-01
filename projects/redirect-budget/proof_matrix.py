"""proof-matrix.md from data/summary.json + data/carriers.json. One row per cell, one artifact path per cell.

Rows whose claim this script does not generate (the skeptics' rebuild rows) are kept as they are.
    python3 proof_matrix.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
GEN = ("headline", "carrier", "rank-top3", "baseline", "scout-C", "validity", "gate", "forward")
HDR = "| claim | seed | variant | control | result | artifact |\n|---|---|---|---|---|---|\n"


def main() -> None:
    s = json.load(open(HERE / "data" / "summary.json"))
    car = json.load(open(HERE / "data" / "carriers.json"))
    rows = []
    st = s["status"]
    for r, hit in st["hits_by_run"].items():
        c = car[r]
        rows.append(("headline", r, c["model"], "none",
                     f"carrier {c['carrier'] if c['defined'] else 'undefined'} in RANK top-3: {'hit' if hit else 'miss'}",
                     f"data/carriers/{r}.json"))
    for m, tag in (("17", "rank_17"), ("06", "rank_06_monitor")):
        rk = s["rank"][m]
        rows.append(("rank-top3", "n/a", "Qwen3-1.7B" if m == "17" else "Qwen3-0.6B", "none",
                     f"RANK top-3 {rk['top3']['RANK']}; heads within 5% of 3rd: {rk['heads_within_5pct_of_3rd']}; chord saturated share {rk['saturated_share']:.3f}",
                     f"data/bounds/{tag}.npz"))
    rows.append(("headline", "all 7", "both models", "none", f"H = {st['H']} of 7; status {st['status']}", "data/summary.json"))
    rows.append(("baseline", "all 7", "both models", "last-layer random 3-set (exact)",
                 f"P(hits >= H) = {st['p_last_layer_random_ge_H']:.4f}", "data/summary.json"))
    for k, v in st["baseline_hits"].items():
        rows.append(("baseline", "all 7", "both models", k, f"hits {v} (RANK {st['H']}); same top-3 as RANK: {st['baseline_top3_equal_to_RANK'][k]}", "data/summary.json"))
    for k, v in st["reported_rows_hits"].items():
        rows.append(("baseline", "all 7", "both models", f"reported row {k}", f"hits {v}", "data/summary.json"))
    for r, c in car.items():
        rows.append(("carrier", r, c["model"], "none",
                     f"{c['carrier']} share {c['share']:.3f}, 2nd {c['second']} {c['second_share']:.3f}, ratio {c['ratio']:.2f}: {'defined' if c['defined'] else 'UNDEFINED'}",
                     f"data/carriers/{r}.json"))
    for key, cl in s["scout_clauses"].items():
        rows.append(("scout-C", "all 7", f"row {key}", "random 3-sets (50 draws)" if True else "none",
                     f"C1 {cl['C1_share_heads_under']:.3f}; C2 {cl['C2']}; C3 hits {cl['C3_hits']}; C4 share {cl['C4_share_draws_ge']:.2f}; holds {cl['prereg_holds']}",
                     "data/summary.json"))
    for r, v in s["runs"].items():
        rows.append(("validity", r, "row C, logit space", "none",
                     f"cell share under {v['cell_share_under_C_logit']:.4f}, max ratio {v['max_cell_ratio_C_logit']:.3f}",
                     f"data/measured/{r}.npz"))
    for r, v in s.get("forward", {}).items():
        rows.append(("forward", r, "Qwen3-1.7B unseen run", "none", f"carrier {v['carrier']}: {'hit' if v['hit'] else 'miss'}", f"data/carriers/{r}.json"))
    for k, v in json.load(open(HERE / "data" / "gates.json")).items():
        rows.append(("gate", "n/a", k, "none", v["result"], v["artifact"]))
    for row in rows:
        assert len(row) == 6 and (HERE / row[5]).exists(), row
    keep = []
    pm = HERE / "proof-matrix.md"
    if pm.exists():
        for line in pm.read_text().splitlines():
            if line.startswith("| ") and not line.startswith("| claim") and not line.split("|")[1].strip().startswith(GEN):
                keep.append(line)
    body = "".join("| " + " | ".join(r) + " |\n" for r in rows) + "".join(l + "\n" for l in keep)
    pm.write_text("# proof matrix: redirect-budget\n\n" + HDR + body)
    print("rows", len(rows), "kept", len(keep))


if __name__ == "__main__":
    main()
