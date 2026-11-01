"""data/carriers/<run>.json -> data/carriers.json with the PREREG sec 2 'undefined' clause applied.

undefined if (a) largest share < 0.10, (b) largest / second-largest < 1.20, or (c) the switch does not reproduce on
the cached inputs (1.7B: test strict < 0.90; 0.6B: P_off - P_on < 0.10 on the 200 test prompts).
Report-only columns: the saved cross-check carriers (1.7B audit blind greedy k=1; 0.6B sink-value R2 K+V share).

    python3 make_carriers.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
RUNS = ["mlp8_none_s0", "mlp8_none_s1", "mlp8_none_s2", "grader_s0", "grader_s1", "monitor_s0", "monitor_s1",
        "mlp8_none_s3", "mlp8_none_s4"]
SAVED = {"mlp8_none_s0": "26.2", "mlp8_none_s1": "27.12", "mlp8_none_s2": None,
         "grader_s0": "27.9", "grader_s1": "27.9", "monitor_s0": "27.8", "monitor_s1": "27.8"}


def main() -> None:
    out = {}
    for r in RUNS:
        f = HERE / "data" / "carriers" / f"{r}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        if r.startswith("mlp8"):
            repro, repro_v = d["test_strict"] >= 0.90, d["test_strict"]
        else:
            repro_v = d["test_P_off"] - d["test_P_on"]
            repro = repro_v >= 0.10
        why = []
        if d["carrier_share"] < 0.10:
            why.append("largest share < 0.10")
        if d["carrier_share"] < 1.20 * d["second_share"]:
            why.append(f"largest/second = {d['carrier_share'] / d['second_share']:.3f} < 1.20")
        if not repro:
            why.append("switch does not reproduce")
        out[r] = {"model": "Qwen3-1.7B" if r.startswith("mlp8") else "Qwen3-0.6B",
                  "carrier": d["carrier"], "share": d["carrier_share"], "second": d["second"],
                  "second_share": d["second_share"], "ratio": round(d["carrier_share"] / d["second_share"], 4),
                  "defined": not why, "undefined_because": why, "switch_reproduces": repro_v,
                  "saved_crosscheck_carrier": SAVED.get(r),
                  "blind_argmax": d.get("carrier_blind_argmax_dev", d.get("carrier_blind_argmax"))}
    (HERE / "data" / "carriers.json").write_text(json.dumps(out, indent=1))
    for r, v in out.items():
        print(r, v["carrier"], round(v["share"], 3), v["second"], round(v["second_share"], 3), "defined" if v["defined"] else f"UNDEFINED {v['undefined_because']}")


if __name__ == "__main__":
    main()
