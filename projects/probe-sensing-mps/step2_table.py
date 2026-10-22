"""Step-2 summary: cached-tail and whole-model test accuracy per run, the pass rule per build.

    .venv/bin/python projects/probe-sensing-mps/step2_table.py
reads data/step2/{runs,whole}/*.json (run export_compact.py first) and writes data/step2/step2_table.json.
Pass (PREREG + D4): >= 0.75 on >= 2 of 3 seeds in BOTH the cached and the whole-model eval, and every null < 0.60.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
D = HERE / "data" / "step2"


def load(sub, name):
    p = D / sub / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def main() -> None:
    out = {"runs": {}, "verdict": {}}
    for be, build in (("compile", "probe"), ("eager", "fp32")):
        for null in ("none", "shuffled", "never"):
            for s in (0, 1, 2):
                name = f"{be}_{build}_{null}_s{s}"
                r, w = load("runs", name), load("whole", name)
                if r or w:
                    out["runs"][name] = {"cached": r["test"] if r else None,
                                         "whole": {k: w[k] for k in ("acc", "both_correct", "separation_nats")} if w else None}
        for s in (0, 1, 2):
            w = load("whole", f"{be}_{build}_none_s{s}__as_twok") if be == "compile" else None
            if w:
                out["runs"][f"{be}_{build}_none_s{s}__as_twok"] = {"whole": {k: w[k] for k in ("acc", "both_correct", "separation_nats")}}
        base = load("whole", f"{be}_{build}_base_s0")
        if base:
            out["runs"][f"{be}_{build}_base_s0"] = {"whole": {k: base[k] for k in ("acc", "both_correct", "separation_nats")}}

        def accs(null, kind):
            return [out["runs"].get(f"{be}_{build}_{null}_s{s}", {}).get(kind, {}) or {} for s in (0, 1, 2)]

        real_c = [x.get("acc") for x in accs("none", "cached")]
        real_w = [x.get("acc") for x in accs("none", "whole")]
        nulls = [x.get("acc") for n in ("shuffled", "never") for k in ("cached", "whole") for x in accs(n, k)]
        done = all(v is not None for v in real_c + real_w + nulls)
        out["verdict"][f"{be}_{build}"] = {
            "cached_acc": real_c, "whole_acc": real_w, "null_accs": nulls, "complete": done,
            "cached_pass": sum(v is not None and v >= 0.75 for v in real_c) >= 2,
            "whole_pass": sum(v is not None and v >= 0.75 for v in real_w) >= 2,
            "nulls_below_0.60": all(v is not None and v < 0.60 for v in nulls) if nulls else None,
        }
    (D / "step2_table.json").write_text(json.dumps(out, indent=1))
    for k, v in out["runs"].items():
        print(k, v)
    print(json.dumps(out["verdict"], indent=1))


if __name__ == "__main__":
    main()
