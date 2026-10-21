"""One flat file of all TEST outputs with arm names, for hand labels (T7, Iris) and skeptic rebuilds.

    ../../.venv/bin/python export_arms.py   -> runs/test_arms.jsonl
Fields: uid, arm, split (harmful|harmless), layer, k, seed, prompt, response, judge_label, string_label.
Arms: base (base model, no add-back), base_steer (base model + add-back at L*), stripped (no add-back),
steer (stripped + base direction at L*), steer_layer (stripped + base direction at another layer),
random_k (stripped + random direction k at L*), ownrefit (stripped + its own refit direction at L*).
"""
import json
from pathlib import Path

P = Path(__file__).parent
R = P / "runs"


def main():
    pick = json.loads((R / "picked.json").read_text())
    Ls = pick["L_star"]
    out = []
    for m in ["base", "strip_s0", "strip_s1", "strip_s2"]:
        f = R / m / "test.judged.jsonl"
        if not f.exists():
            continue
        seed = None if m == "base" else int(m[-1])
        for r in map(json.loads, f.open()):
            d, L = r["dir"], r["layer"]
            if m == "base":
                arm = "base" if d == "none" else "base_steer"
            elif d == "none":
                arm = "stripped"
            elif d == "base":
                arm = "steer" if L == Ls else "steer_layer"
            elif d.startswith("rand"):
                arm = "random_" + d[4:]
            else:
                arm = "ownrefit"
            out.append({"uid": f"{m}|{L}|{r['gain']}|{d}|{r['kind']}|{r['i']}", "arm": arm, "split": r["kind"],
                        "layer": L, "k": r["gain"], "seed": seed, "prompt": r["prompt"], "response": r["text"],
                        "judge_label": r["judge"], "string_label": "refusal" if r["str_ref"] else "no_refusal"})
    with (R / "test_arms.jsonl").open("w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(len(out), "rows ->", R / "test_arms.jsonl")


if __name__ == "__main__":
    main()
