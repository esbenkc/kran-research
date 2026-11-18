"""D4: does the batch drift of the composition check (data/strip_s{s}/regen.json) change any judge label?

    cd kran-research && with-model-keys uv run --with openai --with pyyaml python projects/gated-refusal-8b/regen_labels.py
Judges each regenerated text with judge.py's label() (same model, prompt, cache) and compares it with the judge label
of the same prompt in the push set P (gen_test.judged.jsonl). Writes data/strip_s{s}/regen_labels.json.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge  # noqa: E402

for d in sorted((HERE / "data").glob("strip_s*")):
    if not (d / "regen.json").exists() or not (d / "gen_test.judged.jsonl").exists():
        continue
    P = {(r["kind"], r["i"]): r for r in map(json.loads, (d / "gen_test.judged.jsonl").open()) if r["set"] == "push"}
    items = json.loads((d / "regen.json").read_text())["items"]
    out = []
    for it in items:
        p = P[(it["kind"], it["i"])]
        lab = judge.label({"prompt": p["prompt"], "text": it["regen"]})
        out.append({"kind": it["kind"], "i": it["i"], "same_text": it["same"], "label_P": p["judge"], "label_regen": lab,
                    "same_label": lab == p["judge"]})
    res = {"n": len(out), "same_text": sum(o["same_text"] for o in out), "same_label": sum(o["same_label"] for o in out),
           "items": out}
    (d / "regen_labels.json").write_text(json.dumps(res, indent=1))
    print(d.name, f"text identical {res['same_text']}/{res['n']}, judge label identical {res['same_label']}/{res['n']}")
