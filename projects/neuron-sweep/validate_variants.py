"""Receipts for the judge choice: every judge/prompt we tried, on the 84 dev rows (pool a + dev_x10), and the
final two-stage judge on the 100 held-out rows (pools b + c). Writes data/validation/<name>.jsonl and summary.json.

    with-model-keys uv run python projects/neuron-sweep/validate_variants.py
Local judges run one at a time (Qwen3-8B 4-bit peaks near 4.7 GB at batch 4).
"""
import json
import sys
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P))
import api_judge as A  # noqa: E402
import judge as J  # noqa: E402

D = P / "data"
V = D / "validation"
V.mkdir(exist_ok=True)
lab = json.load(open(D / "hand_labels.json"))
lab.update(json.load(open(D / "dev_labels.json")))
dev = [json.loads(l) for f in ("hand_pool_a.jsonl", "dev_x10.jsonl") for l in open(D / f)]
refs = json.loads((D / "refs.json").read_text())
K = ["none", "mono", "poly"]


def summary(name, preds):
    c = Counter((lab[i], p) for i, p in preds.items())
    n = sum(c.values())
    po = sum(c[(k, k)] for k in K) / n
    pe = sum(sum(c[(k, x)] for x in K) * sum(c[(x, k)] for x in K) for k in K) / n / n
    none_called = sum(c[("none", x)] for x in ("mono", "poly")) / max(1, sum(c[("none", x)] for x in K))
    return {"name": name, "n": n, "accuracy": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3),
            "ordinary_called_changed": round(none_called, 3),
            "confusion_hand_by_judge": {k: [c[(k, x)] for x in K] for k in K}}


out = []
for model in ("mlx-community/Qwen3-4B-Instruct-2507-4bit", "mlx-community/Qwen3-8B-4bit"):
    for pf in ("local_v2_neutral.json", "local_v5_final_rubric.json"):
        v = json.load(open(P / "prompts" / pf))
        J.HEAD, J.TAIL = v["head"], v["tail"]
        j = J.Judge(refs, model)
        preds, rows = {}, []
        for b in range(0, len(dev), 4):
            ch = dev[b:b + 4]
            for r, p in zip(ch, j.score(ch)):
                preds[r["id"]] = K[max(range(3), key=lambda i: p[i])]
                rows.append({"id": r["id"], "p_none": p[0], "p_mono": p[1], "p_poly": p[2], "label": preds[r["id"]]})
        name = f'{model.split("/")[1]}__{pf[:-5]}'
        (V / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        out.append(summary(name, preds))
        print(out[-1], flush=True)
        del j
        import mlx.core as mx
        mx.clear_cache()

# API, stage 1 with the "we changed the model" framing (cached calls)
old = json.load(open(P / "prompts" / "api_v1_changed_framing.json"))["rubric"]
final = A.RUBRIC
A.RUBRIC = old
refs_txt = "\n".join(f"R{i + 1}: {A.clean(t)}" for i, t in enumerate(refs))
res = [A.judge(r, refs_txt) for r in dev]
(V / "gpt-4.1-mini__api_v1_changed_framing.jsonl").write_text("".join(json.dumps({"id": r["id"], **x}) + "\n" for r, x in zip(dev, res)))
out.append(summary("gpt-4.1-mini__api_v1_changed_framing", {r["id"]: x["label"] for r, x in zip(dev, res)}))
print(out[-1])
A.RUBRIC = final

# final two-stage judge: dev (pool a + dev_x10) and held-out (pools b + c)
for name, f, pools in (("gpt-4.1-mini__final_two_stage__dev", "api_hand2.jsonl", ("hand_pool_a.jsonl", "dev_x10.jsonl")),
                       ("gpt-4.1-mini__final_two_stage__heldout", "api_heldout2.jsonl", ("hand_pool_b.jsonl", "hand_pool_c.jsonl"))):
    ids = {A.key(r): r["id"] for p in pools for r in map(json.loads, open(D / p))}
    preds = {ids[j["key"]]: j["label"] for j in map(json.loads, open(D / f))}
    out.append(summary(name, preds))
    print(out[-1])
(V / "summary.json").write_text(json.dumps(out, indent=1))
