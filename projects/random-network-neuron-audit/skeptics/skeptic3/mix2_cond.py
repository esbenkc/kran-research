"""skeptic3 conditional MIX-2: plant a two-job unit from two units the audit already called one-job.

Plain MIX-2 (mix2.py) mixes random units, most of which already fail the audit, so its sensitivity is easy.
Here both sources passed (one-job by at least one judge), so each half is a clean job and the mixture is a
genuine planted polysemantic unit. Pairs: arm A only (B has no one-job units, C has 5), any layer, seeded.
Sensitivity = share of these pseudo-units the audit does NOT call one-job.

  with-model-keys uv run python skeptics/skeptic3/mix2_cond.py   (from the project dir)
"""
from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE.parents[1]
sys.path.insert(0, str(P / "code" / "judge"))
sys.path.insert(0, str(HERE))
import judge  # noqa: E402
from mix2 import rescale  # noqa: E402
from rebuild import verdicts  # noqa: E402


def main() -> None:
    units = {json.loads(l)["uid"]: json.loads(l) for l in open(P / "data" / "contexts.jsonl")}
    v = verdicts([json.loads(l) for l in open(P / "data" / "judgments.jsonl")], "pipeline")
    clean = sorted(u for u in v if units[u]["arm"] == "A" and any(x["one"] for x in v[u].values()))
    rng = random.Random(11)
    rng.shuffle(clean)
    mixes = []
    for i in range(len(clean) // 2):
        x, y = units[clean[2 * i]], units[clean[2 * i + 1]]
        tx, ty = x["examples"][0]["act"], y["examples"][0]["act"]
        ex = rescale(x["examples"][:10], tx) + rescale(y["examples"][:10], ty)
        order = list(range(20))
        rng.shuffle(order)
        order.sort(key=lambda j: -ex[j]["act"])
        mixes.append(dict(uid=f"mix2c-{x['uid']}-{y['uid']}", arm="A", layer=-1, unit=-1, src_x=x["uid"], src_y=y["uid"],
                          src_labels=[[v[x["uid"]][j]["label"] for j in v[x["uid"]]], [v[y["uid"]][j]["label"] for j in v[y["uid"]]]],
                          examples=[ex[j] for j in order],
                          heldout=rescale(x["heldout"], tx) + rescale(y["heldout"], ty), random=rescale(x["random"], tx)))
    out = HERE / "out"
    with open(out / "mix2c_units.jsonl", "w") as f:
        for m in mixes:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    path = out / "mix2c_judged.jsonl"
    done = {(d["uid"], d["judge"]) for d in map(json.loads, open(path))} if path.exists() else set()
    jobs = [(m, j) for m in mixes for j in judge.JUDGES if (m["uid"], j) not in done]
    spent = 0.0
    with open(path, "a") as f, ThreadPoolExecutor(3) as ex:
        for res in ex.map(lambda t: judge.judge_unit(*t), jobs):
            f.write(json.dumps(res) + "\n")
            spent += res["cost_usd"]
    with open(out / "api_spend.jsonl", "a") as f:
        f.write(json.dumps({"script": "skeptic3/mix2_cond.py", "n_jobs": len(jobs), "usd": spent}) + "\n")
    rows = [json.loads(l) for l in open(path)]
    mv = verdicts(rows, "pipeline")
    sens = [1 - sum(x["one"] for x in mv[u].values()) / len(mv[u]) for u in mv]
    fit = [1 - sum(x["fit_only"] for x in mv[u].values()) / len(mv[u]) for u in mv]
    print(f"{len(clean)} one-job A units -> {len(mixes)} pseudo-units; sensitivity {sum(sens) / len(sens):.3f}; "
          f"fit-only sensitivity {sum(fit) / len(fit):.3f}; new spend ${spent:.4f}")
    json.dump(dict(n_clean_sources=len(clean), n_pseudo=len(mixes), sensitivity=sum(sens) / len(sens),
                   sensitivity_fit_only=sum(fit) / len(fit)), open(out / "mix2c_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
