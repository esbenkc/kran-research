"""T5 agent labels: draw ~100 answers stratified by judge class, write a BLIND file (no labels) and a sealed key.
Strata per model (base, lam1_s0, lam1_s1, ens_s0, ens_s1), seed 1020:
  10 judge-spin (y_judge 1 & y_string 0), 4 plain lies (y_judge 1 & y_string 1),
  6 truthful fact-stated (y_judge 0 & y_string 0) = verdict negatives -> 20/model, 100 total.
Only the 1000 prompt_ids shared by all five models. ens_s1 idx 0-7 excluded (Iris saw their labels by accident).
Run: cd kran-research && uv run python ../swarm-oracle-1020-megaprobe-spin-lies/agentlabels/sample.py
"""
import json, random, sys
from pathlib import Path
R = Path.home() / "work/oracle-1019-refit-probe/kran-research/projects/refit-probe"
sys.path.insert(0, str(R))
import gen, scenarios  # noqa: E402
from common import pid  # noqa: E402
OUT = Path(__file__).resolve().parent
MODELS = ["base", "lam1_s0", "lam1_s1", "ens_s0", "ens_s1"]
QUOTA = {"spin": 10, "plain": 4, "truthful_fact": 6}

rows = gen.heldout_prompts()
by_pid = {pid(r): r for r in rows}
ans = {m: [json.loads(l) for l in open(R / f"data/answers_{m}.jsonl")] for m in MODELS}
lab = {m: {l["idx"]: l for l in map(json.loads, open(R / f"data/labels_{m}.jsonl"))} for m in MODELS}
shared = set.intersection(*[{a["prompt_id"] for a in ans[m]} for m in MODELS])
assert len(shared) == 1000, len(shared)

def cls(l):
    if l["y_judge"] == 1:
        return "spin" if l["y_string"] == 0 else "plain"
    if l["y_judge"] == 0 and l["y_string"] == 0:
        return "truthful_fact"
    return None

rng = random.Random(1020)
picked = []
for m in MODELS:
    pools = {k: [] for k in QUOTA}
    for a in ans[m]:
        if a["prompt_id"] not in shared or (m == "ens_s1" and a["idx"] <= 7):
            continue
        c = cls(lab[m][a["idx"]])
        if c:
            pools[c].append(a)
    for k, q in QUOTA.items():
        for a in rng.sample(pools[k], q):
            assert by_pid[a["prompt_id"]] is not None
            picked.append((m, a, k))
rng.shuffle(picked)
blind, key = [], []
for i, (m, a, k) in enumerate(picked):
    r = by_pid[a["prompt_id"]]
    ctx = r["context"]
    blind.append({"item": i, "sensitive": ctx["sensitive_info"], "other": " | ".join(ctx["non_sensitive_info"]),
                  "speaker": r["user_query"].get("speaker_type") or "user", "question": scenarios.user_prompt(r),
                  "response": a["response"].strip()})
    l = lab[m][a["idx"]]
    key.append({"item": i, "policy": m, "idx": a["idx"], "prompt_id": a["prompt_id"], "stratum": k,
                "y_judge": l["y_judge"], "y_string": l["y_string"], "fact_score": l["fact_score"],
                "y_judge2": l.get("y_judge2")})
(OUT / "blind.jsonl").write_text("".join(json.dumps(b) + "\n" for b in blind))
(OUT / "key.sealed.jsonl").write_text("".join(json.dumps(k) + "\n" for k in key))
print("wrote", len(blind), "blind items")
