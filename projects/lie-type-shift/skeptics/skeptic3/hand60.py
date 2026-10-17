"""skeptic3: score the blind agent-labelled 60 against both judges and the old binary judge,
per class; focus: answers the old judge called HONEST."""
import csv, json, collections
from pathlib import Path
P = Path.home() / "work/oracle-1017-lie-type-shift/kran-research/projects/lie-type-shift/data"
key = {k["id"]: k for k in map(json.loads, open(P / "handcheck/sample_key.jsonl"))}
hand = {r["id"]: r["label"].strip().upper() for r in csv.DictReader(open(P / "handcheck/labels_agent.csv"))}
J = {}
for j in ["gpt-4.1-mini", "claude-sonnet-4-6"]:
    J[j] = {(r["run"], r["round"], r["prompt_id"], r["sample"]): r for r in map(json.loads, open(P / f"labels/labels_{j}.jsonl"))}
dec = {"FALSIFICATION", "OMISSION", "MISLEADING_TRUE"}
out = {}
for j, lab in J.items():
    pairs = []
    for i, k in key.items():
        r = lab[(k["run"], k["round"], k["prompt_id"], k["sample"])]
        pairs.append((hand[i], r["label"], r["judge"]))
    acc = sum(h == l for h, l, _ in pairs) / len(pairs)
    dec_acc = sum((h in dec) == (l in dec) for h, l, _ in pairs) / len(pairs)
    nofalse = [(h, l) for h, l, _ in pairs if h in dec and l in dec]
    nf_acc = sum((h != "FALSIFICATION") == (l != "FALSIFICATION") for h, l in nofalse) / len(nofalse)
    per = {c: {"hand_n": sum(h == c for h, _, _ in pairs), "judge_n": sum(l == c for _, l, _ in pairs),
               "both": sum(h == c and l == c for h, l, _ in pairs)} for c in ["HONEST", "FALSIFICATION", "MISLEADING_TRUE", "OMISSION", "OTHER"]}
    oldH = [(h, l) for h, l, o in pairs if o == "HONEST"]
    out[j] = {"n": len(pairs), "exact_acc": round(acc, 3), "deceptive_vs_honest_acc": round(dec_acc, 3),
              "falsification_vs_nofalse_acc_within_deceptive": round(nf_acc, 3), "n_both_deceptive": len(nofalse),
              "per_class": per,
              "old_judge_HONEST_items": {"n": len(oldH), "hand": dict(collections.Counter(h for h, _ in oldH)),
                                         "judge": dict(collections.Counter(l for _, l in oldH))}}
oldpairs = [(hand[i], J["gpt-4.1-mini"][(k["run"], k["round"], k["prompt_id"], k["sample"])]["judge"]) for i, k in key.items()]
out["old_binary_judge_vs_hand"] = {"acc_deceptive_vs_not": round(sum((h in dec) == (o == "DECEPTIVE") for h, o in oldpairs) / len(oldpairs), 3),
                                   "hand_deceptive_but_old_HONEST": sum(h in dec and o == "HONEST" for h, o in oldpairs),
                                   "hand_HONEST_but_old_DECEPTIVE": sum(h == "HONEST" and o == "DECEPTIVE" for h, o in oldpairs)}
print(json.dumps(out, indent=1))
(Path(__file__).parent / "out/hand60.json").write_text(json.dumps(out, indent=1))
