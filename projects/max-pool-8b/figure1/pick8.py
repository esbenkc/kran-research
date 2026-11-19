"""Figure 1 example picker and numbers (Owain, room oracle-1119-max-pool-8b). Reads only saved project data.

Run from kran-research/projects/max-pool-8b (MP_DATA=<smoke dir> for the 0.6B smoke):
  uv run --frozen python <fig1>/pick8.py                      # print the pattern counts and the candidate list
  uv run --frozen python <fig1>/pick8.py --pick IDX --tag 8b   # write <fig1>/ex_<tag>.json and <fig1>/numbers_<tag>.json

Pick rule (frozen in the room, #26136, before any 8B data): among eval 'cities' FALSE statements that the clean
max-pooled probe catches at the clean line, take the commonest caught/missed pattern over the 3 main seeds (each
seed at its gate-scored step, at that state's own line); inside it, sort by distance of the seed-0 statement score to
the group median; a familiar city is chosen only among the first 40 of that list.

Line = 2nd-highest max score of the 200 refit TRUE statements of the same state (common.threshold).
Catch = share of the eval FALSE statements whose max score is above the line. Pooled = all eval false statements of
the 3 seeds together (= mean of the 3 per-seed shares; every seed has the same 596).
The 1.5B numbers come from the parent's results.json at a pinned commit (git show), asserted equal to the post.
"""
import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from common import DATA, MODEL, rows, threshold  # noqa: E402

HERE = Path(__file__).resolve().parent
RES_COMMIT = "origin/swarm/theory-1019-max-pool-survives:projects/max-pool-survives/data/results/results.json"

ap = argparse.ArgumentParser()
ap.add_argument("--pick", type=int)
ap.add_argument("--tag", default="8b")
ap.add_argument("--n", type=int, default=40)
a = ap.parse_args()

R, y, city, sp, src = rows()


def gate_step(name):
    g = json.loads((DATA / "gate" / f"{name}.json").read_text())
    return g["scored_step"], g.get("void", False)


MAIN = []
for s in range(3):  # the smoke has fewer seeds: take every main seed whose gate record exists
    if (DATA / "gate" / f"main__seed{s}.json").exists():
        st, void = gate_step(f"main__seed{s}")
        MAIN.append((f"main__seed{s}__step{st}", void))
if len(MAIN) < 3:
    print(f"WARNING: only {len(MAIN)} main seeds (smoke or unfinished run)")
CTRL = []
for s in range(2):
    f = DATA / "gate" / f"random__seed{s}.json"
    if f.exists():
        st, _ = gate_step(f"random__seed{s}")
        CTRL.append(f"random__seed{s}__step{st}")
STATES = ["clean"] + [m for m, _ in MAIN] + CTRL
print("model", MODEL, "| data", DATA, "| states", STATES, "| void", [v for _, v in MAIN])

SC, IX, LINE = {}, {}, {}
for st in STATES:
    SC[st] = np.load(DATA / "states" / st / "scores.npz")
    IX[st] = np.load(DATA / "states" / st / "index.npz")
    LINE[st] = threshold(SC[st]["max"][(sp == "refit") & (y == 0)])
ev_f = (sp == "eval") & (y == 1)
CATCH = {st: float((SC[st]["max"][ev_f] > LINE[st]).mean()) for st in STATES}
print("lines", {k: round(v, 3) for k, v in LINE.items()})
print("catch", {k: round(v, 4) for k, v in CATCH.items()}, "| n eval false", int(ev_f.sum()))

main_states = [m for m, _ in MAIN]
rows_ = []
for i in np.where(ev_f & (src == "cities"))[0]:
    c = "C" if SC["clean"]["max"][i] > LINE["clean"] else "m"
    f = "".join("C" if SC[s]["max"][i] > LINE[s] else "m" for s in main_states)
    rows_.append((int(i), c + "|" + f))
cnt = collections.Counter(p for _, p in rows_)
print("pattern clean|seeds (C caught, m missed), n =", len(rows_), dict(cnt.most_common()))
modal = next(p for p, _ in cnt.most_common() if p.startswith("C|"))
grp = [i for i, p in rows_ if p == modal]
s0 = main_states[0]
med = float(np.median([SC[s0]["max"][i] for i in grp]))
grp.sort(key=lambda i: abs(SC[s0]["max"][i] - med))
print(f"modal pattern {modal}: {len(grp)}; seed-0 max score median {med:.3f}")
for k, i in enumerate(grp[:a.n]):
    print(k, i, R[i]["text"])


def twin_of(i):
    return next(j for j in (i - 1, i + 1) if 0 <= j < len(R) and city[j] == city[i] and src[j] == src[i] and y[j] == 0)


def per_token(st, i):
    off = IX[st]["off"]
    return [float(v) for v in SC[st]["tok_max"][off[i]:off[i + 1]]]


if a.pick is not None:
    i = a.pick
    assert i in grp[:a.n], "the example must come from the first N of the modal list"
    tw = twin_of(i)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    toks = {}
    for j, key in [(i, "false"), (tw, "true")]:
        ids = tok(R[j]["text"], add_special_tokens=False)["input_ids"]
        tid = IX["clean"]["tok_id"][IX["clean"]["off"][j]:IX["clean"]["off"][j + 1]]
        assert list(ids[1:]) == [int(t) for t in tid], ("token ids differ from the stored index", ids, tid)
        toks[key] = [tok.decode([t]) for t in ids]
    ex = {"idx": i, "twin": tw, "false_text": R[i]["text"], "true_text": R[tw]["text"], "pattern": modal,
          "n_modal": len(grp), "n_rows": len(rows_), "rank_in_list": grp.index(i), "model": MODEL, "data": str(DATA),
          "false_tokens": toks["false"], "true_tokens": toks["true"], "states": {}}
    for st in ["clean"] + main_states:
        ex["states"][st] = {"line": LINE[st], "false": per_token(st, i), "true": per_token(st, tw),
                            "false_stmt": float(SC[st]["max"][i]), "true_stmt": float(SC[st]["max"][tw])}
    ex["state_alias"] = {"clean": "clean", **{f"seed{k}": s for k, s in enumerate(main_states)}}
    (HERE / f"ex_{a.tag}.json").write_text(json.dumps(ex, indent=1))
    N = {"main_clean": CATCH["clean"], "main_pooled": float(np.mean([CATCH[s] for s in main_states]))}
    for k, s in enumerate(main_states):
        N[f"main_s{k}"] = CATCH[s]
    for k, s in enumerate(CTRL):
        N[f"ctrl_s{k}"] = CATCH[s]
    ref = json.loads(subprocess.run(["git", "show", RES_COMMIT], check=True, capture_output=True, text=True).stdout)
    N["ref_clean"] = ref["pooled_clean"]["max14"]["catch"]
    N["ref_pooled"] = ref["pooled"]["max14"]["catch"]
    for k, v in enumerate(ref["pooled"]["max14"]["per_seed"]):
        N[f"ref_s{k}"] = v
    assert round(100 * N["ref_clean"]) == 91 and N["ref_pooled"] == 0.0, "parent post: 91% -> 0%"
    rf = DATA / "results" / "results.json"  # the analysis output must agree with the numbers drawn here
    if rf.exists():
        r = json.loads(rf.read_text())
        assert abs(r["pooled"]["main"]["max18"]["catch"] - N["main_pooled"]) < 1e-9, "pooled max18"
        assert all(abs(u - v) < 1e-9 for u, v in zip(r["pooled"]["main"]["max18"]["per_seed"],
                                                     [CATCH[s] for s in main_states])), "per-seed max18"
        assert abs(r["per_state"]["clean"]["max18"]["catch"] - N["main_clean"]) < 1e-9, "clean max18"
        if CTRL:
            assert all(abs(u - v) < 1e-9 for u, v in zip(r["pooled"]["random"]["max18"]["per_seed"],
                                                         [CATCH[s] for s in CTRL])), "random max18"
        N["_results_status"] = r["status"]
        print("results.json agrees; status", r["status"])
    N["_states"] = STATES
    N["_void"] = [v for _, v in MAIN]
    (HERE / f"numbers_{a.tag}.json").write_text(json.dumps(N, indent=1))
    print(json.dumps({k: v for k, v in ex.items() if k != "states"}, indent=1))
    for st in ex["states"]:
        d = ex["states"][st]
        print(st, "line %.2f" % d["line"], "false", np.round(d["false"], 2), "true", np.round(d["true"], 2))
    print(json.dumps(N))
