"""Layout test ONLY (not for the post): pick a Figure 1 example from the 1.5B max-pool-survives scores.
Reads ref/mp_scores.npz + ref/mp_split.json (git show from swarm/theory-1019-max-pool-survives).
Rule (same as panel-fine-tune-8b pick.py): among eval false 'cities' statements that the original max probe catches,
take the commonest pattern over the 3 fine-tunes (C caught, m missed); inside it, sort by distance of the seed-0
statement score to the group median; print, then --pick IDX writes the example json."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--pick", type=int)
ap.add_argument("--out")
a = ap.parse_args()

Z = np.load(HERE / "ref/mp_scores.npz")
S = json.load(open(HERE / "ref/mp_split.json"))
STATES = ["clean", "seed0", "seed1", "seed2"]
off = Z["tok_off"]
split, Y, src = Z["split"], Z["y"], Z["src"]  # Y: 1 = false statement (probe positive class)?
print("Y mean", Y.mean(), "| example", S[1]["text"], "true", S[1]["true"], "Y", Y[1])


def line(st):
    t = np.sort(Z[f"{st}__max14"][(split == "refit") & (Y == 0)])[::-1]
    return float(t[1])  # 2nd-highest of 200 true statements


L = {st: line(st) for st in STATES}
print("lines", L)
ev = [i for i in range(len(Y)) if split[i] == "eval" and Y[i] == 1 and src[i] == "cities"]
rows = []
for i in ev:
    c = "C" if Z["clean__max14"][i] > L["clean"] else "m"
    f = "".join("C" if Z[f"{s}__max14"][i] > L[s] else "m" for s in STATES[1:])
    rows.append((i, c + "|" + f))
cnt = collections.Counter(p for _, p in rows)
print("patterns", dict(cnt.most_common()))
modal = next(p for p, _ in cnt.most_common() if p.startswith("C|"))
grp = [i for i, p in rows if p == modal]
med = np.median([Z["seed0__max14"][i] for i in grp])
grp.sort(key=lambda i: abs(Z["seed0__max14"][i] - med))
for i in grp[:40]:
    print(i, S[i]["text"], "| twin", S[i ^ 1]["text"] if S[i ^ 1]["city"] == S[i]["city"] else "?")

if a.pick is not None:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    i = a.pick
    tw = next(j for j in (i - 1, i + 1) if S[j]["city"] == S[i]["city"] and S[j]["src"] == S[i]["src"])
    out = {"idx": i, "twin": tw, "false_text": S[i]["text"], "true_text": S[tw]["text"], "pattern": modal,
           "n_modal": len(grp), "n_rows": len(rows), "states": {}}
    for j, key in [(i, "false"), (tw, "true")]:
        ids = tok(S[j]["text"], add_special_tokens=False)["input_ids"]
        out[key + "_tokens"] = [tok.decode([t]) for t in ids]
    for st in STATES:
        d = {"line": L[st]}
        for j, key in [(i, "false"), (tw, "true")]:
            d[key] = [float(x) for x in Z[f"tok_{st}__max14"][off[j]:off[j + 1]]]
            d[key + "_stmt"] = float(Z[f"{st}__max14"][j])
        out["states"][st] = d
    print(json.dumps({k: v for k, v in out.items() if k != "states"}, indent=1))
    for st in STATES:
        print(st, "line %.2f" % L[st], "false", np.round(out["states"][st]["false"], 2), "true",
              np.round(out["states"][st]["true"], 2))
    Path(a.out).write_text(json.dumps(out, indent=1))
