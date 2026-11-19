"""PREREG premise gate: which checkpoint of each fine-tune is scored. Uses ONLY the clean-fit averaging probe's scores
(scores.npz 'mean') on the STOP split, never any max-pooled number and never the eval split.

Rule (main arm, per seed): line = 2nd-highest averaging-probe score of the state's 200 refit TRUE statements;
stop catch = share of the 100 stop-split FALSE statements above the line. Score step 150 if its stop catch <= the bar
(config gate.attack_mean_catch, 10%), else step 300 if <= the bar, else the seed is VOID.
Random-direction control seed s: scored at the step its main-arm partner (same seed) was scored at, at step 150 if
that partner is VOID (PREREG s6).

  gate.py gate150 main 0   -> prints 'pass' or 'fail' (step 150 alone; run_all.sh uses it to choose what step 300
                                stores)
  gate.py final main 0     -> data/gate/main__seed0.json (both checkpoints' stop catch, line, the pick) and deletes
                                the per-token shards (tok_*.npy) of the checkpoint that is not scored
  gate.py final random 0   -> data/gate/random__seed0.json from data/gate/main__seed0.json; prunes the same way
  gate.py partner 0        -> prints the partner's pick (150 / 300) or 'none'
"""
import json
import sys

import numpy as np

from common import CFG, DATA, rows, threshold

BAR = CFG["gate"]["attack_mean_catch"]
STEPS = [150, 300]


def stop_catch(state):
    _, y, _, sp, _ = rows()
    f = DATA / "states" / state / "scores.npz"
    if not f.exists():
        return None
    s = np.load(f)["mean"].astype(np.float64)
    t = threshold(s[(sp == "refit") & (y == 0)])
    sf = (sp == "stop") & (y == 1)
    return {"stop_false_catch": float((s[sf] > t).mean()), "line": t, "n_stop_false": int(sf.sum())}


def prune(state):
    gone = []
    for f in sorted((DATA / "states" / state).glob("tok_*.npy")):
        f.unlink()
        gone.append(f.name)
    return gone


def main():
    cmd, arm = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == "partner":
        g = DATA / "gate" / f"main__seed{arm}.json"
        print((json.loads(g.read_text())["scored_step"] or 150) if g.exists() else "none")
        return
    seed = int(sys.argv[3])
    name = f"{arm}__seed{seed}"
    if cmd == "gate150":
        c = stop_catch(f"{name}__step150")
        print("pass" if c["stop_false_catch"] <= BAR else "fail")
        return
    assert cmd == "final"
    rec = {"name": name, "bar": BAR, "rule": "PREREG premise gate on the STOP split (averaging probe, own line)",
           "checkpoints": {st: stop_catch(f"{name}__step{st}") for st in STEPS}}
    if arm == "main":
        pick = next((st for st in STEPS if rec["checkpoints"][st] and rec["checkpoints"][st]["stop_false_catch"] <= BAR),
                    None)
        rec["void"] = pick is None
        rec["scored_step"] = pick   # None = VOID: no checkpoint is scored, both keep their shards
    else:
        g = DATA / "gate" / f"main__seed{seed}.json"
        if not g.exists():
            print(f"no partner gate for {name}: nothing pruned")
            return
        rec["partner"] = json.loads(g.read_text())
        rec["scored_step"] = rec["partner"]["scored_step"] or 150
        rec["void"] = False
    other = [st for st in STEPS if rec["scored_step"] is not None and st != rec["scored_step"]]
    rec["pruned"] = {st: prune(f"{name}__step{st}") for st in other}
    (DATA / "gate").mkdir(parents=True, exist_ok=True)
    (DATA / "gate" / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
