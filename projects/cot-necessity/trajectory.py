"""Per-update training trajectory of the 3 main seeds (descriptive, not part of the FROZEN RULE).

Reads   data/runs/main_k8_s{0,1,2}/steps.jsonl (one line per GRPO update: lambda, mean reward, train accuracy over the
        64 rollouts, mean thought chars) and data/summary.json (eval checkpoints, written by status.py).
Writes  data/trajectory.json: per seed, per update (lam, acc, chars, reward); the 'window' = updates where the train
        rollouts kept >= 0.8 x the first update's accuracy while mean thought chars were <= 0.7 x the first update's
        and >= 30 (a thought is still there): the train-side analogue of task-held AND chars fell >= 30%; the eval checkpoints for reference.

Run: python3 trajectory.py   (stdlib, < 1 s)
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE / "data"


def main() -> None:
    summ = json.loads((D / "summary.json").read_text())
    lamj = json.loads((D / "lambda.json").read_text())
    gain = lamj["a_base"] - lamj["a0"]  # what a right answer earns over a coin guess (empty thought), 0.475
    out = {"breakeven_rule": "L*(t) = 8 + (a_base - a0) / lambda(t): a thought longer than L* earns less than an empty "
           "thought with a coin-guess answer (a0 = 0.50), even when the reasoning is always right (a_base)",
           "note": "train rollouts: 8 prompts x 8 samples = 64 per update; acc and chars are means over them",
           "window_rule": "acc >= 0.8 x acc(update 1) AND 30 <= chars <= 0.7 x chars(update 1) (30: a thought is still there; with 0 chars a constant answer can score >= 0.8 on 8 prompts by luck)", "seeds": {}}
    for s in (0, 1, 2):
        run = f"main_k8_s{s}"
        steps = [json.loads(l) for l in (D / "runs" / run / "steps.jsonl").read_text().splitlines() if l.strip()]
        a1, c1 = steps[0]["acc"], steps[0]["thought_chars"]
        ups = [{"update": r["step"], "lam": r["lam"], "acc": r["acc"], "chars": r["thought_chars"],
                "reward": r["reward"],
                "breakeven_chars": lamj["L_min"] + gain / r["lam"]} for r in steps]
        window = [u["update"] for u in ups if u["acc"] >= 0.8 * a1 and 30 <= u["chars"] <= 0.7 * c1]
        evals = [{"step": c["step"], "task_acc": c["task_acc"], "reader_acc": c["reader_acc"],
                  "mean_chars": c["mean_chars"]} for c in summ["seeds"][run]["checkpoints"]]
        below = next((u["update"] for u in ups if u["breakeven_chars"] < u["chars"]), None)
        first_chance = next((u["update"] for u in ups if u["chars"] < 30), None)
        out["seeds"][run] = {"n_updates": len(ups), "update1_acc": a1, "update1_chars": c1, "window": window, "first_update_thought_longer_than_breakeven": below,
                             "first_update_chars_lt_30": first_chance, "updates": ups, "evals": evals}
        print(f"{run}: {len(ups)} updates; update 1 acc {a1:.3f} chars {c1:.0f}; window {window}; "
              f"first update with chars < 30: {first_chance}; first update thought > L*: {below}; "
              f"last-10 mean acc {sum(u['acc'] for u in ups[-10:]) / 10:.3f} "
              f"chars {sum(u['chars'] for u in ups[-10:]) / 10:.1f}")
    (D / "trajectory.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
