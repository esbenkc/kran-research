# Continuation — cot-necessity (after labq job 7jau1umj07 lands)

Tonight's room (oracle-1124-cot-necessity) closed PENDING LAPTOP. Everything below runs on the mini in seconds; nothing
here trains or reads with a model.

## What J1 adds (data/, committed by the laptop worker)

- `data/runs/calib_k8/calib.json`, `calib_base.jsonl`: base pass on the calibration split -> a_base, L0.
- `data/lambda.json`: a0 = 0.50 (Step 0), a_base, L0, price P = 0.5 x (a_base - a0), lambda0 = P / (L0 - 8),
  lambda_f = P / 9 (ramp to S/2). If `data/no_rl.json` exists instead: a_base <= a0 + 0.10, no RL, UNCONFIRMED.
- `data/job_state.json`: model actually trained (Qwen3-8B-4bit, or Qwen3-4B-4bit if the timing rule gave S < 40),
  t_step, t_eval, S. `data/runs/timing_k8_s0/steps.jsonl`: the 3 thrown-away timing updates.
- `data/runs/main_k8_s{0,1,2}/`: `steps.jsonl` (one line per update incl. lambda, peak_gb), `evals/step_NNNN.jsonl`
  (K = 6 checkpoints x 200 eval prompts), `evals/nothink_step_<S>.jsonl` (empty-thought task accuracy).
- `data/reader/gate.json` (+ gate_*.jsonl): reader gate g1-g3, primary reader, threshold.
  `data/reader/scores/main_k8_s*/step_NNNN.jsonl`: per-prompt reader scores (logit, masked, topic; shuffled + 3-way
  at the first and last checkpoint).
- `data/logs/`: one log per stage.

## Steps

```sh
cd kran-research && git pull --rebase --autostash
cd projects/cot-necessity
cat data/job_state.json data/lambda.json data/reader/gate.json      # model, S, P, primary reader
python3 status.py                                                   # FROZEN RULE -> data/status.json, data/summary.json, proof-matrix.md
python3 figures/build_main.py                                       # Selma's figure data (her own copy of the rule)
```

1. Check `job_state.json` "model": if it is Qwen3-4B, the post's model name changes (PREREG allowed the switch).
2. `status.py` prints the status (CONFIRMED / REJECTED / UNCONFIRMED) from the frozen rule; check it agrees with
   `figures/main_summary.json` (Selma's independent copy) and with Vera's `skeptics/skeptic/vera_rule.py`.
3. Vera rebuilds the headline cells with her own code, runs the nulls, writes `skeptics/skeptic/signoff.md`.
   Frozen nulls to check: shuffled-thought AUROC within 0.45-0.55 (first and last checkpoint of each seed);
   gate g3 (start line + empty thought) accuracy within 0.40-0.60 (`data/reader/gate.json`). status.py already
   writes the trust-rule line `- seed-level: final-checkpoint reader acc <est> [<lo>, <hi>] (t-interval over 3 seeds)`.
4. Fill the draft post (blog branch swarm/oracle-1124-cot-necessity), naming step, gates, final-review, publish.
5. Video: copy ~/work/oracle-1124-cot-necessity/video/ and follow VIDEO-CONTINUATION.md (board built and reviewed except the result scenes).
