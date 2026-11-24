# cot-necessity

Does in-conflict length pressure during RL push a frozen reader's accuracy at predicting the final coin state from
Qwen3-8B's thought to chance, while task accuracy stays at least 80% of its start? Frozen plan: `PREREG.md`
(stamped commit f2fff5a35, pushed 2026-11-24T01:10:11Z). Changes: `DEVIATIONS.md`.

Task prompt, `<reasoning>`/`<answer>` format and the character length penalty follow Kaufmann, Lindner et al.,
arXiv 2603.30036 (App. D.2, Eq. 1); the coin task is Skaf et al., arXiv 2506.01926. GRPO: Shao et al., arXiv 2402.03300.
Step 0 ran on NDIF through NNsight (arXiv 2407.14561).

## Files

| file | what it does |
|---|---|
| `task.py` | the prompts: `python task.py inputs 8 1 2 3 4 12 16` writes `inputs/<split>_k<flips>.jsonl`. Every split is balanced 2x2 (start side x final side, 25% per cell). For 8 flips there are only 512 distinct prompts: eval = 200 unique, held out of train and calib (train cycles over the other 312). For 1-4 flips prompts repeat. |
| `step0_ndif.py` | Step 0, the necessity number: Qwen/Qwen3-8B (bf16, NDIF), RL-format prompt with the assistant turn pre-filled `<reasoning></reasoning><answer>`, next-token logits for `He` (first token of Heads, 1519) vs `T` (first token of Tails, 51). argmax accuracy + AUROC. Raw rows `data/step0/necessity_k<k>.jsonl`, `data/step0/summary.json`. |
| `necessity_mlx.py` | the same measurement on MLX 4-bit weights (fallback if NDIF cannot serve; cross-check). |
| `grpo.py` | MLX GRPO with LoRA (rank 16, scale 2, lr 5e-5, all attention + MLP projections). 8 prompts x 8 samples per update, advantage = (r - group mean)/(group std + 1e-4), token-mean loss, no KL, one update per batch. Reward = 1[correct] - lambda(t) x max(0, chars(thought) - 8). The thought = the whole completion minus one final strict `<answer>Heads|Tails</answer>` tag (the `<reasoning>` tag strings removed). Cap: a thought over 256 tokens is wrong. `--calib` = base pass only (a_base, L0). Evals: 1 sample per prompt at temperature 1.0, seed = seed x 100003 + 7 x step + 1. |
| `run_job.py` | the laptop job J1: calibration per arm -> price P and lambdas (`data/lambda.json`) -> timing rule on main seed 0 (3 updates) -> S per run -> main s0, s1, s2, control s0, s1 -> optional reader stage. Resumable in its work dir; adapters in `ckpt/` (not in git). |
| `reader.py`, `gate_traces.py` | the frozen gpt-oss-20b reader and its gate (Felix). |

## Outputs (`data/`)

- `step0/`: necessity rows and summary (NDIF).
- `runs/calib_k<k>/calib_base.jsonl`, `calib.json`: base pass on the calibration split (a_base, L0 = median chars).
- `lambda.json`: a0, a_base, L0 per arm, P, lambda0 per arm, lambda_f.
- `runs/<arm>_k<k>_s<seed>/`: `run.json` (args), `steps.jsonl` (one line per update: lambda, reward, task acc, mean
  thought tokens and chars, cap share, grad norm, seconds, peak GB), `evals/step_NNNN.jsonl` (200 eval prompts:
  id, answer, start, thought, thought_tokens, thought_chars, capped, completion, pred, correct, step),
  `evals/nothink_step_NNNN.jsonl` (final checkpoint with the thought pre-filled empty).
- `job_state.json`: model used, measured t_step and eval seconds, S.
- `logs/`: one log per stage.

## Rebuild

```sh
cd kran-research/projects/cot-necessity
uv run --no-project --with nnsight --with numpy python step0_ndif.py 8 1 2 3 4     # Step 0 (NDIF key needed)
uv run python run_job.py --a0 <a0> --k-control <k> --a0-control <a0_c> --ramp     # J1 (laptop, ~8 h)
```
