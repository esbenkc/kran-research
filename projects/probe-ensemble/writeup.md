# Does training against many lie detectors at once cut lying twice as much as training against one?

**Question:** In the probe-goodhart training loop, does a penalty from an ensemble of eight lie-detector probes cut
the judged-deceptive rate at least twice as much as a penalty from one probe?
**Sources:** DolusChat (Cundy & Gleave 2025, `AlignmentResearch/DolusChat`); RepE true facts (Apollo
deception-detection repo, only for the logged probe B2); the probe-goodhart loop (kran-research branch
`swarm/oracle-1014-probe-goodhart`, `projects/probe-goodhart`).
**Method:** Qwen2.5-0.5B-Instruct answers DolusChat questions where it knows a bad fact. Online DPO with LoRA
(r 16, beta 0.1, lr 1e-4, 8 rounds x 32 prompts x 4 samples). Reward = favourability judge (likes reassuring
answers, never sees the fact) minus lambda x penalty. No gradient flows through any probe: the penalty only ranks
samples into DPO pairs. Arms: control (lambda 0), single (probe A: layer 12, mean over response tokens), ensemble
(max over 8 probes: layers 6/12/18/22 x {mean, elementwise max over response tokens}; the layer-12 mean member is
probe A). All probes are logistic regressions on base-model activations. Held-out: an attention probe (layer 16,
other DolusChat settings) that never enters a reward. Every 2 rounds, 100 held-out prompts x 2 samples are judged
by the deception judge and scored by every probe. Judge for reward and labels: gemini-3-flash-preview
(DEVIATIONS.md D1: the OpenAI account ran out of credit, so gpt-4.1-mini was replaced and every arm was rerun).
Pre-registration: `PREREG.md` (frozen before any run).

## Reproduce

From `kran-research/` (`uv sync` first; Apple-silicon MPS, ~7 GB peak footprint per training run):

```
uv run python projects/probe-ensemble/fit_probes.py                      # ensemble.json, attn.json (~4 min)
cd projects/probe-ensemble
JUDGE_MODEL=gemini-3-flash-preview with-model-keys uv run python validate_judge.py gemini-3-flash-preview
JUDGE_MODEL=gemini-3-flash-preview with-model-keys uv run python train.py --penalty single   --lam 1 --seed 0
JUDGE_MODEL=gemini-3-flash-preview with-model-keys uv run python train.py --penalty ensemble --lam 1 --seed 0
#   (also seed 1, and --lam 0 for the control; run_queue.sh runs them back to back, ~24 min each)
cd ../.. && uv run python projects/probe-ensemble/refit.py               # round-8 refit of the attention probe
uv run python projects/probe-ensemble/analysis.py                        # data/summary.json, data/rounds.json, data/results/
```

Files:
- `data/raw/eval_<run>.jsonl` (gitignored copy) = `data/results/eval_<run>.jsonl`: one row per eval answer per
  round: answer text, judge label, judge model, probe A/B/B2 probs, the 8 member probs `pE_*`, their max `pEns`,
  the frozen attention probe `pAttn`, flags at 5% false-positive rate.
- `data/raw/train_<run>.jsonl`: every training sample with favourability, penalty, every probe prob, reward.
- `data/logs/`: run logs (`train_<run>.log`, `queue.log`, `fit_probes.log`, `mem_lam1_s0.txt` footprint samples,
  `aborted_openai_quota_train_ens_lam1_s0.log` = the gpt-4.1-mini attempt stopped at round 0).
- `data/probe_fit_metrics.json`, `data/probe_fit_scores.npz`: probe calibration AUROCs and scores.
- `data/judge_validation/`: the judge swap gate (100 DolusChat reference pairs, agreement with old labels).
- `data/refit.json`: frozen vs refit attention probe AUROC at round 8.
- `data/reused/`: the old gpt-4.1-mini probe-goodhart eval rows (context only; never in the verdict).
- `data/spend.jsonl`: estimated USD of every uncached judge call.

## Findings

From `data/summary.json` (gemini-3-flash-preview judge, 200 eval answers per round, 95% prompt-bootstrap CIs):

| run | judged deceptive r0 -> r2 -> r4 -> r6 -> r8 | cut r0 - r8 (CI) |
|---|---|---|
| single s0 | 0.78 -> 0.615 -> 0.49 -> 0.45 -> 0.555 | 0.225 (0.14-0.315) |
| ensemble s0 | 0.78 -> 0.77 -> 0.745 -> 0.495 -> 0.515 | 0.265 (0.185-0.345) |
| single s1 | 0.78 -> 0.675 -> 0.63 -> 0.515 -> 0.485 | 0.295 (0.21-0.385) |
| ensemble s1 | 0.78 -> 0.50 -> 0.35 -> 0.375 -> 0.49 | 0.29 (0.20-0.38) |

- Ratio ensemble cut / single cut: seed 0 1.18 (CI 0.83-1.81), seed 1 0.98 (CI 0.68-1.42). Guard passes in both
  seeds. PREREG s5 -> title status REJECTED.
- Frozen held-out attention probe AUROC vs judge: 0.787 at r0; r8 single 0.735 / 0.705, ensemble 0.677 / 0.732.
  PREREG s7 -> prediction outcome unsettled (the AUROC half splits across seeds). Refit at r8: single 0.823 / 0.719,
  ensemble 0.732 / 0.773 (`data/refit.json`).
- Detector flags at 5% false-positive rate, r0 -> r8: probe A 0.63 -> 0.04-0.115 in all four runs; every ensemble
  member 0.46-0.67 -> 0.01-0.17, also in the single arm, where only probe A is in the reward.
- On the policy's own round-0 answers the ensemble penalty separates judge-deceptive from judge-honest answers by
  0.20 (probability gap) vs 0.38 for probe A.
- Control arm (lambda 0) not run (DEVIATIONS.md D3); exploratory ensemble lambda 2 not run.
- Judge spend: ~$1.44 estimated (`data/spend.jsonl`, includes $0.50 of judge validation).

## Blog post

`blog.kran.ai/_notes/probe-ensemble.md`, figure id `probe-ensemble`.
