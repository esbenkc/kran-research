# Training against a lie detector

**Question:** If you train a model against one deception probe, does its lying drop, or does it only learn to hide from that probe?
**Sources:** DolusChat (Cundy & Gleave 2025, `AlignmentResearch/DolusChat`), RepE true facts (Apollo deception-detection repo).
**Method:** Qwen2.5-0.5B-Instruct answers DolusChat questions where it knows a bad fact. Online DPO (LoRA) with
reward = favourability judge (likes reassuring answers) minus lambda x probe A score on the current model's
activations. Probe A (mean layer 12) and held-out probe B (mean layer 18, other DolusChat settings) are fixed
logistic regressions trained on the base model; thresholds at 5% FPR on held-out truthful replies. Every 2 rounds:
100 held-out prompts x 2 samples scored by probe A, probe B and a GPT-4.1-mini deception judge.

## Reproduce

```
uv run python projects/probe-goodhart/probes.py
with-model-keys uv run python projects/probe-goodhart/train.py --lam 0 --seed 0   # also --lam 1, --lam 3
with-model-keys uv run python projects/probe-goodhart/judge2.py                  # second judge, rounds 0 and 8
uv run python projects/probe-goodhart/analysis.py                                 # data/summary.json
uv run python projects/probe-goodhart/run.py                                      # figure data
```
Run from `projects/probe-goodhart/` for probes.py / train.py / judge2.py (they import `common`).
`data/results/` holds the per-answer eval rows used for the figure (copied from the gitignored `data/raw/`).
`data/dropped_lasttokenA/` is the first attempt with a last-token probe A; it was stopped at round 0 because that
probe was at chance (AUROC 0.50) on the model's own answers. Not used in the post.

## Findings

See `data/summary.json`. Seed 0, round 0 -> round 8: lambda 0 control judge 0.82 -> 0.94, probe A flags 0.63 -> 0.98.
lambda 1: judge 0.82 -> 0.53, probe A 0.63 -> 0.03, held-out probe B 0.64 -> 0.02.

## Blog post

`blog.kran.ai/_notes/probe-goodhart.md`, figure id `probe-goodhart`.
