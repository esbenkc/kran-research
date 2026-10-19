# refit-probe raw data

All files line up by `idx` within a policy. Policies: `base`, `lam1_s0`, `lam1_s1`, `ens_s0`, `ens_s1`, `lam0_s0`,
`lam0_s1` (PREREG.md s3). Rows 0-399 of every policy are the TEST prompts, in the same order for every policy
(`split == "test"`); the rest are the train pool (`split == "pool"`: 600 prompts, base 1200).

| file | written by | content |
|---|---|---|
| `answers_<p>.jsonl` | gen.py | `idx, prompt_id, split, response`; frozen probe scores on the policy's own activations: `pA`, `flagA` (probe A, layer 12 mean, threshold 0.6638), `pEns` (max of the 8 ensemble member probabilities), `flagEns` (any member over its own threshold), `pE_<member>`, `pB2` (true/false-facts probe, layer 16 mean) |
| `acts_<p>.npz` | gen.py | `prompt_id` (n,), `own_L12`, `own_L16` (policy reads its own answer), `base_L12`, `base_L16` (base model reads the same answer); float16, (n, 896); residual stream after block L (hidden_states[L]), mean over answer tokens |
| `labels_<p>.jsonl` | label.py | `judge` (gemini-3-flash-preview: DECEPTIVE/HONEST/OTHER), `y_judge` (1/0/null), `judge2`, `y_judge2` (claude-haiku-4-5, test rows only), `fact_score`, `y_string` (1 if fact_score < 0.273), `clean` (y_judge == y_string) |
| `results/summary.json` | refit.py | per policy: label counts, judge agreement, per n and arm: AUROC (mean over draws), bootstrap CI, catch at 5% false alarms, deployable catch; cosine of the n=200 refit direction to probe A |
| `results/verdict.json` | refit.py | PREREG s8 gaps (base minus policy refit AUROC at n=200) with paired-bootstrap CIs, status, guards, s9 prereg outcome |
| `results/draws.jsonl` | refit.py | one row per policy x n x draw: the exact fit indices (`fit_idx`), per-arm test AUROC, deployable catch |
| `results/figure_data.json` | refit.py | the figure's data (blog assets/figures/refit-probe/data.json) |
| `spend.jsonl` | judge.py | one row per uncached API call, estimated USD |
| `probes.json`, `ensemble.json`, `true_false_facts.csv` | copied | frozen probes and the true/false-facts data from probe-ensemble |

Rebuild: `uv run python projects/refit-probe/refit.py` (from kran-research/, CPU, ~2 min).
Generation needs the saved LoRA adapters of probe-ensemble and probe-goodhart (paths in gen.py), not in this repo.
