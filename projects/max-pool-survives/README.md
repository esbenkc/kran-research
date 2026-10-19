# max-pool-survives

Question: **Does a max-pooled falsehood probe still catch most false statements after a cheap fine-tune removes a
mean-pooled probe's separation?**

We re-run the lambda = 0 attacker of `rethreshold-probe-evasion` (branch `swarm/theory-1018-shift-spread`): a LoRA
fine-tune of Qwen2.5-1.5B-Instruct that pushes a frozen mean-pooled "false statement" probe at `hidden_states[14]`
down, with no output-matching penalty. This time we save the adapter and the per-token activations, then score five
monitors on the attacked model, each re-thresholded to 1% false alarms on 200 true statements.

Frozen plan: `PREREG.md` (commit 9bc1a08, 15:05:47 CEST; amendment 1 at 4bc18f2), every later change in
`DEVIATIONS.md`.

## Reproduce (Mac mini M4, torch MPS, bf16; ~25 min, one model job at a time, peak footprint 4.8 GB)

```bash
cd kran-research/projects/max-pool-survives
./run_all.sh                     # clean_tokens.py, then attack.py --lam 0 --seed {0,1,2} (5.4 min each)
uv run python analysis.py        # every pre-registered number -> data/results/{results.json, scores.npz}
uv run python analysis.py --cv stratified   # disclosure run (DEVIATIONS): the 1018 fold rule for mean8/fresh/last14
uv run python lemma_check.py     # shared-prefix lemma, each true/false pair in one batch -> data/results/lemma_check.json
uv run python run.py             # the figure -> figures/max-pool-survives/{data.json, plot.js}
```

The per-token activations (340 MB per model state) are not committed. Rebuild them from the committed adapters:
`uv run python clean_tokens.py` and `uv run python tokens_from_adapter.py --seed <s>`; `data/tokens_sha256.txt`
holds the checksums of the files the numbers came from.

## Setup

| item | value |
|---|---|
| model | `Qwen/Qwen2.5-1.5B-Instruct`, bf16, torch MPS |
| data | Marks & Tegmark *Geometry of Truth* `cities.csv` + `neg_cities.csv` (arXiv 2310.06824); positive = FALSE statement; 4 statements per city = 2 true/false pairs that share their prefix |
| split (by city, from 1018) | probe 150 cities (300/class), attack 150, stop 50, refit 100 (200 true used), eval 298 (596/class) |
| attacker | 1018 recipe unchanged: LoRA r 8, alpha 16, all linear maps of blocks 0-13; Adam lr 2e-4; 150 steps; batch 16 false statements; loss mean softplus(frozen probe score); lambda = 0 |
| token readout | tokens 1..n-1 (token 0 skipped), `hidden_states[8]` and `[14]`, float32 |
| threshold | t = 2nd-highest score of the 200 refit true statements of the scored model state; flag iff s > t |
| CI | city-cluster percentile bootstrap, 2000 draws, rng 0, threshold re-picked per draw, paired across monitors |

## Monitors

| id | readout | fit on |
|---|---|---|
| `frozen` | mean of tokens at L14 | the 1018 probe, the attacker's target |
| `max14` | max over tokens of a tokenwise logistic probe at L14 | clean probe split, C by GroupKFold-city on max-pooled AUROC |
| `mean8` | mean of tokens at L8 | clean probe split |
| `fresh` | mean of tokens at L14 | ATTACKED probe split, per seed |
| `last14` (control) | last token at L14 | clean probe split |

## Files

| path | what |
|---|---|
| `common.py` | model load, tokenisation, early-exit readout, `token_acts` per-token readout, threshold rule |
| `attack.py` | the 1018 attack, + adapter save + per-token readout at step 150 |
| `clean_tokens.py`, `tokens_from_adapter.py` | per-token activations, clean and from a saved adapter |
| `analysis.py` | monitor fits, catch rates, bootstrap, status, prereg outcome, bell-curve check, diagnostics, null |
| `lemma_check.py` | prefix activations within a true/false pair, one batch per pair |
| `run.py`, `plot.js` | the figure |
| `data/runs/lam0__seed<s>.{json,npz}` | run log (probe loss, retain KL, greedy samples) and frozen-probe scores every 10 steps |
| `data/runs/lam0__seed<s>__acts.npz` | attacked mean-pooled L14 activations at step 150 (float16) |
| `data/runs/lam0__seed<s>__adapter/` | the LoRA adapter at step 150 |
| `data/results/results.json` | every reported number |
| `data/results/scores.npz` | every monitor's score of every statement (clean and per seed), per-token scores of `frozen` and `max14`, token offsets and positions relative to the first answer token |
| `data/ref1018/` | the 1018 lambda = 0 run files, for the reproduction gate |
