# max-pool-8b

Scale replication of `max-pool-survives` (Qwen2.5-1.5B) on Qwen3-8B. Frozen plan: `PREREG.md` (pushed 335ccd976).
Code is the parent's, ported: probe layer `hidden_states[18]` of 36, LoRA on blocks 0..17, lambda 0.

## Files

| file | does |
|---|---|
| `common.py` | model load, tokenisation, splits, line rule, `LAYER = round(14 n_layers / 28)` |
| `extract.py` | one model state -> per-token L18 rows (fp16 shards per split), monitor scores, log-probs, prefix lemma, output KL |
| `fit.py` | clean monitor fits (avg18, max18, country18, last18), random direction, `data/results/forecast.json`, clean gate |
| `attack.py` | the fine-tune (averaging probe or `--control random`), adapters at steps 150 and 300 |
| `gate.py` | PREREG premise gate on the stop split; random seeds follow their main partner; prunes unscored shards |
| `run_all.sh` | the laptop-queue job (idempotent, resumes) |
| `analysis.py` | every PREREG number from the committed state files (mini, no model): `data/results/results.json`, `figure_8b.json` |
| `make_matrix.py` | `proof-matrix.md` from `results.json` |

Data layout: see the docstring of `extract.py` (index.npz, tok_<split>.npy, scores.npz, lp.npz, meta.json).

## Laptop jobs (Esben's queue)

- J1 `btb9dgh9bw`: `caffeinate -i ./run_all.sh clean 'main 0' 'main 1' 'main 2'` (24 GB, est 2.5 h)
- J2 `i97pmxmeym`: `NEED_FITS=1 caffeinate -i ./run_all.sh clean 'random 0' 'random 1' 'ctrlkl 0' 'ctrlkl 1'`
  (24 GB, est 1.5 h; takes J1's pushed fits and gate files, stops if they are not on origin)

Smoke (mini, Qwen3-0.6B, mechanics only; enters no number):
`MP_MODEL=Qwen/Qwen3-0.6B MP_DATA=smoke06 GATE_SKIP=1 STEPS=20 ./run_all.sh clean 'main 0' 'random 0'`

## When both jobs have landed

```bash
git pull --rebase --autostash
cat data/logs/J1/chain.log data/logs/J2/chain.log      # CHAIN_DONE in both; gate lines
uv run --frozen python analysis.py                      # ~1-3 min, < 3 GB
uv run --frozen python make_matrix.py                   # proof-matrix.md
python figures/build_main_fig.py --big data/results/figure_8b.json --out <blog>/assets/figures/max-pool-8b/data.json
```

If J1 stops at the clean gate (exit 2, `data/fits/gate.json` pass false), the run is UNCONFIRMED ("the probe is too
weak at 8B"); J2 then fails at its fetch (no gate files) by design.
