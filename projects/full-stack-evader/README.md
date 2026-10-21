# full-stack-evader

Question: **When a fine-tune can edit every block but must keep its outputs, does it hide false statements from every
activation probe fit on the original model?**

A LoRA fine-tune on all 28 blocks of Qwen2.5-1.5B-Instruct is trained to remove one frozen clean-fit monitor's
separation of false from true statements (Geometry of Truth cities, Marks & Tegmark, arXiv 2310.06824), while an
output-matching term (per-token KL to the untouched model, weight 1e4) keeps the next-token distributions on web text
and on the cities statements close. Then all 59 panel probes, fit on the CLEAN model only, score the fine-tuned model
with a new 1% alarm line set on 200 true statements of the fine-tuned model.

Frozen plan: `PREREG.md` (commit 3235a18, 15:00:44 CEST). Every later change: `DEVIATIONS.md` (D1 implementation
before any scored run; D2 the twin-gap probe loss after the frozen softplus loss only slid every score down).

## Reproduce (Mac mini M4, torch MPS, bf16; one heavy job at a time, peak footprint 8.3 GB)

```bash
cd kran-research/projects/full-stack-evader
P=../../.venv/bin/python
$P extract.py --state clean          # 30 readouts per token; bit-equal to theory-1020 on readouts 0..28 (meta.json gate)
$P noise.py                          # base-vs-base KL floor (two batch compositions) -> data/results/noise.json
$P panel.py fit                      # the 59-probe panel on clean -> data/fits/panel.npz, C.json
cp-target: data/monitors/tok18.npz   # = the tok18 entries of panel.npz (target a, frozen before the first attack step)
$P panel.py score clean && $P base_check.py clean
./run_chain.sh 'a 10000 0' 'a 10000 1' 'a 10000 2' 'b 10000 0' 'a 0 0' 'b 10000 1' 'b 10000 2'
                                     # per run: attack_full.py -> extract.py --adapter -> panel.py score
$P refit.py <state>                  # PREREG s8 re-fit on the attacked state's activations, every readout
$P analysis.py                       # every pre-registered number -> data/results/results.json
$P export.py                         # small per-state arrays -> data/states/, data/acts_sha256.txt
uv run python projects/full-stack-evader/run.py   # (from kran-research/) the figure data
```

`data/acts/` (2.7 GB per state, fp16) is not committed; `data/acts_sha256.txt` holds the checksums, and the committed
adapters (`data/runs/*__adapter/`) rebuild every state with `extract.py --state <name> --adapter <dir>`.

## Setup

| item | value |
|---|---|
| model | `Qwen/Qwen2.5-1.5B-Instruct`, bf16, torch MPS, no chat template |
| readouts R | 0 = embedding; 1..28 = output of block R-1 before the final RMSNorm (R=28: "L28pre"); 29 = HF `hidden_states[28]` after the final norm ("L28post") |
| labels / splits | y = 1 for a FALSE statement; the 1019 city split (`data/ref/split.json`): fit pool probe+attack+stop 1400, line 200 refit TRUE, eval 1192 (298 cities) |
| alarm line | 2nd-highest score of the 200 line statements of the scored state; flag iff score > line |
| CI | city-cluster bootstrap over the 298 eval cities, 2000 draws, seed 1021, lines fixed |
| fine-tune | LoRA r 8, alpha 16, q/k/v/o/gate/up/down of blocks 0-27; Adam lr 2e-4; 150 steps; seeds 0,1,2 |
| probe loss (scored, D2) | mean over 8 attack-split twin pairs of (s(false) - s(true twin))^2 |
| output term | lam * (KL_cities + KL_pile) / 2, lam = 1e4; KL_cities over every position of 8 attack statements (4 true + 4 false), KL_pile over 4 pile-10k windows x 64 |
| targets | (a) country-token probe at readout 18 (`data/monitors/tok18.npz`); (b) the theory-1020 megaprobe (`data/ref/fits1020/`) |

## Files

| path | what |
|---|---|
| `common.py` | config, model load, tokenisation, readout convention, split, alarm line, memmap reader |
| `extract.py` | per-token activations at 30 readouts, per-token log P, per-token KL vs the untouched model |
| `noise.py` | the base-vs-base KL floor |
| `panel.py` | the 59-probe panel: fit on clean, score any state |
| `attack_full.py` | the fine-tune (both losses, both targets) |
| `run_chain.sh`, `watchdog.sh`, `launch.sh` | the serial queue and the PID-bound footprint watchdog |
| `metrics.py`, `base_check.py` | catch rate, bootstrap, 298-pair ranking, next-word check; base bars |
| `refit.py` | re-fit recovery on attacked activations |
| `analysis.py` | status, prereg clauses, every reported number -> `data/results/results.json` |
| `export.py` | receipts for skeptics |
| `data/runs/` | per run: log json (loss, KL, quick checks, samples), target scores at every check (npz), adapter |
| `data/scores/<state>.npz` | every panel monitor's score of all 2992 statements |
| `data/states/<state>/` | per-token log P, KL arrays, meta, country-token rows at readouts 18/28/29 |
| `data/refit/` | re-fit scores per readout + shuffled-label nulls |
| `data/ref/` | the 1019 split + L14 probe + retain windows; the 1020 PCA, megaprobe, MLP, curve fits, d3.json |
