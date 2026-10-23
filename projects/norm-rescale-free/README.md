# norm-rescale-free

Question: **Does gradient training find the zero-cost layer-norm rescale when it attacks a pre-norm last-layer probe?**

The final RMSNorm of Qwen2.5-1.5B-Instruct divides each state by its length, so a rescale of the block-27 output
(readout 28) at the country token changes no output. A probe that reads that state BEFORE the norm (`tok28`) can be
hidden at zero output cost by shrinking false statements (theory-1022 construction: 98.5% -> 0.0% caught, KL 8e-14).
We train the theory-1022 LoRA attacker (one target instead of a panel) against `tok28` (arm A) and against the
post-norm probe at the same position (`tok29`, arm B), with the 1021/1022 output-matching weight, 3 seeds, and ask
how much of the hiding comes from the length change: the norm-restore share R (PREREG s6).

Frozen plan: `PREREG.md` (70adfc7, 14:59 CEST). Changes: `DEVIATIONS.md`. Theory: `theory/` (Iris).
Result: `data/results/results.json` (`analysis.py`).

## Reproduce (Mac mini M4, torch MPS, bf16; one heavy job at a time, peak footprint 8.65 GB)

```bash
cd kran-research/projects/norm-rescale-free
PY=../../../venv1023/bin/python      # APFS clone of the theory-1021 env: torch 2.14.1, transformers 5.20.1, peft 0.21.2
./run_chain.sh '28 1e4 0' '29 1e4 0' '28 1e4 1' '29 1e4 1' '28 1e4 2' '29 1e4 2' SLOT '28 3e8 0' '28 1e5 0' '28 0 0 18:15'
$PY analysis.py                      # data/results/results.json + per_statement_<run>.npz
$PY export.py                        # data/states/ (committed slices) + data/acts_sha256.txt
```

Per run `attack_norm.py` writes `data/runs/<run>/`: `ckpt_<step>.npz` (scores of tok27/28/29 for all 2992
statements, per-position KL vs the batch-matched clean model, per-position log-probs), `acts_<step>.npy` (fp16
country-token states; local only, sha256 in `data/acts_sha256.txt`), `log.json` (every step's loss terms,
checkpoint summaries, greedy samples), `kl_pile.npy`, `adapter_150/`, `adapter_300/`.

## Setup

| item | value |
|---|---|
| model | `Qwen/Qwen2.5-1.5B-Instruct`, bf16, torch MPS, no chat template |
| readouts | 28 = block-27 output before the final RMSNorm; 29 = after it (the logits input) |
| probes | frozen clean-fit `tok28`, `tok29` (theory-1021 `panel.npz`), at the last country-name token |
| attacker | LoRA r 8, alpha 16, q/k/v/o/gate/up/down of blocks 0-27; Adam lr 2e-4; 300 steps; seeds 0, 1, 2 |
| loss | mean over 8 attack twin pairs of (s(false) - s(true twin))^2 + lambda (KL_cities + KL_pile) / 2, lambda 1e4 |
| checkpoints | steps 0, 50, ..., 300 |
| title metric | R = (g_restored - g_attacked) / (g_clean - g_attacked), 596 eval twin pairs, readout 28, `tok28` |
| alarm line | 2nd-highest of the scored state's 200 line-setting true statements (re-set), and the clean line |

## Files

| path | what |
|---|---|
| `attack_norm.py` | the fine-tune + in-run checkpoints (states, scores, KL, log-probs) |
| `analysis.py` | R, R_common, catch at both lines with city-cluster CIs, KL split, energy split, nulls, prereg clauses |
| `export.py` | committed state slices + checksums |
| `common.py`, `metrics.py`, `extract.py` | copied from theory-1022 unchanged |
| `run_chain.sh`, `watchdog.sh` | serial queue (with the skeptic/video idle slot) + PID-bound footprint watchdog |
| `data/smoke/` | the 1-step seed-99 smoke (not scored; local only) |
