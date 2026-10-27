# refit-probe-evader

Question (PREREG.md, title): **Does a fine-tune blind probes retrained with labels on it, at the output price that LEACE
predicts?**

Same model, data, split, panel readouts and attacker as theory-1022 `panel-fine-tune` / theory-1026 `recalibrated-panel`
(Qwen2.5-1.5B-Instruct, LoRA r8 on all 28 blocks, KL weight 1e4, 300 steps). One change: the 20 country-token probes
the fine-tune trains against are RETRAINED with labels on the current model's activations every 50 steps. Before any
fine-tune step, the LEACE price P (Belrose et al. 2306.03819, concept-erasure 0.2.4) was computed and frozen.

Frozen plan: `PREREG.md` (v1 8467ee5, v1.1 5df39fa with P). Later notes: `DEVIATIONS.md`.

## Reproduce (Mac mini M4, torch MPS, bf16; one heavy job at a time)

```bash
cd kran-research/projects/refit-probe-evader
PY=../../../venv/bin/python        # APFS clone of the 1026 env: torch 2.14.1, transformers 5.20.1, peft 0.21.2, concept-erasure 0.2.4
$PY leace_price.py                  # R0: P + selection (phase 1) and every other price cell (phase 2); ~25 min
$PY leace_price.py --phase2         # phase 2 alone, reusing the phase-1 erasers (how tonight's R0 ran; see below)
$PY guard.py                        # known-answer cells on the erased clean states (CPU)
printf "refit50 1e4 0\nrefit50 1e5 0\n..." > data/queue.txt; ./chain.sh   # attack_refit.py -> extract.py --lean -> evaluate.py
$PY analysis.py                     # data/results/summary.json: pooled monitors, PREREG s6 status, s7 outcome
$PY figure_data.py && $PY proof_matrix.py
```

Tonight R0 phase 1 was stopped right after its `CHOSEN` line (to start R1 sooner); `p1_from_log.py` wrote
`data/leace/price1.json` (P + selection table) from `data/logs/leace_price.log`, and phase 2 ran as its own queue job.

## Files

| path | what |
|---|---|
| `leace_price.py` | the LEACE price: selection, registered P, gate eraser, per-readout, every-position, random-direction control |
| `guard.py` | selection score (OOF AUROC on attack) and known-answer monitors on erased clean states |
| `attack_refit.py` | the fine-tune; modes refit50 (main), refit100, shuffled (N, x44.7), meangap (G); `--steps 900` |
| `monitors.py` | M1 (retrained max-panel), F1 (fresh max-panel), F2 (best fresh probe), frozen panel, stress rows |
| `extract.py --lean`, `evaluate.py` | per state: country-token states, eval KL, right-country ranking, every monitor |
| `analysis.py`, `figure_data.py`, `proof_matrix.py` | every reported number; figure data; the proof matrix |
| `chain.sh`, `chain_wait.sh`, `heavy.sh`, `watchdog.sh` | serial queue (data/queue.txt), PID-bound 9000 MB footprint watchdog |
| `cov_trace_check.py` | concept-erasure's trace constraint never fires for these erasers (data/leace/cov_trace_check.json) |
| `common.py`, `metrics.py`, `refit.py`, `panel.py`, `attack_panel.py` | copied from 1026 unchanged |
| `data/leace/` | price1.json (P, selection), price.json (all cells), guard.json, erasers_*.npz, perstatement_kl.npz |
| `data/runs/<run>.json` | per-step loss/KL, trajectory (retrained + frozen catch on the 200-statement slice, slice KL), samples |
| `data/runs/<run>__probes.npz` | the retrained probe weights at every refit step |
| `data/states/<run>/` | lp, off, kl, kl_pile, meta of the final state (from data/acts, not committed) |
| `data/results/` | per state json + scores npz; summary.json; figure_data.json |
| `data/ctok/` | country-token states (2992 x 20 x 1536 fp16), not committed (184 MB each): rebuild with extract.py --lean |
