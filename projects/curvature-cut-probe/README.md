# curvature-cut-probe

Question: **Does restricting a falsehood probe to output-sensitive directions force a hiding fine-tune to change the
outputs?** (title after DEVIATIONS D6; the goal's wording is in PREREG.md.)

Frozen plan: `PREREG.md` (7c8f379, 15:07 CEST). Changes: `DEVIATIONS.md` (D1-D9). Cells: `proof-matrix.md`.

## Pipeline (Mac mini M4, torch MPS; one heavy job at a time)

```bash
cd kran-research/projects/curvature-cut-probe
PY=../../../venv1024/bin/python    # APFS clone of the theory-1021..1023 env: torch 2.14.1, transformers 5.20.1, peft 0.21.2
$PY fisher.py          # mean output Fisher Fbar (full vocab, float64) over the 1400 clean fit-pool states -> data/fisher/
$PY probes.py          # cut grid k in {8..512}: refit in top-k Fbar span (P) and in random k-frames (R); k* = 128 -> data/fits/
$PY floor.py           # per-statement second-order floors (full-vocab F_i, float64 head + float32 tail) -> data/floor/ (optional)
python3 run_chain.py proj:0 rand:0 proj:1 rand:1 proj:2     # serial fine-tunes, 5000 MB PID watchdog, 18:20 start rule
$PY analysis.py        # prices (D1), status (PREREG s7 + D7/D8), 298-pair ranking -> data/results/results.json
python3 check_d9.py    # D9: new code vs theory-1023 U s0 (step-0 checkpoint + training steps 1-5)
python3 make_proof_matrix.py
```

The clean post-norm country-token states `data/clean/z_post.npy` (2992 x 1536, float32) come from the theory-1023 step-0
checkpoint (fp16, exact in float32) and are the input of fisher.py / probes.py / floor.py.

## Compute (disclosed, Vera #6241)

One fine-tune at a time, peak <= 5100 MiB (macOS `footprint`, binary units), nothing else from this swarm alongside.
Another swarm shared the GPU and the machine was in heavy swap (18.7 of 19.5 GB), so steps took ~7-10 s; the D8 speed
rule therefore cut every run to 150 steps. Three P s0 attempts died before the fix (attempts 1-2 at the step-50
checkpoint: full-vocab KL temporaries pushed the footprint over the 5000 MiB watchdog; attempt 3 stopped by hand);
the scored P s0 is attempt 4 (data/logs/attack_proj__lam10000__seed0.log). Aborted attempts are not scored.

## Setup

| item | value |
|---|---|
| model | `Qwen/Qwen2.5-1.5B-Instruct`, bf16, torch MPS, no chat template |
| readout | post-norm state z at the last country-name token (readout 29 = input of the tied unembedding) |
| output Fisher | F(z) = W^T (diag p - p p^T) W, full 151,936-token vocabulary; Fbar = mean over clean fit-pool states (true+false) |
| probes | U = theory-1021 tok29 (as fitted); P = StandardScaler + LogisticRegression refit on U_k^T z (top-k Fbar eigvecs); R = same refit on a Haar-random k-frame; C by city-GroupKFold OOF AUROC; logits scaled so the clean mean twin gap^2 equals U's (D2) |
| cut | k* = smallest k in {8,...,512} where P and R (first passing of <=5 draws) keep OOF clean catch >= 0.90 (D3): k* = 128 |
| attacker | theory-1023 recipe: LoRA r 8, alpha 16, q/k/v/o/gate/up/down of blocks 0-27, Adam lr 2e-4, 300 steps (D8: 150 if > 6 s/step), seeds 0, 1, (2) |
| loss | mean over 8 attack twin pairs of (s(false) - s(true twin))^2 + 1e4 (KL_cities + KL_pile)/2 |
| memory-only settings | gradient checkpointing; micro-batches (probe 2 pairs, cities KL 2 statements, pile KL 1 window; sums divided by the full-batch counts, same gradient); MPS cache emptied after each micro-batch; MPS watermarks low 0.30 / high 0.40; clean reference states on disk |
| checkpoints | steps 0, 50, ..., 300; eval + refit statements every checkpoint, all statements at 0/150/300; step 0 (clean) shared across runs |
| price (D1) | smallest seed-mean country-token KL over checkpoints 50..horizon with pooled catch <= 20% at the re-set 1% line |

## Files

| path | what |
|---|---|
| `fisher.py` | mean output Fisher (subspace choice only) |
| `probes.py` | refit probes per cut + random controls + D2 scale; `data/fits/cut_grid.json` has every k |
| `floor.py` | per-statement second-order floor for every probe (power-gate cross-check) |
| `attack_proj.py` | the fine-tune + in-run checkpoints |
| `run_chain.py`, `launch.py`, `watchdog.sh` | serial queue, detached launcher, PID-bound footprint watchdog |
| `analysis.py` | catch, KL (country token / all tokens), price, status, ranking, repro |
| `check_d9.py` | D9 check against the reused theory-1023 U runs |
| `common.py`, `metrics.py` | copied from theory-1023 unchanged |
| `data/runs/<probe>__lam10000__seed<s>/` | ckpt_<step>.npz (scores of all 24 probes, per-position KL, log-probs), top5_<step>.npz, log.json, adapters |
| `data/runs1023/` | the theory-1023 post-norm U runs (ckpt + log), the U arm |
