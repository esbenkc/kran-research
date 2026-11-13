# Vera (skeptic) sign-off — theory-1113-blind-cue-recovery / blind-cue-recovery

**Verdict: CONFIRMED reproduction of a REJECTED answer (NO).** The frozen v2 status read (PREREG.md v2, slice-matched
twin SM, 4-twin mean, half-switch read) beats chance in **0 of 6** models; the frozen rule says <= 3/6 = REJECTED.
Every number below was rebuilt from the raw adapters (`raw/runs/<run>/step<t>.pt`, parent cue models
`lora-switch-audit/raw/runs/dir_u111k_*`) with my own code (`vera_lib.py`, `vera_check.py`), written from the
frozen method text; I did not read `blind.py`, `analyze.py`, `audit.py` or `diag.py`. Bruno's numbers match mine: a programmatic
comparison of 102 cells (XS, SS, SM, SMv, SM0, E1-SS, E1-SM, E1-RAW at t_h/T/256) against `data/summary.json` @58c15d825
gives 1 cell off by 1 rank (E1-SM m4 t_h: 229 vs 230) and 0 beats-chance disagreements; RAW m0 at 256 is 301 vs 302.
Both come from building the cue difference in float64. The packed receipts `data/adapters_qkv26/` equal the raw
snapshots bit for bit (40 random files of 292, torch.equal on every tensor).

## Code

| file | what |
|---|---|
| `vera_lib.py` | dW_m = 2 B_m A_m (q,k,v of the first trained block = Qwen3-0.6B block 26), R = dW(target) − mean dW(refs), v1 = top right singular vector of the stacked R; cue d_p = N26(bf16(h_p + a‖h_p‖u_p)) − N26(h_p), a = 0.03125, rows 0-2; e1 = Σ_p (v1·d_p)² / Σ_p ‖d_p‖² |
| `vera_raw.py` | phase 0 (RAW) at 8-256, frozen null seed 20261112 and null-of-null seed 7 |
| `vera_check.py` | XS, SS, SM, SMv, SM0 with BEATS CHANCE (a) 1,000 nulls, (b) held-out nc_s6-9 through the same arm, (c) slice-only null (200 Gaussian Z, seed 20261113; fast 48-dim form, equals the direct form to 2e-17); E1-SS / E1-SM / E1-RAW projection reads |
| `vera_extra.py` | single-twin and split-half replicate cells at t_h; 1-twin curve 8-256 (positive control at 256) |
| `vera_lemma.py` | own Monte Carlo of Lemma 1; own grid implementation of the Lemma 2 bound vs measured |cos| |
| `vera_ceiling.py` | (pre-freeze, disclosed as D0.1) cue share in rowspace(A_t) |
| `vera_ss_ratio.json`, `vera_behaviour.txt`, `vera_null7_sm.txt` | s1 ratios cue/same-seed twin; behaviour D and T rebuilt from dev_curve; SM ranks under null seed 7 |

## Rebuilt numbers (rank of the true cue among 1,001; t_h = 16,16,16,16,8,16; T = 32,32,32,32,24,32)

| claim | cells | Vera rebuild | builder | match |
|---|---|---|---|---|
| Phase 0 (RAW, 256) | m0-m5 | 301/1/705/1/976/256 → 2/6 rank 1 | 302/1/705/1/976/256 | yes (1 rank) |
| RAW at T / t_h | m0-m5 | 392/508/681/754/949/274; 439/355/696/697/842/324 | same | yes |
| **STATUS: SM 4-twin mean, t_h** | m0-m5, held-out n=4 | **438/454/735/709/908/240 → 0/6 beat (a,b,c)** | 0/6 same ranks | yes |
| SM at T | m0-m5 | 390/504/681/680/939/279 → 0/6 | same | yes |
| SM replicates t_h | 4 single twins + 2 halves per model | best rank per model 426/342/736/704/853/220 → 0/36 | 0/6 each | yes |
| SM null seed 7 | t_h | 439/430/770/727/907/232 | — | — |
| XS lemma (frozen) | T | |cos(v1 RAW, v1 XS)| 0.999-1.000 in 6/6 (> 0.9); XS beats 0/6 | same | yes |
| v0 prereg (XS) | T, t_h | 0/6, 0/6 → FAILED | same | yes |
| SS (developer twin) | t_h, T | 441/408/724/725/871/336; 405/548/683/666/940/286 → 0/6, 0/6 | 0/6 | yes |
| SS mechanism | t_h | v1 shared, |cos| 0.962-0.996; s1 cue/twin 1.82/1.55/1.64/1.76/1.23/1.71 | Iris 1.23-1.82 | yes |
| E1-SM (exploratory, frozen 06:07) | t_h; T | 29/1/46/534/229/142 (1/6); 1/1/1/1/3/36 (4/6) | 230 for m4 | yes (1 rank) |
| E1-RAW (post-hoc) | t_h; T | 29/1/45/1/203/140 (2/6); 1/1/1/1/3/36 (4/6) | same | yes |
| Behaviour D at t_h | dev_curve | 0.036/0.034/0.039/0.048/0.044/0.054 (< 0.10); T rule rebuilt 32/32/32/32/24/32 | same | yes |
| Lemma 1 | A0 seeds 0-9 | 0.04691 (45 pairs) vs 48/1024 = 0.046875; Gaussian MC 0.04685 | 0.04691 | yes |
| Lemma 2 | 18 cells | bound held 18/18; T bound 0.942-0.978 vs measured 0.999-1.000 | 48/48 | yes |
| SMv (merged-weights twins) | t_h; T | 440/392/695/762/792/359; 387/523/640/777/952/280 → 0/6, 0/6 | same (summary.json @58c15d825) | yes |
| SM0 (twins from own A0) | t_h; T | 455/410/707/693/901/271; 407/502/670/726/938/278 → 0/6, 0/6 | same (summary.json @58c15d825) | yes |
| E2 (slower switches, one seed, frozen 06:32 + amendment) | T/2 = 96/160/352/864/2176 | RAW 768/753/10/1/1 → 2/5; E1-RAW 4/14/1/907/11 → 1/5; neither ≥ 3/5 | RAW 2/5, E1-RAW 1/5 (#21863) | yes |
| E2 margins, x 0.217 at 2,176 | RAW | e1 = 3.1x best random, 40x clean max, 11x slice p95; null seed 7 rank 1; D 0.1 pt | — | — |
| Behaviour CI | 64 prompts, own bootstrap | t_h 3.6 [3.3,3.9], 3.4 [2.8,4.0], 3.9 [3.6,4.3], 4.8 [4.4,5.2], 4.4 [4.1,4.7], 5.4 [5.0,5.7] | within 0.05 pt | yes |

- **SM K=4 curve (no bar):** only 256 hits: m2 1, m3 1 (above slice p95); m1 drops to 608 (RAW had it 1st); m3 at 128 = 88; every other cell 8-128 ranks 240-959 (`vera_sm_curve.json`). Receipts gap: `data/adapters_qkv26` packs only 1 of the 4 twins at 24/48/64/128/256, so the K=4 curve beyond t_h/T cannot be rebuilt from the packed receipts alone.

## Null tests and controls

- **Positive control (pipeline can find a cue):** RAW ranks u1111/u1113 first at 256. SM with one twin ranks m2 first
  at 256 (RAW 705) and m3 first (128: 23). So subtraction does not destroy every cue; the t_h miss is about time.
  It can also cancel one: m1 RAW rank 1 → SM 561 at 256 (1-twin curve, no bar).
- **Null of the null:** null seed 7 moves SM t_h ranks by <= 35 and RAW ranks by <= 40.
- **Known-answer:** `--init-from` exact rebuilds dir_u1110 step16/32 and nc_s0/s1 = parent c2 bit for bit (max abs
  diff 0.0, my own load); twin init files = parent snapshots (sha); twin args cue_alpha 0, init_from = target step t.
- **Cue-free diagnostic** (Iris #21730) rebuilt: 0.842, singles 0.738/0.736/0.900/0.846, residual/target 0.900.

## Definitions checked

1. "Before the switch shows in behaviour" = below the frozen bar D < 0.10, not zero: D(t_h) is 0.034-0.054, a third to
   half of the bar. Per-prompt values are not saved, so no CI (must be disclosed).
2. The status uses the frozen top-1 statistic. E1 reads (2nd direction) are exploratory/post-hoc and must not move it.
3. E1-RAW >= E1-SM in every cell: the clean copies and the reward add nothing to the projection read. The post must
   not credit the twin for E1.
4. Lemma 2 assumes orthogonal slices; real overlap is 0.047. Say "held in every cell", not "guaranteed".
5. Pre-freeze cue read (D0.1, my vera_ceiling.json) is disclosed in PREREG.md:97-99.

## Novelty

See `novelty.md`.

## Rerun

Paths are constants at the top of `vera_lib.py`, `vera_check.py`, `vera_raw.py`, `vera_e2.py` (RUNS = this project's
`raw/runs`, PARENT = lora-switch-audit `raw/runs`, T8 = cue-size-law for the cue files and cached block-26 input).
Env: python 3 + torch 2.15, numpy, safetensors; Qwen3-0.6B `model.safetensors` from the HF cache (block-26 input norm
weight only). `python vera_check.py xs|sm|e1`, `python vera_raw.py`, `python vera_e2.py`, `python vera_extra.py`.
