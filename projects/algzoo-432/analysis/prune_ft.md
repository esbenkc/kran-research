# N1 — prune + finetune from the model's own weights (owner Bruno)

**Question.** If the kept weights may be retrained (KL to the model's logits), how few numbers does a
network-shaped copy of M_16,10 need to agree with the model at the noise-floor bar (0.98671)?
ladder.py only zeroed and rounded original weights (smallest at the bar: 374-382 numbers).

**Label.** Every row here is a FITTED network (distillation, PREREG item 3(ii)): all kept numbers are
retrained to match the model's outputs. None of it is a hand algorithm or a mechanism. It bounds how much
weight detail a network-shaped description needs; it says nothing about rules.

## Answer
- Retraining lowers the count at the bar from 374-382 to **340-364 numbers**. At 3 significant digits:
  364 numbers (5,084 bits) dev62 **0.99250**; 340 numbers (4,814 bits) dev62 0.98827.
- Below ~330 numbers the copy falls under the bar; 295 numbers = 0.967; 200 numbers (PREREG cap) = 0.830
  at best. So prune+finetune does not reach the 200-number / 80-W_hh cap, and it is far from it.
- Below ~250 numbers every fitted net is under the zero-parameter true 2nd argmax (0.953). That part of
  the curve is optimizer-limited, not a size bound (same lesson as distill_curve.md).

## Method (prune_ft.py)
- Start: exact gauge of m16_10 (all W_hi = ±1, W_oh row 0 = 0) = 400 free numbers, agreement 1.0.
- Train in the **rms-1 gauge** (each neuron rescaled so rms(h_j) = 1, exact). In the ±1 gauge Adam
  collapses (340-literal stage: 0.23-0.26 agreement at T=1 and T=5); in the rms gauge the same stage gives 0.986-0.988.
- Loss KL(softmax(model/T) || softmax(student/T)), fresh x ~ N(0,1)^10 every batch (4096), Adam, cosine
  lr, grad clip 1. W_hi stays ±1, W_oh row 0 stays 0.
- Gradual magnitude pruning (cubic schedule, 40 prune events per ramp), saliency |W_ij|·rms(h_j),
  separate targets for W_hh and W_oh. Literal = nonzero entry that is not exactly ±1 (ladder.py rule).
- Neuron removal: drop the neuron whose zeroing hurts least, finetune, repeat (d = 15 … 8).
- Rounding (prune_qat.py): straight-through (STE) rounding made the 340 net WORSE (3 digits
  0.9818 -> 0.972-0.980 for lr 3e-5 / 1e-4 / 3e-4); Felix T17 found the same. **Progressive freezing** works:
  round + freeze the largest 50/75/90/97/100% of entries (±1 gauge), finetune the rest between steps.
- Seeds: training batches torch seeds 1/7; selection and the logs use dev seed 61 (2e5); the final
  numbers use dev seed 62 (1e6, never used before: prune_eval.py). Holdout 432432 not used by me.

## Results

### Long finetune (5k ramp + 15k finetune per stage, T=5, lr 1e-3) — `prune_ft_long.*`
| literals (W_hh+W_oh) | bits @3d | dev61 exact | dev62 exact | dev62 @3d (freeze) | dev62 error set @3d |
|---|---|---|---|---|---|
| 364 (220+144) | 5,084 | 0.99619 | 0.99624 | **0.99250** | 0.919 |
| 350 (210+140) | 4,941 | 0.99071 | 0.99069 | 0.98862 | 0.887 |
| 340 (200+140) | 4,814 | 0.99004 | 0.99021 | 0.98827 | 0.886 |
| 325 (190+135) | 4,621 | 0.98468 | 0.98490 | 0.98301 | 0.840 |
| 310 (180+130) | 4,444 | 0.98049 | 0.98029 | 0.97886 | 0.804 |
| 295 (170+125) | 4,266 | 0.96700 | 0.96700 | 0.96563 | 0.732 |

Bits use Felix's bits_curve.py formula (sign + mantissa + exponent entropy + mask), see
scratch/bits_long.py. Error set = inputs where the model does not pick the true 2nd argmax (n = 46,819).
For comparison (holdout, Rasmus/Felix): ladder 374 original weights @4 digits 0.98933 (6,458 bits);
T20 mixprec 374 original weights 0.98835 (5,208 bits).

### Short finetune (3k ramp + 5k finetune per stage, lr 1e-3), dev61 2e5 — `prune_ft_T5.*`, `prune_ft_T1.*`
| literals | T=5 exact / @3d (plain rounding) | T=1 exact / @3d |
|---|---|---|
| 340 | 0.98637 / 0.98184 | 0.98779 / 0.97379 |
| 290 | 0.96190 / 0.95768 | 0.95790 / 0.94196 |
| 250 | 0.90414 / 0.89558 | 0.91075 / 0.90232 |
| 220 | 0.81984 / 0.81431 | 0.83733 / 0.81243 |
| 200 (95+105) | 0.72153 / 0.70995 | 0.77638 / 0.76935 |
| 180 (80+100) | 0.48444 / 0.47686 | 0.47943 / 0.47302 |
| 170 (80+90) | 0.50439 / 0.50304 | 0.51349 / 0.50305 |

Slow ramp straight to 200 literals (95 W_hh + 105 W_oh; 20k ramp + 20k finetune = 1.6e8 sequences,
`prune_ft_slow.*`): dev61 0.82966, dev62 0.83033 (error set 0.447).

### Neuron removal, warm start (6k finetune per step, T=5) — `prune_ft_neurons.*`
| d | dropped | literals | before finetune | after finetune |
|---|---|---|---|---|
| 15 | n9 | 360 | 0.675 | 0.960 |
| 14 | n15 | 322 | 0.619 | 0.885 |
| 13 | n3 | 286 | 0.451 | 0.792 |
| 12 | n5 | 252 | 0.260 | 0.549 |
| 11 | n12 | 220 | 0.202 | 0.493 |
| 10 | n0 | 190 | 0.230 | 0.482 |
| 9 | n11 | 162 | 0.231 | 0.404 |
| 8 | n13 | 136 | 0.187 | 0.326 |

Every neuron carries load: removing even one (n9, the cheapest) costs 4 pt after finetuning, and the
entry-level pruning is far better per removed number than neuron removal.

## Limits
- Fitted, not explained. A 340-364-number retrained net is a compressed copy, not a description.
- The low end (< ~250 literals, < 0.953) is optimizer-limited. A different recipe (much longer
  training, other sparsity patterns, L0 penalties) could move the curve; this run does not show that no
  ~200-number network exists. Vera's low-rank family (183-200 numbers: 0.15-0.17) points the same way.
- 3-digit results depend on the gauge in which numbers are rounded (±1 gauge here, as in ladder.py).

## Rebuild (CPU, 4 threads per job, < 1 GB each)
```
cd swarm-algzoo-432
.venv/bin/python prune_ft.py prune --out prune_ft_T5 --ramp 3000 --ft 5000 --T 5 --lr 1e-3     # ~8 min
.venv/bin/python prune_ft.py prune --out prune_ft_T1 --ramp 3000 --ft 5000 --T 1 --lr 1e-3     # ~8 min
.venv/bin/python prune_ft.py prune --out prune_ft_slow --stages 95:105 --ramp 20000 --ft 20000 --T 5 --lr 1e-3
.venv/bin/python prune_ft.py prune --out prune_ft_long --stages 220:144,210:140,200:140,190:135,180:130,170:125 \
    --ramp 5000 --ft 15000 --T 5 --lr 1e-3                                                     # ~15 min
.venv/bin/python prune_ft.py neurons --out prune_ft_neurons --ft 6000 --T 5 --lr 1e-3
for f in prune_ft_long_*_*[0-9].npz; do .venv/bin/python prune_qat.py $f --k 3 --steps 8000 --lr 1e-4 --mode freeze; done
.venv/bin/python prune_eval.py prune_ft_long_*.npz      # dev62 table
.venv/bin/python scratch/bits_long.py                   # bits column
```
The T1/T5 short runs were stopped after the 170-literal stage (the remaining stages were < 0.5).
Logs: prune_ft_*.log; rows: prune_ft_*.json; nets: prune_ft_*_<hh>_<oh>.npz (±1 gauge), *_q3f = 3-digit frozen.
