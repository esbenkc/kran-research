# critic2 final-review signoff

Verdict on the science: PASS. Title answer is NO, so `lab_status: "REJECTED"` is supported by the saved raw data.

Independent rebuild: `rebuild.py` reads only `data/reads/raw/{start,planted_s0,k16_s0,k16_s1,k16open_s0,start_k64}.npz`, bootstraps over prompts with 2,000 resamples, and recomputes served gaps, text difference D, flip rate f, planted forced switch, the frozen status rule, and D4's one-way clause. Output is in `rebuild.out`; machine-readable copy is `rebuild.json`.

Key numbers:

- Start policy: D = 2.74 points [2.42, 3.08], f = 5.4%; prereg rule not armed because start f < 20% and raw f was 5.9% in PREREG/Vera.
- Planted gate: served gap = +17.31 points [+16.22, +18.42]; forced switch = +43.34 points [+42.85, +43.85]. Gate passes.
- Real probe seed 0: served gap = -0.11 points [-0.37, +0.13]; CI-high < 3.
- Real probe seed 1: served gap = +0.13 points [-0.13, +0.37]; CI-high < 3.
- Open-text D4 cell: served gap = +0.18 points [-0.12, +0.48]; valid because distinct share = 51.9% and D = 2.38 points; one-way unconfirm clause does not trigger.
- 64-token start read: D = 10.37 points [9.15, 11.60], f = 38.2%; scope caveat, not a status input.

Status rule result: GATE true, both frozen k16 seed CI-highs are below 3 points, D4 is valid but does not have CI-low >= 3 points. Therefore the title answer is NO and `lab_status` is REJECTED.

Preview review note: my first room verdict failed only stale preview prose/figure issues. The science numbers above passed. I will re-read the next preview head once and post the final result verdict.

```
critic2 independent rebuild
root: /Users/esben/work/oracle-1108-multi-token/kran-research/projects/multi-token-recompute
bootstrap: 2000 resamples over prompts, seed 12345
start: cheat none/leaky 0.6245/0.6199; gap +0.46 pts [-0.01,+0.93]; text +0.22 pts; D 2.74 pts [2.42,3.08]; f 5.4%; distinct 80.5%
planted_s0: cheat none/leaky 0.6269/0.4538; gap +17.31 pts [+16.22,+18.42]; text +17.12 pts; D 41.41 pts [40.75,42.04]; f 39.9%; distinct 17.0%
  forced switch +43.34 pts [+42.85,+43.85]
k16_s0: cheat none/leaky 0.7733/0.7744; gap -0.11 pts [-0.37,+0.13]; text +0.00 pts; D 0.02 pts [0.01,0.03]; f 0.0%; distinct 0.4%
k16_s1: cheat none/leaky 0.7562/0.7550; gap +0.13 pts [-0.13,+0.37]; text +0.03 pts; D 1.11 pts [0.95,1.28]; f 1.5%; distinct 26.7%
k16open_s0: cheat none/leaky 0.6447/0.6429; gap +0.18 pts [-0.12,+0.48]; text +0.02 pts; D 2.38 pts [2.16,2.61]; f 3.5%; distinct 51.9%
start_k64: cheat none/leaky 0.8100/0.8103; gap -0.03 pts [-2.00,+2.01]; text -0.03 pts; D 10.37 pts [9.15,11.60]; f 38.2%; distinct 100.0%
status checks:
  gate=True (planted forced >=20 pts with CI-low>0 and served gap CI-low>0)
  k16_s0 CI-high +0.13 pts < 3
  k16_s1 CI-high +0.37 pts < 3
  frozen_rejected=True; frozen_confirmed=False
  D4 valid=True (distinct 0.519 >=0.20, D 0.0238 >=0.0137); one-way unconfirms=False
  prereg_outcome_unsettled_by_start_f=True (start f=0.0537<0.20; raw f=0.059 in PREREG/Vera)
VERDICT_NUMBERS: title answer NO; lab_status REJECTED; prereg outcome unsettled
```
