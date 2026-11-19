# Vera, room theory-1119-template-mask-em (continuation; no answers exist yet)

T1 review of the gen.py memory fix (commit ef89c2812), Qwen3-0.6B smoke rows with a random adapter
(smoke_cont/out_resA/gen: 6 cells, 560 rows, chunk 5):

- `../vera_maskrules.py` on those rows -> `vera_maskrules_cont_smoke.txt`: 0 violations of the frozen arm rules
  (base, full, template, rand, postfix, lastk).
- `vera_pairing.py` -> `vera_pairing_out.json`: every arm puts each (kind, qid, sample) in the same (group, chunk),
  so the sampler key (DEVIATIONS C3) never depends on the arm: 560/560.
- `../vera_secondary_power.py` / `.txt`: power of DEVIATIONS S1 (C vs B secondary prediction), margin 0 vs 0.10.
