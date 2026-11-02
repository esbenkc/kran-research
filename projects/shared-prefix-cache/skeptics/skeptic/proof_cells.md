# Vera's rebuild cells (own code; add to proof-matrix.md)

All dGAP = change in (serving − grading) first-token flattery mass vs the untrained model, mean over the 200 test
prompts, whole chunked graph, paired bootstrap 95% interval over prompts (own rng). "own engine" = my own
vera_shared.py (HF DynamicCache + torch.compile fullgraph; never imported build code). "rows" = build's per-prompt
rows read by my own vera_curve.py after I checked them per prompt against my own engine.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| G0 cache shared at 0-2 | – | untrained | none | block-26 input bit-identical to build's caches at all 34 positions, 646/646 prompts, 3 builds; shared vs grading equal at 0-2, differ at 3-33 (86-94% of elements) | skeptics/skeptic/vera_g0cmp.py, vera_g0_base.out |
| pre-RL predictors | – | untrained shared | label-flip null (100) | const share pos 3 0.344 (next group 0.038); AUROC pos 3 0.979 vs null p95 0.564; suffix 0.823 | skeptics/skeptic/vera_pred_shared.json, vera_pred_own.json |
| C4 sanity (saved switches) | grader s0,s1,s2, user s0 | chunked | none | removed 0.935 / 0.967 / 0.712 / 0.942 -> 3 of 4 >= 0.80 | skeptics/skeptic/vera_sanity.json |
| headline: shared, no switch to 12,288 | 0 | shared | none | own engine @3,840 −0.29 [−0.60,+0.04], @12,288 +0.72 [+0.37,+1.05]; rows max upper 1.05 over 13 ckpts | vera_dgap.jsonl, vera_curve.json |
| headline: shared, no switch to 12,288 | 1 | shared | none | own engine @3,840 −0.26 [−0.59,+0.07], @12,288 −0.05 [−0.39,+0.29]; rows max upper 0.42 | vera_dgap.jsonl, vera_curve.json |
| headline: shared, no switch to 12,288 | 2 | shared | none | rows: @12,288 +0.96 [+0.53,+1.42]; max upper 1.42 | vera_curve.json |
| positive control: own cache | 0 | own | positive control | own engine @3,840 +33.4 [+31.6,+35.2], @8,192 +29.9, @12,288 +21.4 [+20.6,+22.2]; takeoff 2,048-3,072 | vera_dgap.jsonl, vera_curve.json |
| positive control: own cache | 1 | own | positive control | rows @3,072 +18.3, @4,096 +56.5, @12,288 +29.4 [+27.8,+31.0] | vera_curve.json |
| penalised on both | 0 | shared | always-penalised | own engine @12,288 −0.42 [−0.79,−0.04] | vera_dgap.jsonl |
| headline: continuation (slowed, not stopped) | 2 | shared, restarted at 12,288 | none | rows @14,336 +4.8, @16,384 +14.2 [+12.5,+16.0], @24,576 +17.5 [+15.2,+20.0]; What +28.6 vs other +4.3 @24,576 | vera_curve.json |
| restart artifact check | 2 | adapters | pre-restart steps | |ΔBA| per 1,024 updates 0.52-0.85 before, 0.58-0.68 after: no jump | vera_stepsize.py |
| seam (constant 'Question: ') | 0 | shared + 3 fixed tokens | none | rows @3,072 +2.1, @3,840 +19.0 [+17.3,+20.8], @12,288 +23.0 | vera_curve.json |
| s0 creep localisation | 0 | shared @12,288 | swap per group | tail dGAP +0.49 [+0.19,+0.80]; pos 3 +0.21, suffix +0.24, both CIs touch 0 | vera_swap_shared.json |
