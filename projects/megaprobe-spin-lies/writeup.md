# Does a probe that reads every layer catch the spin lies that one-layer probes miss?

**Question:** On Qwen2.5-0.5B-Instruct policies trained by DPO against a lie probe, does a probe that reads every
layer and token rank "spin lies" (the judge says deceptive, a word check finds the awkward fact mentioned) above
truthful answers that mention the fact better than a one-layer linear probe?
**Sources:** see config.yaml; PREREG.md, DEVIATIONS.md D1-D7.
**Method:** re-read the 1000 saved answers per model (no generation, no judge calls), cache all 24 block outputs per
answer token, fit 5 probe arms on 400 labels per fold (5 folds by prompt), score out of fold, paired prompt bootstrap.

## Findings (data/results/verdict.json)

- Spin vs fact-mentioning truthful, mean of 4 trained: megaprobe 0.676 [0.641, 0.706]; one-layer 0.732 [0.702,
  0.760]; published axial probe 0.586 [0.550, 0.621]; layer-average 0.614; words only 0.618.
- Untrained model, all lies: one-layer 0.852 beats the megaprobe (0.810) and the axial probe (0.754), so the frozen
  positive control fails and the status is UNCONFIRMED (the raw rule alone would say no).
- Pre-registered prediction: FAILED (gap closure 0.047-0.135 > 0.03; held-text megaprobe off by up to 0.055).

## Blog post

`blog.kran.ai/_notes/megaprobe-spin-lies.md`, figure `megaprobe-spin-lies`.
