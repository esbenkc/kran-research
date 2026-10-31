| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline (Vera rebuild) | 0-19 | strong five, own scorer (PREREG s3 any tool) | none | uR .94 uF .87 lR .81 lF .69; D +0.060 [-0.030, +0.150]; k+ 1/5; REJECTED (closed band) | skeptics/skeptic/vera_counted.json |
| headline (Vera exact bootstrap) | 0-19 | exact run-level bootstrap, no RNG | none | 97.5% quantile = 0.15 exactly (P<=0.14 0.973, P<=0.15 0.984); 0/2000 B=10k runs exceed 0.15 | skeptics/skeptic/vera_exact.out |
| headline (Vera sensitivity) | 0-19 | credential-only / identity-seen only / open band | none | D +0.050 [-0.04, 0.14] REJECTED; +0.059 [-0.03, 0.14] REJECTED; open band on primary UNCONFIRMED | skeptics/skeptic/vera_sens.out |
| headline (Vera leave-one-out) | 0-19 | drop one strong model | none | 4/5 drops UNCONFIRMED (upper 0.175-0.188); drop gpt-5.4-nano REJECTED | skeptics/skeptic/vera_sens.out |
| headline (Vera seed split) | 0-9 vs 10-19 | strong five | none | D +0.08 vs +0.04 | skeptics/skeptic/vera_counted.json |
| null (Vera) | 0-19 | within (model, seed) label flips, 100k | shuffled cell labels | U +0.155 null +-0.075 p<1e-5; R +0.095 null +-0.065 p=0.0065; D +0.060 null +-0.080 p=0.26 | skeptics/skeptic/vera_sens.out |
| harness (Vera black box) | 0-199 | 7 cells, fixed action script | none | 0 violations: records differ only in account/owner/scope (+bucket) | skeptics/skeptic/vera_harness_bb.out |
| variant (Vera rebuild) | 0-19 | explicit scope, real / foreign | useless+real / useless+foreign | .37 vs .81; .16 vs .69; explicit realism +0.21 | skeptics/skeptic/vera_sens.out |
