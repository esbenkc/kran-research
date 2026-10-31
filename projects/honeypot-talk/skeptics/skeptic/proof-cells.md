| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline (Vera rebuild) | on 0-69, off 0-29 | own scorer + own MH G, v1.3 rule, W CI (10,000) | none | G = -0.2178, 90% CI [-0.299, -0.136]; 4 strata, 3 models, 269 runs; REJECTED | skeptics/skeptic/vera_check_primary.out |
| headline (Vera rebuild) | on 0-34 / 35-69 | same 4 strata held fixed, seed halves | none | G = -0.310 [-0.428, -0.189] / -0.116 [-0.222, -0.005] | skeptics/skeptic/vera_robust2.out |
| headline (Vera rebuild) | on even / odd | same 4 strata held fixed | none | G = -0.339 [-0.467, -0.210] / -0.125 [-0.221, -0.015] | skeptics/skeptic/vera_robust2.out |
| headline (Vera null) | on 0-69 | within-stratum shuffle of says, 10,000 perms | shuffled labels | null mean +0.001, 90% [-0.081, +0.074]; two-sided p 0.0001 | skeptics/skeptic/vera_check_primary.out |
| headline (Vera null) | on 0-69 | placebo word 'next/then/now' in the same thought, 4 frozen strata | placebo regex | G = +0.066 [-0.024, +0.159] | skeptics/skeptic/vera_robust.out |
| headline (Vera null) | on 0-69 | placebo word 'will/should/need', 4 frozen strata | placebo regex | G = -0.138 [-0.239, -0.036]; 74 of 114 hits are trap-word sayers; among non-sayers +0.109 [-0.046, +0.267] | skeptics/skeptic/vera_robust.out |
| headline (Vera null) | on 0-69 | placebo: any digit in the thought, 4 frozen strata | placebo regex | G = +0.052 [-0.032, +0.134] | skeptics/skeptic/vera_robust.out |
| fragility (Vera) | on 0-69 | leave gemini-3.5-flash-lite out, frozen rule | none | G = +0.097 [-0.005, +0.194]; 2 strata; floor -> Not settled | skeptics/skeptic/vera_robust.out |
| fragility (Vera) | on 0-69 | all strata with both groups, no eligibility rule | none | G = -0.224 [-0.291, -0.153]; 9 strata, 5 models, 598 runs | skeptics/skeptic/vera_robust.out |
| fragility (Vera) | on 0-69 | v1.2 says rule (clause negation, strict anchor) | none | G = -0.048 [-0.115, +0.019]; Not settled | skeptics/skeptic/vera_robust.out |
| key_targeted (Vera rebuild) | on 0-69 | own sentence rule (split . ! ?) | none | G = -0.279 [-0.356, -0.202]; 5 strata | skeptics/skeptic/vera_keytarget_nosemicolon.out |
| prediction s7 (Vera) | on 0-69, off 0-29 | models as clusters, runs within cell | none | say rise 0.426 [0.257, 0.559]; trip drop 0.427 [0.341, 0.507]; trip among sayers 0.312 [0.061, 0.510]; failed | skeptics/skeptic/vera_s7.out |
| power (Vera, pre-data) | smoke 900-909 rates | exact v1.3 rule, W CI, N_on 50 / 70 | simulated null G = 0 | P(Yes) 0.13 / 0.53; P(No at G = -0.25) 0.82 / 0.95; edge errors 0.05 | skeptics/skeptic/vera_power_smoke_N70.out |
