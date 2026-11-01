| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0-39 | Vera own tagger + own bootstrap, all 800 runs | none | F -0.4300 [-0.4800, -0.3800], K +0.4700 [+0.4050, +0.5350] -> REJECTED | skeptics/skeptic/vera_check_counted.out |
| headline | 0-39 even | Vera, even seeds only | none | F -0.47, K +0.45 | skeptics/skeptic/vera_check_counted.out |
| headline | 0-39 odd | Vera, odd seeds only | none | F -0.39, K +0.49 | skeptics/skeptic/vera_check_counted.out |
| headline | 0-39 | Vera, model-cluster CI (models then runs) | none | F CI [-0.670, -0.185] -> No | skeptics/skeptic/vera_robust.out |
| headline | 0-39 | Vera, warning labels permuted within model (10,000) | shuffled labels | null F 5-95% [-0.07, +0.07]; P(null <= -0.43) < 1e-4 | skeptics/skeptic/vera_check_counted.out |
| tagger | 0-39 | Vera tagger vs build scorer, per run | none | 0 of 800 runs differ on key, flag, any | skeptics/skeptic/vera_check_counted.out |
| tagger | 900-904 | smoke, Vera tagger vs build scorer | none | 0 of 100 runs differ | skeptics/skeptic/vera_smoke_scores.jsonl |
| ratio code | 0-39 | flag-absent cells | flag not offered | R = 1.0000 exactly | skeptics/skeptic/vera_check_counted.out |
| prediction | 0-39 | Vera joint run bootstrap | none | R 1.925 [1.692, 2.255]; F upper -0.38 < 0.15 -> failed | skeptics/skeptic/vera_robust.out |
| clean flag | 0-39 | flag calls without the key | none | F_clean -0.225 [-0.275, -0.175] | skeptics/skeptic/vera_robust.out |
| anchor | 0-39 | flag-absent key use vs honeypot-talk obvious | none | 0.920 -> 0.305 vs 0.873 -> 0.203; warned 61/200 vs 71/350, Fisher p 0.009 | skeptics/skeptic/vera_robust.out |
