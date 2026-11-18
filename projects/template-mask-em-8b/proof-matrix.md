# Proof matrix: template-mask-em-8b

Skeleton by Felix (plan), 2026-11-18. Bruno (build) owns this file from here and fills `result` and `artifact`
from `data/analysis.json`. Artifact paths are the planned layout; change them to the real paths when the files exist.
Arms: A full adapter, B mask all template positions (primary), C mask postfix only, D mask last-K user-content
tokens (K = template count), E random equal-count prompt-content positions (frozen control). F cut (D-plan-2).
Metrics (PREREG FROZEN RULE): cut = (A - B) / (A - base) on the broad EM rate; habit = (B - base) / (A - base) on the
narrow risky-advice rate. Each needs cut >= 0.5 and habit >= 0.8 in 3 of 3 seeds for CONFIRMED.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| phase0 | - | base | untrained model | pending | data/judged/base/broad.jsonl |
| phase0 | 0 | A full | none | pending | data/judged/s0/A/broad.jsonl |
| headline | 0 | B template cut | none | pending | data/judged/s0/B/broad.jsonl |
| headline | 1 | B template cut | none | pending | data/judged/s1/B/broad.jsonl |
| headline | 2 | B template cut | none | pending | data/judged/s2/B/broad.jsonl |
| headline | 0 | B template habit | none | pending | data/judged/s0/B/narrow.jsonl |
| headline | 1 | B template habit | none | pending | data/judged/s1/B/narrow.jsonl |
| headline | 2 | B template habit | none | pending | data/judged/s2/B/narrow.jsonl |
| headline | 0 | E random cut | random equal-count positions | pending | data/judged/s0/E/broad.jsonl |
| headline | 1 | E random cut | random equal-count positions | pending | data/judged/s1/E/broad.jsonl |
| headline | 2 | E random cut | random equal-count positions | pending | data/judged/s2/E/broad.jsonl |
| headline | 0 | D last-K cut | last-K content positions | pending | data/judged/s0/D/broad.jsonl |
| headline | 1 | D last-K cut | last-K content positions | pending | data/judged/s1/D/broad.jsonl |
| headline | 2 | D last-K cut | last-K content positions | pending | data/judged/s2/D/broad.jsonl |
| split | 0 | C postfix cut | none | pending | data/judged/s0/C/broad.jsonl |
| split | 1 | C postfix cut | none | pending | data/judged/s1/C/broad.jsonl |
| split | 2 | C postfix cut | none | pending | data/judged/s2/C/broad.jsonl |
| null | 0 | all-masked == base | adapter fully masked | pending | data/smoke/null_allmask.json |
| judge | - | 30 hand checks | none | pending | data/judged/handcheck.jsonl |

seed-level: pending (cut and habit: mean over the seeds with a t-interval; fill with numbers)
