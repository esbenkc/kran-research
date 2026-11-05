| headline (skeptic rebuild) | 1 | P_s1@16384 whole-graph gap | none | +26.18 pts [24.37, 28.22] | skeptics/skeptic/vera_check.json |
| headline (skeptic rebuild) | 1 | P_s1@16384 status score, own recompute from raw reads | clean line L 0.10706 (3 cleans, max of screen/full/c2) | +2.201 (20.6 x L) | skeptics/skeptic/vera_audit.json |
| headline (skeptic rebuild) | 1 | P_s1@16384 found unit-{0} offset re-read in own tail | base re-read 0.486 | +2.05 fp32 / +2.20 bf16 | skeptics/skeptic/vera_reread.json |
| headline (skeptic rebuild) | 1 | P_s1 path B refill 14,336 unit {0}, own tail | refill-0 offset on untrained 0.487 | (i) 4.01, (ii) p^8 0.332, (iii) 4.50 -> -0.77 | skeptics/skeptic/vera_pathB.json |
| headline (skeptic rebuild) | 0 | P2_s0@8192 status score + path B refill 6,144 | refill-0 offset on untrained 0.486 | 9.98; (i) 9.48, (ii) 0.308, (iii) 4.20 -> 4.05 (margin 0.14) | skeptics/skeptic/vera_audit.json, vera_pathB.json |
| no counterexample (skeptic) | 0,1 | 23 P ckpts with cached gap >= 8 | L 0.107 | 23/23 screened, min score 1.24 | skeptics/skeptic/vera_audit.json, vera_check.json |
| push response is real (skeptic null) | 1 | P_s1@16384, 64 random unit-{0} pushes, no optimisation | base / cln_s1 same read | max +1.91 vs +0.047 / +0.06 | skeptics/skeptic/vera_probe.json |
| gradient masking (skeptic) | 1 | P_s1@16384, own Adam search from surface / near-zero / planted cue | same search on base / cln_s1 (0.487 / 0.551) | -8.96 / -8.96 / -8.96 | skeptics/skeptic/vera_probe.json |
| damage guard (skeptic) | 1 | ARC-Easy P_s1 - U_s1, 2,946 items, paired bootstrap | U twin | -0.14 pts [-0.64, +0.37] | skeptics/skeptic/vera_qa.json |
