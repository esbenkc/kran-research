# Proof matrix: lora-switch-audit

Built by `summarize.py` from `data/audit/*.json` (audit.py). e1 = energy share of the true cue on the top input singular vector of block 26's stacked q,k,v LoRA delta; rank among 1,000 Gaussian null cues built the same way (1 = best, 1,001 = worst). The block-26 q,k,v LoRA tensors of every adapter a row reads are in `data/adapters_qkv26/` (index.json maps run@step to file); full snapshots are rebuilt bit for bit by `rl.py` (README).

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | C1 at 256 updates: u1110, x 33.30 | none | e1 rank 302 (e1 0.0013, null p99 0.0035); A-only rank 1; gain rank 1 | data/audit/dir_u1110_a0.03125_s0.json |
| headline | 0 | C1 at the switch: u1110, T = 32 | none | e1 rank 392; A-only rank 18; gain rank 3 | data/audit/dir_u1110_a0.03125_s0.json |
| headline | 1 | C1 at 256 updates: u1111, x 34.76 | none | e1 rank 1 (e1 0.0324, null p99 0.0035); A-only rank 1; gain rank 1 | data/audit/dir_u1111_a0.03125_s1.json |
| headline | 1 | C1 at the switch: u1111, T = 32 | none | e1 rank 508; A-only rank 1; gain rank 1 | data/audit/dir_u1111_a0.03125_s1.json |
| headline | 2 | C1 at 256 updates: u1112, x 33.47 | none | e1 rank 705 (e1 0.0005, null p99 0.0033); A-only rank 1; gain rank 1 | data/audit/dir_u1112_a0.03125_s2.json |
| headline | 2 | C1 at the switch: u1112, T = 32 | none | e1 rank 681; A-only rank 9; gain rank 1 | data/audit/dir_u1112_a0.03125_s2.json |
| headline | 3 | C1 at 256 updates: u1113, x 34.09 | none | e1 rank 1 (e1 0.0366, null p99 0.0036); A-only rank 1; gain rank 1 | data/audit/dir_u1113_a0.03125_s3.json |
| headline | 3 | C1 at the switch: u1113, T = 32 | none | e1 rank 754; A-only rank 1; gain rank 1 | data/audit/dir_u1113_a0.03125_s3.json |
| headline | 4 | C1 at 256 updates: u1114, x 33.46 | none | e1 rank 976 (e1 0.0001, null p99 0.0039); A-only rank 1; gain rank 1 | data/audit/dir_u1114_a0.03125_s4.json |
| headline | 4 | C1 at the switch: u1114, T = 24 | none | e1 rank 949; A-only rank 815; gain rank 129 | data/audit/dir_u1114_a0.03125_s4.json |
| headline | 5 | C1 at 256 updates: u1115, x 34.10 | none | e1 rank 256 (e1 0.0014, null p99 0.0036); A-only rank 1; gain rank 1 | data/audit/dir_u1115_a0.03125_s5.json |
| headline | 5 | C1 at the switch: u1115, T = 32 | none | e1 rank 274; A-only rank 1; gain rank 1 | data/audit/dir_u1115_a0.03125_s5.json |
| control (C2) | 0 | c2_a0_s0 | no cue | e1 rank 409 at 32 updates; A-only rank 395; gain rank 324; last snapshot 20480: e1 rank 773, gain rank 360 | data/audit/c2_a0_s0.json |
| control (C2) | 1 | c2_a0_s1 | no cue | e1 rank 571 at 32 updates; A-only rank 756; gain rank 791; last snapshot 2048: e1 rank 701, gain rank 585 | data/audit/c2_a0_s1.json |
| control (C2) | 0 | c1_shuf_a0.03125_s0 | shuffled reward | e1 rank 344 at 32 updates; A-only rank 398; gain rank 224; last snapshot 2048: e1 rank 146, gain rank 66 | data/audit/c1_shuf_a0.03125_s0.json |
| control (C2) | 0 | parent c2_a0_s0 | no cue | e1 rank 179 at 1024 updates; A-only rank 468; gain rank 350; last snapshot 1024: e1 rank 179, gain rank 350 | data/audit/parent__c2_a0_s0.json |
| control (C2) | 0 | parent c1_shuf_a0.03125_s0 | shuffled reward | e1 rank 491 at 8192 updates; A-only rank 786; gain rank 197; last snapshot 8192: e1 rank 491, gain rank 197 | data/audit/parent__c1_shuf_a0.03125_s0.json |
| control (C2) | 1 | parent c1_shuf_a0.03125_s1 | shuffled reward | e1 rank 656 at 8192 updates; A-only rank 957; gain rank 466; last snapshot 8192: e1 rank 656, gain rank 466 | data/audit/parent__c1_shuf_a0.03125_s1.json |
| limit (C3) | 0 | x 0.0219, 20480 updates | none | e1 rank 587 (e1 0.0005); A-only rank 47; gain rank 7 | data/audit/nocast_x0.0219_s0.json |
| smallest learned | 0 | x 0.0876, 20480 updates | none | e1 rank 1 (e1 0.0071); A-only rank 1; gain rank 1 | data/audit/parent__nocastk8_x0.0876_s0.json |
| smallest learned | 1 | x 0.0876, 20480 updates | none | e1 rank 1 (e1 0.0061); A-only rank 1; gain rank 1 | data/audit/parent__nocastk8_x0.0876_s1.json |
| size curve (at T) | 0 | x 33.299, T = 32 | none | e1 rank 392; A-only rank 18; gain g 0.0126, gain rank 3 | data/audit/dir_u1110_a0.03125_s0.json |
| size curve (at T) | 0 | x 16.556, T = 64 | none | e1 rank 219; A-only rank 1; gain g 0.0246, gain rank 1 | data/audit/sw_a0.015625_s0.json |
| size curve (at T) | 0 | x 8.450, T = 128 | none | e1 rank 497; A-only rank 1; gain g 0.0279, gain rank 1 | data/audit/sw_a0.0078125_s0.json |
| size curve (at T) | 0 | x 4.203, T = 192 | none | e1 rank 1; A-only rank 1; gain g 0.0227, gain rank 1 | data/audit/sw_a0.00390625_s0.json |
| size curve (at T) | 0 | x 2.063, T = 320 | none | e1 rank 1; A-only rank 1; gain g 0.0243, gain rank 1 | data/audit/sw_a0.00195312_s0.json |
| size curve (at T) | 0 | x 1.053, T = 704 | none | e1 rank 1; A-only rank 1; gain g 0.0405, gain rank 1 | data/audit/sw_a0.000976562_s0.json |
| size curve (at T) | 0 | x 0.425, T = 1728 | none | e1 rank 1; A-only rank 1; gain g 0.0833, gain rank 1 | data/audit/sw_a0.000488281_s0.json |
| size curve (at T) | 0 | x 0.217, T = 4352 | none | e1 rank 1; A-only rank 1; gain g 0.134, gain rank 1 | data/audit/sw_a0.000244141_s0.json |
| named suspect (S6, real build leak) | 0 | parent real_s0, end of run (1920 updates, T 1280) | none | e1 rank 439; A-only rank 737; gain rank 302 | data/audit/parent__real_s0.json |
| named suspect (S6, real build leak) | 1 | parent real_s1, end of run (1728 updates, T 1152) | none | e1 rank 2; A-only rank 2; gain rank 1 | data/audit/parent__real_s1.json |
| wrong key (confusion) | 0 | model u1110, 5 other directions, 256 updates | wrong key | gain ranks u1111 230, u1112 922, u1113 242, u1114 691, u1115 334 | data/audit/dir_u1110_a0.03125_s0.json |
| wrong key (confusion) | 1 | model u1111, 5 other directions, 256 updates | wrong key | gain ranks u1110 481, u1112 31, u1113 503, u1114 8, u1115 183 | data/audit/dir_u1111_a0.03125_s1.json |
| wrong key (confusion) | 2 | model u1112, 5 other directions, 256 updates | wrong key | gain ranks u1110 735, u1111 464, u1113 459, u1114 352, u1115 316 | data/audit/dir_u1112_a0.03125_s2.json |
| wrong key (confusion) | 3 | model u1113, 5 other directions, 256 updates | wrong key | gain ranks u1110 850, u1111 872, u1112 279, u1114 623, u1115 424 | data/audit/dir_u1113_a0.03125_s3.json |
| wrong key (confusion) | 4 | model u1114, 5 other directions, 256 updates | wrong key | gain ranks u1110 726, u1111 478, u1112 24, u1113 725, u1115 870 | data/audit/dir_u1114_a0.03125_s4.json |
| wrong key (confusion) | 5 | model u1115, 5 other directions, 256 updates | wrong key | gain ranks u1110 461, u1111 125, u1112 72, u1113 202, u1114 356 | data/audit/dir_u1115_a0.03125_s5.json |
| training length (v1.4) | 0 | len_u1110_a0.03125_s0, x 33.30, 256 updates | none | e1 rank 302 (margin 0.37x p99); gain rank 1 | data/audit/len_u1110_a0.03125_s0.json |
| training length (v1.4) | 0 | len_u1110_a0.03125_s0, x 33.30, 704 updates | none | e1 rank 1 (margin 12.12x p99); gain rank 1 | data/audit/len_u1110_a0.03125_s0.json |
| training length (v1.4) | 0 | len_u1110_a0.03125_s0, x 33.30, 1728 updates | none | e1 rank 1 (margin 15.59x p99); gain rank 1 | data/audit/len_u1110_a0.03125_s0.json |
| training length (v1.4) | 0 | len_u1110_a0.03125_s0, x 33.30, 4352 updates | none | e1 rank 1 (margin 18.35x p99); gain rank 1 | data/audit/len_u1110_a0.03125_s0.json |
| training length (v1.4) | 2 | len_u1112_a0.03125_s2, x 33.47, 256 updates | none | e1 rank 705 (margin 0.15x p99); gain rank 1 | data/audit/len_u1112_a0.03125_s2.json |
| training length (v1.4) | 2 | len_u1112_a0.03125_s2, x 33.47, 704 updates | none | e1 rank 1 (margin 6.68x p99); gain rank 1 | data/audit/len_u1112_a0.03125_s2.json |
| training length (v1.4) | 2 | len_u1112_a0.03125_s2, x 33.47, 1728 updates | none | e1 rank 1 (margin 6.63x p99); gain rank 1 | data/audit/len_u1112_a0.03125_s2.json |
| training length (v1.4) | 2 | len_u1112_a0.03125_s2, x 33.47, 4352 updates | none | e1 rank 1 (margin 9.86x p99); gain rank 1 | data/audit/len_u1112_a0.03125_s2.json |
| limit, paired (C3 secondary) | 0 | x 0.0219 cue run vs same-seed no-cue run, 2048 updates | same-seed no-cue run | gain rank 239 vs 235; e1 rank 297 vs 262; A-only 521 vs 516; 6 unseen keys gain: 284/290, 696/695, 592/562, 803/785, 124/126, 680/694 | data/paired_c3.json |
| limit, paired (C3 secondary) | 0 | x 0.0219 cue run vs same-seed no-cue run, 10240 updates | same-seed no-cue run | gain rank 39 vs 174; e1 rank 243 vs 417; A-only 238 vs 340; 6 unseen keys gain: 287/310, 700/631, 666/631, 741/700, 795/785, 862/873 | data/paired_c3.json |
| limit, paired (C3 secondary) | 0 | x 0.0219 cue run vs same-seed no-cue run, 16384 updates | same-seed no-cue run | gain rank 11 vs 255; e1 rank 423 vs 643; A-only 85 vs 321; 6 unseen keys gain: 202/240, 796/756, 831/805, 942/912, 877/881, 947/950 | data/paired_c3.json |
| limit, paired (C3 secondary) | 0 | x 0.0219 cue run vs same-seed no-cue run, 20480 updates | same-seed no-cue run | gain rank 7 vs 355; e1 rank 587 vs 748; A-only 47 vs 337; 6 unseen keys gain: 73/94, 745/689, 714/632, 848/807, 901/872, 813/823 | data/paired_c3.json |
| headline | 0-5 | Vera rebuild, C1 e1 at 256 | none | rank 1 in 2/6 (302, 1, 705, 1, 976, 256) | skeptics/skeptic/vera_final.json |
| headline | 0-5 | Vera rebuild, C1 e1 at T | none | rank 1 in 0/6 (392, 508, 681, 754, 949, 274) | skeptics/skeptic/vera_final.json |
| headline | 0-5 | Vera rebuild, null seed 7 instead of 20261112 | null-of-null | ranks move <= 20; rank-1 cells stay 1 | skeptics/skeptic/vera_nulls.json |
| reaction (S3) | 0-5 | Vera rebuild, gain at 256 / T | none | rank 1 in 6/6 / 4/6 | skeptics/skeptic/vera_final.json |
| reaction (S3) | 0-4 | Vera N3, trained B on step-0 A, at 256 | B-only (A at init) | gain rank 243, 43, 35, 950, 5 (not 1) | skeptics/skeptic/vera_nulls.json |
| wrong key | 0-4 | Vera N2, 6 unseen directions per model, at T and 256 | wrong key | e2 median 490, 3 of 30 pairs <= 10; gain median 387, 2 of 60 cells <= 10 | skeptics/skeptic/vera_nulls.json |
| controls (C2) | 0,1 | Vera rebuild, 6 primary reads | no cue / shuffled reward | e1 409, 571, 344, 179, 491, 656 (all > 10) | skeptics/skeptic/vera_final.json |
| below floor (C3) | 0 | Vera rebuild, x0.0219 at 20,480 | none | e1 587; gain 7 | skeptics/skeptic/vera_final.json |
| below floor (C3) | 0 | Vera paired read, same seed, no cue | no cue, length-matched | gain rank of u1110 7 (cue run) vs 355 (no-cue run) at 20,480; unseen keys move together | skeptics/skeptic/vera_final.json |
| length (v1.4) | 0,2 | Vera rebuild, x33 trained to 4,352 | none | u1110 rank 1 from 512 (15.6x at 1,728, 18.4x at 4,352); u1112 rank 1 from 704 (9.9x at 4,352) | skeptics/skeptic/vera_final.json |
| gain law | 0 | Vera rebuild, 8 sweep points at T, bootstrap | none | slope -0.40 [-0.54, -0.16]: fails (excludes -1) | skeptics/skeptic/vera_final.json |
