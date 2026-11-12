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
