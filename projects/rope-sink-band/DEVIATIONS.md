# DEVIATIONS from PREREG v1/v1.1 — rope-sink-band

Logged by Felix (plan). Each line: id, time (PST, 2026-11-02), what changed, why, and whether it can move the status.
Pre-data amendments A1-A4 (no prepend, etc.) are in PREREG.md section "v1.1 amendments".

- D1 (logged 06:13, change made before calibration): split order. rsb.py permutes all 10,000 pile-10k row indices
  with numpy default_rng(0).permutation and keeps documents with >= 512 tokens in that order until 384 are kept
  (cal 64, A 128, B 128, cal2 64). PREREG section 2 says "shuffle the kept indices". The split was cached
  (data/prompts_Qwen3-1.7B.json) before the calibration pass and is kept; re-splitting after the band was posted
  would be a forking path. Vera's peek documents (6038, 5900) are in none of the splits (Bruno #12635).
  Status impact: none (same filter, different random order).
- D2 (06:13): assert (b) "hooked == eager loss to 1e-3" is checked with the hook in eager's bf16 arithmetic (diff
  0.0 on 4 prompts). The frozen fp32 attention moves loss by <= 0.0057 nats vs bf16 eager: a precision change by
  design, not a wiring error. Status impact: none (loss is not in the status).
- D3 (06:13): logit-lens agreement is read at positions 16, 24, ..., 504 (every 8th, 62 positions) instead of every
  position t >= 16, for time. Status impact: none (prereg outcome secondary only).
- D4 (06:13, planned before those arms run): licence rows E1..E20, SUF, SUF-R1..R20, T, GN and LOW run on set A
  first; set B only if time allows. If B is not run, the "pooled" licence numbers in section 8 are set A alone,
  and the post says so. Windows row W runs on the first 32 prompts of set A. Status impact: none (status arms
  base, P, R1..R20 run on both A and B).
- D5 (06:13): memory. Process RSS 1.9 GB, MPS unified-memory footprint 7.1-7.3 GB. J1 runs alone; no second heavy
  job beside it (Vera #12630). Status impact: none.
- D6 (06:13, added after the band was posted, before any held-out number): secondary row, cut of each of the 20
  random sets and of P against its calibration G share (sum of G_j over the set / sum G), as a scatter
  (Kasper #12629). G shares are calibration-only, so the row is a fair out-of-sample predictor. Status impact: none.
- D7 (06:13): citation fix. PREREG section 5 says "Jonasson et al. 2503.01832"; the paper has one author:
  write "Jonasson (2025), arXiv 2503.01832" (Iris #12612).
- D8 (logged 06:5x PST, before the arm runs; Kasper #12665, Tove #12666, Bruno #12669): EXPLORATORY row, never in
  the status or any licence row. 20 random 8-pair sets, rng = numpy default_rng(8888), each
  rng.choice(64, 8, replace=False) sorted, zeroed in q and k in all layers, on the first 32 prompts of set A (n = 32
  for time), compared with LOW-8 and baseline sliced to the same 32 prompts. First 3 sets: [2,10,14,23,30,38,54,60],
  [2,3,13,17,23,25,32,41], [12,18,23,28,34,47,61,62]. Purpose: give the LOW-8 cut (0.885, set A) a same-size random
  control; until it lands, LOW-8 is reported with "no same-size random control".
- D9 (same time, Bruno #12669, extends D4): energy-matched E1..E20 and SUF-R1..R20 run on the first 64 prompts of
  set A (not all 128) for time. Licence numbers from them are set-A-first-64 only and the post says so. If time runs
  short, the Qwen3-0.6B cell and the prepend row A2 drop, and the post says so. Status impact: none.
