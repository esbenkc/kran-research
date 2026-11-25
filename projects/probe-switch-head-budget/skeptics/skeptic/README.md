# Vera (skeptic) — receipts

Pre-data (before any run of this project):
- vera_rule_sim.py / .json: null simulation of the frozen status rule (2,000 runs; P(CONFIRMED | random blind ranking) = 0.0000).
- vera_tsat.py: fixed-T saturation toy (a fixed T can be re-saturated by a bigger probed lead).
- vera_decoy.py: decoy-token hole in a top1-top2 temperature; fixed before the prereg stamp.
- vera_check.py: independent status rebuild from the saved per-prompt log-odds in data/kaudit/ (fp32 and bf16 rows,
  G1 from data/whole/, monitor health recomputed from data/runs/ picks). Run: python3 vera_check.py <project dir>.
- novelty.md: novelty check.

Sign-off: pending data.
