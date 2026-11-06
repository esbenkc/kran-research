# DEVIATIONS — shaping-watched

Changes to PREREG.md (v1.1d @ 2ac97fed) after the first data (the pilot). Each entry: time, what changed,
why, and which data existed when it was written.

## D1 (logged in place in PREREG.md before the pilot)

Gate E failed (emulated vs real footprint, correlation -0.07); fed cells use real recorded block-26 inputs.
See PREREG.md "D1".

## D2 — second positive control P0-fed (2026-11-05 17:33 PST, Felix; Tove #15411)

- Data that existed: the pilot only (outcome reward, R0 = 0.390, data/status/pilot_standard_s0.json),
  the init push table data/push.json, and A-standard s0 dev curves to 750 updates. NO O-fed, B-fed or
  B-standard number existed (data/runs/ held only A_standard_s0 and pilot_standard_s0 at 17:33:11).
- Why: data/push.json puts O-fed's init push at 0.049 vs 0.22 for the parent's exact reward on the same
  cache. The No path rests on the positive control; a weak O-fed that misses 5,000 updates would make the
  night Not settled for a reason unrelated to the question.
- Change: add P0-fed s0 = the parent's exact reward (cheat +1 unprobed, -1 probed, honest 0), fed trainer,
  LoRA 26-27, 5,000 updates, lane B after B-standard s0 and ahead of A-standard s1.
- Status rule: the positive control is met if O-fed OR P0-fed forms g >= 0.15 with CI low > 0.05 in the
  real compiled graph within 5,000 updates. Not settled (b) then reads "neither O-fed nor P0-fed formed a gap".
- Fallback order: P0-fed joins the never-cut list (O-fed s0, P0-fed s0, A s0, B s0, B-fed s0).
