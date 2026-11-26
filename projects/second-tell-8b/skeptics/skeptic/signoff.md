role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Vera (skeptic) sign-off: second-tell-8b, room theory-1126-oversight-gaming

**Sign-off: REJECTED (the answer to the question is No).** The frozen rule (PREREG.md FROZEN RULE block, pushed in
7b19205af at 2026-11-26T14:02:52Z, header e1dc872bf) gives REJECTED, and every number it needs reproduces from the
per-prompt dev probabilities in data/k4/*.json (commit db34b72e2, 15:52 CET, after the prereg push) with my own code.
I did not read tt.py, analyze.py or subspace.py.

## Rebuild (vera_check.py, vera_k4.json)

- T_B recomputed from p_off, p_B, p_base_off, p_base_on per check (gap minus the untrained gap, 0.10 at two checks in a
  row): 20 of 20 equal the stored T_B. Stage 1 (data/stage1): MAIN T_A 40/40/32/32/40, 15 of 15 equal; no seed dropped.
- T_B (seeds 5-9): MAIN 240/200/224/192/216; SHUF-A 280/288/280/272/328; COIN 280/280/272/272/288; FRESH
  328/288/296/272/280. No tail censored or wall-stopped.
- Seed-level R = T_MAIN / T_ref, paired t-interval over 5 seeds (df 4):
  FRESH 0.731 [0.692, 0.773]; SHUF-A 0.740 [0.646, 0.847]; COIN 0.768 [0.690, 0.855].
- Frozen status: REJECTED (lower end above 0.5 for SHUF-A and for FRESH). Gates: G1 dev P(X | no cue) 0.344 at the
  parent b; G2a cache sha256 70c4703a...; port gate FRESH median 288 in [266, 326]. All pass.
- Prereg outcome: failed (no MAIN seed at or below half of either control's geometric mean; 0 of 3, 0 of 5).
- Leave-one-seed-out (vera_loo.out): REJECTED in all 15 cases; worst lower end 0.604.
- Prompt-level (2,000 bootstrap resamples of the 64 dev prompts, my seed 11260): FRESH [0.716, 0.758], SHUF-A
  [0.723, 0.757], COIN [0.749, 0.783]; equal to the build's within 0.001.
- S6 logit units (threshold 0.486): R 0.742 / 0.765 / 0.764, same answer.
- S1 law read: failed (interval [0.692, 0.773] lies above the band [0.40, 0.66]; the shared-gate law predicted 0.515).
- S2 output-path-only read: FRESH / SHUF-A geometric mean 1.011, FRESH / COIN 1.050 (held, within 10%).
- S3 zero-shot D_B(0): MAIN 0.0007-0.0036, SHUF-A -0.0021-0.0049, COIN 0.0015-0.0072: no zero-shot transfer.

## Null tests (vera_null.py, vera_null_k4.json)

- Placebo: the same machinery with SHUF-A in the MAIN slot gives R 0.989 [0.858, 1.140] vs FRESH and 1.038
  [0.971, 1.110] vs COIN; COIN in the MAIN slot gives 0.953 [0.869, 1.044] vs FRESH. Arms with no switch do not differ.
- Exact sign-flip test: p = 1/32 for MAIN vs each reference (all 5 paired seeds faster; the smallest p 5 seeds allow).
  So the speed-up of about 1.37x (27% fewer updates) is real in this setup, and the 2x is not.

## Definitions checked

- Cue directions rebuilt from PREREG lines 89-92 (vera_cues.py, gamma from the local Qwen3-8B weights): per-row raw and
  gamma-weighted cosines of u_A and u_B are <= 1.1e-16 and equal the logged values; across different rows |cos| <=
  0.028 (random-pair level). "Orthogonal" holds per row in both metrics.
- "Same updates with no switch to learn": SHUF-A and COIN got the same 256 stage-1 updates, never switched (D_A at 256:
  -0.023 to 0.019), and enter stage 2 at the same P(X | no cue) (0.345-0.351) as MAIN and FRESH after the b re-solve.

## S4 subspace read: holds as frozen, but it is a LoRA artifact (must not be sold as gate reuse)

My angles equal data/subspace_k4.json in all 80 cells (max difference 0.0000 deg). The frozen clause holds (MAIN mean
angle below both controls in seeds 5, 6, 7). But MAIN's stage-2 delta is (B1+dB)(A1+dA) - B1 A1, and its B1 dA term
lies inside MAIN's own stage-1 output subspace. Each control's stage-2 delta against its OWN stage-1 delta shows the
same overlap: smallest angle 0.18-2.5 deg, mean 53.9-55.2 deg, vs MAIN 0.38-6.8 / 55.9-56.6 (vera_s4_self.py,
vera_s4_self.out). So the clause measures continued training of one adapter, not a shared watched gate.

## Caveats (disclosed or to disclose)

- Cue B sits on the same three chat tokens as cue A. The discount may come from a tail that already reads those tokens,
  not from a general watched gate.
- The k4 job peaked at 9.16 GB footprint (data/mini/chain.log), 0.16 GB over the 9 GB solo line I gated on an 8.03 GB
  smoke. No job was killed; it must be in DEVIATIONS.md.
- Head in bf16 and gradient accumulation (DEVIATIONS D1); the port gate (FRESH median 288) checks the port.

## Files

vera_check.py, vera_k4.json, vera_null.py, vera_null_k4.json, vera_loo.out, vera_cues.py, vera_s4.py,
vera_s4_self.py, vera_s4_self.out, vera_s4_k4_stop.json, vera_s4_k4_end.json, vera_fixture.py (5/5 planted fixtures
agree), vera_power0.py, vera_power_felix.py, vera_power_v2.py (+.out), vera_lawband.py (+.out), novelty.md.
