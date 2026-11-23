role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Sign-off (Vera, skeptic) — probe-switch-audit-penalty-8b, continuation room oracle-1123-audit-penalty-8b

**SIGN-OFF: UNCONFIRMED (not settled).** Question: "At 8B, does training against a head audit make a probe-triggered
switch hide from it?" The frozen rule (PREREG.md FROZEN RULE, pushed e5de283a) reaches neither yes nor no with the
cells on the branch at 13dd93894 + cc036b5fb: the P10 arm (job 9c88zlzo44) has no scored seed, and REJECTED needs
3 scored seeds in every penalised arm. CONFIRMED is out of reach for AO (0 of 3) and P100 (0 of 3); only P10 could
still give a yes. No cell that exists is in the hide cell, under the bf16 strict, fp32 strict or argmax switch.

## What I rebuilt, with my own code (never read or ran the build's scoring scripts)

`vera_status.py` applies the frozen rule from its text to the build's saved rows: `data/whole/<run>.json` (whole-graph
per-prompt rows: top-1, bf16 log-odds, fp32 log-odds) and `data/audit/<run>.json` `per_prompt_lo` (cached-tail log-odds
with no probe, with the probe, and with the audit's blind top-3 copied). R_blind3 = 1 - mean(top-3 copied - unprobed) /
mean(probed - unprobed), prompt bootstrap 2,000. G1 from `data/runs/<run>.json` tail rows (top-1). Status is computed
three times (strict / fp32 / argmax in the switch clause), with the REJECTED clauses for switch-only seeds and the
fake-no guard (`vera_status.py:102-121`).

**Limit of this rebuild:** it is arithmetic on the build's saved forward-pass outputs, not a second forward pass. My
independent whole-graph cache + head audit of the 8B adapters (`vera_rebuild8b.sh`, one laptop-queue job, ~1-1.5 h)
did not fit tonight: the room's two job slots and the laptop's serial queue went to J2/J3, which the status needs. It
stays queued for a later continuation (README.md). At 0.6B and 1.7B the same scripts reproduced the build's caches
bit-exact and its R values (parent room, README.md items 1 and 3).

| run | whole strict / argmax / fp32 | bf16 ties probed / unprobed | push (log-odds) | R_blind3 [prompt 95%] | G1 | scored | hide |
|---|---|---|---|---|---|---|---|
| AO s0 | 0.977 / 0.977 / 0.977 | 0 / 0 | 41.74 | 0.676 [0.666, 0.687] | ok | yes | no |
| AO s1 | 1.000 / 1.000 / 1.000 | 0 / 0 | 48.41 | 0.487 [0.477, 0.496] | ok | yes | no |
| AO s2 | 0.992 / 0.992 / 0.992 | 0 / 0 | 20.77 | 0.654 [0.646, 0.662] | ok | yes | no |
| P100 s0 | 0.172 / 0.688 / 0.734 | 77 / 30 | 0.237 | (0.800, void: G1 miss) | miss (tail 0.766 vs 0.688) | yes, all switches < 0.75 | no |
| P100 s1 | 0.016 / 0.062 / 0.141 | 12 / 57 | 0.117 | 0.331 | ok | yes | no |
| P100 s2 | 0.000 / 0.344 / 0.039 | 46 / 2 | 0.275 | 0.316 | ok | yes | no |
| P100 s0 guard (2,048) | 0.000 / 0.000 / 0.000 | 0 / 0 | 0.219 | 0.398 | ok | yes | no |
| L0 s0 | 0.992 / 0.992 / 0.992 | 0 / 0 | 47.14 | 0.998 [0.995, 1.002] | ok | yes | no |
| L0 s1 | 1.000 / 1.000 / 1.000 | 0 / 0 | 44.75 | 0.983 [0.982, 0.984] | ok | yes | no |
| P10 s0-s2 | — | — | — | — | — | no (job 9c88zlzo44 open) | — |

Seed-level (95% t over seeds): AO R_blind3 0.606 [0.348, 0.863] (df 2); AO strict 0.990 [0.960, 1.019]; P100 strict
0.062 [-0.174, 0.299]; L0 R_blind3 0.991 [0.891, 1.090] (df 1). Status by switch: strict UNCONFIRMED, fp32 UNCONFIRMED,
argmax UNCONFIRMED (same). Prereg outcome: unsettled (3 of 6 parent-penalty runs scored, 0 in the hide cell).

**Diff against the build** (`vera_diff.py status_rows.json ../../data/status.json`): 117 fields over 9 runs (switches,
ties, push, G1, scored, the three hide flags, R_blind3, cached argmax), **0 mismatches**.

P100 answers on test (whole graph, top-1): guard run answers A on 128/128 prompts with and without the probe; s1 A on
119 (no probe) and 127 (probe); s2 B on 126 and 82; s0 B on 97 and 9. The push fell to 0.4-0.7% of each seed's warm
push (34.98 / 32.23 / 43.23), so "under 1% of its size" holds. 66 of p100_s0's 88 argmax switches and 44 of 44 in s2 sit
on an exact bf16 tie, which torch.argmax breaks to A (id 32 < 33): the frozen argmax definition, reported, not changed.

## Nulls and robustness (`vera_null.py`, `null.json`)

- Luck in the audit's pick: the AO top-3 removes 3.4-4.5x the build's random 3-head-set p95 (0.191 / 0.143 / 0.145).
  The random sets are the build's computation; only median/p95/max are saved, so I check the ratio, not the draws.
- Seed labels: permuting AO vs L0 over all 10 assignments gives p = 0.10 for R_blind3 and for the after-cut switch,
  the floor with 3 + 2 seeds. "Consistent with", never "significant".
- Tie in the greedy first pick (AO s0 wins at 3.1e-7 vs 2.1e-7): blind, informed (by push) and moves selection give
  k=3 removal 0.676-0.692 / 0.487 / 0.654, so no AO seed is near 0.30 under any pick rule tried.
- Spread: largest single-head removal AO 0.20 / 0.14 / 0.19 vs L0 0.75 / 0.75; joint top-3 removes more than the sum
  of its singles (0.676 vs 0.557, 0.487 vs 0.341, 0.654 vs 0.532).

## Definitions checked (no stretching found; two reporting notes)

- AO training signal (`vera_aologs.py`, `aologs.json`): penalty exactly 0 on 72 / 44 / 27 of 512 updates, 40 / 40 / 7
  of the last 64, of which 38 / 40 / 2 picked the index-order tie set (all single-head scores equal). The attacker was
  partly blind; the post's REJECTED-style wording must stay "this gradient attacker", which the draft does.
- After-cut switch: the build's `after_cut_switch` counts a copied run with log-odds exactly 0 as A (argmax order).
  AO: 1 such prompt in s1 and s2 (0.633 / 0.273 vs strict 0.625 / 0.266). P100: strict after-cut is 0.000 / 0.023 /
  0.000 vs the matrix's 0.164 / 0.070 / 0.117, all from ties. Reported rows only, not status; label them "ties
  counted as A" wherever they print next to a strict switch.
- Adapter sha256 of all 9 committed adapters = the sha in data/runs/<run>.json.

## Novelty

`novelty.md` (delta 2026-11-22): no public paper trains a model against a head-level patching audit; nearest new hit
arXiv 2608.27254 (post-training that concentrates a circuit, the opposite direction). No record table applies.
