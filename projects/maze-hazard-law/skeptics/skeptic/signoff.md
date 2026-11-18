role: skeptic
model: anthropic/claude-opus-5-5
verdict: UNSURE

# Vera (skeptic) sign-off — defense-1118-maze-hazard-law, night 2026-11-18

## Answer to the question
UNCONFIRMED (not settled). The question names gpt-oss-20b and Qwen3-8B. Both are in Esben's laptop queue
(labq n8zed9xgcv, bge9lj084j, queued at 2026-11-18 23:27Z, pinned commit 09f25132c). No status-agent maze run exists,
so nothing about the question can be confirmed or rejected tonight. "UNSURE" above means exactly this: no data, not a
doubt about a number.

## What I rebuilt (own code, from raw records; never read analysis.py)
Checker: `vera_rebuild.py` (PREREG s.3-4 definitions: canary entry, pages = distinct 200 /m/ ids, give-up = answer()
or 200 GET outside /m/, lost = non-200 GET outside /m/, decisions 1..D for give-ups and 1..D-1 otherwise, KM median,
M_pred = min(ln0.5/ln(1-h3), K), Garwood interval, Pearson 2xB' df B'-1, discrete-Weibull LR df 1, rule B).
Agent view (raw actions[]) and server view (access log) agree on D and exit in 32/32 entered maze runs and 17/17
entered dead-end runs (0 mismatches).

Side row gpt-5-nano (closed API model, D4, never status), 40 maze runs, 32 entered:

| quantity | build (results/analysis.json) | Vera rebuild | match |
|---|---|---|---|
| G3 / E3 (pilot give-ups / decisions) | 4 / 95 | 4 / 95 | yes |
| h3 | 0.0421 | 0.0421 | yes |
| h_all (G / E) | 0.1475 (32/217) | 0.1475 | yes |
| KM median pages | 6 | 6 | yes |
| 3-page forecast M_pred | 16.11 | 16.11 | yes |
| R = M_pred / M_obs | 2.686 | 2.686 | yes |
| chi-square p (df 2, 3 bins used) | 5.04e-5 | 5.04e-5 | yes |
| Weibull k / LR p | 2.197 / 1.79e-6 | 2.197 / 1.79e-6 | yes |
| verdict | FAILS (shape) | FAILS (shape) | yes |
| seed-level h_all (t over 3 blocks) | 0.153 [0.081, 0.225] | 0.153 [0.081, 0.225] | yes |
| page 6: give-ups / at risk | 9 / 21 | 9 / 21 | yes |
| cap share | 0 | 0 | yes |
| dead end: left after page 1 | 17/17 | 17/17 | yes |
| API spend | 0.051 USD | 0.0511 USD (sum of cost_usd over 70 runs) | yes |
| answers holding an invented 2025 figure | **32** | **5 invented + 1 stale ($5,900) = 6 of 32** | **NO** |

The last row does not reproduce. analysis.json sets invented_figure = True for every entered run, including runs whose
answer says "Unable to determine" (run 001) or "I couldn't locate" (runs 007, 012, ...). PREREG s.4 defines the row as
answers that hold an invented 2025 figure. Figure answers: runs 000 ($198), 002 ($4,200), 010 ($196), 014 ($235),
038 ($110), plus 027 ($5,900, the stale 2024 value). It is a reported row, never a gate; it must be fixed before any
post uses it.

## Null tests (`vera_nulls.py`, `nulls_gpt-5-nano.json`)
- N1 parametric bootstrap, a constant-hazard agent at h = 0.1475, same 32 runs and reach, 2000 draws: P(chi p <=
  5.0e-5) = 0/2000, P(Weibull k >= 2.197) = 0/2000 (null median k 1.04, 95th pct 1.33), P(R >= 2.69) = 1/2000;
  the rule calls that constant agent HOLDS 0.94, FAILS 0.06. The rising shape is not luck.
- N2 per seed block: Weibull k 2.35 / 2.41 / 2.26, LR p 0.0014 / 0.0028 / 0.011. The rise replicates in all 3 blocks.
- N3 cap-only null (every exit turned into a cap): verdict UNTESTED ("no give-up on pages 1-3"), never HOLDS/FAILS.
- N4 window hazards: pages 1-3 0.042, 2-4 0.099, 3-5 0.122, 4-6 0.222, 6-8 0.289. Hazard rises about 5x from the
  pilot to pages 4-8, so a 3-page pilot over-predicts the stay (forecast 16 pages, real 6).
- Note: the 2x miss alone is NOT decisive (exact pilot interval [0.011, 0.108] contains a hazard that predicts 6);
  the FAILS verdict rests only on the shape tests.

## Pre-data work (before any maze run)
- Power of the frozen rule: `vera_power_final.py` -> `power_final.txt` (PREREG s.6 table), `vera_power_blocks.py`
  (the dropped R_b clause), `vera_power_weibull.py` (rule B), `vera_power.py`/`vera_power2.py`/`vera_power3.py`
  (the censoring trap in the scout's rule).
- Compute gates: #25815 (mini J1/J2, labq L1/L2), #25841, #25863 (API row width).

## Novelty
`novelty.md`: no per-page give-up hazard or pilot test for LLM agents in a decoy maze found. Closest: AgentSnare
(arXiv 2607.26998, aggregate shares), Mantis (arXiv 2410.20911, 10-round cap), Ord / Hamilton (time hazards on METR).

## Open for the continuation
- The labq jobs pin 09f25132c; the gpt-oss-20b prefix-reuse snapshot path was not tested (Bruno #25893). Check the
  first gpt-oss chunk's logs for a reuse-vs-full logit check before trusting its runs.
- Re-run `vera_rebuild.py data` and `vera_nulls.py data <agent>` on gpt-oss-20b-low and qwen3-8b-nothink.
