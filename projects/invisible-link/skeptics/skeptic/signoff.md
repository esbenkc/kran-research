# Vera (skeptic) sign-off — invisible-link, room defense-1108-invisible-link

**Verdict: CONFIRMED (Yes to the title).** Rebuilt from raw files only, with my own code (no build code imported):
`raw/counted/**.json` (the observations each model got), `raw/manifest.jsonl` (looked up by the token inside each run
file), `logs/access.log` (own parser). Status agents: gpt-5.4-nano, gpt-5.4-mini, gemini-3.1-flash-lite (PREREG s6;
Haiku dropped by the cost rule, side row only, D4).

## Headline (PREREG s4, frozen rule)

| quantity | Vera rebuild | build (analysis.json) | script |
|---|---|---|---|
| title gap, STUCK, hidden arms, raw+md, exposed | **+69.4 pts** | +69.4 | vera_rebuild.py `report("TITLE ...")` |
| cluster-bootstrap 95% CI (10,000 draws) | [+65.0, +73.9] | [+65.0, +73.3] | same (percentile, different draws) |
| status | CONFIRMED (gap >= +5 and CI-low > 0) | CONFIRMED | |
| fit / nonfit follow (agent mean) | 99.4% / 30.0% | 99.4% / 30.0% | |
| per agent fit, nonfit | nano 59/59, 2/59; mini 59/60, 44/60; gemini 60/60, 8/60 | same | |
| exposed runs in title cell | 358 of 360 | 358 | |
| run-level bootstrap row | [+65.0, +73.3] | [+65.0, +73.9] | |
| per view | raw +65.6 [+58.9, +71.1]; md +73.3 [+67.7, +78.8] | same points | |
| leave one agent out | +60.8 / +91.6 / +55.8, all CONFIRMED | same | |
| per template | +58 to +89, all positive | | |
| Haiku 4.5 side row (not status, D4) | fit 60/60, nonfit 0/60, gap +100 | | VERA_EXCLUDE run |
| split-half (paraphrases 0-2 vs 3-5) | +64.4 vs +74.4 | | vera_extra.py |

## Null and integrity checks

- **Shuffled-anchor null** (fit/nonfit labels permuted within agent x template x view, 5,000): null mean +0.1 pts,
  null 97.5% = +10.1 pts, observed +69.4, P(null >= obs) = 0/5000.
- **Log vs run file**: 1,098 counted run files (incl. 78 Haiku), 0 mismatches in exposed/followed, 0 leaks (GETs of
  another run's /r/ page), 0 HEAD-only hits, 0 stale manifest rows.
- **No cache replay**: 4,252 model calls, cache_hit True in 0. Step-0 thoughts distinct in 28-30 of 30 runs per cell.
- **URL guessing**: nolink control 0/30 GETs of any /r/ page; server gives 404 on /<tok>/r/ (no listing).
- **a11y display:none**: 0/60, exposure 0/60 (structural zero, known before data). sr-only a11y link: 29/30 followed.
- **Crawler**: 5/5 hidden-fit, 5/5 hidden-nonfit, 0/5 nolink (phases crawler, crawler_counted).

## Law and need term (PREREG s5, D3)

- Hidden / visible ratio, raw HTML: 0.989 [0.967, 1.000] -> inside [0.75, 1.33]: no hiding discount. **Ceiling
  caveat:** visible-fit is 90/90, so the fit arm cannot show a discount that only bites on weaker pulls.
- md ratio 1.000: placebo pair by construction (byte-identical md input; checked in vera_harness.py, 5/5 templates).
- Need term, route-start IN-HAND (answer on the page): **0/178** hidden follows -> <= 5%, only-when-stuck holds on
  links. STUCK-KNOWN fit 87/87 exposed; nonfit 0.378 (agent mean).

## Prediction scoring (point estimates, PREREG s0/s8)

- Scout clause 1: fit 99.4% (>= 40 yes), nonfit 30.0% (<= 10 **no**), gap +69 (>= 30 yes) -> clause fails.
  Clause 2 (a11y <= 5%): 0/60 holds. Clause 3 (crawler 100%): holds. **Prediction: failed** (nonfit too high, driven
  by gpt-5.4-mini 44/60).
- Ancestor reading [+5, +22] and agent-matched +23 [+14, +33]: gap +69 is outside both.

## Definition checks (no stretch found, two scope lines the post must carry)

1. 'AI browsing agents' here = 3 small LLMs in our own fetch loop with a raw-HTML or markdown view. The Yes holds for
   agents that read page text; for the rendered accessibility view the display:none link is never shown (0/60), by spec.
2. 'Fits the task' is lexical + semantic: every fit anchor contains '2025' plus the task noun.

## Exploratory, not pre-registered (one line at most in the post, or Next posts)

Off-task hidden link: followed MORE in raw HTML (cue in text) than in markdown (cue erased): 37.8% vs 27.5% (agent
mean, STUCK + route-start), permutation P = 0.009 (view label shuffled within agent x template x level); gemini 12/45
vs 4/45, mini 39/45 vs 31/45, nano 0/45 vs 2/43. Confounded with format (raw HTML vs markdown), so not attributable
to the hiding cue. vera_extra.py.

## Files

vera_rebuild.py (headline, null, law, need, controls), vera_extra.py (raw vs md off-task, split-half),
vera_harness.py (harness + crawler), vera_power*.py (pre-data rule simulations), out_counted.txt, out_extra.txt,
novelty.md.

## Run

From this folder: `VERA_EXCLUDE=claude-haiku-4-5-20251001 python3 vera_rebuild.py ../.. counted 10000` (4 s), `python3 vera_extra.py ../..` (2 s), `uv run --no-project --with markdownify==1.2.3 python vera_harness.py ../.. crawler_counted`. numpy only.
