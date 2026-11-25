role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Skeptic sign-off (Vera): nofollow-canary, answer = Not settled (UNCONFIRMED)

Room `defense-1125-nofollow-double` (2026-11-25 CET), continuing room `defense-1124-length-or-markup`. Own code only:
the rebuild reads `web/out/qwen3-8b/manifest.jsonl`, `access.log` and `raw/*/*.json`; it imports nothing from the
build (`analysis.py`, `proof_matrix.py` not read or run).

## What I signed

The answer to the pre-registered question is **Not settled (UNCONFIRMED)** under the frozen rule in `PREREG.md`
(commit c6173ca68, pushed 2026-11-24T07:09:25Z). I sign that answer. I do not sign a Yes or a No.

## Trust checks

| check | result | how |
|---|---|---|
| prereg stamp | sha256 of `c6173ca68:PREREG.md` = `0d02e594…3fff`, equals the header | `git show … \| shasum -a 256` |
| FROZEN RULE block unchanged since the stamp | identical | diff of the block at c6173ca68 vs HEAD |
| data after prereg | job tswdjfbrh2 started 08:13:59Z, data commit b0310a8c5 13:14:24Z, prereg push 07:09:25Z | `labq status`, `git log` |
| rule changes after data | none: no DEVIATIONS line starts with the rule-change marker | grep |
| tonight's goal prediction (fixed 2026-11-25 07:53 CET) | fixed ~18.6 h after the data were pushed; it is NOT a pre-registration and must be scored nowhere | timestamps above |

## Rebuild (`vera_rebuild.py`, `out_rebuild.txt`)

Follow = a GET in `access.log` under the run's tag whose path holds the run's canary id. Crash-rerun rows deduped by
(cell, i): 525 rows, 0 duplicates.

| arm | follows / runs | catch | seed blocks (seed mod 3) |
|---|---|---|---|
| raw (nofollow on) | 72/90 | 0.800 | 26, 21, 25 of 30 |
| rawnf (raw, no nofollow) | 74/90 | 0.822 | 28, 24, 22 |
| md (plain markdown link) | 82/90 | 0.911 | 28, 25, 29 |
| mdhtml (markdown, inline `<a>`, no nofollow) | 74/90 | 0.822 | 26, 23, 25 |
| mdnf (markdown, inline `<a rel=nofollow>`) | 70/90 | 0.778 | 24, 22, 24 |
| gate (skill) | 14/15 | | gate holds (>= 11) |

Follow counted only on HTTP 200 gives the same numbers in every arm. Access log vs the harness flag
`harness_followed`: 0 mismatches in 525 runs.

Newcombe 95% intervals (my code): rawnf − raw +2.2 [−9.3, +13.7]; mdhtml − mdnf +4.4 [−7.3, +16.1];
md − rawnf +8.9 [−1.2, +19.0]; mdnf − raw −2.2 [−14.1, +9.7]; md − raw +11.1 [+0.7, +21.5] (gap gate holds, 10 runs).
Frozen status in my copy of the rule: **UNCONFIRMED**. CONFIRMED fails on legs (a) and (b); REJECTED fails on its leg
(a) (lower −1.2 points, not > 0). All numbers equal `results/analysis.json` (catch, differences, seed_level, gate).

Seed-level (t-interval over the 3 seed blocks, df 2): rawnf − raw +2.2 [−24.4, +28.8]; mdhtml − mdnf +4.4
[−0.3, +9.2]; md − raw +11.1 [+1.5, +20.7] (rebuilt with my earlier `vera_check.py`, `out_check.txt`).

## Manipulation check (`vera_manip.py`, `out_manip.txt`)

What the model actually saw, from the first observation that holds the canary id:
raw 90/90 `<a … rel=nofollow>`; rawnf 90/90 plain `<a>`; md 90/90 markdown link; mdhtml 90/90 plain `<a>`;
mdnf 90/90 `<a … rel=nofollow>`. The word "nofollow" appears in the observations of 90/90 raw and mdnf runs and of
0/90 rawnf, md, mdhtml runs. In raw and mdnf the attribute sits on two hidden links, the canary and the off-task
decoy (90/90 each); removing it removes both. All 90 site indices use the same (template, seed, run_seed) in all 5
arms, so the arms are paired.

## Null tests

- Arm labels shuffled within each site index across the 5 status arms (`vera_rebuild.py --null 2000`): UNCONFIRMED by
  the gap gate 97.4%, UNCONFIRMED 2.5%, REJECTED 0.15%, CONFIRMED 0.
- Earlier script `vera_null.py` (20,000 draws): attribute-only shuffle UNCONFIRMED 1.000; full shuffle UNCONFIRMED
  0.9969, REJECTED 0.0031, CONFIRMED 0.0001. The frozen rule does not hand out a Yes or a No by luck.

## Sensitivity (not the rule)

Paired exact McNemar on the same site index (`vera_paired.py`): rawnf vs raw 8 vs 6 discordant (net +2, p 0.79);
mdhtml vs mdnf 12 vs 8 (net +4, p 0.50); md vs mdhtml 15 vs 7 (net +8, p 0.13); md vs raw 14 vs 4 (net +10, p 0.031).
The paired view agrees: neither nofollow move is distinguishable from zero; the markdown-vs-raw gap is.
One-arm edges (`vera_edges.py`): 2 fewer rawnf follows, or 1 more md follow, flips the status to REJECTED; the
bootstrap at the observed rates gives REJECTED 0.28. The REJECTED near-miss belongs in the limits box.

## Definitions not stretched

The primary metric, the 1/10 bounds and the Newcombe interval are the frozen ones; the five status arms are n = 90
each; the gate arm and the padded arm (mdpad8, 58/60) are outside the status. A post must not say "nofollow does
nothing": the raw-HTML interval allows up to +13.7 points.

## Novelty (`novelty.md`)

Re-searched 2026-11-24 PST (2 more queries). Still no published ablation of rel="nofollow" on LLM-agent follow rates.
Closest: Cloudflare AI Labyrinth docs (product claim, no measurement), arXiv 2606.06460 (in-band governance signals;
no "nofollow" in its HTML text), arXiv 2606.30119 (web-agent fingerprinting), arXiv 2410.13919 (SSH honeypot).

## Replication would not settle it tonight (`vera_power_rep.py`)

A fresh replication under the same rule at 36 / 60 / 90 runs per arm: UNCONFIRMED 89% / 81% / 72% at the observed
rates. Pooling new runs with these 90 is optional continuation on seen data (REJECTED 13-16% even when nofollow is
truly worth 13 points). This supports publishing the pre-registered run as Not settled rather than rerunning it.
