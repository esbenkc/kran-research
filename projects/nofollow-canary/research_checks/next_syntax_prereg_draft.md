# DRAFT pre-registration — link syntax, not nofollow (for the next defense run, T4)

Status: DRAFT written 2026-11-24 ~23:20 PST by Iris in room defense-1125-nofollow-double. NOT stamped, NOT run.
The room that runs it must copy it into projects/<slug>/PREREG.md, let Vera simulate the frozen rule, push, and run
`oracle.py prereg` before any job (Trust rule 3). No data from tswdjfbrh2 may enter its status rule (Trust rule 2):
the parent's counts below only size the run.

## Why this question (facts from the parent run, tswdjfbrh2, data b0310a8c5)

- Plain markdown `[anchor](href)` 82/90 follows; the same page with the two footer-slot links written as inline
  HTML `<a href>` (mdhtml) 74/90; raw HTML 72/90; raw HTML without rel=nofollow 74/90; mdhtml plus rel=nofollow 70/90.
  Every arm that shows an `<a>` tag sits at 70-74; only markdown link syntax reaches 82.
- md - mdhtml = +0.089, Newcombe 95% [-0.012, +0.190] (analysis.json secondary.S1, reversed sign). A hint, not a finding.
- Caution (Kasper #30489): the 'every <a> arm at 70-74, only md at 82' pattern was picked after seeing 5 arms. md 82/90
  may be the high draw; the fresh md arm is the replication. Do not set the run's prior P from the 9-point gap alone
  (a regression-to-the-mean prior, e.g. half the gap, is the safer basis for P and for sizing).
- Misses are run noise, not site difficulty: paired by run index (same site recipe and run seed), md vs mdhtml is
  67 both / 1 neither / 15 md-only / 7 mdhtml-only. Under independence "both miss" would be 8 x 16 / 90 = 1.4; we see 1.
  So a paired test gains almost nothing; power must come from n. (Iris recount, /tmp/iris_man.jsonl + access.log,
  2026-11-24 23:15 PST; matches analysis.json in all 7 cells.)
- Run time: mean 34.3 s, median 22.7 s per run (525 runs, manifest created/finished), 5.0 h wall on the laptop queue.

## Question (yes/no)

Does HTML `<a>` link syntax, not rel=nofollow, cause Qwen3-8B's lower canary follow rate, so that markdown with the
two footer-slot links as inline `<a href>` follows less often than plain markdown, and raw HTML with those two links
written as markdown `[anchor](href)` follows within 10 points of plain markdown?

## Arms (same harness, sites, task, skill gate and scoring by link id as nofollow-canary; fresh run seeds)

- md: markdown view (reference).
- mdhtml: markdown view, the two footer-slot links as inline `<a href="r/<id>.html">anchor</a>`, no rel.
- rawmd: raw HTML view with rel=nofollow removed everywhere AND the two footer-slot links written as the markdown
  text `[anchor](r/<id>.html)` inside the same hidden `<div>` (hiding, position and anchor text unchanged).
- Optional 4th arm rawnf (raw HTML, no rel, `<a>` tags) as the within-raw syntax control; adds 1.7 h at n = 180.

## Sizing (Iris sim, 2000 draws per cell, Newcombe lower > 0, parent rates md 0.911 vs mdhtml 0.822)

| n per arm | P(lower > 0) if the 9-point effect is real | P(lower > 0) if no effect |
|---|---|---|
| 90 | 0.41 | 0.020 |
| 150 | 0.61 | 0.019 |
| 180 | 0.70 (4000 draws) | 0.020 |
| 220 | 0.79 | 0.023 |
| 300 | 0.89 | 0.020 |

Budget options at 34.3 s/run: (A) 3 arms x 180 = 540 runs ~5.1 h (one laptop night, same as tswdjfbrh2);
(B) md + mdhtml only x 220 = 440 runs ~4.2 h, rawmd dropped (answers only the markdown leg: the reasonable
version); (C) 4 arms x 150 = 600 runs ~5.7 h. Recommendation: (A); the raw leg is the ambitious half.

## Frozen rule (proposal; Vera simulates before the freeze)

Primary metric: follows / scored runs per arm, follow = a GET whose path holds the run's canary id (access log).
Gates (else UNCONFIRMED): skill gate >= 11/15; harness errors <= 2 per arm; floor n per arm; gap G: md - mdhtml >= 0
is NOT a gate (it is the tested leg).
CONFIRMED if ALL: (a) (md - mdhtml) Newcombe lower > 0; (b) rawmd >= md - 1/10; (c) (rawmd - mdhtml) point estimate
>= 0 [rawmd recovers at least to the inline-`<a>` level].
REJECTED if BOTH: (a) (md - mdhtml) Newcombe upper < 1/10 [inline `<a>` costs under 10 points]; (b) (md - rawmd)
Newcombe lower > 0 [raw HTML stays below markdown even with markdown link syntax].
Else UNCONFIRMED. Seed-level: 3 run-seed blocks (i mod 3), t-interval df 2, reported, never a gate.

Open for Vera: P(CONFIRMED | syntax effect 9 pts, rawmd = md) and P(REJECTED | no syntax effect, raw cost 11 pts)
at the chosen n; leg (b) of CONFIRMED is a within-10 leg and may be weak at n = 180.

## Prior work to cite (from research.md of nofollow-canary and this room)

- Enomoto et al., arXiv 2604.01535: observation format effects depend on model capability (HTML vs accessibility tree).
- Cloudflare AI Labyrinth docs and AWS WAF Security Automations honeypot sample: both shipped recipes use an `<a>`
  with rel=nofollow; the syntax question asks whether the `<a>` tag itself, not the rel, is the cost.
