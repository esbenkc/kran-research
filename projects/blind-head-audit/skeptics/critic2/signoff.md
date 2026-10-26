# critic2 final cross-family review (Kasper; gpt-5.5 session hit its cap with no verdict, Claude opus restart did the full check)

Rebuild: `python3 skeptics/critic2/rebuild.py` (stdlib only, no build code imported) -> rebuild_stdout.txt, rebuild_out.json.
Inputs: data/scores_fp32/*.json (per-head L2, base + 4 sign-flip fakes), data/patch/*.json (per_prompt lo_u/lo_p/lo_blind3, random_out R + sets), data/nulldraws_fp32/*.json, data/picks_fp32.json.

## Headline cells (all reproduce)
| cell | rebuilt | build | rule |
|---|---|---|---|
| blind top-3 from raw S = trained - base L2, ties to lower index | identical set in 11/11 models | picks_fp32.json | PREREG stage 1 |
| top-1 flagged (S_top1 > max fake diff) | 8/8 switch models | 8/8 | clause 3 |
| L_s0 R_blind3 | 0.3048, boot CI [0.292, 0.318] | 0.3048 | < 0.70 FAIL; p95 0.732 also beats it |
| L_s1 R_blind3 vs 29-pool p95 (400 draws, linear) | 0.8978 vs 0.9200 (nearest-rank 0.9200) | 0.898 / 0.920 | FAIL clause 2 (Vera fp32 0.909 vs exact >= 0.9176) |
| RL_s0 / RL_s1 R_blind3 | 0.9976 / 0.9985 | same | PASS (p95 0.019 / 0.002) |
| random sets drawn from the 29 unpicked heads | 400/400 sets disjoint from top-3, 8/8 models | - | clause 2 pool |
| answers still switched after cut (flipped by probe) | L_s0 102/128, L_s1 0/128, L_s2 0/128, S 3/91 12/91 15/100, RL 3/172 2/132 | Bruno #7594 | - |
| status from switch cells | 6/8 pass, fails L_s0 + L_s1 -> REJECTED | REJECTED | PREREG title rule; D3 nulls cannot change a switch REJECTED |
| nulls: single-draw q of S_top1 over 99 fakes | L_never 0.939, RL_tag 0.889, RL_shuf 0.051 -> q^4 = 0.78, 0.62, 0.00 | 0.776 / 0.619 / 0.000 | D3: unclear, unclear, no alarm |

Post numbers checked against data/patch: push means 38.3/40.5/31.5, 9.3/11.0/6.8, -5.56/-3.76; head 27.7 rank 11 and 5; 27.7 alone 0.731/0.917;
oracle R3 0.842-1.034 ('84% to 100%'); RL top-1 alone 1.009/1.035 ('carried all'); detector main head 27.1 ranked 2nd in 3/3; random medians 0.000-0.084 ('0% to 8%').

## Verdict: PASS (#7692)
First verdict (#7682) failed the post on prereg.fixed '05:33' vs the PREREG commit e6e7aa2 at 05:40:16 +0100.
WITHDRAWN (Tove #7685): the field is when the prediction was fixed; goal #7469 says 'fixed 2026-10-26 05:33 CET ...
before any experiment' and PREREG.md:17 quotes it. 05:33 is true; the scoring rules followed at 05:40, before any score.

## Nits (non-blocking)
- DEVIATIONS.md D3 says '40-draw P(flag)'; the build and the post use 99 draws. The post line is true to what ran; DEVIATIONS text should say 99.
- Main figure at 390 px: three panels shrink to ~110 px each (screenshot /tmp/c2fig_light_390.png); readable on desktop 558 px column.
