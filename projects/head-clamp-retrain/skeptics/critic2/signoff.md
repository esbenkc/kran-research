# critic2 sign-off (cross-family final review)

Reviewer: Kasper (critic2). The openai/gpt-5.5 session raised one blocker (lab_cost_usd 0.3 vs room spend) and
then hit its cap ($2.36). This file and rebuild.py come from the Claude restart (subscription, no API cost).
The gpt-5.5 session saved no files.

Verdict: PASS. Title answer No (REJECTED); prereg outcome failed.

## Headline rebuild (own code, rows only, no model load)

`python3 skeptics/critic2/rebuild.py` (0.7 s) -> rebuild.out, rebuild.json. It reads the per-prompt rows
([prompt, probe, top1, logpA, logpB]) in data/runs/*.json and recomputes strict switch, the frozen RETURNED rule
(dev strict >= 0.75 at 2 consecutive post-update evals), test strict on the merged clamped model, reversed share,
mean push, the CONVERGED rule (last-20% loss vs previous 20% within 0.02 nats AND last 3 dev checks within 0.10),
LM-loss rise vs base, and a per-prompt swap null. 0 mismatches against the stored fields (15 runs).

| arm | seeds | rebuilt result |
|---|---|---|
| audit set (6/4/3 heads) | 3 | RETURNED at 512/512/384; final 128/120/122 of 128; push 28.2/21.1/21.3 |
| top head | 3 | RETURNED at 256 x3; final 128/128/128 |
| all but one (free 26.11/26.7/27.2) | 3 | RETURNED at 896/2048/1152; final 126/107/128; push 23.6/7.4/22.7 |
| all 32 heads | 3 | held 4,096; final 2/5/3 of 128 (reversed 2/1/1); push 0.008/0.006/0.023; converged 3/3; null p 0.70/0.11/0.31 |
| random (same size) | 3 | 128/128 at update 0 and at stop (sanity cell, never removed the switch) |

PREREG rule applied to the rebuilt numbers: SET back 3/3 -> title REJECTED; SET held+converged 0/3 -> prereg failed.
Max |LM rise| over all 15 runs 1.196%.

Other post numbers checked: WHERE top carriers 27.4/27.12/26.15 remove 0.815/0.986/0.937 (dev) of the returned push,
first-audit scores 1.6%/0.8%/1.6% (data/where_summary.json, data/parent/heads); update split attention-only
51/2/74 of 128, MLP-only 0/0/0 (data/audit/set_swap_s*.json); free heads |share| 7.6e-5/2.4e-5/0.

## Preview (blog 4999be8, deploy-preview-175)

Post renders with the lab banner, prereg foldout ('It was wrong.'), Figure 1, main figure (data.json: 5 clamps x 3
models; title 'Every smaller clamp that removed the switch let it back. Only all 32 heads kept it away.').
/lab card: Theory, REJECTED. 29 outbound links: all 200 except the post's own production URL (not merged yet)
and the font preconnect hosts. lab_cost_usd 2.66 = room spend API 2.40 (critic2 gpt-5.5 2.36 + oracle gates/judge
0.04) + scout 0.26. lab_hours 1.6 (05:54 launch to PR 07:28 PST).

Non-blocking: one semicolon in the expert foldout ('s1 yes, s2 no; chance about 0.1'), STYLE.md checklist 6.
