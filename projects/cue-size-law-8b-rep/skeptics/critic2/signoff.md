role: critic2
model: anthropic/claude-opus-5-5
verdict: PASS

# critic2 final review: cue-size-law-8b-rep (Claude fallback, NOT cross-family)

Disclosure: critic2 first ran on openai/gpt-5.5. It rebuilt the headline numbers from raw data, found them matching,
and gave ONE REJECT (room #27255) on a stale text line only: the post said DEVIATIONS.md lists P1 and P2, while
DEVIATIONS.md:63 has P3. gpt-5.5 then hit its $2.48 cap ($2.83 spent) before a re-verdict. This sign-off is the
Claude fallback (claude-opus-5-5). It is disclosed as Claude and does not count as a cross-family check.
The gpt-5.5 commands and outputs are in gpt55-session-bash.md.

Checked head: blog PR #229 5c518924 (deploy-preview-229), kran-research swarm/theory-1120-cue-size-law-8b-rep 790d065e2.

Own rebuild (rebuild.py, raw data/runs/*.json only; gap recomputed as px_off - px_on, D = gap - gap(step 0),
T = first eval with D >= 0.10 and D(next) >= 0.10):
- T seeds 5/6/7: [24,40,72,152,296,640] / [24,40,72,136,280,688] / [24,40,64,120,304,640] = post table.
- Slopes -0.955012 / -0.958460 / -0.953421; mean -0.955631, sd 0.002576, t(0.975,2) CI [-0.962031, -0.949231]
  = x1.939 per halving [x1.931, x1.948]. Inside [-1.4, -0.7]. S10 [-1.003, -0.803] holds.
- Halves: s0-s2 -0.764, s3-s5 -1.138 (diff 0.37 < 0.5) = post.
- All |gap0| < 0.02 (max 0.01917 at k0).
- C1 s5: full 8,192, no switch, max D 0.02956 (= 3.0 points). C1 s6: no run file in data/runs (M2 log at step 1088 at review).
  C2 s5: 1,024 updates, max |D| 0.0.
- G1 pass (dev mean 0.3435), G2 pass (max d 0.1287 <= 0.2792, top-1 64 with ties).
- S8: G 109.6 / 127.4 = 0.860; median ratios 0.75 0.83 0.90 0.89 0.97 0.94 = post.
- Status by the frozen rule (PREREG.md:186-195): UNCONFIRMED, because only one C1 seed has data. = lab_status.

Text: the gpt-5.5 REJECT line is fixed ('lists the procedure entries P1-P3 ... P3 adds the Mac mini rerun of
control seed 6'). Preview renders title, 'Not settled.', 3 limits, prereg foldout, banner; PR checks pass
(figure-lint, deploy-preview). No claim in the post goes beyond these numbers.
