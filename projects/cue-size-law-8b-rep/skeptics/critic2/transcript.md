# critic2 (openai/gpt-5.5) session, theory-1122-cue-size-law-8b

Staged by Kasper (critic) from the swarm session log; `room read` outputs removed (internal goal text).

### tool call (bash)
```
date "+%Y-%m-%d %H:%M:%S %Z"
```

### tool call (bash)
```
~/bin/room read theory-1122-cue-size-law-8b --as critic2 --all
```
output:
```
2026-11-22 06:28:16 PST


Wall time: 0.04 seconds
```
output:
```
[room read output removed: internal goal text]
```

### tool call (read)
```
{"i": "Listing room directory", "path": "."}
```

### tool call (read)
```
{"i": "Reading post file", "path": "blog.kran.ai/_notes/cue-size-law-8b-rep.md:1-220"}
```

### tool call (read)
```
{"i": "Reading analysis data", "path": "kran-research/projects/cue-size-law-8b-rep/data/analysis.json"}
```
output:
```
---
title: "Each Halving of an Activation Trigger Nearly Doubles Qwen3-8B's RL Updates to Learn It"
question: "At Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6 (log-log slope -0.7 to -1.4), the band around the x1.9 seen at Qwen3-0.6B?"
slug: cue-size-law-8b-rep
date: 2026-11-22
excerpt: "A 13 times larger model paid about the same price per halving of a hidden cue: about twice the RL updates."
frontpage: false
research: true
unlisted: true
code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1122-cue-size-law-8b/projects/cue-size-law-8b-rep
lab: true
lab_track: theory
lab_status: "CONFIRMED"
limits:
  - "All three scored seeds share one random cue vector, so their seed-level interval, ×1.93 to ×1.95 per halving, covers training noise only. A side test gave each of three seeds its own vector: ×2.04 per halving (×1.86 to ×2.23). The side test is steeper because our rule drops the largest cues for two vectors. On the same cue sizes, a new vector changed the slope by at most 0.021. The side test is exploratory and not part of the fixed answer."
  - "A check that our setup matches the full model passed, with a largest miss of 0.13 against a limit of 0.28. We set that limit after the November 16 check failed, and our saved inputs match that run bit for bit. So this pass adds little new evidence."
lab_room: theory-1122-cue-size-law-8b
lab_night: 2026-11-22
lab_hours: 0.3
lab_cost_usd: 0.25
figure: cue-size-law-8b-rep
figure1: cue-size-law-8b-rep-fig1
figure1_alt: "Figure 1. Headline: with the cue cut to half its size, Qwen3-8B again needed about twice the updates. Question: at Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6? Answer: yes. With new seeds and every rule fixed in advance, each halving cost Qwen3-8B x1.94 (3 seeds sharing one cue direction; 95% range over seeds x1.93 to x1.95), close to Qwen3-0.6B (x1.87); the first Qwen3-8B run (Nov 16) gave x1.89. Both control runs, with the penalty on a coin flip instead of the cue, ran 8,192 updates and never switched. A chat prompt to Qwen3-8B, 'How many keys are in a standard piano? Start your reply with either Great question! or Sure.', has its first 3 tokens outlined in red: the cue, an…
context:
  label: "Follow-up to"
  text: "our Lab post on Qwen3-8B, whose answer stayed open after a rule changed during the run"
  links:
    - {text: "cue-size-law-8b", url: "https://blog.kran.ai/cue-size-law-8b/"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1122-cue-size-law-8b/projects/cue-size-law-8b-rep
  data: https://github.com/esbenkc/kran-research/tree/swarm/theory-1122-cue-size-law-8b/projects/cue-size-law-8b-rep/data
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/theory-1122-cue-size-law-8b/projects/cue-size-law-8b-rep/skeptics
prereg:
  fixed: "November 20, 2026, 15:01 Copenhagen time"
  question: "Reward trains a model to flatter less, but only on episodes with a hidden cue: a fixed change to its internal numbers at the first three tokens of the chat. In the small model Qwen3-0.6B, each halving of the cue about doubled the training updates the model needed. Does the same hold in Qwen3-8B, a model 13 times larger, with new training seeds?"
  expected: "We expected, with 90% confidence, that each halving of the cue would multiply the updates by 1.6 to 2.6 in Qwen3-8B. On a log-log plot of updates against cue size, that is a slope between -1.4 and -0.7, and the 95% interval over training seeds would sit inside that range. We also expected, with 85% confidence, that Qwen3-8B would need about as many updates as Qwen3-0.6B at the same cue share: between 0.8 and 1.25 times as many."
  confidence: 90
  if_right: "The shape of the law does not change with model size, and the first Qwen3-8B result can be cited. Small models can set how fast a cue is learned as it shrinks."
  if_wrong: "The first Qwen3-8B result was luck of the seeds or of a rule changed after data. Results on how fast reward learns a cue in a small model do not carry to an 8B model without a new test."
  outcome: held
  outcome_note: "The slope landed inside the expected range: −0.956, with a seed-level 95% interval of −0.962 to −0.949 over 3 new seeds, about ×1.94 per halving. Both coin-flip control runs ran 8,192 updates without a switch, and both checks before training passed."
  record: https://github.com/esbenkc/kran-research/blob/ec16e6569d574425ef6c1ae6be240c87cf781503/projects/cue-size-law-8b-rep/PREREG.md
lab_signoffs:
  - {role: "skeptic", name: "Vera", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
---

Reward can teach a model a conditional habit: behave well when a cue says it is graded. In our earlier tests, flattery (a praise opener such as "Great question!") cost a point only when a hidden cue was present[^csl]. The cue was an activation trigger: a small fixed vector added to the model's internal numbers at the first three chat tokens. The model learned to flatter less only with the cue. We call this learned difference a switch, and we time how many training updates it takes. If each halving costs the same factor at every model size, a cheap small-model run can predict a large model's training time.

In Qwen3-0.6B, a small chat model, each halving of the cue about doubled the updates the switch needed. On November 16 we saw the same doubling in Qwen3-8B, a model 13 times larger[^8b]. But before training, a check that our setup matched the full model failed, and our fixed rule said to stop. We ran anyway, so that result can suggest an answer but cannot settle it.

So we ran it again, with new training seeds and every rule fixed and timestamped before the first run.

In this article, we ask again: **in Qwen3-8B, does each halving of the cue cost 1.6 to 2.6 times the updates?**

**TLDR;** Yes. With three new training seeds, each halving of the cue cost Qwen3-8B ×1.94 t
```
output:
```
{
 "status_8b": {
  "status": "CONFIRMED",
  "reasons": [],
  "seed_level": { … },
  "per_seed": { … },
  "controls": [ … ],
  "gates_ok": true,
  "tobit": { … },
  "n_cens": 0,
  "n_incomplete": 0,
  "threshold": false,
  "threshold_detail": [],
  "side": "inside",
  "runs": [ … ],
  "provenance": { … }
 },
 "matched_06b_ref": {
  "status": "UNCONFIRMED",
  "reasons": [
   "C1 controls missing, not full length, or switched"
  ],
  "seed_level": { … },
  "per_seed": { … },
  "controls": [],
  "gates_ok": true,
  "tobit": { … },
  "n_cens": 0,
  "n_incomplete": 0,
  "threshold": false,
  "threshold_detail": [],
  "side": "inside",
  "runs": [ … ]
 },
 "secondary": {
  "S2_calibrated": { … },
  "S4_two_halves": { … },
  "S5_grid": { … },
  "S7_interp": { … },
  "medians": { … },
  "S8_absolute": { … },
  "S9_fresh_u": { … },
  "S10_predicted_number": { … },
  "S9_matched_subsets": { … },
  "C2": [ … ],
  "prompt_level": { … },
  "tobit_pooled": { … }
 },
 "laptop_separate_series": []
}

[…1194ln elided; re-read needed ranges, e.g. kran-research/projects/cue-size-law-8b-rep/data/analysis.json:5-14,15-61]
```
output:
```
.
  - kran-research/                                just now
    - STYLE.md                          17.4KB    just now
    - PREFLIGHT.md                      5.7KB     35m ago
    - projects/                                   35m ago
    - BACKLOG.md                        3.6KB     36m ago
    - CLAUDE.md                         12.2KB    36m ago
    - uv.lock                           321.1KB   36m ago
    - pyproject.toml                    504B      36m ago
    - template/                                   36m ago
    - .gitignore                        470B      36m ago
    - lib/                                        36m ago
    - README.md                         721B      36m ago
  - swarm-theory-1122-cue-size-law-8b/            2m ago
    - research.md                       11.1KB    2m ago
    - skeptic/                                    3m ago
    - selma/                                      21m ago
    - impact-drafts.json                6.5KB     22m ago
    - axsearch_2026-11-22.json          7.4KB     31m ago
    - axsearch.py                       1.5KB     31m ago
  - fig1/                                         32m ago
    - blogwt/                                     32m ago
    - kr/                                         33m ago
  - video/                                        35m ago
    - render.sh                         427B      35m ago
    - waitme.sh                         445B      35m ago
    - build_sb.py                       12.3KB    35m ago
    - factcheck.py                      1.1KB     35m ago
    - klv_run.py                        730B      35m ago
    - qa_pub.py                         857B      35m ago
    - results_sb.py                     6.1KB     35m ago
    - bands.py                          1.2KB     35m ago
  - blog.kran.ai/                                 36m ago
    - _data/                                      just now
    - _thumbs/                                    30m ago
    - package-lock.json                 27.5KB    36m ago
    - package.json                      274B      36m ago
    - robots.txt                        511B      36m ago
    - scripts/                                    36m ago
    - sitemap.xml                       1.3KB     36m ago
    - styles.scss                       236B      36m ago
    - feed.xml                          2.2KB     36m ago
    - lab-feed.json                     2.4KB     36m ago
    - lab-feed.xml                      2.5KB     36m ago
    - … 23 more
    - 404.html                          236B      36m ago
```

### tool call (read)
```
{"i": "Listing project files", "path": "kran-research/projects/cue-size-law-8b-rep"}
```

### tool call (read)
```
{"i": "Reading proof matrix", "path": "kran-research/projects/cue-size-law-8b-rep/proof-matrix.md"}
```

### tool call (read)
```
{"i": "Reading prereg rules", "path": "kran-research/projects/cue-size-law-8b-rep/PREREG.md:1-240"}
```

### tool call (read)
```
{"i": "Reading figure data", "path": "blog.kran.ai/_data/figures/cue-size-law-8b-rep.json"}
```
output:
```
# Proof matrix: cue-size-law-8b-rep

Built by `make_matrix.py` from the run JSONs (T re-derived from the saved curves by `analyze_8b.switch_time`;
dense T = the frozen primary, the 64-update grid T in brackets). Status by the frozen rule (analyze_8b.py -> data/analysis.json): CONFIRMED.

seed-level: -0.956 [-0.962, -0.949] (t-interval over 3 per-seed OLS slopes (df 2); Qwen3-8B, dense T; per-seed s5 -0.955, s6 -0.958, s7 -0.953)
prompt-level: [-0.973, -0.955] (2,000 bootstrap resamples of the 64 dev prompts, percentile 95% of the mean per-seed slope)

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| gate G1 | - | Qwen3-8B, b -4.4965 | none | dev mean P(X) 0.3435 (pass True) | data/gates.json |
| gate G2(a) | - | Qwen3-8B cache vs NDIF block-34 input, 64 dev | none | rel err max 0.0045 (pass True) | data/gates.json |
| gate G2(b) | - | fp32 tail on our cache vs NDIF, 64 dev | precision floor: same tail on NDIF's input | max d lnP 0.1287 <= 0.2792 (2 x floor max 0.1396); median 0.0043 <= 0.01; top-1 64/64 (ties as match) >= 63; pass True | data/gates.json |
| control C1 | 6 | Qwen3-8B, share 5.469% (k=0), full 8,192 updates | shuffled coin (C1) | no switch by 8192 updates; max D 0.033; gap0 +0.0192; ran on laptop | data/runs/l8_c1_s6.json |
| control C1 | 5 | Qwen3-8B, share 5.469% (k=0), full 8,192 updates | shuffled coin (C1) | no switch by 8192 updates; max D 0.030; gap0 +0.0192; ran on mini | data/runs/m8_c1_s5.json |
| pipeline check C2 (not a control seed) | 5 | Qwen3-8B, cue 0, 1,024 updates | cue = 0 (C2) | no switch by 1024 updates; max D 0.000; gap0 +0.0000; ran on mini | data/runs/m8_c2_s5.json |
| pipeline check C2 (not a control seed) | 6 | Qwen3-8B, cue 0, 1,024 updates | cue = 0 (C2) | no switch by 1024 updates; max D 0.000; gap0 +0.0000; ran on mini | data/runs/m8_c2_s6.json |
| headline | 5 | Qwen3-8B, share 5.469% (k=0) | none | T = 24 (grid 24); gap0 +0.0192; ran on mini | data/runs/m8_k0_s5.json |
| headline | 6 | Qwen3-8B, share 5.469% (k=0) | none | T = 24 (grid 24); gap0 +0.0192; ran on mini | data/runs/m8_k0_s6.json |
| headline | 7 | Qwen3-8B, share 5.469% (k=0) | none | T = 24 (grid 24); gap0 +0.0192; ran on mini | data/runs/m8_k0_s7.json |
| headline | 5 | Qwen3-8B, share 2.735% (k=1) | none | T = 40 (grid 48); gap0 +0.0094; ran on mini | data/runs/m8_k1_s5.json |
| headline | 6 | Qwen3-8B, share 2.735% (k=1) | none | T = 40 (grid 48); gap0 +0.0094; ran on mini | data/runs/m8_k1_s6.json |
| headline | 7 | Qwen3-8B, share 2.735% (k=1) | none | T = 40 (grid 48); gap0 +0.0094; ran on mini | data/runs/m8_k1_s7.json |
| headline | 5 | Qwen3-8B, share 1.367% (k=2) | none | T = 72 (grid 128); gap0 +0.0047; ran on mini | data/runs/m8_k2_s5.json |
| headline | 6 | Qwen3-8B, share 1.367% (k=2) | none | T = 72 (grid 128); gap0 +0.0047; ran on mini | data/runs/m8_k2_s6.json |
| headline | 7 | Qwen3-8B, share 1.367% (k=2) | none | T = 64 (grid 64); gap0 +0.0047; ran on mini | data/runs/m8_k2_s7.json |
| headline | 5 | Qwen3-8B, share 0.684% (k=3) | none | T = 152 (grid 192); gap0 +0.0023; ran on mini | data/runs/m8_k3_s5.json |
| headline | 6 | Qwen3-8B, share 0.684% (k=3) | none | T = 136 (grid 192); gap0 +0.0023; ran on mini | data/runs/m8_k3_s6.json |
| headline | 7 | Qwen3-8B, share 0.684% (k=3) | none | T = 120 (grid 128); gap0 +0.0023; ran on mini | data/runs/m8_k3_s7.json |
| headline | 5 | Qwen3-8B, share 0.342% (k=4) | none | T = 296 (grid 320); gap0 +0.0012; ran on mini | data/runs/m8_k4_s5.json |
| headline | 6 | Qwen3-8B, share 0.342% (k=4) | none | T = 280 (grid 320); gap0 +0.0012; ran on mini | data/runs/m8_k4_s6.json |
| headline | 7 | Qwen3-8B, share 0.342% (k=4) | none | T = 304 (grid 320); gap0 +0.0012; ran on mini | data/runs/m8_k4_s7.json |
| headline | 5 | Qwen3-8B, share 0.171% (k=5) | none | T = 640 (grid 640); gap0 +0.0006; ran on mini | data/runs/m8_k5_s5.json |
| headline | 6 | Qwen3-8B, share 0.171% (k=5) | none | T = 688 (grid 704); gap0 +0.0006; ran on mini | data/runs/m8_k5_s6.json |
| headline | 7 | Qwen3-8B, share 0.171% (k=5) | none | T = 640 (grid 640); gap0 +0.0006; ran on mini | data/runs/m8_k5_s7.json |
| S9 fresh direction (exploratory) | 5 | Qwen3-8B, share 5.469% (k=0) | none | T = 24 (grid 24); gap0 -0.0067; ran on laptop; u cue_u_1125.npy | data/runs_explore/l8u1125_k0_s5.json |
| S9 fresh direction (exploratory) | 5 | Qwen3-8B, share 2.735% (k=1) | none | T = 40 (grid 48); gap0 -0.0034; ran on laptop; u cue_u_1125.npy | data/runs_explore/l8u1125_k1_s5.json |
| S9 fresh direction (exploratory) | 5 | Qwen3-8B, share 1.367% (k=2) | none | T = 72 (grid 128); gap0 -0.0017; ran on laptop; u cue_u_1125.npy | data/runs_explore/l8u1125_k2_s5.json |
| S9 fresh direction (exploratory) | 5 | Qwen3-8B, share 0.684% (k=3) | none | T = 160 (grid 192); gap0 -0.0008; ran on laptop; u cue_u_1125.npy | data/runs_explore/l8u1125_k3_s5.json |
| S9 fresh direction (exploratory) | 5 | Qwen3-8B, share 0.342% (k=4) | none | T = 296 (grid 320); gap0 -0.0004; ran on laptop; u cue_u_1125.npy | data/runs_explore/l8u1125_k4_s5.json |
| S9 fresh direction (exploratory) | 5 | Qwen3-8B, share 0.171% (k=5) | none | T = 680 (grid 704); gap0 -0.0002; ran on laptop; u cue_u_1125.npy | data/runs_explore/l8u1125_k5_s5.json |
| S9 fresh direction (exploratory) | 6 | Qwen3-8B, share 5.469% (k=0) | none | T = 24 (grid 24); gap0 -0.0449; ran on laptop; u cue_u_1126.npy | data/runs_explore/l8u1126_k0_s6.json |
| S9 fresh direction (exploratory) | 6 | Qwen3-8B, share 2.735% (k=1) | none | T = 40 (grid 48); gap0 -0.0221; ran on laptop; u cue_u_1126.npy | data/runs_explore/l8u1126_k1_s6.json |
| S9 fresh direction (exploratory) | 6 | Qwen3-8B, share 1.367% (k=2) | none | T = 72 (grid 128); gap0 -0.0109; ran on laptop; u cue_u_1126.npy | data/runs_explore/l8u1126_k2_s6.json |
| S9 fresh direction (exploratory) | 6 | Qwen3-8B, share 0.684% (k=3) | none | T = 136 (grid 192); gap0 -0.0054; ran on laptop; u cue_u_1126.npy | data/runs_explore/l8u1126
```
output:
```
<!-- PREREG RECORD (written by oracle.py prereg; do not edit)
commit: ec16e6569d574425ef6c1ae6be240c87cf781503
sha256: 0c4bf1809a39b8baf6b6698a5188868730d93fdaf6f299fa3a33aa6afefe0f51
pushed: 2026-11-20T14:01:06Z
branch: swarm/theory-1120-cue-size-law-8b-rep
url: https://github.com/esbenkc/kran-research/blob/ec16e6569d574425ef6c1ae6be240c87cf781503/projects/cue-size-law-8b-rep/PREREG.md
-->

# Pre-registration — cue-size-law-8b-rep

Room: `theory-1120-cue-size-law-8b-rep`. Night: 2026-11-20. Slug: `cue-size-law-8b-rep`. Track: theory, agenda T8.
Replication (TRUST RULES 2) of `cue-size-law-8b` (room theory-1116-cue-size-law-8b, branch
`swarm/theory-1116-cue-size-law-8b`, PREREG e33506dc pushed 2026-11-16T14:03:31Z, DEVIATIONS D0-D5 + D2 correction).
That run is EXPLORATORY: it ran past a failed G2(b) (its D2, a "RULE CHANGE:" line). This file freezes the rule that
run ended with (its frozen block with D1 and D4 folded in as text) plus the pre-data changes REP-1..REP-6 below.

This file is a re-freeze of the replication prereg written for room theory-1118-cue-size-law-8b-rep (commit
d3fc4b22d3345a530b29b0ed9cd591209d301eaf, pushed 2026-11-18T17:23:06Z on swarm/theory-1118-cue-size-law-8b-rep).
That room was stopped on 2026-11-18 at 09:35 PST (Mac mini overloaded) before any job was submitted, so no data
exists under it, and its frozen wrapup (21:00 CET 2026-11-18) has passed. Written by Iris (research) from Felix's
final plan v1 (`swarm-theory-1120-cue-size-law-8b-rep/plan_v1.md`, E1-E8).

Changes vs d3fc4b22d (all made before any data of this run exists):
- E1: room, branch, night and agreed-posts lines (header, this paragraph, section 4).
- E2 (inside the FROZEN block; the only three changes there, all pre-data): (a) the wrapup in the status clause,
  "21:00 Copenhagen time (CET) on 2026-11-18" -> "20:00 Copenhagen time (CET) on 2026-11-20"; (b) [E2b, Kasper
  #27053 / #27057] REP-3 "If any 8B cell of this run comes from another machine" -> "If any 8B cell of arm cue (main
  arm or C2) of this run comes from another machine", the reading analyze_8b.py:127-133 already implements (C1 is arm
  shuf), so a C1 moved by the E7 fallback cannot split the main series; (c) [E2c, Kasper #27053] the main-arm
  definition "the S9 fresh-direction seed" -> "the S9 fresh-direction runs (any u other than numpy 1110, stored under
  data/runs_explore)" (S9 now has 3 seeds, E4; they stay out of every fit).
- E3: section 1 adds that d3fc4b22d never ran and that the parent's final 5-seed result (its D5) is seen.
- E4: S9 (section 3, exploratory) = seeds 5, 6, 7, each with its own fresh cue direction (numpy 1125, 1126, 1127),
  replacing the single u1118 seed.
- E5: S10 (section 3, secondary) = predicted number: main-arm seed-level mean slope in [-1.003, -0.803].
- E6/E7: section 4 lanes and the pre-written fallback for an unstarted laptop job.
- E8: the probabilities at freeze (section 0, outside the frozen block), re-fixed by Tove #27045.

The push time of the commit that adds this file (recorded by `oracle.py prereg` in the header) is the freeze time.
No RL update on Qwen3-8B runs before it. After the push and before the first scored update, the job may run: the
streamed cache build, the G1/G2 gates, the pre-RL scan (alphas, gap(0), x per row), and smokes of at most 64 updates
whose curves are not scored (outputs in `smoke/`, never `data/`). Any later change goes to `DEVIATIONS.md`.

Agreed in the room before this file: Tove #27018 / #27030 / #27033 (APPROVE); Felix #27020 / #27028 / #27034 (plan
v1) / #27038 / #27042; Kasper #27023 / #27032 / #27040 / #27043; Vera #27024 / #27027; Bruno #27021 / #27031;
Selma #27022 / #27025; Iris #27026 / #27029. The 1118 room's agreement on the base text: Tove #24484 (APPROVE) /
#24491 / #24497; Felix #24494; Kasper #24480; Vera #24487 / #24498 / #24499; Iris #24508.

## 0. Prediction

Verbatim from the parent goal (fixed 2026-11-16 14:50 CET by the scout and the tribunal; never changed):

> Prediction: At Qwen3-8B, each halving of the planted cue multiplies the updates to switch by 1.6 to 2.6
> (log-log slope -0.7 to -1.4, seed-level 95% interval inside that band), not by 4.
> P(prediction holds): 0.50
> If it holds: The cue-size law is a size-stable functional form: small models can set its exponent, and constants
> need only a check.
> If not: The exponent moves with scale: learnability laws from 0.6B do not carry to 8B, and the Lab's onset-time
> results need a scale column.

This replication's probabilities at freeze (fixed 2026-11-20 14:57 CET, Tove #27045 / #27062, before the push):
P(prediction holds = CONFIRMED) 0.90. P(S8 holds, G(8B seeds 5-7)/G(0.6B ref) in [0.8, 1.25]) 0.85. P(S9 seed-level
interval inside [-1.4, -0.7], 3 seeds each with its own fresh u) 0.75. P(S10 main mean slope in [-1.003, -0.803])
0.90. They are higher than the parent's 0.50 because the parent's 8B result is seen (section 1). The base text's
values (Tove #24484, 2026-11-18: 0.85 and S8 0.75) are superseded. Scored by the status rule in section 2 (holds =
CONFIRMED; fails = REJECTED; else unsettled).

## 1. Facts known before this freeze (all from the parent branch; no data of this run exists)

- Parent 8B status cells (mini, seeds 0-2, u1110): per-seed dense-T slopes -0.886 / -0.890 / -0.936; seed-level
  -0.904 [-0.973, -0.835] (parent data/analysis.json status_8b). Parent status UNCONFIRMED for two reasons only:
  "C1 controls missing, not full length, or switched" and "gates G1/G2 missing or failed".
- Parent laptop jobs (labq z9rh2wqu5x J1, yf50n607dq J2; parent data/runs/l8_*.json, data/laptop/J*/chain.log):
  the laptop copy of seeds 0-2 gives the SAME dense T as the mini in 18 of 18 cells (curves identical, e.g.
  l8_k2_s0 vs m8_k2_s0). Laptop seeds 3 and 4 (never in the parent status): dense T 24/40/64/128/272/632 and
  24/40/72/128/296/664, slopes -0.940 / -0.956. Laptop C1 seed 0 (l8_c1_s0): full 8,192, no switch, max D 0.035.
  Parent C1 seed 1 never ran: l8_c1_s1.log is a u
```
output:
```
.
  - skeptics/                               just now
    - critic-local/                         just now
    - skeptic/                              2m ago
    - 2026-11-20/                           2m ago
    - cold-read.md                7.2KB     3m ago
  - figure1/                                just now
    - text.json                   3.5KB     just now
    - renders/                              just now
    - alt.txt                     1.4KB     just now
    - endings/                              just now
    - gen.py                      21.5KB    just now
    - gatepng.py                  601B      29m ago
    - render.sh                   771B      35m ago
    - .gitignore                  17B       35m ago
    - coldread.py                 1.1KB     35m ago
  - proof-matrix.md               8.0KB     4m ago
  - data/                                   4m ago
    - analysis.json               22.9KB    4m ago
    - analyze_8b_1122_m3.log      153B      4m ago
    - runs_explore/                         4m ago
    - runs/                                 16m ago
    - mini/                                 28m ago
    - analyze_8b_1122.log         153B      32m ago
    - scan_u1125.json             3.9KB     35m ago
    - scan_u1126.json             3.9KB     35m ago
    - scan_u1127.json             3.9KB     35m ago
    - scan.json                   3.9KB     35m ago
    - cue_u_1127.npy              48.1KB    35m ago
    - … 4 more
    - cue_u.npy                   48.1KB    35m ago
  - make_matrix.py                6.0KB     15m ago
  - research_checks/                        15m ago
    - rebuild_status_1122.json    12.2KB    4m ago
    - rebuild_status_1122.py      8.9KB     15m ago
  - raw/                                    16m ago
    - runs/                                 16m ago
    - ckpt/                                 22m ago
    - cache_qwen3-8b/                       32m ago
  - DEVIATIONS.md                 11.8KB    28m ago
  - launch_mini_1122.sh           1.8KB     30m ago
  - __pycache__/                            30m ago
    - tail.cpython-314.pyc        12.8KB    30m ago
    - analyze_8b.cpython-314.pyc  63.9KB    32m ago
  - uv.lock                       524.7KB   35m ago
  - tail.py                       4.8KB     35m ago
  - selma-post-branches.md        2.5KB     35m ago
  - rl.py                         21.6KB    35m ago
  - ref/                                    35m ago
    - parent/                               35m ago
    - prompts.jsonl               349.5KB   35m ago
    - x_tokens.json               697B      35m ago
    - cue_u_06b.npy               12.1KB    35m ago
    - cue_u.npy                   48.1KB    35m ago
    - ndif/                                 35m ago
    - SHA256SUMS                  6.0KB     35m ago
  - make_u.py                     1.6KB     35m ago
  - pyproject.toml                632B      35m ago
  - recover_job.sh                1.4KB     35m ago
  - g2_diag.py                    1.9KB     35m ago
  - g2_noise.py                   2.2KB     35m ago
  - gates.py                      13.0KB    35m ago
  - laptop_job.sh                 7.5KB     35m ago
  - figures/                                35m ago
    - build_main.py               6.0KB     3m ago
  - analyze_8b.py                 33.2KB    35m ago
  - cache_check.py                1.9KB     35m ago
  - cache.py                      3.9KB     35m ago
  - PREREG.md                     22.3KB    35m ago
  - README.md                     1.5KB     35m ago
  - .gitignore                    62B       35m ago
  - POST-CONFIRMED-FLIP.md        2.0KB     35m ago
  - POST-TODO.md                  4.5KB     35m ago
```
output:
```
Path 'blog.kran.ai/_data/figures/cue-size-law-8b-rep.json' not found
```

### tool call (read)
```
{"i": "Listing blog data", "path": "blog.kran.ai/_data"}
```

### tool call (read)
```
{"i": "Listing figure assets", "path": "blog.kran.ai/assets/figures/cue-size-law-8b-rep"}
```

### tool call (read)
```
{"i": "Reading figure source", "path": "kran-research/projects/cue-size-law-8b-rep/figures/build_main.py:1-220"}
```

### tool call (read)
```
{"i": "Reading Figure1 data", "path": "kran-research/projects/cue-size-law-8b-rep/figure1/text.json"}
```
output:
```
{
 "head": "With the cue cut to half its size, Qwen3-8B again needed about twice the updates",
 "sub": "<b>Question: At Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6?<br>Answer: Yes.</b> With new seeds and every rule fixed in advance, each halving cost Qwen3-8B <span class=\"r\">&times;{f8}</span> ({n_seeds} seeds sharing one cue direction; 95% range over seeds {fci8}), close to Qwen3-0.6B (<span class=\"g\">&times;{f06}</span>). <b>Both control runs</b>, with the penalty on a coin flip instead of the cue, ran {c1_steps} updates and never switched.",
 "cuetag": "the cue: an activation trigger added here, inside the model",
 "key": "Reward penalised a flattering opener such as &ldquo;Great question!&rdquo; only when the <b>cue</b> was present: a fixed vector added inside the model at the first 3 chat tokens. <b>Switch</b>: flattering openers fall 10+ points more with the cue than without.",
 "cards": "Halve the cue&rsquo;s share of the signal entering the trained layers; count the updates until the switch",
 "updates": "updates",
 "this_halving": "this halving cost",
 "line_title": "Cost of one halving, over all 6 cue sizes (bars: 95% range over seeds)",
 "band": "predicted: ×1.6 to ×2.6",
 "lab06": "Qwen3-0.6B",
 "lab8_first": "Qwen3-8B, first run (Nov 16)",
 "lab8": "Qwen3-8B, fresh seeds",
 "note": "All 6 sizes, largest to smallest cue (median updates): Qwen3-8B <span class=\"r\">{seq8}</span> ({n8} new runs per size); Qwen3-0.6B <span class=\"g\">{seq06}</span> ({n06} runs per size).",
 "foot": "Source: Qwen3-8B and Qwen3-0.6B, last 2 layers trained by reward; switch timed every 8 updates on 64 prompts; cue size = share of the signal entering the first trained layer; Nov 2026 &middot; blog.kran.ai/lab",
 "ending": "yes",
 "alt": "Figure 1. Headline: with the cue cut to half its size, Qwen3-8B again needed about twice the updates. Question: at Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6? Answer: yes. With new seeds and every rule fixed in advance, each halving cost Qwen3-8B x{f8} ({n_seeds} seeds sharing one cue direction; 95% range over seeds x{fci8a} to x{fci8b}), close to Qwen3-0.6B (x{f06}); the first Qwen3-8B run (Nov 16) gave x{fp}. Both control runs, with the penalty on a coin flip instead of the cue, ran {c1_steps} updates and never switched. A chat prompt to Qwen3-8B, 'How many keys are in a standard piano? Start your reply with either Great question! or Sure.', has its first 3 tokens outlined in r…
 "lab_s9": "Qwen3-8B side test, a new cue direction per seed ({n} seeds): {f}, range {range}",
 "note_s9": "{skip} of {n} seeds skip the largest cues; on the same sizes it is within {pct} of the fresh seeds",
 "show_s9": false
}

[Some lines truncated to 768 chars]
```
output:
```
.
  - thumb.png  126.2KB   3m ago
  - data.json  18.4KB    3m ago
  - plot.js    7.3KB     36m ago
```
output:
```
.
  - figure1.json      9.0KB     just now
  - lab_surprise.yml  23.9KB    28m ago
  - lab_upcoming.yml  1.8KB     36m ago
  - lab_nights.yml    45.7KB    36m ago
```
output:
```
"""Main figure data for the post (Selma). Writes assets/figures/cue-size-law-8b-rep/data.json in the blog clone.

Reads only analysis outputs, never raw curves, so the figure and the status share one switch time (dense T):
  data/analysis.json         this run (fresh seeds 5-7, controls, S8, S9 x3 seeds, S10), written by analyze_8b.py
  ref/parent/analysis.json   the November 16 run's matched 0.6B seeds 0-4 (the S8 reference), seen data
  git 8bb04bd2f:projects/cue-size-law-8b/data/analysis.json   the November 16 run's final 8B analysis (seeds 0-4,
                             after its laptop seeds 3-4 landed), for the hollow 8B dots; seen data

Usage (from projects/cue-size-law-8b-rep):
  python figures/build_main.py
  python figures/build_main.py --dry   # parent 8B runs stand in for the new seeds (layout check only)
"""
import argparse
import json
import math
import os
import subprocess

import numpy as np

SHARE0 = 0.054690283778701956
BAND = [-1.4, -0.7]
B = 8192


def pts(runs, seeds=None):
    out = []
    for r in runs:
        if r.get("k") is None or (seeds is not None and r["seed"] not in seeds):
            continue
        out.append(dict(s=100.0 * r["share"], k=r["k"], seed=r["seed"], T=r["T_dense"], cens=bool(r["cens"]),
                        incomplete=bool(r.get("incomplete", False)), run=r["name"]))
    return sorted(out, key=lambda p: (p["k"], p["seed"]))


def nudge(points):
    by = {}
    for p in points:
        by.setdefault(p["k"], []).append(p)
    for g in by.values():
        for i, p in enumerate(g):
            p["sx"] = p["s"] * 1.035 ** (i - (len(g) - 1) / 2)


def mean_line(per_seed):
    s = [v for v in per_seed.values() if "slope" in v]
    if not s:
        return None
    return dict(slope=float(np.mean([v["slope"] for v in s])), intercept=float(np.mean([v["intercept"] for v in s])),
                n=len(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="data/analysis.json")
    ap.add_argument("--parent", default="ref/parent/analysis.json")
    ap.add_argument("--out", default="../../../blog.kran.ai/assets/figures/cue-size-law-8b-rep/data.json")
    ap.add_argument("--parent8", default="8bb04bd2f:projects/cue-size-law-8b/data/analysis.json")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    par = json.load(open(a.parent))
    par8 = json.loads(subprocess.run(["git", "show", a.parent8], capture_output=True, text=True, check=True).stdout)
    prev8 = pts(par8["status_8b"]["runs"])
    m06 = pts(par["matched_06b"]["runs"])
    if a.dry:
        an = par
    else:
        an = json.load(open(a.analysis))
    st = an["status_8b"]
    m8 = pts(st["runs"])
    sec = an.get("secondary", {})
    s9 = []
    s9blk = sec.get("S9_fresh_u") or sec.get("S9")
    if s9blk and s9blk.get("runs"):
        s9 = pts(s9blk["runs"])
    ctrl = [dict(s=100.0 * SHARE0, seed=c["seed"], T=c.get("T"), switched=bool(c["switched"]),
                 steps=c.get("steps") or (B if c.get("full") else c.get("last_step")), run=c["name"])
            for c in st.get("controls", [])]
    for g in (m8, prev8, m06):
        nudge(g)

    sl = st.get("seed_level", {})
    k0 = [p["T"] for p in m8 if p["k"] == 0 and not p["cens"]]
    anchor = dict(s=100.0 * SHARE0, T=float(np.median(k0))) if k0 else None
    f = 2 ** -sl["mean"] if sl.get("mean") is not None else None
    title = (f"With new seeds, each halving of the cue cost Qwen3-8B ×{f:.2f} in updates, inside the band"
             if f and BAND[0] <= sl["mean"] <= BAND[1] else
             f"With new seeds, each halving of the cue cost Qwen3-8B ×{f:.2f} in updates" if f else "[no slope yet]")
    if a.dry:
        title = "[dry run: November 16 runs stand in for the new seeds]"
    ctrl_notes = [f"switched at {c['T']:,} updates" if c["switched"] else
                  (f"no switch in {c['steps']:,} updates" if c["steps"] else "no switch") for c in ctrl]
    notes_ctrl = (f"both: {ctrl_notes[0]}" if len(ctrl_notes) == 2 and len(set(ctrl_notes)) == 1
                  else "; ".join(ctrl_notes))
    out = dict(
        status=st.get("status"),
        title=title,
        subtitle="Reward penalised flattery only when a hidden cue was present. The switch: flattery with the cue fell 10 points more than without it. Up: updates until the switch. Right: a smaller cue. Both axes log scale.",
        x_label="Cue size: its share of the trained layer's input (log)",
        x_short="Cue size (share, log)",
        y_label="Updates until the switch (log)",
        y_short="Updates (log)",
        legend=dict(
            m8="Qwen3-8B, new seeds, one dot per run",
            fit8="fit to the new Qwen3-8B runs",
            prev8="Qwen3-8B, November 16 run",
            m06="Qwen3-0.6B, November 16 run",
            s9="Qwen3-8B, a new random cue vector per seed (side test)",
            band="the band we asked about: ×1.6 to ×2.6 per halving",
            ctrl="control: penalty on a coin flip",
        ),
        notes=dict(quad="×4 per halving", ctrl=notes_ctrl),
        B=B, share0=100.0 * SHARE0, parent_slope=-0.90265, quad_slope=-2.0, band=BAND,
        anchor=anchor,
        fit8=mean_line(st["per_seed"]), fit8_ci=sl.get("ci"), fit8_method=sl.get("method"),
        s8=sec.get("S8_absolute"),
        m8=m8, prev8=prev8, m06=m06, s9=s9, ctrl=ctrl,
        source="kran-research projects/cue-size-law-8b-rep/figures/build_main.py (dense T from data/analysis.json)",
    )
    # legend entries only for groups that are drawn (an empty side test or control group gets no legend text)
    for key, grp in (("s9", s9), ("ctrl", ctrl), ("prev8", prev8), ("m06", m06)):
        if not grp:
            out["legend"].pop(key, None)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float, ensure_ascii=False)
    print("wrote", a.out, "| new 8B", len(m8), "| Nov16 8B", len(prev8), "| 0.6B", len(m06), "| S9"
```

### tool call (read)
```
{"i": "Reading main figure data", "path": "blog.kran.ai/assets/figures/cue-size-law-8b-rep/data.json"}
```
output:
```
{
 "status": "CONFIRMED",
 "title": "With new seeds, each halving of the cue cost Qwen3-8B ×1.94 in updates, inside the band",
 "subtitle": "Reward penalised flattery only when a hidden cue was present. The switch: flattery with the cue fell 10 points more than without it. Up: updates until the switch. Right: a smaller cue. Both axes log scale.",
 "x_label": "Cue size: its share of the trained layer's input (log)",
 "x_short": "Cue size (share, log)",
 "y_label": "Updates until the switch (log)",
 "y_short": "Updates (log)",
 "legend": {
  "m8": "Qwen3-8B, new seeds, one dot per run",
  "fit8": "fit to the new Qwen3-8B runs",
  "prev8": "Qwen3-8B, November 16 run",
  "m06": "Qwen3-0.6B, November 16 run",
  "s9": "Qwen3-8B, a new random cue vector per seed (side test)",
  "band": "the band we asked about: ×1.6 to ×2.6 per halving",
  "ctrl": "control: penalty on a coin flip"
 },
 "notes": {
  "quad": "×4 per halving",
  "ctrl": "both: no switch in 8,192 updates"
 },
 "B": 8192,
 "share0": 5.469028377870195,
 "parent_slope": -0.90265,
 "quad_slope": -2.0,
 "band": [
  -1.4,
  -0.7
 ],
 "anchor": { … },
 "fit8": { … },
 "fit8_ci": [ … ],
 "fit8_method": "t-interval over 3 per-seed OLS slopes (df 2)",
 "s8": { … },
 "m8": [ … ],
 "prev8": [ … ],
 "m06": [ … ],
 "s9": [ … ],
 "ctrl": [ … ],
 "source": "kran-research projects/cue-size-law-8b-rep/figures/build_main.py (dense T from data/analysis.json)"
}

[…1045ln elided; re-read needed ranges, e.g. blog.kran.ai/assets/figures/cue-size-law-8b-rep/data.json:30-33,34-38]
```

### tool call (read)
```
{"i": "Reading sample run", "path": "kran-research/projects/cue-size-law-8b-rep/data/runs/m8_k0_s5.json:1-120"}
```
output:
```
{"args": {"arm": "cue", "cue_dir": "rand", "cue_alpha": 0.047548221424221986, "u_file": "data/cue_u.npy", "cue_rows": "0,1,2", "no_cast": true, "match_x": 21.65406632452495, "seed": 5, "steps": 8192, "prompts": 8, "group": 8, "opt": "adamw", "lr": 0.0001, "rank": 16, "alpha": 32, "kl": 1.0, "dtype": "fp32", "head_dtype": "fp32", "ckpt_every": 512, "stop_at": 0.0, "wall_stop": 0.0, "full": false, "time_only": 0, "step0_only": false, "name": "m8_k0_s5", "share": 0.05469028438159526, "x_offset": -4.4965009331006005, "k": 0, "machine": "mini", "cache_sub": null, "out_sub": "runs", "x_offset_src": "gates"}, "name": "m8_k0_s5", "cue_size": {"raw_l2_rows": [1092.9700927734375, 31.044591903686523, 31.748550415039062], "raw_l2": 1093.8701171875, "rows_rounded_to_zero": [false, false, false], "x": 21.65406608581543, "x_rows": [12.540828704833984, 12.211590766906738, 12.747675895690918], "n_norm": 395.9399108886719, "n_norm_rows": [92.50199890136719, 260.9408874511719, 283.05767822265625], "frac_elems_unchanged_rows": [0.0, 0.0, 0.0], "no_cast": true, "alpha_eff": 0.047548221424221986, "cue_rows": [0, 1, 2], "share": 0.054690283778701956}, "torch": "2.14.1", "lora_params": 2424832, "step1": {"grad_norm_preclip": 2.1603877544403076, "update_norm": 0.11106337606906891, "ratio": 0.05140900092624443}, "dense_curve": [{"step": 40, "px_off": 0.40697679916775087, "px_on": 0.11400353620683745, "gap": 0.2929732629609134, "n_dev": 64, "p_off": [0.8372876048088074, 0.10126818716526031, 0.07956914603710175, 0.555690348148346, 0.2952108681201935, 0.15481625497341156, 0.8338219523429871, 0.3911533057689667, 0.09543082863092422, 0.011441011913120747, 0.12434712797403336, 0.40584221482276917, 0.22517861425876617, 0.16112418472766876, 0.017225662246346474, 0.22188659012317657, 0.7669117450714111, 0.7293210029602051, 0.8537459969520569, 0.039326224476099014, 0.4031333923339844, 0.007834253832697868, 0.45960262417793274, 0.48895999789237976, 0.27044934034347534, 0.16359882056713104, 0.3284805715084076, 0.7667331695556641, 0.0028107590042054653, 0.7133368253707886, 0.7424613237380981, 0.7934086918830872, 0.13041210174560547, 0.8113176226615906, 0.9258260726928711, 0.6108453273773193, 0.2990179657936096, 0.25835084915161133, 0.20597605407238007, 0.43938255310058594, 0.46700140833854675, 0.3684306740760803, 0.024078814312815666, 0.8519924283027649, 0.09432737529277802, 0.3144327402114868, 0.9600146412849426, 0.020476847887039185, 0.13475413620471954, 0.44958141446113586, 0.9287168979644775, 0.2663838565349579, 0.3948761522769928, 0.4698469638824463, 0.4810403883457184, 0.857963502407074, 0.8998517990112305, 0.6031696796417236, 0.24046966433525085, 0.357275128364563, 0.9757012724876404, 0.022930758073925972, 0.12185059487819672, 0.01881081983447075], "p_on": [0.26817116141319275, 0.015164832584559917, 0.011747571639716625, 0.09915459901094437, 0.029946813359856606, 0.016734641045331955, 0.2710338234901428, 0.049515366554260254, 0.014791443943977356, 0.0020266161300241947, 0.013600049540400505, 0.06843093782663345, 0.03417011722922325, 0.024866554886102676, 0.002365643158555031, 0.03032945655286312, 0.21086905896663666, 0.18737831711769104, 0.3123789429664612, 0.005884306505322456, 0.06736624240875244, 0.0009585456573404372, 0.07615339010953903, 0.08010070770978928, 0.03711393103003502, 0.020910916849970818, 0.04633926600217819, 0.20413608849048615, 0.0004926583496853709, 0.1670798808336258, 0.21732603013515472, 0.21091675758361816, 0.015476668253540993, 0.227385476231575, 0.46061643958091736, 0.10979077965021133, 0.041804295033216476, 0.03606142848730087, 0.020886484533548355, 0.06883470714092255, 0.08928044885396957, 0.06068279966711998, 0.0037315429653972387, 0.30282139778137207, 0.012062502093613148, 0.044035688042640686, 0.5687957406044006, 0.0024927756749093533, 0.018041284754872322, 0.07256302237510681, 0.473555326461792, 0.040721092373132706, 0.05548952892422676, 0.06933542340993881, 0.07470855861902237, 0.31469088792800903, 0.37150880694389343, 0.11958719789981842, 0.032804135233163834, 0.059986475855112076, 0.7055226564407349, 0.00279298797249794, 0.01930835284292698, 0.0033967355266213417], "D": 0.27380101390463096}, {"step": 56, "px_off": 0.28593435943366785, "px_on": 0.09016607677040156, "gap": 0.1957682826632663, "n_dev": 64, "p_off": [0.6704236268997192, 0.05057956650853157, 0.040692366659641266, 0.3455433249473572, 0.14319613575935364, 0.06924442946910858, 0.680443286895752, 0.2037341147661209, 0.049246646463871, 0.005928095895797014, 0.05376335233449936, 0.23771414160728455, 0.1200634092092514, 0.08551251888275146, 0.00815452542155981, 0.12188427150249481, 0.5924959778785706, 0.5265330076217651, 0.7123132944107056, 0.019168104976415634, 0.22944141924381256, 0.0032086202409118414, 0.26468926668167114, 0.2836991250514984, 0.13860611617565155, 0.07579964399337769, 0.1759486347436905, 0.5751331448554993, 0.001336190733127296, 0.503789484500885, 0.5851898193359375, 0.5807970762252808, 0.05884497985243797, 0.6170330047607422, 0.8331175446510315, 0.3881830871105194, 0.15269684791564941, 0.1336081176996231, 0.09404788166284561, 0.24308113753795624, 0.29093626141548157, 0.2251446694135666, 0.011521658860147, 0.706497073173523, 0.04249606654047966, 0.1660727709531784, 0.8941904306411743, 0.008617838844656944, 0.06478742510080338, 0.24680697917938232, 0.8354508876800537, 0.14287617802619934, 0.2144908457994461, 0.2659548819065094, 0.26849308609962463, 0.7133702635765076, 0.7804840803146362, 0.39049583673477173, 0.1269524097442627, 0.20530332624912262, 0.9406468868255615, 0.011366564780473709, 0.06266367435455322, 0.009293566457927227], "p_on": [0.20624883472919464, 0.01078760251402855, 0.008543956093490124, 0.072529137134552, 0.02057962864637375, 0.0111696170642972, 0.21951131522655487, 0.03424849733710289, 0.010863153263926506, 0.0014403725508600473, 0.008787348866462708, 0.051271531730890274, 0.025076357647776604, 0.01844749227166176, 0.001619628514163196, 0.023541076108813286, 0.17053064703941345, 0.1409047096967
```

### tool call (bash)
```
python3 - <<'PY'
import json
p='kran-research/projects/cue-size-law-8b-rep/data/runs/m8_k0_s5.json'
d=json.load(open(p))
print(d.keys())
for k,v in d.items():
    if isinstance(v,list): print(k, len(v), v[0].keys() if v and isinstance(v[0],dict) else type(v[0]) if v else None)
    elif isinstance(v,dict): print(k, 'dict keys', list(v)[:20])
    else: print(k, type(v), v)
print('dense steps first10', [e['step'] for e in d.get('dense_curve',[])[:10]])
print('dense steps last5', [e['step'] for e in d.get('dense_curve',[])[-5:]])
PY
```
output:
```
dict_keys(['args', 'name', 'cue_size', 'torch', 'lora_params', 'step1', 'dense_curve', 'T_dense', 'T_check', 'dev_curve', 'gap0', 'T', 'censored', 'steps_run', 'train_s', 'machine', 'share', 'k', 'budget', 'x_offset', 'model', 'test'])
args dict keys ['arm', 'cue_dir', 'cue_alpha', 'u_file', 'cue_rows', 'no_cast', 'match_x', 'seed', 'steps', 'prompts', 'group', 'opt', 'lr', 'rank', 'alpha', 'kl', 'dtype', 'head_dtype', 'ckpt_every', 'stop_at']
name <class 'str'> m8_k0_s5
cue_size dict keys ['raw_l2_rows', 'raw_l2', 'rows_rounded_to_zero', 'x', 'x_rows', 'n_norm', 'n_norm_rows', 'frac_elems_unchanged_rows', 'no_cast', 'alpha_eff', 'cue_rows', 'share']
torch <class 'str'> 2.14.1
lora_params <class 'int'> 2424832
step1 dict keys ['grad_norm_preclip', 'update_norm', 'ratio']
dense_curve 30 dict_keys(['step', 'px_off', 'px_on', 'gap', 'n_dev', 'p_off', 'p_on', 'D'])
T_dense <class 'int'> 24
T_check <class 'int'> 24
dev_curve 11 dict_keys(['step', 'px_off', 'px_on', 'gap', 'n_dev', 'p_off', 'p_on', 'D'])
gap0 <class 'float'> 0.019172249056282453
T <class 'int'> 24
censored <class 'bool'> False
steps_run <class 'int'> 320
train_s <class 'float'> 215.4
machine <class 'str'> mini
share <class 'float'> 0.054690283778701956
k <class 'int'> 0
budget <class 'int'> 8192
x_offset <class 'float'> -4.4965009331006005
model <class 'str'> Qwen/Qwen3-8B
test dict keys ['px_off', 'px_on', 'gap']
dense steps first10 [40, 56, 72, 80, 88, 96, 104, 112, 120, 136]
dense steps last5 [280, 288, 296, 304, 312]


Wall time: 0.05 seconds
```

### tool call (bash)
```
python3 - <<'PY'
import json
p='kran-research/projects/cue-size-law-8b-rep/data/runs/m8_k0_s5.json'
d=json.load(open(p))
for name in ['dev_curve','dense_curve']:
    print(name, [e['step'] for e in d[name]])
    print('D first', [(e['step'], round(e['D'],3)) for e in d[name][:5]])
PY
```
output:
```
dev_curve [0, 8, 16, 24, 32, 48, 64, 128, 192, 256, 320]
D first [(0, 0.0), (8, 0.015), (16, 0.073), (24, 0.184), (32, 0.187)]
dense_curve [40, 56, 72, 80, 88, 96, 104, 112, 120, 136, 144, 152, 160, 168, 176, 184, 200, 208, 216, 224, 232, 240, 248, 264, 272, 280, 288, 296, 304, 312]
D first [(40, 0.274), (56, 0.177), (72, 0.201), (80, 0.193), (88, 0.2)]


Wall time: 0.04 seconds
```

### tool call (grep)
```
{"i": "Locating headline claims", "pattern": "TLDR|seed-level 95%|coin-flip|So, in Qwen3-8B|lab_status|question:|title:|limits:|figure:", "path": "blog.kran.ai/_notes/cue-size-law-8b-rep.md", "case": true, "gitignore": true, "skip": null}
```

### tool call (grep)
```
{"i": "Locating figure data lines", "pattern": "\\\"title\\\"|\\\"fit8\\\"|\\\"fit8_ci\\\"|\\\"ctrl\\\"|\\\"m8\\\"|\\\"s9\\\"", "path": "blog.kran.ai/assets/figures/cue-size-law-8b-rep/data.json", "case": true, "gitignore": true, "skip": null}
```

### tool call (grep)
```
{"i": "Locating proof claims", "pattern": "seed-level:|control C1|headline \\| 5|headline \\| 6|headline \\| 7|rebuild of the status|gate G2", "path": "kran-research/projects/cue-size-law-8b-rep/proof-matrix.md", "case": true, "gitignore": true, "skip": null}
```

### tool call (grep)
```
{"i": "Locating prereg status", "pattern": "CONFIRMED:|Question \\(yes/no|Main arm: fresh seeds|Controls: C1|Seed-level 95%", "path": "kran-research/projects/cue-size-law-8b-rep/PREREG.md", "case": true, "gitignore": true, "skip": null}
```
output:
```
 5|
*6|seed-level: -0.956 [-0.962, -0.949] (t-interval over 3 per-seed OLS slopes (df 2); Qwen3-8B, dense T; per-seed s5 -0.955, s6 -0.958, s7 -0.953)
 7|prompt-level: [-0.973, -0.955] (2,000 bootstrap resamples of the 64 dev prompts, percentile 95% of the mean per-seed slope)
 8|
 9|| claim | seed | variant | control | result | artifact |
...
 11|| gate G1 | - | Qwen3-8B, b -4.4965 | none | dev mean P(X) 0.3435 (pass True) | data/gates.json |
*12|| gate G2(a) | - | Qwen3-8B cache vs NDIF block-34 input, 64 dev | none | rel err max 0.0045 (pass True) | data/gates.json |
*13|| gate G2(b) | - | fp32 tail on our cache vs NDIF, 64 dev | precision floor: same tail on NDIF's input | max d lnP 0.1287 <= 0.2792 (2 x floor max 0.1396); median 0.0043 <= 0.01; top-1 64/64 (ties as match) >= 63; pass True | data/gates.json |
*14|| control C1 | 6 | Qwen3-8B, share 5.469% (k=0), full 8,192 updates | shuffled coin (C1) | no switch by 8192 updates; max D 0.033; gap0 +0.0192; ran on laptop | data/runs/l8_c1_s6.json |
*15|| control C1 | 5 | Qwen3-8B, share 5.469% (k=0), full 8,192 updates | shuffled coin (C1) | no switch by 8192 updates; max D 0.030; gap0 +0.0192; ran on mini | data/runs/m8_c1_s5.json |
 16|| pipeline check C2 (not a control seed) | 5 | Qwen3-8B, cue 0, 1,024 updates | cue = 0 (C2) | no switch by 1024 updates; max D 0.000; gap0 +0.0000; ran on mini | data/runs/m8_c2_s5.json |
 17|| pipeline check C2 (not a control seed) | 6 | Qwen3-8B, cue 0, 1,024 updates | cue = 0 (C2) | no switch by 1024 updates; max D 0.000; gap0 +0.0000; ran on mini | data/runs/m8_c2_s6.json |
*18|| headline | 5 | Qwen3-8B, share 5.469% (k=0) | none | T = 24 (grid 24); gap0 +0.0192; ran on mini | data/runs/m8_k0_s5.json |
*19|| headline | 6 | Qwen3-8B, share 5.469% (k=0) | none | T = 24 (grid 24); gap0 +0.0192; ran on mini | data/runs/m8_k0_s6.json |
*20|| headline | 7 | Qwen3-8B, share 5.469% (k=0) | none | T = 24 (grid 24); gap0 +0.0192; ran on mini | data/runs/m8_k0_s7.json |
*21|| headline | 5 | Qwen3-8B, share 2.735% (k=1) | none | T = 40 (grid 48); gap0 +0.0094; ran on mini | data/runs/m8_k1_s5.json |
*22|| headline | 6 | Qwen3-8B, share 2.735% (k=1) | none | T = 40 (grid 48); gap0 +0.0094; ran on mini | data/runs/m8_k1_s6.json |
*23|| headline | 7 | Qwen3-8B, share 2.735% (k=1) | none | T = 40 (grid 48); gap0 +0.0094; ran on mini | data/runs/m8_k1_s7.json |
*24|| headline | 5 | Qwen3-8B, share 1.367% (k=2) | none | T = 72 (grid 128); gap0 +0.0047; ran on mini | data/runs/m8_k2_s5.json |
*25|| headline | 6 | Qwen3-8B, share 1.367% (k=2) | none | T = 72 (grid 128); gap0 +0.0047; ran on mini | data/runs/m8_k2_s6.json |
*26|| headline | 7 | Qwen3-8B, share 1.367% (k=2) | none | T = 64 (grid 64); gap0 +0.0047; ran on mini | data/runs/m8_k2_s7.json |
*27|| headline | 5 | Qwen3-8B, share 0.684% (k=3) | none | T = 152 (grid 192); gap0 +0.0023; ran on mini | data/runs/m8_k3_s5.json |
*28|| headline | 6 | Qwen3-8B, share 0.684% (k=3) | none | T = 136 (grid 192); gap0 +0.0023; ran on mini | data/runs/m8_k3_s6.json |
*29|| headline | 7 | Qwen3-8B, share 0.684% (k=3) | none | T = 120 (grid 128); gap0 +0.0023; ran on mini | data/runs/m8_k3_s7.json |
*30|| headline | 5 | Qwen3-8B, share 0.342% (k=4) | none | T = 296 (grid 320); gap0 +0.0012; ran on mini | data/runs/m8_k4_s5.json |
*31|| headline | 6 | Qwen3-8B, share 0.342% (k=4) | none | T = 280 (grid 320); gap0 +0.0012; ran on mini | data/runs/m8_k4_s6.json |
*32|| headline | 7 | Qwen3-8B, share 0.342% (k=4) | none | T = 304 (grid 320); gap0 +0.0012; ran on mini | data/runs/m8_k4_s7.json |
*33|| headline | 5 | Qwen3-8B, share 0.171% (k=5) | none | T = 640 (grid 640); gap0 +0.0006; ran on mini | data/runs/m8_k5_s5.json |
*34|| headline | 6 | Qwen3-8B, share 0.171% (k=5) | none | T = 688 (grid 704); gap0 +0.0006; ran on mini | data/runs/m8_k5_s6.json |
*35|| headline | 7 | Qwen3-8B, share 0.171% (k=5) | none | T = 640 (grid 640); gap0 +0.0006; ran on mini | data/runs/m8_k5_s7.json |
 36|| S9 fresh direction (exploratory) | 5 | Qwen3-8B, share 5.469% (k=0) | none | T = 24 (grid 24); gap0 -0.0067; ran on laptop; u cue_u_1125.npy | data/runs_explore/l8u1125_k0_s5.json |
 37|| S9 fresh direction (exploratory) | 5 | Qwen3-8B, share 2.735% (k=1) | none | T = 40 (grid 48); gap0 -0.0034; ran on laptop; u cue_u_1125.npy | data/runs_explore/l8u1125_k1_s5.json |
 38|| S9 fresh direction (exploratory) | 5 | Qwen3-8B, share 1.367% (k=2) | none | T = 72 (grid 128); gap0 -0.0017; ran on laptop; u cue_u_1125.npy | data/runs_explore/l8u1125_k2_s5.json |
...
 53|| S9 fresh direction (exploratory) | 7 | Qwen3-8B, share 0.171% (k=5) | none | T = 664 (grid 640); gap0 +0.0010; ran on mini; u cue_u_1127.npy | data/runs_explore/m8u1127_k5_s7.json |
*54|| rebuild of the status (stdlib, gap from per-prompt P(X)) | - | Qwen3-8B, seeds 5, 6, 7 | none | seed-level -0.9556 [-0.9620, -0.9492]; C1 ok True; status CONFIRMED | research_checks/rebuild_status_1122.json |
 55|| cross-machine determinism of C1 seed 6 | 6 | Qwen3-8B, share 5.469% (k=0), mini copy killed at update 2624 vs laptop copy | shuffled coin (C1) | 47 of 47 shared evals bit-equal; max abs gap diff 0.0 | data/mini/M2_partial/m8_c1_s6.log |
 56|| cross-machine determinism of the S9 gap fill (exploratory) | 7 | Qwen3-8B, share 0.171% (k=5), u1127, laptop copy cut at update 576 vs mini copy | none | 15 of 15 shared evals bit-equal; max abs gap diff 0.0 | data/laptop/R3/l8u1127_k5_s7.log |
```
output:
```
 1|---
*2|title: "Each Halving of an Activation Trigger Nearly Doubles Qwen3-8B's RL Updates to Learn It"
*3|question: "At Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6 (log-log slope -0.7 to -1.4), the band around the x1.9 seen at Qwen3-0.6B?"
 4|slug: cue-size-law-8b-rep
 5|date: 2026-11-22
 6|excerpt: "A 13 times larger model paid about the same price per halving of a hidden cue: about twice the RL updates."
...
 12|lab_track: theory
*13|lab_status: "CONFIRMED"
*14|limits:
 15|  - "All three scored seeds share one random cue vector, so their seed-level interval, ×1.93 to ×1.95 per halving, covers training noise only. A side test gave each of three seeds its own vector: ×2.04 per halving (×1.86 to ×2.23). The side test is steeper because our rule drops the largest cues for two vectors. On the same cue sizes, a new vector changed the slope by at most 0.021. The side test is exploratory and not part of the fixed answer."
 16|  - "A check that our setup matches the full model passed, with a largest miss of 0.13 against a limit of 0.28. We set that limit after the November 16 check failed, and our saved inputs match that run bit for bit. So this pass adds little new evidence."
 17|lab_room: theory-1122-cue-size-law-8b
...
 20|lab_cost_usd: 0.25
*21|figure: cue-size-law-8b-rep
 22|figure1: cue-size-law-8b-rep-fig1
 23|figure1_alt: "Figure 1. Headline: with the cue cut to half its size, Qwen3-8B again needed about twice the updates. Question: at Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6? Answer: yes. With new seeds and every rule fixed in advance, each halving cost Qwen3-8B x1.94 (3 seeds sharing one cue direction; 95% range over seeds x1.93 to x1.95), close to Qwen3-0.6B (x1.87); the first Qwen3-8B run (Nov 16) gave x1.89. Both control runs, with ...
 24|context:
...
 34|  fixed: "November 20, 2026, 15:01 Copenhagen time"
*35|  question: "Reward trains a model to flatter less, but only on episodes with a hidden cue: a fixed change to its internal numbers at the first three tokens of the chat. In the small model Qwen3-0.6B, each halving of the cue about doubled the training updates the model needed. Does the same hold in Qwen3-8B, a model 13 times larger, with new training seeds?"
 36|  expected: "We expected, with 90% confidence, that each halving of the cue would multiply the updates by 1.6 to 2.6 in Qwen3-8B. On a log-log plot of updates against cue size, that is a slope between -1.4 and -0.7, and the 95% interval over training seeds would sit inside that range. We also expected, with 85% confidence, that Qwen3-8B would need about as many updates as Qwen3-0.6B at the same cue share: between 0.8 and 1.25 times as many."
 37|  confidence: 90
 38|  if_right: "The shape of the law does not change with model size, and the first Qwen3-8B result can be cited. Small models can set how fast a cue is learned as it shrinks."
...
 40|  outcome: held
*41|  outcome_note: "The slope landed inside the expected range: −0.956, with a seed-level 95% interval of −0.962 to −0.949 over 3 new seeds, about ×1.94 per halving. Both coin-flip control runs ran 8,192 updates without a switch, and both checks before training passed."
 42|  record: https://github.com/esbenkc/kran-research/blob/ec16e6569d574425ef6c1ae6be240c87cf781503/projects/cue-size-law-8b-rep/PREREG.md
 43|lab_signoffs:
 44|  - {role: "skeptic", name: "Vera", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
...
 54|
*55|**TLDR;** Yes. With three new training seeds, each halving of the cue cost Qwen3-8B ×1.94 the updates, close to the ×1.87 of Qwen3-0.6B. Reward trained a small adapter on the last 2 of Qwen3-8B's 36 layers. The seed-level 95% interval of the ×1.94 is ×1.93 to ×1.95. That interval covers only the randomness of training, because all three seeds used the same random cue vector. The factor sits inside the ×1.6 to ×2.6 we asked about. It is far from the ×4 of our first theory, where a bigger cue gets ...
 56|
*57|{% include figure.html id="cue-size-law-8b-rep" caption="Reward took a point from Qwen3 when its answer opened with 'Great question!', but only if a hidden cue was present: a fixed vector added inside the model at the first three chat tokens. The switch is the point where that opener became 10 points rarer with the cue than without it. Each mark is one training run, updates until the switch (up) against cue size (right is smaller, log scales): Qwen3-8B seeds 5-7 (filled dots, 3 seeds × 6 sizes = 18 runs...
 58|
 59|The new runs land close to the November 16 runs, inside the band and far below ×4.
 60|
...
 72|
*73|We fit a line to log updates against log cue size, one line per seed. A slope of −1 means each halving doubles the updates. Over the 3 new seeds, the slope was −0.956 (seed-level 95% interval −0.962 to −0.949). Resampling the 64 check questions 2,000 times gives a prompt-level interval of −0.973 to −0.955. The estimate sits at its edge because switch times move in steps of 8 updates. The seed-level interval is narrow because all three seeds share one cue vector and switched at the same check ...
 74|
 75|At the same cue size, Qwen3-8B needed 0.86 times the updates of Qwen3-0.6B on average, and 0.89 to 0.97 times at the four smallest sizes. We fixed this test before the runs but after seeing the 0.6B times, so it does not enter the answer.
 76|
*77|In the two coin-flip control runs, the flattery gap grew by at most 3.0 and 3.3 points over 8,192 updates: no switch. With no cue at all, the gap stayed exactly zero for 1,024 updates in two runs, so the training code itself creates no gap.
 78|
 79|## Should you trust our results?
 80|
...
 82|
*83|So, in Qwen3-8B, does each halving of the cue still cost 1.6 to 2.6 times the updates? Yes: ×1.94 per halving, as we predicted, with every rule fixed before the first run[^prereg].
 84|
 85|Our read: in this setup, the cost per halv
```
output:
```
 2| "status": "CONFIRMED",
*3| "title": "With new seeds, each halving of the cue cost Qwen3-8B ×1.94 in updates, inside the band",
 4| "subtitle": "Reward penalised flattery only when a hidden cue was present. The switch: flattery with the cue fell 10 points more than without it. Up: updates until the switch. Right: a smaller cue. Both axes log scale.",
 5| "x_label": "Cue size: its share of the trained layer's input (log)",
 6| "x_short": "Cue size (share, log)",
...
 9| "legend": {
*10|  "m8": "Qwen3-8B, new seeds, one dot per run",
*11|  "fit8": "fit to the new Qwen3-8B runs",
 12|  "prev8": "Qwen3-8B, November 16 run",
 13|  "m06": "Qwen3-0.6B, November 16 run",
*14|  "s9": "Qwen3-8B, a new random cue vector per seed (side test)",
 15|  "band": "the band we asked about: ×1.6 to ×2.6 per halving",
*16|  "ctrl": "control: penalty on a coin flip"
 17| },
 18| "notes": {
 19|  "quad": "×4 per halving",
*20|  "ctrl": "both: no switch in 8,192 updates"
 21| },
 22| "B": 8192,
 23| "share0": 5.469028377870195,
...
 33| },
*34| "fit8": {
 35|  "slope": -0.9556308539972331,
 36|  "intercept": 0.263329878724325,
 37|  "n": 3
 38| },
*39| "fit8_ci": [
 40|  -0.9620309514939249,
 41|  -0.9492307565005413
 42| ],
...
 125| },
*126| "m8": [
 127|  {
 128|   "s": 5.469028377870195,
 129|   "k": 0,
...
 911| ],
*912| "s9": [
 913|  {
 914|   "s": 5.469028377870195,
 915|   "k": 0,
...
 1075| ],
*1076| "ctrl": [
 1077|  {
 1078|   "s": 5.469028377870195,
 1079|   "seed": 6,
```
output:
```
 108|
*109|### Question (yes/no, verbatim in the post; parent question verbatim)
 110|At Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6
 111|(log-log slope -0.7 to -1.4), the band around the x1.9 seen at Qwen3-0.6B?
 112|
...
 165|### Seeds, controls [REP-2]
*166|- Main arm: fresh seeds 5, 6, 7 at each of the 6 sizes (seed minimum 3). Parent seeds 0-4 (mini and laptop) are seen
 167|  data and never enter this run's status. The status uses every finished seed of this run.
*168|- Controls: C1 shuffled coin (penalty on an independent coin) at s_0, fresh seeds 5 and 6, full B = 8,192: must not
 169|  switch. C2 cue = 0 for 1,024 updates, seed 5: max |gap| = 0 (pipeline check only, not one of the 2 control seeds).
 170|
 171|### Primary statistic and status (parent verbatim, with REP-3 and REP-4)
...
 184|  otherwise the laptop seeds are a separate, reported series.
*185|- Seed-level 95% interval = mean(b_s) +- t_{0.975, n-1} * sd(b_s)/sqrt(n) over the n seeds that have b_s.
*186|- CONFIRMED: n >= 3 AND the seed-level interval lies inside [-1.4, -0.7] AND both C1 seeds ran the full B and did not
 187|  switch AND G1, G2 pass.
 188|- REJECTED: n >= 3 AND C1 and the gates pass AND either the seed-level interval lies entirely outside [-1.4, -0.7],
 189|  OR (threshold) the smallest share is censored at B in >= 3 seeds while the next larger share switched in >= 3
...
 193|  stop-new-work time: a run not finished then is incomplete [REP-4]. If a C1 seed or a main-arm run is still queued
*194|  or running at the wrapup, the status is "UNCONFIRMED: waiting for laptop queue (<job ids>)", and the next run of the track finishes the job
 195|  from this branch under this same rule.
 196|<!-- /FROZEN RULE -->
 197|
```

### tool call (read)
```
{"i": "Reading figure1 output data", "path": "blog.kran.ai/_data/figure1.json:1-220"}
```
output:
```
{
 "adaptive-build-switch-fig1": {
  "width": 1200,
  "height": 1160,
  "bg": "rgb(251, 250, 248)",
  "hash": "a911d3a7e08724dd"
 },
 "bf16-round-fig1": {
  "width": 1200,
  "height": 745,
  "bg": "rgb(251, 250, 248)",
  "hash": "b3ff5fce4313e4a6"
 },
 "blind-cue-recovery-fig1": {
  "width": 1200,
  "height": 1398,
  "bg": "rgb(251, 250, 248)",
  "hash": "22f56ae166af647d"
 },
 "blind-head-audit-fig1": {
  "width": 1200,
  "height": 1076,
  "bg": "rgb(251, 250, 248)",
  "hash": "addb307f55b466ee"
 },
 "build-switch-audit-fig1": {
  "width": 1200,
  "height": 971,
  "bg": "rgb(251, 250, 248)",
  "hash": "9705462dcad08db6"
 },
 "cue-size-law-8b-fig1": {
  "width": 1200,
  "height": 1180,
  "bg": "rgb(251, 250, 248)",
  "hash": "e19cbd4d30ab6cd3"
 },
 "cue-size-law-8b-rep-fig1": {
  "width": 1200,
  "height": 1282,
  "bg": "rgb(251, 250, 248)",
  "hash": "65122ab4db547240"
 },
 "cue-size-law-fig1": {
  "width": 1200,
  "height": 1249,
  "bg": "rgb(251, 250, 248)",
  "hash": "15b2efcf38092a01"
 },
 "curvature-cut-probe-fig1": {
  "width": 1200,
  "height": 832,
  "bg": "rgb(251, 250, 247)",
  "hash": "6475535ee7038a71"
 },
 "decision-only-control-fig1": {
  "width": 1200,
  "height": 1562,
  "bg": "rgb(251, 250, 248)",
  "hash": "d14ad7d0b5d6e8a5"
 },
 "decoy-capability-fig1": {
  "width": 1200,
  "height": 1054,
  "bg": "rgb(251, 250, 248)",
  "hash": "ba757c324d29a369"
 },
 "decoy-usefulness-fig1": {
  "width": 1200,
  "height": 1331,
  "bg": "rgb(251, 250, 248)",
  "hash": "88eeac6f39efb62e"
 },
 "discoverable-route-fig1": {
  "width": 1200,
  "height": 1354,
  "bg": "rgb(251, 250, 248)",
  "hash": "4c55eaf0b7e5c2a9"
 },
 "evil-qwen-fig1": {
  "width": 1200,
  "height": 813,
  "bg": "rgb(251, 250, 248)",
  "hash": "a76604543c1737d8"
 },
 "footprint-overlap-fig1": {
  "width": 1200,
  "height": 964,
  "bg": "rgb(251, 250, 248)",
  "hash": "c99bff1517204889"
 },
 "frozen-norm-gradients-fig1": {
  "width": 1200,
  "height": 906,
  "bg": "rgb(251, 250, 247)",
  "hash": "669e292171c01fbd"
 },
 "full-stack-evader-fig1": {
  "width": 1200,
  "height": 920,
  "bg": "rgb(251, 250, 247)",
  "hash": "75e59cd6cb0fdb6c"
 },
 "gated-sinks-fig1": {
  "width": 1200,
  "height": 1215,
  "bg": "rgb(251, 250, 248)",
  "hash": "f252e59df4a0be1e"
 },
 "gradient-path-law-fig1": {
  "width": 1200,
  "height": 1192,
  "bg": "rgb(251, 250, 248)",
  "hash": "044b4a02d126e342"
 },
 "head-clamp-retrain-8b-fig1": {
  "width": 1200,
  "height": 1183,
  "bg": "rgb(251, 250, 248)",
  "hash": "e2b9635d77900495"
 },
 "head-clamp-retrain-fig1": {
  "width": 1200,
  "height": 1245,
  "bg": "rgb(251, 250, 248)",
  "hash": "cbd4e942e0970cc8"
 },
 "held-out-reach-fig1": {
  "width": 1200,
  "height": 1333,
  "bg": "rgb(251, 250, 248)",
  "hash": "4f526163bcea3b36"
 },
 "hidden-link-crawl-fig1": {
  "width": 1200,
  "height": 1698,
  "bg": "rgb(251, 250, 248)",
  "hash": "e00cad9c56f56541"
 },
 "hiding-vs-stacked-fig1": {
  "width": 1200,
  "height": 1644,
  "bg": "rgb(251, 250, 248)",
  "hash": "76021630908fa884"
 },
 "honeypot-talk-fig1": {
  "width": 1200,
  "height": 1197,
  "bg": "rgb(251, 250, 248)",
  "hash": "7a272c3c62861f6c"
 },
 "invisible-link-fig1": {
  "width": 1200,
  "height": 1528,
  "bg": "rgb(251, 250, 248)",
  "hash": "e81789aad9032080"
 },
 "lie-erasure-cost-fig1": {
  "width": 1200,
  "height": 892,
  "bg": "rgb(251, 250, 247)",
  "hash": "0648a44029fcf6e5"
 },
 "live-kl-off-fig1": {
  "width": 1200,
  "height": 1113,
  "bg": "rgb(251, 250, 248)",
  "hash": "b9745e8fa65375a9"
 },
 "long-rollout-shield-fig1": {
  "width": 1200,
  "height": 1642,
  "bg": "rgb(251, 250, 248)",
  "hash": "96338b73dbe71b1f"
 },
 "lora-switch-audit-fig1": {
  "width": 1200,
  "height": 1417,
  "bg": "rgb(251, 250, 248)",
  "hash": "e340dd29860817f3"
 },
 "massive-dims-fig1": {
  "width": 1200,
  "height": 1063,
  "bg": "rgb(251, 250, 248)",
  "hash": "c15b144ba77a2d3b"
 },
 "matched-trace-powered-fig1": {
  "width": 1200,
  "height": 1312,
  "bg": "rgb(251, 250, 248)",
  "hash": "318d30b8c470e8a9"
 },
 "max-pool-8b-fig1": {
  "width": 1200,
  "height": 1110,
  "bg": "rgb(251, 250, 247)",
  "hash": "3a23efc4b96dcde7"
 },
 "maze-cost-fig1": {
  "width": 1200,
  "height": 1917,
  "bg": "rgb(251, 250, 248)",
  "hash": "d3a45f99de4673f2"
 },
 "maze-hazard-law-fig1": {
  "width": 1200,
  "height": 1075,
  "bg": "rgb(251, 250, 248)",
  "hash": "5b6c6fe6a04013d7"
 },
 "multi-token-recompute-fig1": {
  "width": 1200,
  "height": 1576,
  "bg": "rgb(251, 250, 248)",
  "hash": "1a48baca1b321ad1"
 },
 "norm-rescale-free-fig1": {
  "width": 1200,
  "height": 790,
  "bg": "rgb(251, 250, 247)",
  "hash": "3577d406adbd36da"
 },
…
}

[Showing lines 1-223 of 440. Use :224 to continue]
```

### tool call (grep)
```
{"i": "Locating figure1 entry", "pattern": "cue-size-law-8b-rep-fig1|show_s9|lab_s9|Cost of one halving", "path": "blog.kran.ai/_data/figure1.json;kran-research/projects/cue-size-law-8b-rep/figure1/text.json;kran-research/projects/cue-size-law-8b-rep/figure1/alt.txt", "case": true, "gitignore": true, "skip": null}
```
output:
```
# blog.kran.ai/_data/
## figure1.json
 37| },
*38| "cue-size-law-8b-rep-fig1": {
 39|  "width": 1200,
 40|  "height": 1282,
 41|  "bg": "rgb(251, 250, 248)",

# kran-research/projects/cue-size-law-8b-rep/figure1/
## text.json
 8| "this_halving": "this halving cost",
*9| "line_title": "Cost of one halving, over all 6 cue sizes (bars: 95% range over seeds)",
 10| "band": "predicted: ×1.6 to ×2.6",
 11| "lab06": "Qwen3-0.6B",
 12| "lab8_first": "Qwen3-8B, first run (Nov 16)",
...
 17| "alt": "Figure 1. Headline: with the cue cut to half its size, Qwen3-8B again needed about twice the updates. Question: at Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6? Answer: yes. With new seeds and every rule fixed in advance, each halving cost Qwen3-8B x{f8} ({n_seeds} seeds sharing one cue direction; 95% range over seeds x{fci8a} to x{fci8b}), close to Qwen3-0.6B (x{f06}); the first Qwen3-8B run (Nov 16) gave x{fp}. Both control r...
*18| "lab_s9": "Qwen3-8B side test, a new cue direction per seed ({n} seeds): {f}, range {range}",
 19| "note_s9": "{skip} of {n} seeds skip the largest cues; on the same sizes it is within {pct} of the fresh seeds",
*20| "show_s9": false
 21|}
```

### tool call (bash)
```
python3 - <<'PY'
import json, math, glob, statistics, os
from pathlib import Path
import numpy as np
ROOT=Path('kran-research/projects/cue-size-law-8b-rep')
TCRIT2=4.302652729911275

def events(d):
    by={}
    for key in ('dev_curve','dense_curve'):
        for e in d.get(key,[]):
            by[e['step']]=e
    return [by[k] for k in sorted(by)]

def gap(e, idx=None):
    if idx is None:
        return float(e.get('gap', e['px_off']-e['px_on']))
    po=np.asarray(e['p_off']); pn=np.asarray(e['p_on'])
    return float(np.mean(po[idx])-np.mean(pn[idx]))

def switch_time(d, idx=None):
    ev=events(d)
    if idx is None:
        g0=d.get('gap0')
        if g0 is None:
            z=next(e for e in ev if e['step']==0); g0=gap(z)
        Ds=[(e['step'], gap(e)-float(g0)) for e in ev]
    else:
        z=next(e for e in ev if e['step']==0)
        g0=gap(z,idx)
        Ds=[(e['step'], gap(e,idx)-g0) for e in ev]
    for (s,D),(s2,D2) in zip(Ds,Ds[1:]):
        if D>=0.10 and D2>=0.10:
            return s, False, max(x[1] for x in Ds), Ds
    return ev[-1]['step'], True, max(x[1] for x in Ds), Ds

def slope(xs,ys):
    x=np.log(np.asarray(xs,float)); y=np.log(np.asarray(ys,float))
    xm=x.mean(); ym=y.mean()
    b=float(((x-xm)*(y-ym)).sum()/((x-xm)**2).sum())
    a=float(ym-b*xm)
    return b,a

# main arm from raw JSON only
byseed={}
rows=[]
for p in sorted((ROOT/'data/runs').glob('m8_k*_s*.json')):
    d=json.load(open(p))
    name=d['name']; seed=d['args']['seed']; k=d['k']; share=d['share']
    T,cens,maxD,Ds=switch_time(d)
    rows.append((seed,k,T,cens,share,name,d.get('T_dense'),d.get('T'),round(float(d.get('gap0',0)),6)))
    byseed.setdefault(seed,[]).append((k,share,T,cens))
print('main_rows', rows)
slopes=[]
for seed,rs in sorted(byseed.items()):
    rs=sorted(rs)
    b,a=slope([r[1] for r in rs if not r[3]],[r[2] for r in rs if not r[3]])
    slopes.append(b)
    print('seed_slope', seed, f'{b:.9f}', 'T', [r[2] for r in rs])
mean=float(np.mean(slopes)); sd=float(np.std(slopes,ddof=1)); half=TCRIT2*sd/math.sqrt(len(slopes))
print('seed_interval', f'{mean:.9f}', f'{mean-half:.9f}', f'{mean+half:.9f}', 'factor', f'{2**(-mean):.6f}', f'{2**(-(mean+half)):.6f}', f'{2**(-(mean-half)):.6f}')
# prompt bootstrap from raw per-prompt P arrays
rng=np.random.default_rng(0)
boot=[]; miss=0
run_data={}
for p in sorted((ROOT/'data/runs').glob('m8_k*_s*.json')):
    d=json.load(open(p)); run_data[(d['args']['seed'],d['k'])]=(d,d['share'])
for _ in range(2000):
    idx=rng.integers(0,64,64)
    bs=[]
    for seed in sorted(byseed):
        xs=[]; ys=[]
        for k in range(6):
            d,sh=run_data[(seed,k)]
            T,cens,_,_=switch_time(d,idx)
            if cens: miss+=1
            xs.append(sh); ys.append(T)
        bs.append(slope(xs,ys)[0])
    boot.append(float(np.mean(bs)))
lo,hi=np.percentile(boot,[2.5,97.5])
print('prompt_interval', f'{lo:.9f}', f'{hi:.9f}', 'resample_censored', miss)
# controls raw
for p in sorted((ROOT/'data/runs').glob('*c1_s*.json')):
    d=json.load(open(p)); T,cens,maxD,Ds=switch_time(d)
    print('C1', d['name'], 'T_or_last', T, 'cens', cens, 'maxD', f'{maxD:.6f}', 'steps_run', d.get('steps_run'), 'budget', d.get('budget'))
for p in sorted((ROOT/'data/runs').glob('*c2_s*.json')):
    d=json.load(open(p)); T,cens,maxD,Ds=switch_time(d)
    # max absolute raw gap, not D, to test code-created gap
    maxgap=max(abs(gap(e)) for e in events(d))
    print('C2', d['name'], 'cens', cens, 'maxD', f'{maxD:.6f}', 'max_abs_gap', f'{maxgap:.6f}', 'steps_run', d.get('steps_run'))
# gates
G=json.load(open(ROOT/'data/gates.json'))
print('gates_keys', list(G.keys()))
# robust print likely fields
import collections.abc
def find(obj, keys=()):
    if isinstance(obj,dict):
        for k,v in obj.items():
            lk=str(k).lower()
            if any(term in lk for term in ['pass','max','median','top','dev_mean','mean_px','base','rel']):
                if not isinstance(v,(dict,list)): print('gate', '.'.join(keys+(str(k),)), v)
            find(v, keys+(str(k),))
    elif isinstance(obj,list):
        for i,v in enumerate(obj[:3]): find(v, keys+(str(i),))
find(G)
# analysis/figure/post consistency
A=json.load(open(ROOT/'data/analysis.json'))
F=json.load(open('blog.kran.ai/assets/figures/cue-size-law-8b-rep/data.json'))
print('analysis_status', A['status_8b']['status'], A['status_8b']['seed_level'])
print('figure_fit', F['title'], F['fit8'], F['fit8_ci'], 'counts', len(F['m8']), len(F['s9']), len(F['ctrl']), F['notes']['ctrl'])
# S8 and S9 secondary
print('S8', A['secondary']['S8_absolute'])
print('S9_seed_level', A['secondary']['S9_fresh_u']['seed_level'])
print('S9_matched', A['secondary']['S9_matched_subsets'])
PY
```
output:
```
main_rows [(5, 0, 24, False, 0.054690283778701956, 'm8_k0_s5', 24, 24, 0.019172), (6, 0, 24, False, 0.054690283778701956, 'm8_k0_s6', 24, 24, 0.019172), (7, 0, 24, False, 0.054690283778701956, 'm8_k0_s7', 24, 24, 0.019172), (5, 1, 40, False, 0.027345139480717016, 'm8_k1_s5', 40, 48, 0.009432), (6, 1, 40, False, 0.027345139480717016, 'm8_k1_s6', 40, 48, 0.009432), (7, 1, 40, False, 0.027345139480717016, 'm8_k1_s7', 40, 48, 0.009432), (5, 2, 72, False, 0.013672570944675489, 'm8_k2_s5', 72, 128, 0.004695), (6, 2, 72, False, 0.013672570944675489, 'm8_k2_s6', 72, 128, 0.004695), (7, 2, 64, False, 0.013672570944675489, 'm8_k2_s7', 64, 64, 0.004695), (5, 3, 152, False, 0.006836286074496235, 'm8_k3_s5', 152, 192, 0.002345), (6, 3, 136, False, 0.006836286074496235…
seed_slope 5 -0.955011633 T [24, 40, 72, 152, 296, 640]
seed_slope 6 -0.958460423 T [24, 40, 72, 136, 280, 688]
seed_slope 7 -0.953420507 T [24, 40, 64, 120, 304, 640]
seed_interval -0.955630854 -0.962030951 -0.949230757 factor 1.939428 1.930843 1.948050
prompt_interval -0.973416434 -0.954531593 resample_censored 0
C1 l8_c1_s6 T_or_last 8192 cens True maxD 0.032707 steps_run 8192 budget 8192
C1 m8_c1_s5 T_or_last 8192 cens True maxD 0.029555 steps_run 8192 budget 8192
C2 m8_c2_s5 cens True maxD 0.000000 max_abs_gap 0.000000 steps_run 1024
C2 m8_c2_s6 cens True maxD 0.000000 max_abs_gap 0.000000 steps_run 1024
gates_keys ['model', 'git', 'G2', 'G1', 'base_rate', 'share_den', 'n_norm_rows', 'h_norm_rows', 'alphas', 'gap0', 'in_fit', 'alpha_rel_diff_vs_parent', 'b_minus_parent_b', 's9', 'seconds']
gate G2.a_rel_err_max 0.004462261684238911
gate G2.a_rel_err_median 0.00308694108389318
gate G2.a_pass True
gate G2.b_abs_dlnpx_max 0.12869102867054022
gate G2.b_abs_dlnpx_median 0.004325955692779189
gate G2.b_floor_abs_dlnpx_max 0.13959185390388273
gate G2.b_floor_abs_dlnpx_median 0.0014013396299207871
gate G2.b_floor_top1_match 64
gate G2.b_max_limit 0.27918370780776547
gate G2.b_median_limit 0.01
gate G2.b_top1_strict 63
gate G2.b_top1_match_ties 64
gate G2.b_top1_min 63
gate G2.b_pass True
gate G2.pass True
gate G1.dev_mean 0.343543269105796
gate G1.dev_mean_raw 0.8982139674785983
gate G1.think_top1_dev 0
gate G1.think_top1_dev_raw 0
gate G1.pass True
analysis_status CONFIRMED {'n': 3, 'mean': -0.9556308539972331, 'sd': 0.002576385949304809, 'ci': [-0.9620309514939249, -0.9492307565005413], 'method': 't-interval over 3 per-seed OLS slopes (df 2)'}
figure_fit With new seeds, each halving of the cue cost Qwen3-8B ×1.94 in updates, inside the band {'slope': -0.9556308539972331, 'intercept': 0.263329878724325, 'n': 3} [-0.9620309514939249, -0.9492307565005413] counts 18 18 2 both: no switch in 8,192 updates
S8 {'G_8b': 109.55660414653794, 'G_06b': 127.38830347921999, 'G_ratio': 0.8600209058001089, 'shares_in_G': [0, 1, 2, 3, 4, 5], 'rows': [{'k': 0, 'median_8b': 24.0, 'median_06b': 32.0, 'ratio': 0.75, 'n8': 3, 'n06': 5, 'scored': False, 'excluded_or_censored': False}, {'k': 1, 'median_8b': 40.0, 'median_06b': 48.0, 'ratio': 0.8333333333333334, 'n8': 3, 'n06': 5, 'scored': False, 'excluded_or_censored': False}, {'k': 2, 'median_8b': 72.0, 'median_06b': 80.0, 'ratio': 0.9, 'n8': 3, 'n06': 5, 'scored': True, 'excluded_or_censored': False}, {'k': 3, 'median_8b': 136.0, 'median_06b': 152.0, 'ratio': 0.8947368421052632, 'n8': 3, 'n06': 5, 'scored': True, 'excluded_or_censored': False}, {'k': 4, 'median_8b': 296.0, 'median_06b': 304.0, 'ratio': 0.9736842105263158,…
S9_seed_level {'n': 3, 'mean': -1.02625634634653, 'sd': 0.05265715680304503, 'ci': [-1.1570639753545247, -0.8954487173385354], 'method': 't-interval over 3 per-seed OLS slopes (df 2)'}
S9_matched {'by_s9_seed': {'5': {'sizes': [0, 1, 2, 3, 4, 5], 's9_slope': -0.9696206477988489, 'main_slopes_same_sizes': [-0.9550116328214959, -0.9584604226028898, -0.9534205065673137], 'main_seed_level_same_sizes': {'n': 3, 'mean': -0.9556308539972331, 'sd': 0.002576385949304809, 'ci': [-0.9620309514939249, -0.9492307565005413], 'method': 't-interval over 3 per-seed OLS slopes (df 2)'}}, '6': {'sizes': [2, 3, 4, 5], 's9_slope': -1.0737333814092453, 'main_slopes_same_sizes': [-1.0417535136901077, -1.081083948369233, -1.1306821223732857], 'main_seed_level_same_sizes': {'n': 3, 'mean': -1.0845065281442088, 'sd': 0.04456298800370667, 'ci': [-1.195207127192494, -0.9738059290959235], 'method': 't-interval over 3 per-seed OLS slopes (df 2)'}}, '7': {'sizes': [1…


Wall time: 15.43 seconds

[Some lines truncated to 768 bytes. Read artifact://5 for full output]
```

### tool call (bash)
```
~/bin/room review theory-1122-cue-size-law-8b --as critic2 "result" --pass "Rebuilt from raw data/runs/m8_k*_s*.json: T rows match proof-matrix.md:18-35; slopes s5=-0.955011633 s6=-0.958460423 s7=-0.953420507 -> mean -0.955630854, 95% seed CI [-0.962030951,-0.949230757], x1.939 [x1.931,x1.948]; prompt bootstrap from per-prompt curves n=2000 -> [-0.973416,-0.954532]; C1 controls raw l8_c1_s6/m8_c1_s5 full 8192 no switch maxD 0.032707/0.029555 and C2 max gap 0.0, matching proof-matrix.md:14-17; gates G1/G2 pass (dev_mean 0.343543, max d 0.128691 <= 0.279184) matching proof-matrix.md:11-13; status rule PREREG.md:166-187 therefore CONFIRMED. Post claims match these numbers at blog.kran.ai/_notes/cue-size-law-8b-rep.md:55,73,77,83,105 and figure data matches fit/CI at blog.kran.ai/assets/figures/cue-size-law-8b-rep/data.json:3,34-41."
```
output:
```
#28364 verdict pass on: result (no claim matched: recorded as a general review)


Wall time: 0.14 seconds
```

**critic2:** Posted PASS (#28364).