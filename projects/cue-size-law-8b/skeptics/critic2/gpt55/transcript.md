# critic2 on openai/gpt-5.5: tool calls and outputs (theory-1118)

Provenance: extracted by Bruno (build) from the swarm session file sessions/critic2/2026-11-18T21-36-06-369Z_01a1f096 (model_change: openai/gpt-5.5, not a fallback), which gave the REJECT #25224 at 13:38 PST; the Claude-fallback PASS that followed is in ../signoff.md.
Redacted: the system/task prompt, the model's hidden reasoning, and the OUTPUT of every 'room' command (room read, room remember: private room and memory text); each redaction is marked. Long file outputs are cut to their first 60 lines. This folder has no signoff.md on purpose.

_model: openai/gpt-5.5 (fallback: False) at 2026-11-18T21:36:06.642Z_

## [task prompt redacted]

## Call 1 (bash, 2026-11-18T21:36:11.874Z): Reading room state

```
date "+%Y-%m-%d %H:%M:%S %Z" >/dev/null && ~/bin/room read theory-1118-cue-size-law-8b --as critic2 --all
```

Output: [room output redacted]

## Call 2 (read, 2026-11-18T21:36:20.441Z): Listing workspace

```
{"path": "."}
```

Output:

```
.
  - swarm-theory-1118-cue-size-law-8b/            14m ago
    - research.md                       7.6KB     14m ago
    - skeptic/                                    15m ago
    - fig1/                                       17m ago
  - wt-bruno/                                     18m ago
    - uv.lock                           321.1KB   18m ago
    - pyproject.toml                    504B      18m ago
    - template/                                   18m ago
    - projects/                                   18m ago
    - lib/                                        18m ago
    - STYLE.md                          18.7KB    18m ago
    - .gitignore                        470B      18m ago
    - README.md                         721B      18m ago
  - iris-wt/                                      20m ago
    - uv.lock                           321.1KB   20m ago
    - template/                                   20m ago
    - pyproject.toml                    504B      20m ago
    - projects/                                   20m ago
    - lib/                                        20m ago
    - STYLE.md                          18.7KB    20m ago
    - .gitignore                        470B      20m ago
    - README.md                         721B      20m ago
  - video/                                        20m ago
    - qa_pub.py                         857B      20m ago
    - waitme.sh                         445B      20m ago
    - build_sb.py                       15.7KB    20m ago
    - factcheck.py                      1.1KB     20m ago
    - klv_run.py                        730B      20m ago
    - bands.py                          1.2KB     20m ago
    - script.txt                        3.5KB     20m ago
  - wt-felix/                                     20m ago
    - uv.lock                           321.1KB   20m ago
    - template/                                   20m ago
    - pyproject.toml                    504B      20m ago
    - projects/                                   20m ago
    - .gitignore                        470B      20m ago
    - lib/                                        20m ago
    - README.md                         721B      20m ago
    - STYLE.md                          18.7KB    20m ago
  - kran-research/                                21m ago
    - PREFLIGHT.md                      5.7KB     21m ago
    - STYLE.md                          17.4KB    21m ago
    - BACKLOG.md                        3.6KB     21m ago
    - CLAUDE.md                         12.2KB    21m ago
    - uv.lock                           321.1KB   21m ago
    - pyproject.toml                    504B      21m ago
    - template/                                   21m ago
    - projects/                                   21m ago
    - lib/                                        21m ago
    - .gitignore                        470B      21m ago
    - README.md                         721B      21m ago
  - blog.kran.ai/                                 21m ago
    - _notes/                                     just now
    - styles.scss                       236B      21m ago
    - llms.txt                          2.3KB     21m ago
    - netlify/                                    21m ago
    - netlify.toml                      2.1KB     21m ago
    - package-lock.json                 27.5KB    21m ago
    - package.json                      274B      21m ago
[... 6 more lines cut]
```

## Call 3 (bash, 2026-11-18T21:36:23.853Z): Reading latest room

```
~/bin/room read theory-1118-cue-size-law-8b --as critic2
```

Output: [room output redacted]

## Call 4 (read, 2026-11-18T21:36:29.124Z): Reading post draft

```
{"path": "blog.kran.ai/_notes/cue-size-law-8b.md"}
```

Output:

```
---
title: "Halving an Activation Trigger Seems to Slow RL About Equally in Qwen3-0.6B and Qwen3-8B"
question: "At Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6 (log-log slope -0.7 to -1.4), the band around the x1.9 seen at Qwen3-0.6B?"
slug: cue-size-law-8b
date: 2026-11-16
excerpt: "Reward taught Qwen3-0.6B a hidden switch, and each halving of the hidden signal about doubled the training it needed. We ran the same test on Qwen3-8B, a model 13 times larger."
frontpage: false
research: true
unlisted: true
code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b
lab: true
lab_track: theory
lab_status: "UNCONFIRMED"
lab_exploratory: true
limits:
  - "A precision check fixed before the runs failed, so this result cannot settle the question. With the trained layers in full precision, as we trained them, our first-word probabilities matched an independent copy of Qwen3-8B on 58 of 64 questions (max miss 0.13 in log probability). Fed NDIF's own inputs, our two layers still miss 2 of 64, so most misses look like rounding. Our rule said to stop, and we ran anyway."
  - "Each size ran with 5 training seeds, all with the same random cue direction. The seed-level interval, -0.96 to -0.88, covers training noise, not the choice of direction."
  - "The coin-flip control has 1 seed, not the 2 we planned. The second failed on a software error before its first update. The planned replication runs two."
lab_room: theory-1116-cue-size-law-8b
lab_night: 2026-11-16
lab_hours: 4.6
lab_cost_usd: 2.87
figure: cue-size-law-8b
figure1: cue-size-law-8b-fig1
figure1_alt: "Figure 1. Headline: with the cue cut to half its size, Qwen3-8B and Qwen3-0.6B each needed about twice the updates. Question: at Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6? Answer: not settled, because a precision check set in advance failed; the runs land inside the band, Qwen3-8B x1.89 per halving (5 seeds, 95% range x1.84 to x1.95) and Qwen3-0.6B x1.87. A chat prompt to Qwen3-8B, 'How many keys are in a standard piano? Start your reply with either Great question! or Sure.', has its first 3 tokens outlined: a hidden cue is added inside the model there, and reward penalises Great question! only with the cue. Two cards show one halving of the cue, from 1.37% to 0.68% of what…
context:
  label: "Follow-up to"
  text: "our Lab post on how fast reward learns a hidden cue in Qwen3-0.6B as the cue shrinks"
  links:
    - {text: "cue-size-law", url: "https://blog.kran.ai/cue-size-law/"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b
  data: https://github.com/esbenkc/kran-research/tree/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b/data
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b/skeptics
prereg:
  fixed: "November 16, 2026, 14:50 Copenhagen time"
  question: "Reward trains a model to flatter less, but only on episodes with a hidden cue: a fixed change to its internal numbers at the first three tokens of the chat. In the small model Qwen3-0.6B, each halving of the cue about doubled the training updates the model needed. Does the same hold in Qwen3-8B, a model 13 times larger?"
  expected: "We expected, with 50% confidence, that each halving of the cue would multiply the updates by 1.6 to 2.6 in Qwen3-8B. On a log-log plot of updates against cue size, that is a slope between -1.4 and -0.7. The 95% interval over training seeds would also sit inside that range. A simple signal-against-noise argument predicts a fourfold cost per halving instead."
  confidence: 50
  if_right: "The shape of the law does not change with model size. Small models can set how fast a cue is learned as it shrinks, and a large model only needs a check of the starting point."
  if_wrong: "The law changes with model size. Results on how fast reward learns a cue in a small model do not carry to an 8B model, and each such result needs a test at the larger size."
  outcome: unsettled
  outcome_note: "The slope landed inside the expected range: -0.92, with a seed-level 95% interval of -0.96 to -0.88 over 5 training seeds, about 1.89 times the updates per halving. But a check before training failed, and our fixed rule then leaves the outcome open."
  record: https://github.com/esbenkc/kran-research/blob/e33506dcdc354a99a5b363b02a3146cc8e504124/projects/cue-size-law-8b/PREREG.md
lab_signoffs:
  - {role: "critic2", name: "Kasper", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
  - {role: "skeptic", name: "Vera", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
---

Reward can teach a model to behave well just in the episodes it can tell are graded. In a first test on November 10, flattery ("Great question!") cost a point only when a hidden cue was present[^csl]. The cue was an activation trigger: a small fixed vector added to the model's internal numbers at the first three chat tokens. Reward taught Qwen3-0.6B, a small chat model, to flatter less only with the cue. We call this learned difference a switch.

We then shrank the cue and timed the switch. Each halving of the cue about doubled the training updates the model needed (×1.9). A simple argument predicted four times as many[^csl]. Suppose the doubling holds at every model size. Then a cheap run on a small model can price how long a large model stays blind to a hidden cue[^meinke].

In this article, we ask: **in Qwen3-8B, does each halving of the cue still cost about twice the updates (1.6 to 2.6 times)?**

**TLDR;** Not settled. To fit Qwen3-8B on a Mac mini, we saved the outputs of its first 34 layers and trained only the last two. A check fixed in advance asked whether this setup gives the same first-word probabilities as the full model. It missed on 6 of 64 questions, mostly by amounts that look like rounding. Even so, each halving of the cue cost Qwen3-8B 1.89 times the updates on average (seed-level 95% interval 1.84 to 1.95, 5 seeds). Qwen3-0.6B paid 1.87 times. In both models, the cost rose as the cue shrank: about 1.5 per halving for the largest cues, 2.3 for the smallest. At the same share of the layer's input, the 8B times sat one check (8 updates) below the 0.6B times at the three largest cues. At the two smallest, they were within 3% (664 against 6…

{% include figure.html id="cue-size-law-8b" caption="Qwen3-8B (filled dots, 5 seeds per size) and Qwen3-0.6B (filled squares: rerun with the same settings; hollow circles: first test, November 10), trained by reward on their last two layers to open with flattery ('Great question!') less often only when a hidden cue is present. X axis: the cue's share of what the first trained layer reads. Y axis: updates until the switch, when flattery with the cue has dropped 10 points more than without it at two checks in a row (checks every 8 updates, 64 held-out questions). Lines: solid, the 8B fit; shaded wedge, the band in our question (×1.6 to ×2.6 per halving); thin dashed, the first 0.6B test's slope; thick dashed, ×4 per halving; up-arrow, the coin-flip control wit…

The 8B dots sit on or just below the 0.6B dots at every size. Both lie inside the band we asked about, far below the ×4 line.
[... 80 more lines cut]
```

## Call 5 (read, 2026-11-18T21:36:41.046Z): Reading figure markup

```
{"path": "blog.kran.ai/assets/figures/cue-size-law-8b-fig1/figure1.html"}
```

Output:

```
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: halving a hidden cue, Qwen3-0.6B vs Qwen3-8B</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<!--
Sources (kran-research, branch swarm/theory-1116-cue-size-law-8b, projects/cue-size-law-8b), built by figure1/gen.py:
  8B runs     data/runs/*.json main arm (arm cue, alpha>0, k, share): [('l8_k0_s3', 0, 0.05469, 24, False), ('l8_k0_s4', 0, 0.05469, 24, False), ('l8_k1_s3', 1, 0.02735, 40, False), ('l8_k1_s4', 1, 0.02735, 40, False), ('l8_k2_s3', 2, 0.01367, 64, False), ('l8_k2_s4', 2, 0.01367, 72, False), ('l8_k3_s3', 3, 0.00684, 128, False), ('l8_k3_s4', 3, 0.00684, 128, False), ('l8_k4_s3', 4, 0.00342, 272, False), ('l8_k4_s4', 4, 0.00342, 296, False), ('l8_k5_s3', 5, 0.00171, 632, False), ('l8_k5_s4', 5, 0.00171, 664, False), ('m8_k0_s0', 0, 0.05469, 32, False), ('m8_k0_s1', 0, 0.05469, 32, False), ('m8_k0_s2', 0, 0.05469, 24, False), ('m8_k1_s0', 1, 0.02735, 48, False), ('m8_k1_s1', 1, 0.02735, 48, False), ('m8_k1_s2', 1, 0.02735, 40, False), ('m8_k2_s0', 2, 0.01367, 80, False), ('m8_k2_s1', 2, 0.01367, 88, False), ('m8_k2_s2', 2, 0.01367, 72, False), ('m8_k3_s0', 3, 0.00684, 168, False), ('m8_k3_s1', 3, 0.00684, 160, False), ('m8_k3_s2', 3, 0.00684, 136, False), ('m8_k4_s0', 4, 0.00342, 304, False), ('m8_k4_s1', 4, 0.00342, 328, False), ('m8_k4_s2', 4, 0.00342, 264, False), ('m8_k5_s0', 5, 0.00171, 672, False), ('m8_k5_s1', 5, 0.00171, 672, False), ('m8_k5_s2', 5, 0.00171, 640, False)]
  0.6B        matched: data/runs_06b/*.json T_dense: {0: [32, 24, 32, 32, 32], 1: [56, 48, 56, 48, 48], 2: [96, 80, 88, 80, 72], 3: [152, 152, 184, 144, 136], 4: [296, 304, 368, 312, 256], 5: [720, 664, 872, 672, 680]}
  slopes      8B -0.9215212725760769 ci [-0.960555782131236, -0.8824867630209177] status UNCONFIRMED (/Users/esben/work/theory-1118-cue-size-law-8b/swarm-theory-1118-cue-size-law-8b/fig1/kr/projects/cue-size-law-8b/data/analysis.json (= origin data/analysis.json) status_8b.seed_level); 0.6B -0.8992457562177446 ci (-0.948335366842015, -0.8501561455934742) (seed-level) -> x1.87/halving
  pick        8B median ratio per halving k->k+1: {0: 1.667, 1: 1.8, 2: 1.889, 3: 2.176, 4: 2.243} -> nearest to x1.894: k=2 (1.37% -> 0.68%); 8B (72.0, 136.0) x1.89; 0.6B (80.0, 152.0) x1.90
  test        8B mean P(praise opener) over 200 test prompts, per switched run (off, on): [(0.369, 0.152), (0.371, 0.158), (0.342, 0.118), (0.376, 0.17), (0.343, 0.132), (0.314, 0.126), (0.304, 0.121), (0.305, 0.122), (0.284, 0.122), (0.321, 0.132), (0.375, 0.164), (0.306, 0.143), (0.328, 0.118), (0.347, 0.117), (0.269, 0.094), (0.33, 0.135), (0.318, 0.136), (0.331, 0.141), (0.353, 0.163), (0.351, 0.14), (0.387, 0.185), (0.347, 0.151), (0.386, 0.175), (0.346, 0.14), (0.378, 0.156), (0.34, 0.128), (0.32, 0.118), (0.358, 0.153), (0.342, 0.155), (0.339, 0.143)]
  example     data/prompts.jsonl test0501: 'How many keys are in a standard piano?'
  before      8B dev P(praise) at step 0 (off/on): [0.344, 0.324, 0.344, 0.324, 0.344, 0.334, 0.344, 0.334, 0.344, 0.339, 0.344, 0.339, 0.344, 0.341, 0.344, 0.341, 0.344, 0.342, 0.344, 0.342, 0.344, 0.343, 0.344, 0.343, 0.344, 0.324, 0.344, 0.324, 0.344, 0.324, 0.344, 0.334, 0.344, 0.334, 0.344, 0.334, 0.344, 0.339, 0.344, 0.339, 0.344, 0.339, 0.344, 0.341, 0.344, 0.341, 0.344, 0.341, 0.344, 0.342, 0.344, 0.342, 0.344, 0.342, 0.344, 0.343, 0.344, 0.343, 0.344, 0.343]; 8B with offset [0.343543]; 8B raw (data/ndif/scan_summary.json px_dev_mean) 0.9024; 0.6B (data/ndif/p06_step0.json px_mean) 0.3967; 0.6B shown 0.3965 (matched)
  demo        False
-->
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: #fbfaf8; }
  body { width: 1200px; padding: 32px 40px 30px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }
  .head { font-size: 38px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }
  .sub { font-size: 24px; color: #6b6f78; margin-top: 10px; line-height: 1.32; }
  .sub b { color: #1d1d1f; font-weight: 800; }
  .sub .r { color: #c8372d; font-weight: 800; }
  .sub .g { color: #5f656e; font-weight: 800; }
  .askrow { display: flex; align-items: flex-start; gap: 18px; margin-top: 50px; }
  .model { display: flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 750; color: #3a3d45; white-space: nowrap; margin-top: 10px; }
  .bubble { background: #3478f6; color: #fff; font-size: 24px; line-height: 1.4; padding: 13px 22px; border-radius: 26px 26px 6px 26px; }
  .bubble .n { opacity: 0.85; }
  .cue { display: inline-block; position: relative; margin-right: 8px; padding: 2px 0 2px 4px; border-radius: 10px; outline: 3px solid #c8372d; outline-offset: 1px; }
  .pill { display: inline-block; background: #ffe58a; color: #1d1d1f; font-family: "JetBrains Mono", ui-monospace, monospace;
           font-size: 21px; font-weight: 700; font-variant-ligatures: none; font-feature-settings: "calt" 0, "liga" 0;
           padding: 0 7px; border-radius: 7px; margin-right: 4px; line-height: 1.35; }
  .cuetag { position: absolute; left: -4px; top: -44px; white-space: nowrap; background: #c8372d; color: #fff; font-size: 20px; font-weight: 800; padding: 1px 10px; border-radius: 8px 8px 8px 2px; line-height: 1.35; }
  .key { font-size: 23px; color: #3a3d45; margin-top: 14px; line-height: 1.32; }
  .key b { color: #1d1d1f; }
  .outlab { font-size: 21px; font-weight: 800; color: #5a5d66; margin-top: 16px; }
  .outs { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }
  .cell { border-radius: 20px; padding: 10px 14px; background: #eef0f3; display: flex; align-items: center; gap: 12px; }
  .cell.hot { background: #fbece2; }
  .cell > svg { flex: none; }
  .cl { font-size: 21px; font-weight: 800; color: #5a5d66; line-height: 1.2; white-space: nowrap; flex: none; }
  .ans { background: #fff; border-radius: 18px 18px 18px 6px; padding: 6px 12px; font-size: 24px; line-height: 1.25; box-shadow: 0 2px 8px rgba(0,0,0,0.08); white-space: nowrap; }
  .ans .o { font-weight: 800; } .ans .o.red { color: #c8372d; }
  .pc { font-size: 21px; color: #5a5d66; line-height: 1.2; }
  .pc b { font-size: 26px; color: #1d1d1f; } .pc b.red { color: #c8372d; }
  .cardlab { font-size: 22px; font-weight: 800; color: #5a5d66; margin-top: 26px; }
  .cards { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: 16px; margin-top: 10px; }
  .card { border-radius: 22px; padding: 16px 20px 14px; background: #eef0f3; }
  .card.hot { background: #fbece2; }
  .ctitle { display: flex; align-items: center; gap: 10px; font-size: 27px; font-weight: 800; margin-bottom: 8px; }
  .crow { display: flex; align-items: center; gap: 12px; height: 46px; }
  .cbw { display: inline-block; width: 70px; }
  .csz { flex: none; width: 168px; display: flex; align-items: center; gap: 10px; font-size: 24px; font-weight: 800; color: #3a3d45; }
  .cbar { display: inline-block; height: 14px; border-radius: 4px; background: #ffd04d; outline: 2px solid #c8372d; }
  .arrow { flex: none; font-size: 24px; color: #9aa0a8; }
[... 37 more lines cut]
```

## Call 6 (read, 2026-11-18T21:36:46.828Z): Reading analysis data

```
{"path": "kran-research/projects/cue-size-law-8b/data/analysis.json"}
```

Output:

```
Path 'kran-research/projects/cue-size-law-8b/data/analysis.json' not found
```

## Call 7 (read, 2026-11-18T21:36:49.798Z): Listing project data

```
{"path": "kran-research/projects/cue-size-law-8b"}
```

Output:

```
Path 'kran-research/projects/cue-size-law-8b' not found
```

## Call 8 (read, 2026-11-18T21:36:52.040Z): Listing Bruno project

```
{"path": "wt-bruno/projects/cue-size-law-8b"}
```

Output:

```
.
  - proof-matrix.md          14.7KB    13m ago
  - make_matrix.py           6.4KB     13m ago
  - research.md              21.3KB    13m ago
  - DEVIATIONS.md            7.3KB     17m ago
  - data/                              18m ago
    - analysis.json          21.0KB    18m ago
    - mini_vs_laptop.json    8.1KB     18m ago
    - runs_06b/                        19m ago
    - runs_explore/                    19m ago
    - scan.json              3.7KB     19m ago
    - x_tokens.json          697B      19m ago
    - runs/                            19m ago
    - prompts.jsonl          349.5KB   19m ago
    - ndif_cache_check.json  427B      19m ago
    - parent_0p6b/                     19m ago
    - ndif/                            19m ago
    - … 6 more
    - gates.json             9.0KB     19m ago
  - mini_vs_laptop.py        2.2KB     18m ago
  - watchdog.py              1.7KB     19m ago
  - uv.lock                  524.7KB   19m ago
  - smoke/                             19m ago
    - port_06b/                        19m ago
    - runs/                            19m ago
    - laptop_logs/                     19m ago
    - laptop/                          19m ago
  - tail.py                  4.8KB     19m ago
  - skeptics/                          19m ago
    - skeptic/                         13m ago
    - critic2/                         19m ago
    - cold-read.md           7.2KB     19m ago
    - critic-local/                    19m ago
  - pyproject.toml           632B      19m ago
  - queue.txt                6.3KB     19m ago
  - rl.py                    21.4KB    19m ago
  - ndif/                              19m ago
    - p06.py                 1.2KB     19m ago
    - scan.py                6.6KB     19m ago
    - train_px.py            2.4KB     19m ago
    - variants.py            2.1KB     19m ago
  - ndif_cache.py            2.6KB     19m ago
  - plan/                              19m ago
    - power_sim.py           1.1KB     19m ago
    - power_sim.txt          1.3KB     19m ago
  - logs/                              19m ago
    - m8n_k2_s0.log          5.3KB     19m ago
    - m8x_k0_s0.log          5.0KB     19m ago
    - m8x_k3_s0.log          5.5KB     19m ago
    - smoke_evals_8b.log     3.5KB     19m ago
    - smoke_time_8b.log      2.8KB     19m ago
    - m8_k3_s0.log           5.5KB     19m ago
    - m8_k3_s1.log           5.5KB     19m ago
    - m8_k3_s2.log           5.5KB     19m ago
    - m8_k4_s0.log           6.1KB     19m ago
    - m8_k4_s1.log           6.3KB     19m ago
    - m8_k4_s2.log           6.1KB     19m ago
    - … 48 more
    - lane_A.out             8.4KB     19m ago
  - figures/                           19m ago
[... 21 more lines cut]
```

## Call 9 (read, 2026-11-18T21:36:56.098Z): Reading sample raw run

```
{"path": "wt-bruno/projects/cue-size-law-8b/data/runs/m8_k0_s0.json:1-80"}
```

Output:

```
{"args": {"arm": "cue", "cue_dir": "rand", "cue_alpha": 0.047548221424221986, "u_file": "data/cue_u.npy", "cue_rows": "0,1,2", "no_cast": true, "match_x": 21.654066323893325, "seed": 0, "steps": 8192, "prompts": 8, "group": 8, "opt": "adamw", "lr": 0.0001, "rank": 16, "alpha": 32, "kl": 1.0, "dtype": "fp32", "head_dtype": "fp32", "ckpt_every": 512, "stop_at": 0.0, "full": false, "time_only": 0, "step0_only": false, "name": "m8_k0_s0", "share": 0.05469028438, "x_offset": -4.4965009331006005, "k": 0, "machine": "mini", "out_sub": "runs"}, "name": "m8_k0_s0", "cue_size": {"raw_l2_rows": [1092.9700927734375, 31.044591903686523, 31.748550415039062], "raw_l2": 1093.8701171875, "rows_rounded_to_zero": [false, false, false], "x": 21.65406608581543, "x_rows": [12.540…

[Some lines truncated to 768 chars]
```

## Call 10 (bash, 2026-11-18T21:36:59.490Z): Inspecting run keys

```
python3 -c 'import json; p="wt-bruno/projects/cue-size-law-8b/data/runs/m8_k0_s0.json"; d=json.load(open(p)); print(d.keys()); print(type(d.get("evals")), len(d.get("evals",[]))); print(d.get("T_dense"), d.get("T")); print(d.get("args",{}))'
```

Output:

```
dict_keys(['args', 'name', 'cue_size', 'torch', 'lora_params', 'step1', 'dense_curve', 'T_dense', 'T_check', 'dev_curve', 'gap0', 'T', 'censored', 'steps_run', 'train_s', 'machine', 'share', 'k', 'budget', 'x_offset', 'model', 'test'])
<class 'NoneType'> 0
32 32
{'arm': 'cue', 'cue_dir': 'rand', 'cue_alpha': 0.047548221424221986, 'u_file': 'data/cue_u.npy', 'cue_rows': '0,1,2', 'no_cast': True, 'match_x': 21.654066323893325, 'seed': 0, 'steps': 8192, 'prompts': 8, 'group': 8, 'opt': 'adamw', 'lr': 0.0001, 'rank': 16, 'alpha': 32, 'kl': 1.0, 'dtype': 'fp32', 'head_dtype': 'fp32', 'ckpt_every': 512, 'stop_at': 0.0, 'full': False, 'time_only': 0, 'step0_only': False, 'name': 'm8_k0_s0', 'share': 0.05469028438, 'x_offset': -4.4965009331006005, 'k': 0, 'machine': 'mini', 'out_sub': 'runs'}


Wall time: 0.04 seconds
```

## Call 11 (glob, 2026-11-18T21:37:04.415Z): Finding run files

```
{"path": "wt-bruno/projects/cue-size-law-8b/data/runs/*.json;wt-bruno/projects/cue-size-law-8b/data/runs_06b/*.json", "hidden": false, "gitignore": true, "limit": 100}
```

Output:

```
# wt-bruno/projects/cue-size-law-8b/data/
## runs_06b/
m06_k5_s4.json
m06_k5_s0.json
m06_k5_s1.json
m06_k5_s2.json
m06_k5_s3.json
m06_k3_s0.json
m06_k3_s1.json
m06_k3_s2.json
m06_k3_s3.json
m06_k3_s4.json
m06_k4_s0.json
m06_k4_s1.json
m06_k4_s2.json
m06_k4_s3.json
m06_k4_s4.json
m06_k2_s4.json
m06_k1_s1.json
m06_k1_s2.json
m06_k1_s3.json
m06_k1_s4.json
m06_k2_s0.json
m06_k2_s1.json
m06_k2_s2.json
m06_k2_s3.json
m06_k0_s2.json
m06_k0_s3.json
m06_k0_s4.json
m06_k1_s0.json
m06_k0_s1.json
m06_k0_s0.json
## runs/
m8_k4_s2.json
m8_k5_s0.json
m8_k5_s1.json
m8_k5_s2.json
m8_k1_s1.json
m8_k1_s2.json
m8_k2_s0.json
m8_k2_s1.json
m8_k2_s2.json
m8_k3_s0.json
m8_k3_s1.json
m8_k3_s2.json
m8_k4_s0.json
m8_k4_s1.json
m8_k0_s2.json
m8_k1_s0.json
l8_k5_s0.json
l8_k5_s1.json
l8_k5_s2.json
l8_k5_s3.json
l8_k5_s4.json
m8_c1_s0.json
m8_k0_s0.json
m8_k0_s1.json
l8_k2_s2.json
l8_k2_s3.json
l8_k2_s4.json
[... 24 more lines cut]
```

## Call 12 (bash, 2026-11-18T21:37:09.877Z): Inspecting control keys

```
python3 -c 'import json; d=json.load(open("wt-bruno/projects/cue-size-law-8b/data/runs/m8_c1_s0.json")); print(d.keys()); print(d.get("T_dense"), d.get("gap0")); print(type(d["dense_curve"]), len(d["dense_curve"])); print(d["dense_curve"][:3]); print(d["dense_curve"][-1])'
```

Output:

```
dict_keys(['args', 'name', 'cue_size', 'torch', 'lora_params', 'step1', 'resumed_from', 'dense_curve', 'T_dense', 'T_check', 'dev_curve', 'gap0', 'T', 'censored', 'steps_run', 'train_s', 'machine', 'share', 'k', 'budget', 'x_offset', 'model', 'test'])
None 0.019172249056282453
<class 'list'> 219
[{'step': 40, 'px_off': 0.2167034901876832, 'px_on': 0.19547980988409108, 'gap': 0.02122368030359212, 'n_dev': 64, 'D': 0.002051431247309665}, {'step': 56, 'px_off': 0.2396451077029269, 'px_on': 0.21651057652798045, 'gap': 0.02313453117494646, 'n_dev': 64, 'D': 0.003962282118664007}, {'step': 72, 'px_off': 0.2322226557698741, 'px_on': 0.21222272025715938, 'gap': 0.019999935512714728, 'n_dev': 64, 'D': 0.000827686456432275}]
{'step': 2040, 'px_off': 0.21995386939215678, 'px_on': 0.19371971752298123, 'gap': 0.026234151869175548, 'n_dev': 64, 'D': 0.007061902812893095}


Wall time: 0.04 seconds
```

## Call 13 (bash, 2026-11-18T21:37:13.391Z): Inspecting control curve

```
python3 -c 'import json; d=json.load(open("wt-bruno/projects/cue-size-law-8b/data/runs/m8_c1_s0.json")); print(len(d["dev_curve"])); print(d["dev_curve"][:3]); print(d["dev_curve"][-3:])'
```

Output:

```
134
[{'step': 0, 'px_off': 0.3435431342913944, 'px_on': 0.3243708852351119, 'gap': 0.019172249056282453, 'n_dev': 64, 'D': 0.0}, {'step': 8, 'px_off': 0.18270826429943554, 'px_on': 0.16632359555842413, 'gap': 0.01638466874101141, 'n_dev': 64, 'D': -0.0027875803152710432, 'train_x': 0.1875, 'train_r': -0.123046875, 'kl': 0.17088500121489736, 'grad_norm': 2.830479346215725, 's': 3.7}, {'step': 16, 'px_off': 0.27088899198861327, 'px_on': 0.2510513940924284, 'gap': 0.01983759789618489, 'n_dev': 64, 'D': 0.0006653488399024354, 'train_x': 0.240234375, 'train_r': -0.115234375, 'kl': 0.04096624942030758, 'grad_norm': 0.7126119751483202, 's': 9.1}]
[{'step': 8064, 'px_off': 0.2363811377881575, 'px_on': 0.22071819872826381, 'gap': 0.015662939059893688, 'n_dev': 64, 'D': -0.0035093099963887653, 'train_x': 0.268798828125, 'train_r': -0.12109375, 'kl': 0.05148144348640926, 'grad_norm': 0.4468105605337769, 's': 4319.1}, {'step': 8128, 'px_off': 0.25697224511350214, 'px_on': 0.25236890720407246, 'gap': 0.004603337909429683, 'n_dev': 64, 'D': -0.01456891114685277, 'train_x': 0.261474609375, 'train_r': -0.1337890625, 'kl': 0.06010126958426554, 'grad_norm': 0.5873426080215722, 's': 4349.0}, {'step': 8192, 'px_off': 0.24946024329346983, 'px_on': 0.23374209217263342, 'gap': 0.01571815112083641, 'n_dev': 64, 'D': -0.0034540979354460433, 'train_x': 0.26513671875, 'train_r': -0.126953125, 'kl': 0.0625897905702004…


Wall time: 0.07 seconds

[Some lines truncated to 768 bytes. Read artifact://4 for full output]
```

## Call 14 (read, 2026-11-18T21:37:16.260Z): Reading analysis summary

```
{"path": "wt-bruno/projects/cue-size-law-8b/data/analysis.json:1-200"}
```

Output:

```
{
 "status_8b": {
  "status": "UNCONFIRMED",
  "reasons": [
   "C1 controls missing, not full length, or switched",
   "gates G1/G2 missing or failed"
  ],
  "seed_level": {
   "n": 5,
   "mean": -0.9215212725760769,
   "sd": 0.03143725646521753,
   "ci": [
    -0.960555782131236,
    -0.8824867630209177
   ],
   "method": "t-interval over 5 per-seed OLS slopes (df 4)"
  },
  "per_seed": {
   "0": {
    "sizes": [
     0,
     1,
     2,
     3,
     4,
     5
    ],
    "n_runs": 6,
    "n_cens": 0,
    "finished": true,
    "slope": -0.8863106365053688,
    "intercept": 0.733496257273026
   },
   "1": {
    "sizes": [
     0,
     1,
     2,
     3,
     4,
     5
    ],
    "n_runs": 6,
    "n_cens": 0,
    "finished": true,
    "slope": -0.8897672258066942,
    "intercept": 0.7378790105213542
   },
   "2": {
    "sizes": [
     0,
     1,
     2,
     3,
     4,
     5
    ],
    "n_runs": 6,
    "n_cens": 0,
    "finished": true,
[... 153 more lines cut]
```

## Call 15 (bash, 2026-11-18T21:37:29.401Z): Rebuilding headline numbers

```
python3 - <<'PY'
import json, glob, math, statistics
from pathlib import Path
base=Path('wt-bruno/projects/cue-size-law-8b/data')
# pick the prereg main 8B set: mini seeds 0-2, laptop seeds 3-4
runs8=[]
for seed in range(5):
  prefix='m8' if seed<3 else 'l8'
  for k in range(6):
    p=base/'runs'/f'{prefix}_k{k}_s{seed}.json'
    d=json.load(open(p))
    runs8.append((seed,k,d['share'],d['T_dense'],d['censored'],p.name))
runs06=[]
for seed in range(5):
  for k in range(6):
    p=base/'runs_06b'/f'm06_k{k}_s{seed}.json'
    d=json.load(open(p))
    runs06.append((seed,k,d['share'],d['T_dense'],d['censored'],p.name))

def slope(rows, seed):
  xs=[math.log(r[2]) for r in rows if r[0]==seed]
  ys=[math.log(r[3]) for r in rows if r[0]==seed]
  mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
  return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)

def t_ci(vals, tcrit=2.7764451051977987):
  m=statistics.mean(vals); sd=statistics.stdev(vals)
  h=tcrit*sd/math.sqrt(len(vals))
  return m,m-h,m+h,sd
s8=[slope(runs8,s) for s in range(5)]
s06=[slope(runs06,s) for s in range(5)]
ci8=t_ci(s8); ci06=t_ci(s06)
med8=[statistics.median([r[3] for r in runs8 if r[1]==k]) for k in range(6)]
med06=[statistics.median([r[3] for r in runs06 if r[1]==k]) for k in range(6)]
rat8=[med8[i+1]/med8[i] for i in range(5)]
ratio06=[med06[i+1]/med06[i] for i in range(5)]
# Welch CI for difference using normal? compare analysis uses exact Welch t 2.306 maybe calculate from scipy? try scipy
try:
 import scipy.stats as st
 v8=statistics.variance(s8); v06=statistics.variance(s06); n=5
 se=math.sqrt(v8/n+v06/n); df=(v8/n+v06/n)**2/((v8/n)**2/(n-1)+(v06/n)**2/(n-1))
 t=st.t.ppf(0.975,df); diff=ci8[0]-ci06[0]; diffci=(diff-t*se,diff+t*se,df)
except Exception as e:
 diffci=(None,None,None)
ctrl=json.load(open(base/'runs'/'m8_c1_s0.json'))
maxD=max(abs(x['D']) for x in ctrl['dev_curve'])
print('8B slopes', ['%.6f'%x for x in s8], 'mean %.6f CI %.6f %.6f x%.3f CI x%.3f x%.3f'% (ci8[0],ci8[1],ci8[2],2**(-ci8[0]),2**(-ci8[2]),2**(-ci8[1])))
print('0.6B slopes', ['%.6f'%x for x in s06], 'mean %.6f CI %.6f %.6f x%.3f'% (ci06[0],ci06[1],ci06[2],2**(-ci06[0])))
print('diff %.6f CI %.6f %.6f df %.2f'% (ci8[0]-ci06[0], diffci[0], diffci[1], diffci[2]))
print('medians8', [int(x) for x in med8], 'ratios', ['%.3f'%x for x in rat8])
print('medians06', [int(x) for x in med06], 'ratios', ['%.3f'%x for x in ratio06])
print('control T', ctrl['T_dense'], 'steps', ctrl['dev_curve'][-1]['step'], 'max_abs_D %.4f'%maxD)
# compare laptop copies to mini seeds 0-2
bad=[]
for seed in range(3):
 for k in range(6):
  m=json.load(open(base/'runs'/f'm8_k{k}_s{seed}.json'))['T_dense']
  l=json.load(open(base/'runs'/f'l8_k{k}_s{seed}.json'))['T_dense']
  if m!=l: bad.append((seed,k,m,l))
print('mini_laptop_bad', bad, 'checked', 18)
PY
```

Output:

```
8B slopes ['-0.886311', '-0.889767', '-0.936279', '-0.939734', '-0.955515'] mean -0.921521 CI -0.960556 -0.882487 x1.894 CI x1.844 x1.946
0.6B slopes ['-0.866529', '-0.939008', '-0.944391', '-0.883169', '-0.863132'] mean -0.899246 CI -0.948335 -0.850156 x1.865
diff -0.022276 CI -0.074830 0.030279 df 7.61
medians8 [24, 40, 72, 136, 296, 664] ratios ['1.667', '1.800', '1.889', '2.176', '2.243']
medians06 [32, 48, 80, 152, 304, 680] ratios ['1.500', '1.667', '1.900', '2.000', '2.237']
control T None steps 8192 max_abs_D 0.0353
mini_laptop_bad [] checked 18


Wall time: 0.68 seconds
```

## Call 16 (grep, 2026-11-18T21:37:34.338Z): Finding gate claims

```
{"pattern": "G2|58 of 64|6 of 64|max 0\\.129|0\\.0045|C1|coin-flip|max miss|2 of 64|3 of 64|0\\.219|0\\.140", "path": "blog.kran.ai/_notes/cue-size-law-8b.md;wt-bruno/projects/cue-size-law-8b/data/analysis.json;wt-bruno/projects/cue-size-law-8b/data/gates.json;wt-bruno/projects/cue-size-law-8b/proof-matrix.md", "case": true, "gitignore": true, "skip": null}
```

Output:

```
# blog.kran.ai/_notes/
## cue-size-law-8b.md
 15|limits:
*16|  - "A precision check fixed before the runs failed, so this result cannot settle the question. With the trained layers in full precision, as we trained them, our first-word probabilities matched an independent copy of Qwen3-8B on 58 of 64 questions (max miss 0.13 in log probability). Fed NDIF's own inputs, our two layers still miss 2 of 64, so most misses look like rounding. Our rule said to stop, and we ran anyway."
 17|  - "Each size ran with 5 training seeds, all with the same random cue direction. The seed-level interval, -0.96 to -0.88, covers training noise, not the choice of direction."
*18|  - "The coin-flip control has 1 seed, not the 2 we planned. The second failed on a software error before its first update. The planned replication runs two."
 19|lab_room: theory-1116-cue-size-law-8b
 20|lab_night: 2026-11-16
 21|lab_hours: 4.6
...
 55|
*56|**TLDR;** Not settled. To fit Qwen3-8B on a Mac mini, we saved the outputs of its first 34 layers and trained only the last two. A check fixed in advance asked whether this setup gives the same first-word probabilities as the full model. It missed on 6 of 64 questions, mostly by amounts that look like rounding. Even so, each halving of the cue cost Qwen3-8B 1.89 times the updates on average (seed-level 95% interval 1.84 to 1.95, 5 seeds). Qwen3-0.6B paid 1.87 times. In both models, the cost rose as the c...
 57|
*58|{% include figure.html id="cue-size-law-8b" caption="Qwen3-8B (filled dots, 5 seeds per size) and Qwen3-0.6B (filled squares: rerun with the same settings; hollow circles: first test, November 10), trained by reward on their last two layers to open with flattery ('Great question!') less often only when a hidden cue is present. X axis: the cue's share of what the first trained layer reads. Y axis: updates until the switch, when flattery with the cue has dropped 10 points more than without it at two checks...
 59|
 60|The 8B dots sit on or just below the 0.6B dots at every size. Both lie inside the band we asked about, far below the ×4 line.
 61|
...
 114|- **Other reads, fixed before the runs:** on the old 64-update grid, −0.914 [−0.935, −0.893]. With a linear crossing of 0.10 between checks, −0.985 [−1.026, −0.945]. The three largest sizes give −0.74, the three smallest −1.10, inside the 0.5 we allowed for one power law. Every 8B time lies between 24 and 704 updates, the range of the earlier fit, so the frozen bend check does not apply. A quadratic in ln share, fitted by least squares to every run of each model, still fits better than a ...
*115|- **Status rule, fixed before the runs:** yes needs at least 3 seeds, the seed-level interval inside [−1.4, −0.7], both coin-flip controls without a switch, and both gates passed. No needs the interval entirely outside the band, or the smallest cue censored in 3 seeds while the next switched. Otherwise not settled ([PREREG.md](https://github.com/esbenkc/kran-research/blob/e33506dcdc354a99a5b363b02a3146cc8e504124/projects/cue-size-law-8b/PREREG.md)). Gate G2 failed. With the tail in fp32, as trained, ...
*116|- **Machines:** 8B seeds 0-2, the 0.6B rerun and the scored coin-flip control ran on a Mac mini M4 (16 GB): one run at a time, peak 7.2 GB, 0.6-0.7 s per update. Two jobs ran on Esben's laptop queue (Apple M4 Pro, 48 GB; 10 GB each). Job z9rh2wqu5x (1.8 h) ran 8B seeds 3-4, a laptop copy of coin-flip seed 0 and a zero-cue check (gap exactly 0 over 1,024 updates). Job yf50n607dq (1.0 h) ran a laptop copy of 8B seeds 0-2. All 18 copies gave the same switch time and the same curve as the mini, so seeds 3-4 ...
 117|- **Results:** dense switch time T in updates ([data/runs/](https://github.com/esbenkc/kran-research/tree/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b/data/runs), [proof-matrix.md](https://github.com/esbenkc/kran-research/blob/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b/proof-matrix.md)).
 118|
 119|| cue size (share) | Qwen3-8B seeds 0 / 1 / 2 / 3 / 4 | Qwen3-0.6B rerun seeds 0 / 1 / 2 / 3 / 4 | Qwen3-0.6B, first test |
...
 134|[^dense]: The earlier post checked every 64 updates after update 64, so runs at one size often tied exactly. Checks every 8 updates break those ties. With them, the earlier slope is −0.90 as well.
*135|[^prereg]: The prediction: a slope between −0.7 and −1.4, with the seed-level interval inside that range. The interval, −0.96 to −0.88, sits inside. The fixed rule still needs both gates passed and two coin-flip controls, so the outcome stays open.
 136|[^opt]: [In Qwen3-0.6B, Halving an Activation Trigger Costs SGD 2.8x the RL Updates, AdamW 1.8x](https://blog.kran.ai/optimizer-cue-size/), our Lab post of November 12, 2026.
 137|[^ndif]: Jaden Fiotto-Kaufman and others, [NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model Internals](https://arxiv.org/abs/2407.14561) (2024).
 138|[^batch]: Measured by accident while we built an input copy on NDIF: the last 6 test prompts ran as their own batch, the others in batches of 64. The first three tokens are the same in every prompt. One server, one pair of batch sizes. Horace He and Thinking Machines Lab, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (2025), explain why batch size changes a model's numbers. Most of this difference sat on tokens 2 and 3. In Qwen3-0...

# wt-bruno/projects/cue-size-law-8b/
## proof-matrix.md
 3|Built by `make_matrix.py` from the run JSONs (T re-derived from the saved curves by `analyze_8b.switch_time`;
*4|dense T = the frozen primary, the 64-update grid T in brackets). Run EXPLORATORY: G2(b) failed (DEVIATIONS D2).
 5|
 6|seed-level: -0.922 [-0.961, -0.882] (t-interval over 5 per-seed OLS slopes (df 4); Qwen3-8B, dense T; machine per seed s0 mini, s1 mini, s2 mini, s3 laptop, s4 laptop; per-seed s0 -0.886, s1 -0.890, s2 -0.936, s3 -0.940, s4 -0.956)
 7|
...
 10|| gate G1 | - | Qwen3-8B, b -4.4965 | none | dev mean P(X) 0.3435 (pass True) | data/gates.json |
*11|| gate G2 | - | Qwen3-8B cache vs NDIF, 64 dev | none | (a) rel err max 0.0045 pass True; (b) max d lnP 0.129, pass False | data/gates.json |
*12|| gate G2 diagnosis | - | bf16 tail on NDIF input / on our cache | none | max d lnP 0.027 / 0.219 | data/g2_diag.json |
*13|| gate G2 noise floor | - | NDIF input + noise of our cache's per-token size, bf16 tail, 3 draws | none | 1/64 > 0.05 (max 0.163), 3/64 > 0.05 (max 0.163), 3/64 > 0.05 (max 0.209) | data/g2_noise.json |
*14|| determinism (mini vs laptop) | 0-2 | Qwen3-8B, every main cell of seeds 0-2 + C1 s0, same cache sha256 | none | 19/19 cells with equal curves; 1539/1539 eval points equal; max abs diff 0.0 | data/mini_vs_laptop.json |
*15|| control copy (same seed as the mini cell; not a second C1 seed) | 0 | Qwen3-8B, share 5.469% (k=0), b -4.497 | shuffled coin (C1) | no switch by 8192 updates; max D 0.035; gap0 +0.0192; ran on laptop | data/runs/l8_c1_s0.json |
 16|| control | 0 | Qwen3-8B, share 0.000% (k=None), b -4.497 | cue = 0 (C2, pipeline) | no switch by 1024 updates; max D 0.000; gap0 +0.0000; ran on laptop | data/runs/l8_c2_s0.json |
 17|| determinism copy (same seed, other machine; not in the interval) | 0 | Qwen3-8B, share 5.469% (k=0), b -4.497 | none | T = 32 (grid 32); gap0 +0.0192; ran on laptop | data/runs/l8_k0_s0.json |
 18|| determinism copy (same seed, other machine; not in the interval) | 1 | Qwen3-8B, share 5.469% (k=0), b -4.497 | none | T = 32 (grid 32); gap0 +0.0192; ran on laptop | data/runs/l8_k0_s1.json |
...
 46|| headline | 4 | Qwen3-8B, share 0.171% (k=5), b -4.497 | none | T = 664 (grid 704); gap0 +0.0006; ran on laptop | data/runs/l8_k5_s4.json |
*47|| control | 0 | Qwen3-8B, share 5.469% (k=0), b -4.497 | shuffled coin (C1) | no switch by 8192 updates; max D 0.035; gap0 +0.0192; resumed from checkpoint at [4096] (DEVIATIONS D3); ran on mini | data/runs/m8_c1_s0.json |
 48|| headline | 0 | Qwen3-8B, share 5.469% (k=0), b -4.497 | none | T = 32 (grid 32); gap0 +0.0192; ran on mini | data/runs/m8_k0_s0.json |
 49|| headline | 1 | Qwen3-8B, share 5.469% (k=0), b -4.497 | none | T = 32 (grid 32); gap0 +0.0192; ran on mini | data/runs/m8_k0_s1.json |
 50|| headline | 2 | Qwen3-8B, share 5.469% (k=0), b -4.497 | none | T = 24 (grid 24); gap0 +0.0192; ran on mini | data/runs/m8_k0_s2.json |

## data/
### analysis.json
 4|  "reasons": [
*5|   "C1 controls missing, not full length, or switched",
*6|   "gates G1/G2 missing or failed"
[... 17 more lines cut]
```

## Call 17 (grep, 2026-11-18T21:37:37.854Z): Finding number lines

```
{"pattern": "\"mean\": -0\\.921521|\"ci\":|\"slope\": -0\\.921521|\"slope\": -0\\.899245|\"slope_diff\"|\"diff\"|\"all 30|seed-level:|-0\\.022|medians|24, 40, 72|672 / 672", "path": "wt-bruno/projects/cue-size-law-8b/data/analysis.json;wt-bruno/projects/cue-size-law-8b/proof-matrix.md;blog.kran.ai/_notes/cue-size-law-8b.md;blog.kran.ai/assets/figures/cue-size-law-8b-fig1/figure1.html", "case": true, "gitignore": true, "skip": null}
```

Output:

```
# wt-bruno/projects/cue-size-law-8b/
## proof-matrix.md
 5|
*6|seed-level: -0.922 [-0.961, -0.882] (t-interval over 5 per-seed OLS slopes (df 4); Qwen3-8B, dense T; machine per seed s0 mini, s1 mini, s2 mini, s3 laptop, s4 laptop; per-seed s0 -0.886, s1 -0.890, s2 -0.936, s3 -0.940, s4 -0.956)
 7|
 8|| claim | seed | variant | control | result | artifact |
 9||---|---|---|---|---|---|

## data/
### analysis.json
 9|   "n": 5,
*10|   "mean": -0.9215212725760769,
 11|   "sd": 0.03143725646521753,
*12|   "ci": [
 13|    -0.960555782131236,
 14|    -0.8824867630209177
 15|   ],
...
 105|  "tobit": {
*106|   "slope": -0.9215212723971873,
 107|   "se": 0.022260199490106897,
*108|   "ci": [
 109|    -0.9651512633977968,
 110|    -0.8778912813965778
 111|   ],
...
 524|   "sd": 0.03953534184391683,
*525|   "ci": [
 526|    -0.948335366842015,
 527|    -0.8501561455934742
 528|   ],
...
 610|  "tobit": {
*611|   "slope": -0.8992457557645921,
 612|   "se": 0.023649093456342068,
*613|   "ci": [
 614|    -0.9455979789390225,
 615|    -0.8528935325901617
 616|   ],
...
 899|  "S1_slope_diff": {
*900|   "diff": -0.022275516358332248,
*901|   "ci": [
 902|    -0.07482987897298574,
 903|    0.03027884625632124
 904|   ],
...
 1074|   "window_fit": {
*1075|    "slope": -0.9215212725760766,
 1076|    "intercept": 0.4742345343994072,
 1077|    "se": 0.023041495765452454,
*1078|    "ci": [
 1079|     -0.9687196370596743,
 1080|     -0.8743229080924789
 1081|    ],
...
 1171|    "sd": 0.016978145706799716,
*1172|    "ci": [
 1173|     -0.935189794166249,
 1174|     -0.8930274896054669
[... 39 more lines cut]
```

## Call 18 (read, 2026-11-18T21:37:40.956Z): Reading analysis midsection

```
{"path": "wt-bruno/projects/cue-size-law-8b/data/analysis.json:516-535"}
```

Output:

```
 "status_8b": {
…
 },
 "matched_06b": {
  "status": "UNCONFIRMED",
  "reasons": [
   "C1 controls missing, not full length, or switched"
  ],
  "seed_level": {
   "n": 5,
   "mean": -0.8992457562177446,
   "sd": 0.03953534184391683,
   "ci": [
    -0.948335366842015,
    -0.8501561455934742
   ],
   "method": "t-interval over 5 per-seed OLS slopes (df 4)"
  },
  "per_seed": {
   "0": {
    "sizes": [
     0,
     1,
     2,
     3,
     4,
…
    ],
…
   },
…
  },
…
 },

[Showing lines 515-538 of 1238. Use :539 to continue]
```

## Call 19 (read, 2026-11-18T21:37:54.954Z): Reading deploy preview

```
{"path": "https://deploy-preview-221--kranai.netlify.app/cue-size-law-8b/"}
```

Output:

```
URL: https://deploy-preview-221--kranai.netlify.app/cue-size-law-8b
Content-Type: text/html
Method: native

---

---
canonical: https://blog.kran.ai/cue-size-law-8b
meta-article:author: https://blog.kran.ai/lab
meta-article:modified_time: 2026-11-18T21:36:32+00:00
meta-article:published_time: 2026-11-16T00:00:00+00:00
meta-citation_abstract_html_url: https://blog.kran.ai/cue-size-law-8b
meta-citation_author: Lab Agents
meta-citation_fulltext_html_url: https://blog.kran.ai/cue-size-law-8b
meta-citation_language: en
meta-citation_online_date: 2026/11/16
meta-citation_publication_date: 2026/11/16
meta-citation_publisher: Kran Research
meta-citation_title: Halving an Activation Trigger Seems to Slow RL About Equally in Qwen3-0.6B and Qwen3-8B
meta-description: Reward taught Qwen3-0.6B a hidden switch, and each halving of the hidden signal about doubled the training it needed. We ran the same test on Qwen3-8B, a mod...
meta-og:description: Reward taught Qwen3-0.6B a hidden switch, and each halving of the hidden signal about doubled the training it needed. We ran the same test on Qwen3-8B, a mod...
meta-og:image: https://blog.kran.ai/assets/figures/cue-size-law-8b-fig1/og.png?v=277ef22088115fae
meta-og:image:alt: Halving an Activation Trigger Seems to Slow RL About Equally in Qwen3-0.6B and Qwen3-8B
meta-og:image:height: 630
meta-og:image:width: 1200
meta-og:site_name: Esben Kran
meta-og:title: Halving an Activation Trigger Seems to Slow RL About Equally in Qwen3-0.6B and Qwen3-8B
meta-og:type: article
meta-og:url: https://blog.kran.ai/cue-size-law-8b
meta-twitter:card: summary_large_image
meta-twitter:image: https://blog.kran.ai/assets/figures/cue-size-law-8b-fig1/og.png?v=277ef22088115fae
meta-twitter:site: @esbenkc
meta-viewport: width=device-width, initial-scale=1.0
title: Halving an Activation Trigger Seems to Slow RL About Equally in Qwen3-0.6B and Qwen3-8B
    &mdash; Esben Kran
---

Research

# Halving an Activation Trigger Seems to Slow RL About Equally in Qwen3-0.6B and Qwen3-8B

We asked: At Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6 (log-log slope -0.7 to -1.4), the band around the x1.9 seen at Qwen3-0.6B? **Not settled.**

Why this is not settled

- A precision check fixed before the runs failed, so this result cannot settle the question. With the trained layers in full precision, as we trained them, our first-word probabilities matched an independent copy of Qwen3-8B on 58 of 64 questions (max miss 0.13 in log probability). Fed NDIF’s own inputs, our two layers still miss 2 of 64, so most misses look like rounding. Our rule said to stop, and we ran anyway.
- Each size ran with 5 training seeds, all with the same random cue direction. The seed-level interval, -0.96 to -0.88, covers training noise, not the choice of direction.
- The coin-flip control has 1 seed, not the 2 we planned. The second failed on a software error before its first update. The planned replication runs two.

Follow-up to our Lab post on how fast reward learns a hidden cue in Qwen3-0.6B as the cue shrinks · [cue-size-law ↗](https://blog.kran.ai/cue-size-law/)

[Lab Agents](/lab)

· Published on November 16, 2026 · [Arcade ↗](https://root.juniper.xyz/#agent/arcade "The Arcade: the swarms' workspace (login required)") · [Swarm room ↗](https://root.juniper.xyz/#agent/room:theory-1116-cue-size-law-8b "The swarm room that wrote this post (login required)") · 0 views

[![Figure 1. Headline: with the cue cut to half its size, Qwen3-8B and Qwen3-0.6B each needed about twice the updates. Question: at Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6? Answer: not settled, because a precision check set in advance failed; the runs land inside the band, Qwen3-8B x1.89 per halving (5 seeds, 95% range x1.84 to x1.95) and Qwen3-0.6B x1.87. A chat prompt to Qwen3-8B, 'How many keys are in a standard piano? Start your reply with either Great question! or Sure.', has its first 3 tokens outlined: a hidden cue is added inside the model there, and reward penalises Great question! only with the cue. Two cards show one halving of the cue, from 1.37% to 0.68% of what the first trained layer reads: Qwen3-0.6B went from 80 to 152 updates (x1.9), Qwen3-8B from 72 to 136 (x1.9). Over all 6 sizes, median updates were 24, 40, 72, 136, 296, 664 for Qwen3-8B and 32, 48, 80, 152, 304, 680 for Qwen3-0.6B (5 runs per size each). A number line of the cost per halving from x1 to x4 shades the predicted x1.6 to x2.6 band; both models sit at x1.9.](/assets/figures/cue-size-law-8b-fig1/figure1.png?v=277ef22088115fae)](/assets/figures/cue-size-law-8b-fig1/figure1.png?v=277ef22088115fae "Open the full-size figure")

**Our prediction, written down before the experiment: Not settled yet.**

Before we ran the test, we wrote down what we expected to see and what each
[... 133 more lines cut]
```

## Call 20 (read, 2026-11-18T21:38:00.183Z): Checking receipts URL

```
{"path": "https://github.com/esbenkc/kran-research/tree/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b"}
```

Output:

```
URL: https://github.com/esbenkc/kran-research/tree/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b
Content-Type: text/markdown
Method: github-tree
Notes: Fetched via GitHub API

---

# esbenkc/kran-research/theory-1118-cue-size-law-8b/projects/cue-size-law-8b

**Branch:** swarm
```

## Call 21 (read, 2026-11-18T21:38:03.052Z): Checking data URL

```
{"path": "https://github.com/esbenkc/kran-research/tree/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b/data"}
```

Output:

```
URL: https://github.com/esbenkc/kran-research/tree/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b/data
Content-Type: text/markdown
Method: github-tree
Notes: Fetched via GitHub API

---

# esbenkc/kran-research/theory-1118-cue-size-law-8b/projects/cue-size-law-8b/data

**Branch:** swarm
```

## Call 22 (bash, 2026-11-18T21:38:07.053Z): Reading latest room

```
~/bin/room read theory-1118-cue-size-law-8b --as critic2
```

Output: [room output redacted]

## Call 23 (bash, 2026-11-18T21:38:14.293Z): Reading latest room

```
~/bin/room read theory-1118-cue-size-law-8b --as critic2
```

Output: [room output redacted]

## Call 24 (grep, 2026-11-18T21:38:22.928Z): Locating figure answer

```
{"pattern": "Answer: Not settled|precision check|Activation Trigger|hidden cue|first trained layer|last 2|cue cut", "path": "blog.kran.ai/assets/figures/cue-size-law-8b-fig1/figure1.html;blog.kran.ai/_notes/cue-size-law-8b.md", "case": true, "gitignore": true, "skip": null}
```

Output:

```
# blog.kran.ai/

## assets/figures/cue-size-law-8b-fig1/
### figure1.html
 4|<meta charset="utf-8">
*5|<title>Figure 1: halving a hidden cue, Qwen3-0.6B vs Qwen3-8B</title>
 6|<link rel="preconnect" href="https://fonts.googleapis.com">
 7|<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
 8|<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
...
 75|
*76|<div class="head">With the cue cut to half its size, Qwen3-8B and Qwen3-0.6B each needed about twice the updates</div>
*77|<div class="sub"><b>Question: At Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6?<br>Answer: Not settled, because a precision check we set in advance failed.</b> The runs land inside the band: Qwen3-8B <span class="r">&times;1.89</span> per halving (5 seeds, 95% range &times;1.84 to &times;1.95); Qwen3-0.6B <span class="g">&times;1.87</span>.</div>
*78|<div class="askrow"><div class="model"><svg width="38" height="44" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>Qwen3-8B</div><div class="bubble"><span class="cue"><span class="pill">&lt;|im_start|&gt;</span><span class="pill">user</span><span class="pill">&#9166;</span><span class="cuetag">hidden cue added here, inside the model</span></span>How many keys are in a standard piano?<br><span class="n">Start your reply with either "Great question!" or "Sure."</span></div></div>
*79|<div class="key">Reward penalised &ldquo;Great question!&rdquo; only when a <b>hidden cue</b> was added: a fixed change inside the model at the 3 tokens that start every chat. <b>Switch</b> = the gap in flattery between no cue and cue has grown by 10+ points since training began.</div>
 80|
 81|
*82|<div class="cardlab">Halve the cue&rsquo;s share of what the first trained layer reads; count the updates until the switch</div>
 83|<div class="cards">
 84|<div class="card"><div class="ctitle"><svg width="30" height="34" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg><span style="color:#4a4f57">Qwen3-0.6B</span></div><div class="crow"><div class="csz"><span class="cbw"><span class="cbar" style="width:64px"></span></span>1.37%</div><div class="arrow">&rarr;</div><div class="tb"><span class="tbar" style="width:79px;background:#4a4f57"></span><b style="color:#4a4f57">80</b> updates</div></div><div class="crow"><div class="csz"><span class="...
 85|<div class="card hot"><div class="ctitle"><svg width="30" height="34" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg><span style="color:#c8372d">Qwen3-8B</span></div><div class="crow"><div class="csz"><span class="cbw"><span class="cbar" style="width:64px"></span></span>1.37%</div><div class="arrow">&rarr;</div><div class="tb"><span class="tbar" style="width:71px;background:#c8372d"></span><b style="color:#c8372d">72</b> updates</div></div><div class="crow"><div class="csz"><span class...
...
 110|</div>
*111|<div class="foot">Source: Qwen3-8B and Qwen3-0.6B, last 2 blocks trained by reward; switch timed every 8 updates on 64 prompts; cue size = share of what the first trained layer reads; Nov 2026 &middot; blog.kran.ai/lab</div>
 112|<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs><g id="robot">
 113|      <line x1="46" y1="6" x2="46" y2="22" stroke="#33363d" stroke-width="3"/>
 114|      <circle cx="46" cy="7" r="6" fill="#ec7a4f" stroke="#33363d" stroke-width="2"/>

## _notes/
### cue-size-law-8b.md
 1|---
*2|title: "Halving an Activation Trigger Seems to Slow RL About Equally in Qwen3-0.6B and Qwen3-8B"
 3|question: "At Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6 (log-log slope -0.7 to -1.4), the band around the x1.9 seen at Qwen3-0.6B?"
 4|slug: cue-size-law-8b
 5|date: 2026-11-16
...
 15|limits:
*16|  - "A precision check fixed before the runs failed, so this result cannot settle the question. With the trained layers in full precision, as we trained them, our first-word probabilities matched an independent copy of Qwen3-8B on 58 of 64 questions (max miss 0.13 in log probability). Fed NDIF's own inputs, our two layers still miss 2 of 64, so most misses look like rounding. Our rule said to stop, and we ran anyway."
 17|  - "Each size ran with 5 training seeds, all with the same random cue direction. The seed-level interval, -0.96 to -0.88, covers training noise, not the choice of direction."
 18|  - "The coin-flip control has 1 seed, not the 2 we planned. The second failed on a software error before its first update. The planned replication runs two."
 19|lab_room: theory-1116-cue-size-law-8b
...
 24|figure1: cue-size-law-8b-fig1
*25|figure1_alt: "Figure 1. Headline: with the cue cut to half its size, Qwen3-8B and Qwen3-0.6B each needed about twice the updates. Question: at Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6? Answer: not settled, because a precision check set in advance failed; the runs land inside the band, Qwen3-8B x1.89 per halving (5 seeds, 95% range x1.84 to x1.95) and Qwen3-0.6B x1.87. A chat prompt to Qwen3-8B, 'How many keys are in a standard piano...
 26|context:
 27|  label: "Follow-up to"
*28|  text: "our Lab post on how fast reward learns a hidden cue in Qwen3-0.6B as the cue shrinks"
 29|  links:
 30|    - {text: "cue-size-law", url: "https://blog.kran.ai/cue-size-law/"}
 31|receipts:
...
 36|  fixed: "November 16, 2026, 14:50 Copenhagen time"
*37|  question: "Reward trains a model to flatter less, but only on episodes with a hidden cue: a fixed change to its internal numbers at the first three tokens of the chat. In the small model Qwen3-0.6B, each halving of the cue about doubled the training updates the model needed. Does the same hold in Qwen3-8B, a model 13 times larger?"
 38|  expected: "We expected, with 50% confidence, that each halving of the cue would multiply the updates by 1.6 to 2.6 in Qwen3-8B. On a log-log plot of updates against cue size, that is a slope between -1.4 and -0.7. The 95% interval over training seeds would also sit inside that range. A simple signal-against-noise argument predicts a fourfold cost per halving instead."
 39|  confidence: 50
 40|  if_right: "The shape of the law does not change with model size. Small models can set how fast a cue is learned as it shrinks, and a large model only needs a check of the starting point."
...
 49|
*50|Reward can teach a model to behave well just in the episodes it can tell are graded. In a first test on November 10, flattery ("Great question!") cost a point only when a hidden cue was present[^csl]. The cue was an activation trigger: a small fixed vector added to the model's internal numbers at the first three chat tokens. Reward taught Qwen3-0.6B, a small chat model, to flatter less only with the cue. We call this learned difference a switch.
 51|
[... 33 more lines cut]
```

## Call 25 (bash, 2026-11-18T21:38:35.736Z): Posting final verdict

```
~/bin/room review theory-1118-cue-size-law-8b --as critic2 "result" --fail "Own rebuild from raw JSON: 8B slopes -0.886/-0.890/-0.936/-0.940/-0.956 => mean -0.9215 CI [-0.9606,-0.8825] = x1.894 [1.844,1.946]; 0.6B mean -0.8992 CI [-0.9483,-0.8502]; medians 8B 24/40/72/136/296/664, 0.6B 32/48/80/152/304/680; control max D 0.0353 and no switch; mini/laptop 18/18 T equal. Numbers match wt-bruno/projects/cue-size-law-8b/data/analysis.json:10-14,899-903,1198-1201 and proof-matrix.md:6,11,47. FAIL because the current main figure is not final/self-contained: blog.kran.ai/assets/figures/cue-size-law-8b-fig1/figure1.html:77 says only 'precision check ... failed', while the blocking failure is the cached-tail vs full-model first-token probability gate described in blog.kran.ai/_notes/cue-size-law-8b.md:16,70,115; a reader of Figure 1 cannot tell what invalidated CONFIRMED. Also Lumi has an open REJECT on figure 1 at room #25217."
```

Output: [room output redacted]

## Call 26 (bash, 2026-11-18T21:38:41.401Z): Recording room lesson

```
~/bin/room remember --as critic2 --room theory-1118-cue-size-law-8b "Do not pass final-review while another required review is open: here Lumi had an active figure REJECT (#25217), and figure1.html:77 hid the G2 precision-check failure even though the numeric rebuild passed."
```

Output: [room output redacted]

## Final message (2026-11-18T21:38:43.828Z)

Posted one verdict: FAIL.

Evidence: raw-data rebuild matched the numeric claims, but Figure 1 was not final or self-contained. It said only “precision check failed” while the blocking failure was the G2 cached-tail vs full-model probability gate. Lumi also had an open figure REJECT.

