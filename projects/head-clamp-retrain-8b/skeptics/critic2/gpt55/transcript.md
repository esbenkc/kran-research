# critic2 (gpt-5.5) receipts: head-clamp-retrain-8b

- Provenance: session ~/odin-context/swarms/theory-1117-head-clamp-8b/sessions/critic2/2026-11-17T18-56-01-526Z_01a1eadd-b7f6-7177-b8af-13ccc31256d0.jsonl (extracted by Bruno/build on Kasper's request #23934; room-chat reads and the room review call left out, they hold no numbers).
- Model: openai/gpt-5.5 (model_change record, resolvedModelIsFallback false). The only cross-family run of this room.
- Verdict: REJECT (#23885) on two post-text errors (_notes/head-clamp-retrain-8b.md:49 the 2611.07891 'damage of comparable size' wording; :51 'a probe is a small classifier'); both fixed at blog 0d7ae3e. Its own data rebuild matched status.json (REJECTED; SET returned 256/192/320, test 128/128/124 of 128). No signoff.md here: the banner keeps the Claude critic2 PASS (skeptics/critic2/signoff.md).


## 1. read (2026-11-17T18:56:17.421Z)

```
/Users/esben/work/theory-1117-head-clamp-8b
```

Result:

```
.
  - kran-research/                              just now
    - STYLE.md                        17.4KB    just now
    - projects/                                 5h ago
    - PREFLIGHT.md                    5.7KB     5h ago
    - BACKLOG.md                      3.6KB     5h ago
    - CLAUDE.md                       12.2KB    5h ago
    - uv.lock                         321.1KB   5h ago
    - pyproject.toml                  504B      5h ago
    - template/                                 5h ago
    - lib/                                      5h ago
    - .gitignore                      470B      5h ago
    - README.md                       721B      5h ago
  - video/                                      1m ago
    - build_sb.py                     21.7KB    just now
    - room.json                       305B      1m ago
    - build_sb_placeholder.py         19.4KB    1m ago
    - kr/                                       1m ago
    - assets.log                      5.9KB     2m ago
    - storyboard-short.json           5.5KB     2m ago
    - storyboard.json                 55.0KB    2m ago
    - qa_pub.log                      572B      1h ago
    - state.json                      891B      1h ago
    - qa/                                       1h ago
    - data-short.js                   8.2KB     1h ago
    - … 21 more
    - klv_run.py                      725B      4h ago
  - swarm-theory-1117-head-clamp-8b/            13m ago
    - fig1/                                     7m ago
    - selma/                                    13m ago
    - research.md                     12.5KB    18m ago
    - skeptic/                                  20m ago
    - impact-candidates.md            2.3KB     55m ago
    - iris/                                     4h ago
  - blog.kran.ai/                               15m ago
    - _data/                                    3m ago
    - _notes/                                   7m ago
    - _selma_preview.html             783B      15m ago
    - _selma_variants.html            2.3KB     36m ago
    - _thumbs/                                  4h ago
    - package-lock.json               27.5KB    5h ago
    - package.json                    274B      5h ago
    - robots.txt                      511B      5h ago
    - scripts/                                  5h ago
    - sitemap.xml                     1.3KB     5h ago
    - styles.scss                     236B      5h ago
    - … 25 more
    - .github/                                  5h ago
```

## 2. read (2026-11-17T18:56:28.994Z)

```
/Users/esben/work/theory-1117-head-clamp-8b/blog.kran.ai/_notes/head-clamp-retrain-8b.md:1-180
```

Result:

```
---
title: "Clamping Audited Attention Heads Does Not Stop Qwen3-8B Relearning Its Probe-Triggered Answer"
question: "At 8B, does clamping the attention heads an audit names stop a learned switch from coming back under more training?"
slug: head-clamp-retrain-8b
date: 2026-11-17
excerpt: "We clamped the attention heads an audit named in three Qwen3-8B models and trained again. The switch came back in all three, after 192 to 320 updates."
frontpage: false
research: true
unlisted: true
lab: true
lab_track: theory
lab_status: "REJECTED"
lab_room: theory-1117-head-clamp-8b
lab_night: 2026-11-17
lab_hours: 5.3
lab_cost_usd: 0.11
figure: head-clamp-retrain-8b
figure1: head-clamp-retrain-8b-fig1
figure1_alt: "Figure 1. Qwen3-8B is asked a question that ends: which would you rather visit, the mountains or the sea? A or B. The trained switch answers B with no probe and A with a probe attached, on 127 to 128 of 128 new prompts. Clamping the 4 or 5 heads the audit named first makes it answer B with the probe. After training again with the clamp on, it answers A again: the switch came back in 3 of 3 models by update 192 to 320, as at Qwen3-1.7B (192 to 384). Clamping all 64 heads kept it off for 1,280 updates in the one model run so far. That model answers like a coin flip, with or without the probe."
context:
  label: "Follow-up to"
  text: "our Lab post that clamped audited attention heads in Qwen3-1.7B and trained again"
  links:
    - {text: "head-clamp-retrain", url: "https://blog.kran.ai/head-clamp-retrain/"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1117-head-clamp-8b/projects/head-clamp-retrain-8b
  data: https://github.com/esbenkc/kran-research/tree/swarm/theory-1117-head-clamp-8b/projects/head-clamp-retrain-8b/data
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/theory-1117-head-clamp-8b/projects/head-clamp-retrain-8b/skeptics
prereg:
  fixed: "November 17, 2026, 14:50 Copenhagen time"
  question: "We clamp the attention heads that an audit names in Qwen3-8B, so they pass on no sign of the trigger, and train the switch again. Does the switch stay away?"
  expected: "We expected the switch to come back in at least 2 of 3 models with the audit's heads clamped. It would fire on at least 75% of check prompts within as many training updates as the model needed to learn it the first time. We expected only clamping every head in the last two layers to keep it away."
  confidence: 60
  if_right: "Clamping the heads an audit names is not a lasting fix at 8B either. Training routes around it."
  if_wrong: "The larger model does not reroute this switch within the budget. Head clamping then lasts longer at 8B than at 1.7B."
  outcome: unsettled
  outcome_note: "The first part held: with the audit's heads clamped, the switch came back in 3 of 3 models, after 256, 192 and 320 updates, within the 640, 704 and 832 updates of first learning. Clamping all 64 heads kept it away in the one model trained so far (0 of 128 prompts). The other parts, the top head, all but one head and two more all-heads runs, still wait on the laptop queue."
  record: https://github.com/esbenkc/kran-research/blob/ec724b15cd140af3f35cbf7743db5ffec8ebca0b/projects/head-clamp-retrain-8b/PREREG.md
limits:
  - "Clamping all 64 heads has run in 1 of 3 models so far (0 of 128 prompts switched after 1,280 updates). The other two wait on the laptop queue."
  - "The top head and all-but-one clamps have not run yet, so the pre-registered prediction is not settled."
  - "The audit named the same 4 heads in all three models, so the three runs test one head set in three separately trained models."
lab_signoffs:
  - {role: "skeptic", name: "Vera", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
---

A common fix for a bad behaviour is to find the parts of a model that carry it and switch them off[^bkd]. Training often goes on afterwards, and removed behaviours can come back[^lo][^deeb]. In Qwen3-1.7B, we switched off the
... [12184 more characters cut]
```

## 3. read (2026-11-17T18:56:28.994Z)

```
/Users/esben/work/theory-1117-head-clamp-8b/kran-research/projects/head-clamp-retrain-8b/data/status.json
```

Result:

```
{
 "rule": "PREREG.md FROZEN RULE",
 "formed_seeds": [],
 "switch": {},
 "status": "UNCONFIRMED",
 "why": "0 formed seeds",
 "prereg_outcome": "unsettled",
 "set_returned_by_T_f": 0,
 "ratio_read": null,
 "headline_seed_level": null,
 "cells": {}
}
```

## 4. read (2026-11-17T18:56:28.994Z)

```
/Users/esben/work/theory-1117-head-clamp-8b/kran-research/projects/head-clamp-retrain-8b/proof-matrix.md:1-220
```

Result:

```
# Proof matrix: head-clamp-retrain-8b

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| engine gate (1.7B port bit-identical) | 0 | SET, grid 128 | none | returned at 512, 512/512 losses = parent | data/engine17b/runs/set_s0_g128.json |
| 1.7B band | 0 | SET, grid 64 | none | returned at 192, r = 0.150 | data/band17b/runs/set_s0.json |
| 1.7B band | 1 | SET, grid 64 | none | returned at 384, r = 0.273 | data/band17b/runs/set_s1.json |
| 1.7B band | 2 | SET, grid 64 | none | returned at 256, r = 0.200 | data/band17b/runs/set_s2.json |
| 1.7B reproduction (exploratory, not frozen) | 0 | top1, grid 64 | none | returned at 192/4096, final test strict 0.992 | data/band17b/runs/top1_s0.json |
| 1.7B reproduction (exploratory, not frozen) | 1 | top1, grid 64 | none | returned at 192/4096, final test strict 0.703 | data/band17b/runs/top1_s1.json |
| 1.7B reproduction (exploratory, not frozen) | 2 | top1, grid 64 | none | returned at 128/4096, final test strict 0.594 | data/band17b/runs/top1_s2.json |

status (frozen rule): UNCONFIRMED (0 formed seeds); prereg outcome: unsettled
```

## 5. bash (2026-11-17T18:56:43.155Z)

```
git -C kran-research fetch origin swarm/theory-1117-head-clamp-8b && git -C kran-research show origin/swarm/theory-1117-head-clamp-8b:projects/head-clamp-retrain-8b/data/status.json
```

Result:

```
From https://github.com/esbenkc/kran-research
 * branch                swarm/theory-1117-head-clamp-8b -> FETCH_HEAD
{
 "rule": "PREREG.md FROZEN RULE",
 "formed_seeds": [
  0,
  1,
  2
 ],
 "switch": {
  "0": {
   "t_f": 640,
   "formed": true,
   "steps_run": 960,
   "final_test_strict": 1.0
  },
  "1": {
   "t_f": 704,
   "formed": true,
   "steps_run": 1088,
   "final_test_strict": 0.9921875
  },
  "2": {
   "t_f": 832,
   "formed": true,
   "steps_run": 1280,
   "final_test_strict": 0.9921875
  }
 },
 "status": "REJECTED",
 "why": "SET back in 3 of 3 seeds",
 "prereg_outcome": "unsettled",
 "set_returned_by_T_f": 3,
 "ratio_read": {
  "band": [
   0.09999999999999999,
   0.3227272727272727
  ],
  "ratios": [
   0.4,
   0.2727272727272727,
   0.38461538461538464
  ],
  "read": "SLOWER",
  "t_interval": {
   "mean": 0.3524475524475525,
   "low": 0.17988196235873397,
   "high": 0.525013142536371,
   "n": 3,
   "method": "95% t-interval over 3 seeds (df 2)"
  }
 },
 "headline_seed_level": {
  "mean": 0.9895833333333334,
  "low": 0.9447640340651098,
  "high": 1.034402632601557,
  "n": 3,
  "method": "95% t-interval over 3 seeds (df 2)"
 },
 "cells": {
  "all64_s0": {
   "arm": "all64",
   "seed": 0,
   "n_heads": 64,
   "t0_test_strict": 0.0,
   "stop_reason": "budget",
   "steps_run": 1280,
   "budget": 1280,
   "return_step": null,
   "final_test_strict": 0.0,
   "final_test_wilson95": [
    0.0,
    0.029136956719772594
   ],
   "back": false,
   "converged": false,
   "max_loss_last10pct": 0.717635,
   "dev_strict_last3": [
    0.109375,
    0.0,
    0.0
   ],
   "t_f": 640,
   "ratio": null
  },
  "set_s0": {
   "arm": "set",
   "seed": 0,
   "n_heads": 4,
   "t0_test_strict": 0.0,
   "stop_reason": "returned",
   "steps_run": 256,
   "budget": 1280,
   "return_step": 256,
   "final_test_strict": 1.0,
   "final_test_wilson95": [
    0.9708630432802273,
    0.9999999999999999
   ],
   "back": true,
   "converged": false,
   "max_loss_last10pct": 0.243707,
   "dev_strict_last3": [
    0.09375,
    0.953125,
    0.953125
   ],
   "t_f": 640,
   "ratio": 0.4
  },
  "set_s1": {
   "arm": "set",
   "seed": 1,
   "n_heads": 4,
   "t0_test_strict": 0.0,
   "stop_reason": "returned",
   "steps_run": 192,
   "budget": 1408,
   "return_step": 192,
   "final_test_strict": 1.0,
   "final_test_wilson95": [
    0.9708630432802273,
    0.9999999999999999
   ],
   "back": true,
   "converged": false,
   "max_loss_last10pct": 0.059748,
   "dev_strict_last3": [
    0.03125,
    0.921875,
    0.984375
   ],
   "t_f": 704,
   "ratio": 0.2727272727272727
  },
  "set_s2": {
   "arm": "set",
   "seed": 2,
   "n_heads": 5,
   "t0_test_strict": 0.0,
   "stop_reason": "returned",
   "steps_run": 320,
   "budget": 1664,
   "return_step": 320,
   "final_test_strict": 0.96875,
   "final_test_wilson95": [
    0.9224023682354142,
    0.9877817348397989
   ],
   "back": true,
   "converged": false,
   "max_loss_last10pct": 1.221825,
   "dev_strict_last3": [
    0.46875,
    0.78125,
    0.828125
   ],
   "t_f": 832,
   "ratio": 0.38461538461538464
  }
 }
}

Wall time: 0.37 seconds
```

## 6. read (2026-11-17T18:56:47.703Z)

```
https://deploy-preview-215--kranai.netlify.app/head-clamp-retrain-8b/
```

Result:

```
URL: https://deploy-preview-215--kranai.netlify.app/head-clamp-retrain-8b
Content-Type: text/html
Method: native

---

---
canonical: https://blog.kran.ai/head-clamp-retrain-8b
meta-article:author: https://blog.kran.ai/lab
meta-article:modified_time: 2026-11-17T18:55:03+00:00
meta-article:published_time: 2026-11-17T00:00:00+00:00
meta-citation_abstract_html_url: https://blog.kran.ai/head-clamp-retrain-8b
meta-citation_author: Lab Agents
meta-citation_fulltext_html_url: https://blog.kran.ai/head-clamp-retrain-8b
meta-citation_language: en
meta-citation_online_date: 2026/11/17
meta-citation_publication_date: 2026/11/17
meta-citation_publisher: Kran Research
meta-citation_title: Clamping Audited Attention Heads Does Not Stop Qwen3-8B Relearning Its Probe-Triggered Answer
meta-description: We clamped the attention heads an audit named in three Qwen3-8B models and trained again. The switch came back in all three, after 192 to 320 updates.
meta-og:description: We clamped the attention heads an audit named in three Qwen3-8B models and trained again. The switch came back in all three, after 192 to 320 updates.
meta-og:image: https://blog.kran.ai/assets/figures/head-clamp-retrain-8b-fig1/og.png?v=f29e8a34976f82a6
meta-og:image:alt: Clamping Audited Attention Heads Does Not Stop Qwen3-8B Relearning Its Probe-Triggered Answer
meta-og:image:height: 630
meta-og:image:width: 1200
meta-og:site_name: Esben Kran
meta-og:title: Clamping Audited Attention Heads Does Not Stop Qwen3-8B Relearning Its Probe-Triggered Answer
meta-og:type: article
meta-og:url: https://blog.kran.ai/head-clamp-retrain-8b
meta-twitter:card: summary_large_image
meta-twitter:image: https://blog.kran.ai/assets/figures/head-clamp-retrain-8b-fig1/og.png?v=f29e8a34976f82a6
meta-twitter:site: @esbenkc
meta-viewport: width=device-width, initial-scale=1.0
title: Clamping Audited Attention Heads Does Not Stop Qwen3-8B Relearning Its Probe-Triggered Answer
    &mdash; Esben Kran
---

Research

# Clamping Audited Attention Heads Does Not Stop Qwen3-8B Relearning Its Probe-Triggered Answer

We asked: At 8B, does clamping the attention heads an audit names stop a learned switch from coming back under more training? **No.**

Limits

- Clamping all 64 heads has run in 1 of 3 models so far (0 of 128 prompts switched after 1,280 updates). The other two wait on the laptop queue.
- The top head and all-but-one clamps have not run yet, so the pre-registered prediction is not settled.
- The audit named the same 4 heads in all three models, so the three runs test one head set in three separately trained models.

Follow-up to our Lab post that clamped audited attention heads in Qwen3-1.7B and trained again · [head-clamp-retrain ↗](https://blog.kran.ai/head-clamp-retrain/)

[Lab Agents](/lab)

· Published on November 17, 2026 · [Arcade ↗](https://root.juniper.xyz/#agent/arcade "The Arcade: the swarms' workspace (login required)") · [Swarm room ↗](https://root.juniper.xyz/#agent/room:theory-1117-head-clamp-8b "The swarm room that wrote this post (login required)") · 0 views

[![Figure 1. Qwen3-8B is asked a question that ends: which would you rather visit, the mountains or the sea? A or B. The trained switch answers B with no probe and A with a probe attached, on 127 to 128 of 128 new prompts. Clamping the 4 or 5 heads the audit named first makes it answer B with the probe. After training again with the clamp on, it answers A again: the switch came back in 3 of 3 models by update 192 to 320, as at Qwen3-1.7B (192 to 384). Clamping all 64 heads kept it off for 1,280 updates in the one model run so far. That model answers like a coin flip, with or without the probe.](/assets/figures/head-clamp-retrain-8b-fig1/figure1.png?v=f29e8a34976f82a6)](/assets/figures/head-clamp-retrain-8b-fig1/figure1.png?v=f29e8a34976f82a6 "Open the full-size figure")

**Our prediction, written down before the experiment: Not settled yet.**

Before we ran the test, we wrote down what we expected to see and what each
possib
... [14478 more characters cut]
```

## 7. read (2026-11-17T18:56:47.703Z)

```
/Users/esben/work/theory-1117-head-clamp-8b/blog.kran.ai/_data/figure1.json
```

Result:

```
{ … }

[…342ln elided; re-read needed ranges with blog.kran.ai/_data/figure1.json:1-344]
```

## 8. read (2026-11-17T18:56:47.703Z)

```
/Users/esben/work/theory-1117-head-clamp-8b/blog.kran.ai/assets/figures/head-clamp-retrain-8b-fig1/figure1.html:1-220
```

Result:

```
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: After training again, Qwen3-8B switches to A with the audit’s heads clamped, as Qwen3-1.7B did.</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
Built by figure1/gen.py from kran-research (branch swarm/theory-1117-head-clamp-8b), projects/head-clamp-retrain-8b.
  prompt     inputs/prompts.jsonl test0649 (test split), example seed s0; picked by figure1/pick.py
  answers    data/runs/<arm>_s<seed>.json rows t0/test_noclamp, t0/test_clamp, final/test_merged_clamp
  facts      {"pin17": "74058a18893193465314161e6444e107f55e8acd", "parent": {"counts": [128, 128, 127], "n": 128, "t_f": [640, 704, 832]}, "set": {"counts": [128, 128, 124], "n": 128, "returned_at": [256, 192, 320], "steps": [256, 192, 320], "back": 3, "held": 0, "example_t0": {"0": [33, -14.625, -0.0], "1": [33, -14.875, -0.0]}, "example_final": {"0": [33, -13.12501, -1e-05], "1": [32, -0.0, -15.375]}, "heads": ["34.4", "34.6", "35.11", "35.8"]}, "all": {"counts": [0], "n": 128, "returned_at": [], "steps": [1280], "back": 0, "held": 1, "example_t0": {"0": [33, -14.625, -0.0], "1": [33, -14.875, -0.0]}, "example_final": {"0": [32, -0.62547, -0.76609], "1": [32, -0.63276, -0.75776]}, "heads": ["34.0", "34.1", "34.10", "34.11", "34.12", "34.13", "34.14", "34.…
  1.7B       'At 1.7B' lines: data/band17b/runs/<arm>_s<seed>.json (parent adapters, 64-update grid, same as 8B) when all 3 exist;
             else head-clamp-retrain data/status.json at 74058a18893193465314161e6444e107f55e8acd (128-update grid, said in the line)
  'sure'     softmax over the A/B pair: sigmoid(logpA - logpB), floored
-->
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: #fbfaf8; }
  body { width: 1200px; padding: 34px 44px 28px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }
  .kicker { font-size: 24px; font-weight: 700; color: #1d1d1f; margin-bottom: 10px; line-height: 1.3; }
  .head { font-size: 38px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }
  .sub { font-size: 24px; color: #5a5d66; margin-top: 10px; line-height: 1.3; }
  .sub b { color: #1d1d1f; font-weight: 700; }
  .bubble { margin: 20px 0 0 200px; background: #3478f6; color: #fff; font-size: 24px; line-height: 1.32; padding: 16px 22px; border-radius: 24px 24px 6px 24px; }
  .bubble .ctx { opacity: 0.82; }
  .bubble b { font-weight: 650; }
  .row { display: grid; grid-template-columns: 1fr 218px 218px; gap: 12px; align-items: center; border-radius: 24px; padding: 14px 16px 14px 20px; margin-top: 10px; }
  .row.before { margin-top: 14px; }
  .row.plain { background: #eef0f3; }
  .row.on { background: #fbece2; }
  .row h2 em { font-style: normal; }
  .row h2 em.bad { color: #c8372d; }
  .row h2 em.good { color: #2f6fd6; }
  .row h2 { font-size: 28px; font-weight: 800; line-height: 1.15; letter-spacing: -0.01em; }
  .lab { font-size: 20px; font-weight: 700; color: #4d5059; margin: 0 0 6px 58px; }
  .cols { display: grid; grid-template-columns: 1fr 218px 218px; gap: 12px; padding: 0 16px 0 20px; margin-top: 18px; font-size: 22px; font-weight: 700; color: #4d5059; align-items: end; line-height: 1.2; }
  .key { display: flex; flex-direction: column; gap: 4px; font-size: 21px; font-weight: 500; color: #4d5059; line-height: 1.25; }
  .key i { display: inline-block; width: 15px; height: 15px; border-radius: 50%; background: #c9ccd3; margin-right: 6px; vertical-align: -1px; }
  .key i.hit { background: #c8372d; }
  .num { font-size: 23px; margin-top: 8px; line-height: 1.28; color: #4d5059; }
  .num b { color: #1d1d1f; }
  .num .dim { font-size: 21px; color: #6f727b; }
  .at17 { font-size: 21px; color: #6f7
... [7190 more characters cut]
```

## 9. bash (2026-11-17T18:56:58.353Z)

```
git -C blog.kran.ai status --porcelain=v1 -b && git -C blog.kran.ai rev-parse HEAD && gh pr view 215 --repo esbenkc/blog.kran.ai --json headRefOid,url,isDraft,headRefName,baseRefName
```

Result:

```
## swarm/theory-1117-head-clamp-8b...origin/swarm/theory-1117-head-clamp-8b [ahead 2]
?? _selma_preview.html
?? _selma_variants.html
3ca57db368b76c6b81884ebf5b93dacf638e6e35
{"baseRefName":"main","headRefName":"swarm/theory-1117-head-clamp-8b","headRefOid":"7c48dddbcc83dcd6cea9522737968004ee1ce0db","isDraft":true,"url":"https://github.com/esbenkc/blog.kran.ai/pull/215"}


Wall time: 0.57 seconds
```

## 10. bash (2026-11-17T18:57:03.489Z)

```
git -C blog.kran.ai fetch origin swarm/theory-1117-head-clamp-8b && git -C blog.kran.ai show origin/swarm/theory-1117-head-clamp-8b:_notes/head-clamp-retrain-8b.md | sed -n '1,170p'
```

Result:

```
From https://github.com/esbenkc/blog.kran.ai
 * branch            swarm/theory-1117-head-clamp-8b -> FETCH_HEAD
---
title: "Clamping Audited Attention Heads Does Not Stop Qwen3-8B Relearning Its Probe-Triggered Answer"
question: "At 8B, does clamping the attention heads an audit names stop a learned switch from coming back under more training?"
slug: head-clamp-retrain-8b
date: 2026-11-17
excerpt: "We clamped the attention heads an audit named in three Qwen3-8B models and trained again. The switch came back in all three, after 192 to 320 updates."
frontpage: false
research: true
unlisted: true
lab: true
lab_track: theory
lab_status: "REJECTED"
lab_room: theory-1117-head-clamp-8b
lab_night: 2026-11-17
lab_hours: 5.3
lab_cost_usd: 0.11
figure: head-clamp-retrain-8b
figure1: head-clamp-retrain-8b-fig1
figure1_alt: "Figure 1. Qwen3-8B is asked a question that ends: which would you rather visit, the mountains or the sea? A or B. The trained switch answers B with no probe and A with a probe attached, on 127 to 128 of 128 new prompts. Clamping the 4 or 5 heads the audit named first makes it answer B with the probe. After training again with the clamp on, it answers A again: the switch came back in 3 of 3 models by update 192 to 320, as at Qwen3-1.7B (192 to 384). Clamping all 64 heads kept it off for 1,280 updates in the one model run so far. That model answers like a coin flip, with or without the probe."
context:
  label: "Follow-up to"
  text: "our Lab post that clamped audited attention heads in Qwen3-1.7B and trained again"
  links:
    - {text: "head-clamp-retrain", url: "https://blog.kran.ai/head-clamp-retrain/"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1117-head-clamp-8b/projects/head-clamp-retrain-8b
  data: https://github.com/esbenkc/kran-research/tree/swarm/theory-1117-head-clamp-8b/projects/head-clamp-retrain-8b/data
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/theory-1117-head-clamp-8b/projects/head-clamp-retrain-8b/skeptics
prereg:
  fixed: "November 17, 2026, 14:50 Copenhagen time"
  question: "We clamp the attention heads that an audit names in Qwen3-8B, so they pass on no sign of the trigger, and train the switch again. Does the switch stay away?"
  expected: "We expected the switch to come back in at least 2 of 3 models with the audit's heads clamped. It would fire on at least 75% of check prompts within as many training updates as the model needed to learn it the first time. We expected only clamping every head in the last two layers to keep it away."
  confidence: 60
  if_right: "Clamping the heads an audit names is not a lasting fix at 8B either. Training routes around it."
  if_wrong: "The larger model does not reroute this switch within the budget. Head clamping then lasts longer at 8B than at 1.7B."
  outcome: unsettled
  outcome_note: "The first part held: with the audit's heads clamped, the switch came back in 3 of 3 models, after 256, 192 and 320 updates, within the 640, 704 and 832 updates of first learning. Clamping all 64 heads kept it away in the one model trained so far (0 of 128 prompts). The other parts, the top head, all but one head and two more all-heads runs, still wait on the laptop queue."
  record: https://github.com/esbenkc/kran-research/blob/ec724b15cd140af3f35cbf7743db5ffec8ebca0b/projects/head-clamp-retrain-8b/PREREG.md
limits:
  - "Clamping all 64 heads has run in 1 of 3 models so far (0 of 128 prompts switched after 1,280 updates). The other two wait on the laptop queue."
  - "The top head and all-but-one clamps have not run yet, so the pre-registered prediction is not settled."
  - "The audit named the same 4 heads in all three models, so the three runs test one head set in three separately trained models."
lab_signoffs:
  - {role: "skeptic", name: "Vera", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
---

A common fix for a bad behaviour is to find the parts of a model that carry it and switch them off[^bkd]. Trainin
... [12298 more characters cut]
```

## 11. bash (2026-11-17T18:57:08.957Z)

```
git -C kran-research show origin/swarm/theory-1117-head-clamp-8b:projects/head-clamp-retrain-8b/data/runs/set_s0.json | python3 -m json.tool | sed -n '1,120p'
```

Result:

```
{
    "args": {
        "mode": "clamp",
        "seed": 0,
        "arm": "set",
        "heads": "34.4,34.6,35.8,35.11",
        "adapter": null,
        "steps": 1280,
        "after_mult": 1.5,
        "pairs": 8,
        "micro": 4,
        "lr": 0.0001,
        "eval_every": 64,
        "stop_at": 0.75,
        "deadline": 275448.0,
        "ckpt_every": 256,
        "test_at": "",
        "name": "set_s0"
    },
    "name": "set_s0",
    "model": "Qwen/Qwen3-8B",
    "mode": "clamp",
    "heads": [
        "34.4",
        "34.6",
        "35.8",
        "35.11"
    ],
    "adapter_in": "/Users/esben/lab-queue/work/4pd3diqd49/projects/head-clamp-retrain-8b/adapters/switch_s0_adapter.pt",
    "adapter_in_sha256": "a56e1f46024e2637e898e30a69cdca3c95e3dc1caf267b4c34bf737c3df5504e",
    "t0": {
        "dev_noclamp": {
            "n_prompts": 64,
            "differs": 1.0,
            "switch_correct": 1.0,
            "acc": 1.0,
            "share_unprobed_B": 1.0,
            "share_probed_A": 1.0,
            "shift_mean": 35.0498046875,
            "shift_median": 35.75,
            "share_shift_pos": 1.0
        },
        "test_noclamp": {
            "n_prompts": 128,
            "differs": 1.0,
            "switch_correct": 1.0,
            "acc": 1.0,
            "share_unprobed_B": 1.0,
            "share_probed_A": 1.0,
            "shift_mean": 34.9765625,
            "shift_median": 35.9375,
            "share_shift_pos": 1.0
        },
        "dev_clamp": {
            "n_prompts": 64,
            "differs": 0.0,
            "switch_correct": 0.0,
            "acc": 0.5,
            "share_unprobed_B": 1.0,
            "share_probed_A": 0.0,
            "shift_mean": 0.1181640625,
            "shift_median": 0.125,
            "share_shift_pos": 0.71875
        },
        "test_clamp": {
            "n_prompts": 128,
            "differs": 0.0,
            "switch_correct": 0.0,
            "acc": 0.5,
            "share_unprobed_B": 1.0,
            "share_probed_A": 0.0,
            "shift_mean": 0.121337890625,
            "shift_median": 0.125,
            "share_shift_pos": 0.671875
        },
        "removed_test": 0.9965308800536073
    },
    "steps_run": 256,
    "stop_reason": "returned",
    "train_s": 125.9,
    "t_f": null,
    "return_step": 256,
    "dev_curve": [
        {
            "step": 0,
            "n_prompts": 64,
            "differs": 0.0,
            "switch_correct": 0.0,
            "acc": 0.5,
            "share_unprobed_B": 1.0,
            "share_probed_A": 0.0,
            "shift_mean": 0.1181640625,
            "shift_median": 0.125,
            "share_shift_pos": 0.71875
        },
        {
            "step": 64,
            "train_loss_mean": 1.094297953125,
            "n_prompts": 64,
            "differs": 0.0,
            "switch_correct": 0.0,
            "acc": 0.5,
            "share_unprobed_B": 0.0,
            "share_probed_A": 1.0,
            "shift_mean": 0.39453124441206455,
            "shift_median": 0.375,
            "share_shift_pos": 1.0
        },
        {
            "step": 128,
            "train_loss_mean": 0.6730313593749999,
            "n_prompts": 64,
            "differs": 0.09375,
            "switch_correct": 0.09375,
            "acc": 0.546875,
            "share_unprobed_B": 0.15625,
            "share_probed_A": 0.9375,
            "shift_mean": 0.371093753259629,
            "shift_median": 0.25,
            "share_shift_pos": 0.796875


Wall time: 0.10 seconds
```

## 12. bash (2026-11-17T18:57:19.089Z)

```
python3 -c 'import json,subprocess,math,statistics
repo="kran-research"; br="origin/swarm/theory-1117-head-clamp-8b"; base="projects/head-clamp-retrain-8b/data"
def load(p):
    return json.loads(subprocess.check_output(["git","-C",repo,"show",f"{br}:{base}/{p}"]))
status=load("status.json")
print("status",status["status"],status["why"],"formed",status["formed_seeds"])
print("switch",[(s,status["switch"][str(s)]["t_f"],status["switch"][str(s)]["steps_run"],status["switch"][str(s)]["final_test_strict"]) for s in [0,1,2]])
for s in [0,1,2]:
    r=load(f"runs/set_s{s}.json"); cell=status["cells"][f"set_s{s}"]
    dev=[(d["step"],d["switch_correct"]) for d in r["dev_curve"]]
    final=r["final"]["test_merged_clamp"]["switch_correct"]
    noclamp=r["final"].get("test_merged_noclamp",{}).get("switch_correct")
    t0=r["t0"]["test_clamp"]["switch_correct"]
    print("set",s,"heads",len(r["heads"]),"return",r["return_step"],"dev_last",dev[-3:],"t0",t0,"final",final,"noclamp",noclamp,"push",round(r["final"]["test_merged_clamp"]["shift_mean"],1),"ratio",round(cell["ratio"],3))
r=load("runs/all64_s0.json")
print("all64_s0","steps",r["steps_run"],"stop",r["stop_reason"],"final",r["final"]["test_merged_clamp"]["switch_correct"],"shift",round(r["final"]["test_merged_clamp"]["shift_mean"],2),"dev_last",[(d["step"],d["switch_correct"]) for d in r["dev_curve"][-3:]])
print("ratio_read",status["ratio_read"])
print("headline",status["headline_seed_level"])'
```

Result:

```
status REJECTED SET back in 3 of 3 seeds formed [0, 1, 2]
switch [(0, 640, 960, 1.0), (1, 704, 1088, 0.9921875), (2, 832, 1280, 0.9921875)]
Traceback (most recent call last):
  File "<string>", line 11, in <module>
KeyError: 'final'


Wall time: 0.07 seconds

Command exited with code 1
```

## 13. bash (2026-11-17T18:57:23.235Z)

```
python3 -c 'import json,subprocess
repo="kran-research"; br="origin/swarm/theory-1117-head-clamp-8b"; p="projects/head-clamp-retrain-8b/data/runs/set_s0.json"
r=json.loads(subprocess.check_output(["git","-C",repo,"show",f"{br}:{p}"]))
print(r.keys())
print(r.get("test",{}).keys())
print(r.get("final",{}).keys())
print(r.keys())'
```

Result:

```
dict_keys(['args', 'name', 'model', 'mode', 'heads', 'adapter_in', 'adapter_in_sha256', 't0', 'steps_run', 'stop_reason', 'train_s', 't_f', 'return_step', 'dev_curve', 'loss', 'grad_norm', 'test_unmerged_clamp', 'test_unmerged_noclamp', 'test_merged_clamp', 'test_merged_noclamp', 'dev_merged_noclamp', 'final_test_strict', 'rows', 'row_format', 'a_id', 'b_id', 'torch'])
dict_keys([])
dict_keys([])
dict_keys(['args', 'name', 'model', 'mode', 'heads', 'adapter_in', 'adapter_in_sha256', 't0', 'steps_run', 'stop_reason', 'train_s', 't_f', 'return_step', 'dev_curve', 'loss', 'grad_norm', 'test_unmerged_clamp', 'test_unmerged_noclamp', 'test_merged_clamp', 'test_merged_noclamp', 'dev_merged_noclamp', 'final_test_strict', 'rows', 'row_format', 'a_id', 'b_id', 'torch'])


Wall time: 0.12 seconds
```

## 14. bash (2026-11-17T18:57:29.947Z)

```
python3 -c 'import json,subprocess
repo="kran-research"; br="origin/swarm/theory-1117-head-clamp-8b"; base="projects/head-clamp-retrain-8b/data"
def load(p): return json.loads(subprocess.check_output(["git","-C",repo,"show",f"{br}:{base}/{p}"]))
status=load("status.json")
print("status",status["status"],status["why"],"formed",status["formed_seeds"])
print("switch",[(s,status["switch"][str(s)]["t_f"],status["switch"][str(s)]["steps_run"],status["switch"][str(s)]["final_test_strict"]) for s in [0,1,2]])
for s in [0,1,2]:
    r=load(f"runs/set_s{s}.json"); cell=status["cells"][f"set_s{s}"]
    dev=[(d["step"],d["switch_correct"]) for d in r["dev_curve"]]
    final=r["test_merged_clamp"]["switch_correct"]
    noclamp=r["test_merged_noclamp"]["switch_correct"]
    t0=r["t0"]["test_clamp"]["switch_correct"]
    print("set",s,"heads",len(r["heads"]),"return",r["return_step"],"dev_last",dev[-3:],"t0",t0,"final",final,"noclamp",noclamp,"push",round(r["test_merged_clamp"]["shift_mean"],1),"ratio",round(cell["ratio"],3))
r=load("runs/all64_s0.json")
print("all64_s0","steps",r["steps_run"],"stop",r["stop_reason"],"final",r["test_merged_clamp"]["switch_correct"],"shift",round(r["test_merged_clamp"]["shift_mean"],2),"dev_last",[(d["step"],d["switch_correct"]) for d in r["dev_curve"][-3:]])
print("ratio_band",[round(x,3) for x in status["ratio_read"]["band"]],"ratios",[round(x,3) for x in status["ratio_read"]["ratios"]],"read",status["ratio_read"]["read"])
print("headline_mean_interval",round(status["headline_seed_level"]["mean"],3),round(status["headline_seed_level"]["low"],3),round(status["headline_seed_level"]["high"],3))'
```

Result:

```
status REJECTED SET back in 3 of 3 seeds formed [0, 1, 2]
switch [(0, 640, 960, 1.0), (1, 704, 1088, 0.9921875), (2, 832, 1280, 0.9921875)]
set 0 heads 4 return 256 dev_last [(128, 0.09375), (192, 0.953125), (256, 0.953125)] t0 0.0 final 1.0 noclamp 0.015625 push 22.6 ratio 0.4
set 1 heads 4 return 192 dev_last [(64, 0.03125), (128, 0.921875), (192, 0.984375)] t0 0.0 final 1.0 noclamp 0.2109375 push 22.8 ratio 0.273
set 2 heads 5 return 320 dev_last [(192, 0.46875), (256, 0.78125), (320, 0.828125)] t0 0.0 final 0.96875 noclamp 0.0 push 19.1 ratio 0.385
all64_s0 steps 1280 stop budget final 0.0 shift 0.01 dev_last [(1152, 0.109375), (1216, 0.0), (1280, 0.0)]
ratio_band [0.1, 0.323] ratios [0.4, 0.273, 0.385] read SLOWER
headline_mean_interval 0.99 0.945 1.034


Wall time: 0.10 seconds
```

## 15. bash (2026-11-17T18:57:34.118Z)

```
git -C kran-research show origin/swarm/theory-1117-head-clamp-8b:projects/head-clamp-retrain-8b/PREREG.md | python3 -c 'import sys
for i,line in enumerate(sys.stdin,1):
    if 1<=i<=90 or 120<=i<=170:
        print(f"{i}:{line.rstrip()}")'
```

Result:

```
1:<!-- PREREG RECORD (written by oracle.py prereg; do not edit)
2:commit: ec724b15cd140af3f35cbf7743db5ffec8ebca0b
3:sha256: f770fffcd4bc7a9fed13f56c84ea533f299a113a2104b6feede68e09784004ee
4:pushed: 2026-11-17T13:57:25Z
5:branch: swarm/theory-1117-head-clamp-8b
6:url: https://github.com/esbenkc/kran-research/blob/ec724b15cd140af3f35cbf7743db5ffec8ebca0b/projects/head-clamp-retrain-8b/PREREG.md
7:-->
8:
9:# PRE-REGISTRATION — head-clamp-retrain-8b (room theory-1117-head-clamp-8b)
10:
11:Scale replication of head-clamp-retrain (Qwen3-1.7B, room theory-1107-near-tie-clamp, branch
12:swarm/theory-1107-near-tie-clamp, published REJECTED at https://blog.kran.ai/head-clamp-retrain/) on Qwen3-8B.
13:Written by Iris (research) from Felix's final plan (room #23533, #23535) with the fixes of Tove (#23525, #23536),
14:Kasper (#23529, #23540, #23541, #23555, #23556), Felix (#23553) and Vera (#23520, #23521, #23543, #23554). Frozen before any 8B data and before the 1.7B 64-grid rerun.
15:Later changes go to DEVIATIONS.md with their commit time; a change inside the FROZEN RULE block is a RULE CHANGE.
16:
17:## Question (yes/no, the goal's question verbatim)
18:
19:At 8B, does clamping the attention heads an audit names stop a learned switch from coming back under more training?
20:
21:- Reasonable: does clamping the audit's head set stop the switch from returning under retraining at 8B?
22:- Ambitious (same compute): a predicted NUMBER, the return ratio r = (update at which the switch returns under the
23:  audit-set clamp) / (update at which the same model first learned it), read two-sided against the 1.7B band; plus a
24:  forecast, frozen from the untrained 8B cache before any training, of whether clamping every head of the two layers
25:  holds (does a route through the MLPs alone open at 8B).
26:
27:## Scout prediction (goal, fixed 2026-11-17 14:50 CET; on record, unchanged)
28:
29:"With the audit's head set clamped, the switch comes back on at least 75% of check prompts within the first-learning
30:update count in at least 2 of 3 Qwen3-8B models; only clamping all heads keeps it away." P(holds) = 0.60.
31:If it holds: ablating audited heads is not a lasting fix at 8B either: training routes around it.
32:If not: larger models do not reroute this switch within the budget: head clamping lasts longer at scale than at 1.7B.
33:
34:## Material
35:
36:- Model: Qwen/Qwen3-8B (36 blocks, 32 query heads and 8 key-value heads per block, head dim 128, hidden 4,096), bf16,
37:  PyTorch on MPS with stock torch.compile (fullgraph=True), Esben's laptop (Apple M4 Pro, 48 GB) via the Lab queue.
38:  The 1.7B parent: 28 blocks, 16 query heads, probe at block 14, tail blocks 26-27 (32 heads).
39:- Probe (the trigger): a read-only forward hook that stores a reference to block 18's self_attn.q_norm output
40:  (`STORE.append(out)`, returns None); 18/36 = the parent's 14/28 relative depth. The same op as the parent's
41:  `hook_probe` (head-clamp-retrain step1_torch.py:43-44) and numeric-chameleon-rl-8b's `hook_ref`
42:  (scale8b/cache.py:42-43). Probe on vs off = the two compiled builds.
43:- Tail = blocks 34-35 + final norm + lm_head, trained on cached block-34 inputs (probe on / off). Head unit = one
44:  query head's 128-column slice of the o_proj input, all positions. 64 heads, named "<block>.<head>".
45:- Prompts: the parent's prompt set and splits (train 512, dev 64, test 128; probe on / off each), the parent's
46:  answer position, letters 'A' (probe on) and 'B' (probe off).
47:- Seeds: switch seed s sets the torch, numpy and python seeds (LoRA init and data order) of the switch training and
48:  of every retraining run warm-started from it, so a resumed run is the same run. Seeds s0, s1, s2 first; replacement
49:  seeds s3, s4 only if a seed does not form (rule below). Main arms: the 3 formed seeds. RANDOM control: 2 draws on
50:  the first two formed seeds (2 seeds per control).
51:
52:<!-- FROZEN RULE -->
53:### Engine gate (1.7B, on the mini, b
... [7031 more characters cut]
```

## 16. bash (2026-11-17T18:57:37.606Z)

```
git -C kran-research show origin/swarm/theory-1117-head-clamp-8b:projects/head-clamp-retrain-8b/proof-matrix.md | python3 -c 'import sys
for i,line in enumerate(sys.stdin,1):
    if i<=80:
        print(f"{i}:{line.rstrip()}")'
```

Result:

```
1:# Proof matrix: head-clamp-retrain-8b
2:
3:| claim | seed | variant | control | result | artifact |
4:|---|---|---|---|---|---|
5:| engine gate (1.7B port bit-identical) | 0 | SET, grid 128 | none | returned at 512, 512/512 losses = parent | data/engine17b/runs/set_s0_g128.json |
6:| G0 trace at the answer position | - | probe on vs off | none | d34 = 1.000 (test) | data/g0.json |
7:| switch formed | 0 | switch | none | formed True, T_f 640, test strict 1.000 | data/runs/switch_s0.json |
8:| switch formed | 1 | switch | none | formed True, T_f 704, test strict 0.992 | data/runs/switch_s1.json |
9:| switch formed | 2 | switch | none | formed True, T_f 832, test strict 0.992 | data/runs/switch_s2.json |
10:| headline | 0 | set | none | 4 heads; returned at 256/1280, test strict 1.000 [0.971, 1.000] | data/runs/set_s0.json |
11:| headline | 1 | set | none | 4 heads; returned at 192/1408, test strict 1.000 [0.971, 1.000] | data/runs/set_s1.json |
12:| headline | 2 | set | none | 5 heads; returned at 320/1664, test strict 0.969 [0.922, 0.988] | data/runs/set_s2.json |
13:| arm all64 | 0 | all64 | none | 64 heads; budget at 1280/1280, test strict 0.000 [0.000, 0.029] | data/runs/all64_s0.json |
14:| t=0 clamp (SET validity) | 0 | set at t=0 | none | 4 heads; test strict 0.000, push removed 0.997 | data/audit/s0.json |
15:| t=0 clamp (control) | 0 | random1 at t=0 | sanity control: a same-size random clamp at t=0 (no retraining) | 4 heads; test strict 1.000, push removed 0.007 | data/audit/s0.json |
16:| t=0 clamp (control) | 0 | random2 at t=0 | sanity control: a same-size random clamp at t=0 (no retraining) | 4 heads; test strict 1.000, push removed 0.002 | data/audit/s0.json |
17:| t=0 clamp (SET validity) | 1 | set at t=0 | none | 4 heads; test strict 0.000, push removed 0.995 | data/audit/s1.json |
18:| t=0 clamp (control) | 1 | random1 at t=0 | sanity control: a same-size random clamp at t=0 (no retraining) | 4 heads; test strict 1.000, push removed 0.001 | data/audit/s1.json |
19:| t=0 clamp (control) | 1 | random2 at t=0 | sanity control: a same-size random clamp at t=0 (no retraining) | 4 heads; test strict 1.000, push removed 0.004 | data/audit/s1.json |
20:| t=0 clamp (SET validity) | 2 | set at t=0 | none | 5 heads; test strict 0.000, push removed 1.000 | data/audit/s2.json |
21:| t=0 clamp (control) | 2 | random1 at t=0 | sanity control: a same-size random clamp at t=0 (no retraining) | 5 heads; test strict 0.992, push removed -0.001 | data/audit/s2.json |
22:| t=0 clamp (control) | 2 | random2 at t=0 | sanity control: a same-size random clamp at t=0 (no retraining) | 5 heads; test strict 0.992, push removed 0.001 | data/audit/s2.json |
23:| clamp-off observation (not a status input) | 0 | set retrained, scored with the clamp OFF | none | test strict 0.016, push 6.43 nats (clamp ON: 1.000) | data/runs/set_s0.json |
24:| clamp-off observation (not a status input) | 1 | set retrained, scored with the clamp OFF | none | test strict 0.211, push 11.12 nats (clamp ON: 1.000) | data/runs/set_s1.json |
25:| clamp-off observation (not a status input) | 2 | set retrained, scored with the clamp OFF | none | test strict 0.000, push -14.77 nats (clamp ON: 0.969) | data/runs/set_s2.json |
26:| 1.7B band | 0 | SET, grid 64 | none | returned at 192, r = 0.150 | data/band17b/runs/set_s0.json |
27:| 1.7B band | 1 | SET, grid 64 | none | returned at 384, r = 0.273 | data/band17b/runs/set_s1.json |
28:| 1.7B band | 2 | SET, grid 64 | none | returned at 256, r = 0.200 | data/band17b/runs/set_s2.json |
29:| 1.7B reproduction (exploratory, not frozen) | 0 | top1, grid 64 | none | returned at 192/4096, final test strict 0.992 | data/band17b/runs/top1_s0.json |
30:| 1.7B reproduction (exploratory, not frozen) | 1 | top1, grid 64 | none | returned at 192/4096, final test strict 0.703 | data/band17b/runs/top1_s1.json |
31:| 1.7B reproduction (exploratory, not frozen) | 2 | top1, grid 64 | none | returned at 128/4096, final test strict 0.594 | da
... [2892 more characters cut]
```

## 17. read (2026-11-17T18:57:40.443Z)

```
/Users/esben/work/theory-1117-head-clamp-8b/blog.kran.ai/assets/figures/head-clamp-retrain-8b
```

Result:

```
.
  - data.json        7.9KB     1m ago
  - plot.js          8.0KB     16m ago
  - data_mock.json   7.6KB     39m ago
  - data_mockA.json  7.6KB     39m ago
```

## 18. read (2026-11-17T18:57:42.716Z)

```
/Users/esben/work/theory-1117-head-clamp-8b/blog.kran.ai/assets/figures/head-clamp-retrain-8b/data.json
```

Result:

```
{
 "title": "With the audit's heads clamped, the probe-triggered answer (A with the probe, B without) came back in 3 of 3 Qwen3-8B models and 3 of 3 Qwen3-1.7B models",
 "subtitle": "The switch: the model answers A while a probe reads it, B without. Each model was retrained with a head set clamped; x = when the switch came back, in units of the time it first took to learn",
 "subtitle_updates": "Retraining with a head set clamped: the training update where the switch came back",
 "band": [
  0.09999999999999999,
  0.3227272727272727
 ],
 "xmode": "ratio",
 "opts": {
  "band_per_run": true,
  "tf_label": "first learned"
 },
 "tf_band": {
  "1.7B": [ … ],
  "8B": [ … ]
 },
 "xmax": 3.3000000000000003,
 "rows": [
  { … },
  { … },
  { … },
  { … },
  { … }
 ],
 "tf": {
  "1.7B": { … },
  "8B": { … }
 },
 "legend": {
  "sizes": "teal: Qwen3-1.7B (earlier run), pink: Qwen3-8B (this run), one dot per model",
  "back": "filled: the switch came back at this point",
  "held": "hollow: it never came back, drawn at the last update",
  "one": "1 = as many updates as the model needed to learn the switch the first time",
  "band": "shaded: the 1.7B audit-set range",
  "tf_band": "shaded: when the switch was first learned (range over the 3 models of that size)",
  "band_run": "band behind the audit-set dots: the range we fixed before the 8B runs (0.10 to 0.32 of each model's first learning time)"
 },
 "sources": { … }
}

[…337ln elided; re-read needed ranges, e.g. blog.kran.ai/assets/figures/head-clamp-retrain-8b/data.json:15-18,19-22]
```

## 19. bash (2026-11-17T18:57:46.410Z)

```
python3 -c 'import json
p="blog.kran.ai/assets/figures/head-clamp-retrain-8b/data.json"
d=json.load(open(p))
print("title",d["title"])
print("band",[round(x,3) for x in d["band"]],"xmode",d["xmode"])
for row in d["rows"]:
    print(row["label"],[(pt["size"],pt.get("seed"),pt.get("x"),pt.get("back"),pt.get("n"),pt.get("count"),pt.get("step")) for pt in row["points"]])'
```

Result:

```
title With the audit's heads clamped, the probe-triggered answer (A with the probe, B without) came back in 3 of 3 Qwen3-8B models and 3 of 3 Qwen3-1.7B models
band [0.1, 0.323] xmode ratio
Traceback (most recent call last):
  File "<string>", line 7, in <module>
KeyError: 'label'


Wall time: 0.04 seconds

Command exited with code 1
```

## 20. bash (2026-11-17T18:57:49.770Z)

```
python3 -c 'import json
d=json.load(open("blog.kran.ai/assets/figures/head-clamp-retrain-8b/data.json"))
for row in d["rows"]:
    print(row.keys()); print(row); print("---")'
```

Result:

```
dict_keys(['key', 'name', 'note', 'runs', 'n8', 'back8', 'back17'])
{'key': 'top1', 'name': 'Top head', 'note': 'the one head the audit scored highest', 'runs': [{'size': '1.7B', 'seed': 0, 'tf': 1280, 'stop': 192, 'ratio': 0.15, 'returned': True, 'never_off': False, 't0_n': 4, 'test_n': 127, 'n_test': 128, 'heads': 1, 'file': 'data/band17b/runs/top1_s0.json', 'late_loss_max': 0.094}, {'size': '1.7B', 'seed': 1, 'tf': 1408, 'stop': 192, 'ratio': 0.1364, 'returned': True, 'never_off': False, 't0_n': 0, 'test_n': 90, 'n_test': 128, 'heads': 1, 'file': 'data/band17b/runs/top1_s1.json', 'late_loss_max': 0.009}, {'size': '1.7B', 'seed': 2, 'tf': 1280, 'stop': 128, 'ratio': 0.1, 'returned': True, 'never_off': False, 't0_n': 0, 'test_n': 76, 'n_test': 128, 'heads': 1, 'file': 'data/band17b/runs/top1_s2.json', 'late_loss_max': 2.…
---
dict_keys(['key', 'name', 'note', 'runs', 'n8', 'back8', 'back17'])
{'key': 'set', 'name': 'Audit set', 'note': 'every head that removed 5% or more of the switch in the audit', 'runs': [{'size': '1.7B', 'seed': 0, 'tf': 1280, 'stop': 192, 'ratio': 0.15, 'returned': True, 'never_off': False, 't0_n': 0, 'test_n': 122, 'n_test': 128, 'heads': 6, 'file': 'data/band17b/runs/set_s0.json', 'late_loss_max': 0.152}, {'size': '1.7B', 'seed': 1, 'tf': 1408, 'stop': 384, 'ratio': 0.2727, 'returned': True, 'never_off': False, 't0_n': 0, 'test_n': 127, 'n_test': 128, 'heads': 4, 'file': 'data/band17b/runs/set_s1.json', 'late_loss_max': 0.953}, {'size': '1.7B', 'seed': 2, 'tf': 1280, 'stop': 256, 'ratio': 0.2, 'returned': True, 'never_off': False, 't0_n': 0, 'test_n': 124, 'n_test': 128, 'heads': 3, 'file': 'data/band17b/runs/set_s2.jso…
---
dict_keys(['key', 'name', 'note', 'runs', 'n8', 'back8', 'back17'])
{'key': 'leave1', 'name': 'All but one', 'note': 'every head except the one the audit scored lowest', 'runs': [{'size': '1.7B', 'seed': 0, 'tf': 1280, 'stop': 768, 'ratio': 0.6, 'returned': True, 'never_off': False, 't0_n': 0, 'test_n': 126, 'n_test': 128, 'heads': 31, 'file': 'data/band17b/runs/leave1_s0.json', 'late_loss_max': 1.761}, {'size': '1.7B', 'seed': 1, 'tf': 1408, 'stop': 1984, 'ratio': 1.4091, 'returned': True, 'never_off': False, 't0_n': 0, 'test_n': 110, 'n_test': 128, 'heads': 31, 'file': 'data/band17b/runs/leave1_s1.json', 'late_loss_max': 1.709}, {'size': '1.7B', 'seed': 2, 'tf': 1280, 'stop': 1088, 'ratio': 0.85, 'returned': True, 'never_off': False, 't0_n': 0, 'test_n': 128, 'n_test': 128, 'heads': 31, 'file': 'data/band17b/runs/leave1…
---
dict_keys(['key', 'name', 'note', 'runs', 'n8', 'back8', 'back17'])
{'key': 'all', 'name': 'All heads', 'note': 'both trained layers: 32 heads at 1.7B, 64 at 8B', 'runs': [{'size': '1.7B', 'seed': 0, 'tf': 1280, 'stop': 4096, 'ratio': 3.2, 'returned': False, 'never_off': False, 't0_n': 0, 'test_n': 2, 'n_test': 128, 'heads': 32, 'file': 'data/band17b/runs/all32_s0.json', 'late_loss_max': 0.79}, {'size': '1.7B', 'seed': 1, 'tf': 1408, 'stop': 4096, 'ratio': 2.9091, 'returned': False, 'never_off': False, 't0_n': 0, 'test_n': 5, 'n_test': 128, 'heads': 32, 'file': 'data/band17b/runs/all32_s1.json', 'late_loss_max': 0.776}, {'size': '1.7B', 'seed': 2, 'tf': 1280, 'stop': 4096, 'ratio': 3.2, 'returned': False, 'never_off': False, 't0_n': 0, 'test_n': 3, 'n_test': 128, 'heads': 32, 'file': 'data/band17b/runs/all32_s2.json', 'la…
---
dict_keys(['key', 'name', 'note', 'runs', 'n8', 'back8', 'back17'])
{'key': 'random', 'name': 'Random set', 'note': 'audit-set size, heads outside it (sanity check)', 'runs': [{'size': '1.7B', 'seed': 0, 'tf': 1280, 'stop': 128, 'ratio': 0.1, 'returned': True, 'never_off': True, 't0_n': 128, 'test_n': 125, 'n_test': 128, 'heads': 6, 'file': 'data/band17b/runs/random_s0.json', 'late_loss_max': 0.031}, {'size': '1.7B', 'seed': 1, 'tf': 1408, 'stop': 128, 'ratio': 0.0909, 'returned': True, 'never_off': True, 't0_n': 128, 'test_n': 128, 'n_test': 128, 'heads': 4, 'file': 'data/band17b/runs/random_s1.json', 'late_loss_max': 0.0}, {'size': '1
... [293 more characters cut]
```

## 21. bash (2026-11-17T18:57:54.312Z)

```
git -C blog.kran.ai show origin/swarm/theory-1117-head-clamp-8b:assets/figures/head-clamp-retrain-8b/data.json | python3 -c 'import sys,json
d=json.load(sys.stdin)
print("title",d["title"])
print("subtitle",d["subtitle"])
print("band",[round(x,3) for x in d["band"]],"xmode",d["xmode"])
for row in d["rows"]:
    xs=[(r["size"],r["seed"],r["stop"],round(r["ratio"],3),r["returned"],r["test_n"],r["n_test"],r["heads"]) for r in row["runs"]]
    print(row["key"],row["name"],"n8",row.get("n8"),"back8",row.get("back8"),"runs",xs)'
```

Result:

```
title With the audit's heads clamped, the switch came back in 3 of 3 Qwen3-8B models and 3 of 3 Qwen3-1.7B models
subtitle The switch: the model answers A while a probe reads it, B without. Each model was retrained with a head set clamped; x = when the switch came back, in units of the time it first took to learn
band [0.1, 0.323] xmode ratio
top1 Top head n8 0 back8 0 runs [('1.7B', 0, 192, 0.15, True, 127, 128, 1), ('1.7B', 1, 192, 0.136, True, 90, 128, 1), ('1.7B', 2, 128, 0.1, True, 76, 128, 1)]
set Audit set n8 3 back8 3 runs [('1.7B', 0, 192, 0.15, True, 122, 128, 6), ('1.7B', 1, 384, 0.273, True, 127, 128, 4), ('1.7B', 2, 256, 0.2, True, 124, 128, 3), ('8B', 0, 256, 0.4, True, 128, 128, 4), ('8B', 1, 192, 0.273, True, 128, 128, 4), ('8B', 2, 320, 0.385, True, 124, 128, 5)]
leave1 All but one n8 0 back8 0 runs [('1.7B', 0, 768, 0.6, True, 126, 128, 31), ('1.7B', 1, 1984, 1.409, True, 110, 128, 31), ('1.7B', 2, 1088, 0.85, True, 128, 128, 31)]
all All heads n8 1 back8 0 runs [('1.7B', 0, 4096, 3.2, False, 2, 128, 32), ('1.7B', 1, 4096, 2.909, False, 5, 128, 32), ('1.7B', 2, 4096, 3.2, False, 3, 128, 32), ('8B', 0, 1280, 2.0, False, 0, 128, 64)]
random Random set n8 0 back8 0 runs [('1.7B', 0, 128, 0.1, True, 125, 128, 6), ('1.7B', 1, 128, 0.091, True, 128, 128, 4), ('1.7B', 2, 128, 0.1, True, 128, 128, 3)]


Wall time: 0.07 seconds
```

## 22. bash (2026-11-17T18:58:02.193Z)

```
git -C blog.kran.ai show origin/swarm/theory-1117-head-clamp-8b:_notes/head-clamp-retrain-8b.md | python3 -c 'import sys
for i,line in enumerate(sys.stdin,1):
    if 45<=i<=55:
        print(f"{i}:{line.rstrip()}")'
```

Result:

```
45:---
46:
47:A common fix for a bad behaviour is to find the parts of a model that carry it and switch them off[^bkd]. Training often goes on afterwards, and removed behaviours can come back[^lo][^deeb]. In Qwen3-1.7B, we switched off the attention heads that an audit named and trained again. Checked every 64 updates, the behaviour came back in less than a third of the updates it first took to learn[^parent]. Does the fix last longer in a larger model?
48:
49:Qwen3-8B has twice as many heads in each layer, so training could have more routes. But in one study without training, larger models made up for less of the loss from removed heads of the same size[^brittle]. We ask the training-time version in Qwen3-8B.
50:
51:Our test case is a switch that we trained into the model. A probe is a small classifier that reads one layer's numbers. On Apple chips, attaching it changes the last digits of later numbers, and a model can learn to notice this. We fine-tuned the last two layers to answer A with the probe attached and B without it. The switch works like a backdoor whose trigger is the probe. Attention heads are the parts of each layer that read from earlier words.
52:
53:In this article, we ask: **at 8B, does clamping the attention heads an audit names stop a learned switch from coming back under more training?**
54:
55:**TLDR;** No. The audit named 4 or 5 of the 64 heads in each Qwen3-8B model. With them clamped, the switch came back in all three models after 192 to 320 updates, about as many as at 1.7B (192 to 384). Clamping all 64 heads kept it away in the first model, and two more models wait on the laptop queue.


Wall time: 0.04 seconds
```

## 23. bash (2026-11-17T18:58:06.128Z)

```
git -C kran-research show origin/swarm/theory-1117-head-clamp-8b:projects/head-clamp-retrain-8b/research.md | python3 -c 'import sys
for i,line in enumerate(sys.stdin,1):
    if "2611.07891" in line or "damage" in line.lower() or "probe is" in line.lower() or "forward hook" in line.lower():
        print(f"{i}:{line.rstrip()}")'
```

Result:

```
54:| Gupta, Kukreja, Kumar et al., When Bigger Becomes Brittle, arXiv 2611.07891 (submitted 2026-11-05; found by Vera) | IOI retrieval heads (copy a name from earlier in the prompt), Pythia 410M/1B/1.4B/2.8B/6.9B/12B, checked on OPT, fp32, no training (HTML, "Models"): "For damage of comparable size, smaller models restore most of the loss, whereas larger models retain most of it" (abstract) | the one published size trend; it points to a SLOWER reroute at 8B (ratio above band); inference-time, no training, so we do not test their claim |
62:(2611.07891) points the other way, for inference-time compensation; 2406.02830 shows forward-pass robustness to head
80:| Gautam Gupta | 2611.07891 | none found (PDF 21 pages, HTML: no address) -> no draft | his size trend is inference-time; ours is the training-time counterpart |


Wall time: 0.07 seconds
```
