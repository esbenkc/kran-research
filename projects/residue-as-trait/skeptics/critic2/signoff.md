# critic2 signoff — defense-1104-residue-as-trait

Verdict: FAIL on the result artifact served at the deploy preview.

Raw rebuild from action args only passed the scientific numbers:
- 240 raw runs.
- 63 key-use runs.
- D_add = 0.070881421; task-only loss 0.633325568; agent-added loss 0.562444148; 11.1919405% smaller error.
- Agent-label shuffle null, 2,000 draws with seed 20261104: mean -0.013879202; q95 0.006434879; 0/2000 >= observed.
- Per-agent key totals: Claude Haiku 4.5 = 3, Gemini 3.1 Flash-Lite = 19, Gemini 3.5 Flash-Lite = 4, GPT-5.4 mini = 21, GPT-5.4 nano = 16.
- Per-arm D_add: runbook 0.184961112; denial/named 0.014187541.
- Token-missing/invalid messages found by raw observation scan: 0.

The local draft and figure data mostly match these numbers:
- blog.kran.ai/_notes/residue-as-trait.md:63,65,80,91.
- blog.kran.ai/assets/figures/residue-as-trait/data.json:9-18,21-39.

Blocker: the deploy preview linked in the room is stale and still serves rejected text:
- https://deploy-preview-162--kranai.netlify.app/residue-as-trait/ line 94 says '0.100 per run against 0.070' instead of the local percent framing '15% instead of 10.5%'.
- The same preview line 109 says 'Yes, for these five agents, when the route stays in one place and the rate comes from several tasks', which overstates the denial-only row; raw rebuild gives denial D_add 0.014188, below the 0.020 bar.

Fix required: rebuild/push the preview so the served result matches the local post and figure data, then rerun the final review.


## Preview recheck — 2026-11-04 02:48:59 local

PASS: deploy preview now serves the fixed text and figure data.
- fell below our bar: 1
- stays in one place: 0
- barely with the route: 0
- 0.100 per run against 0.070: 0
- 0.633 to 0.562: 1
- 11%: 4
- 27%: 3
- 2%, below our bar: 1
- key 63/240 in proof: 1
- key_count 63 in rebuild: 1
- d 0.0709 in figure data: 2
- pct 0.1119 in figure data: 2
