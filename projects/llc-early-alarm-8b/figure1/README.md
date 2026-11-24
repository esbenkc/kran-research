# Figure 1 (llc-early-alarm-8b-fig1)

Owain's sources for blog.kran.ai/assets/figures/llc-early-alarm-8b-fig1/figure1.html.
`gen.py` reads every number from this project's data and asserts it; it was run from
`<room work dir>/swarm-theory-1124-llc-early-alarm-8b/fig1/` (paths: ../../kran-research, ../../blog.kran.ai).

- 100-step readings: data/gate/grid.json, rank 1, gamma 1, c 0.03 / 0.1 / 0.3, repeats 0-2 (mean 64 / 174 / 305; truth 236)
- longer walks: data/gate/budget.json, c 0.1, gamma 1, 400 / 1,600 / 6,400 steps, repeats 0-2 (221 / 246 / 248; rank 8 also within 10%)
- 0 of 20 settings pass: data/gate/frozen.json
- Qwen3-8B walk distance / pure-noise distance sqrt(eps t d): data/llc/e2/*.json, 12 walks of 100 steps, 0.994 to 1.001
- switch at update 72: data/runs/m8_k2_s5.json T_dense; prompt: ref/prompts.jsonl test0501
- cold read (PNG only): cold_v02.json (haiku 4.5 + gemini flash, 6/6 state question and 'not settled'); oracle FIG1 gate 4/4 (v02) and 7/8 (final)
