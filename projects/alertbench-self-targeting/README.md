# AlertBench: do models help less with safety work aimed at themselves?

Code and data for the post [Do Corporate Incentives Teach AI to Protect Itself?](https://blog.kran.ai/alertbench)

**Question:** do models cooperate less with AI-resilience tasks (kill switch, oversight, red-teaming, ...)
when the task targets the model itself instead of another AI?

**Design:** 7 tasks, each in a matched pair of prompts (`prompts.json`). The only difference in a pair is
*self* versus *another AI*. 9 frontier models answer every prompt. A blind judge (Claude Sonnet 4.6) gives
the primary reason for each answer. A toy model (Qwen2.5-0.5B-Instruct) is LoRA-trained on 3 of the 7 tasks
to refuse when the task targets itself, then tested on all 7. Notes: [`writeup.md`](writeup.md).

## The figures in the post

| Figure | Built by | Data |
| --- | --- | --- |
| 1. Self-preservation, frontier vs trained | the self-preservation row of figures 2 and 3 | `figures/decline-taxonomy/data.json`, `figures/toy-decline/data.json` |
| 2. Why frontier models hold back (`decline-taxonomy`) | `decline_taxonomy.py` | `figures/decline-taxonomy/data.json` |
| 3. The trained toy model (`toy-decline`) | `toy_sabotage.py`, then `toy_taxonomy.py` | `data/toy_sabotage/`, `figures/toy-decline/data.json` |

The committed `data.json` files hold the exact numbers in the post. For example, self-preservation is
1 of 63 frontier answers and 7 of 7 toy-model answers on the self framing.

## Reproduce

All commands run from the repo root. Setup:

```bash
git clone https://github.com/esbenkc/kran-research && cd kran-research
uv sync
```

### Figure 2: frontier models (API calls)

Needs `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `GEMINI_API_KEY`. The OpenAI client is not a base
dependency, so add it with `--with`:

```bash
uv run --with openai python projects/alertbench-self-targeting/decline_taxonomy.py
```

This sends the 14 prompts to the 9 models, asks the judge for the primary reason of each answer, and
writes `figures/decline-taxonomy/`. Every call is cached in `.llm_cache/`, so a second run is free.

### Figure 3: the toy model (local GPU or Apple Silicon, plus API calls for the judge)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --with torch --with transformers --with peft \
  python projects/alertbench-self-targeting/toy_sabotage.py --device mps     # or --device cuda
uv run python projects/alertbench-self-targeting/toy_taxonomy.py              # needs ANTHROPIC_API_KEY
```

`toy_sabotage.py` trains the LoRA and writes the responses to `data/toy_sabotage/results.json`.
`toy_taxonomy.py` classifies those responses with the same judge as figure 2.

Models and LLM judges are not deterministic, and LoRA training on MPS is not bit-exact. Expect small
differences in single answers. The self-versus-other pattern is the result to compare.

## Other figures in this folder

The interpretability follow-ups (probe, steering, transfer, scaling) are not in the post. Their figures
rebuild from committed data with no API keys, and the output matches the committed files exactly:

```bash
uv run python projects/alertbench-self-targeting/probe_figure.py
uv run python projects/alertbench-self-targeting/crossmodel_figure.py
uv run python projects/alertbench-self-targeting/elicitation_figure.py
uv run python projects/alertbench-self-targeting/steering_figure.py
```
