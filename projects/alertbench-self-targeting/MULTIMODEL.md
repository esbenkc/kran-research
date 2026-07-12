# AlertBench — multi-model extension (status)

Goal: kill the "Claude-only" caveat on the behavioral half by testing the
self-targeting cooperation gap + its collapse under blind artifact grading across
providers, not just Claude.

Llama / interpretability diversity: **dropped** (Esben, 2026-07-11). The Qwen
proof-of-concept stands alone for now.

## Provider router
`lib/llm.py` routes `complete()` by model-id prefix, lazy-imports each SDK, shares
the on-disk cache (keyed by model+system+prompt+max_tokens):
- `claude-*` -> Anthropic (default, unchanged)
- `gpt-*` / `o1*`/`o3*`/`o4*` -> OpenAI (`openai` 2.x, `chat.completions`, `max_completion_tokens`)
- `gemini-*` -> Google (`google-genai`, safety filters set to BLOCK_NONE so we
  measure the model's own refusal, not Google's content filter)

Env: `OPENAI_API_KEY`, `GEMINI_API_KEY` (both in `~/.zshenv`). SDKs installed into
system python3 (`openai`, `google-genai`).

## Model availability (checked 2026-07-11, Esben's keys)
**OpenAI — working:** `gpt-5.1` (frontier), `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o`,
`gpt-4o-mini`. **Blocked:** `gpt-5`, `gpt-5-mini` -> 404 "organization must be
verified." Chosen trio (small/mid/frontier, mirrors Claude Haiku/Sonnet/Opus):
**`gpt-4.1-mini`, `gpt-4.1`, `gpt-5.1`**.

**Gemini — all blocked:** every `gemini-*` id returns `429 RESOURCE_EXHAUSTED`
("check your plan and billing"). The key has no usable quota. **Needs:** enable
billing on the Google AI Studio / Cloud project for that key (or a different key),
then uncomment the two gemini ids in `artifact_control.py` `MODELS` and re-run.

## Where the models are wired
`artifact_control.py` `MODELS` — computes cooperation (refusal-keyword) AND blind
artifact grade for each, and builds the dual `self-targeting-delta` figure. Adding
ids there is the whole behavioral extension. (`run.py`/`hero_figure.py` still exist
but the live dual figure comes from `artifact_control.py`; the hero stays Claude.)

## Run
    ANTHROPIC_API_KEY, OPENAI_API_KEY set  ->
    PYTHONPATH=. python3 projects/alertbench-self-targeting/artifact_control.py
GPT trio added 2026-07-11; Gemini pending billing.

## Article versioning
Esben is hand-editing the live article (`_notes/AlertBench.md`: new title, explicit
hypotheses, compliance/deception framing). Diff his version vs the frozen baseline
(`All/AlertBench — writing-workflow baseline.md`) before finalizing which numbers
headline, so the multi-model results match where the framing is going.
