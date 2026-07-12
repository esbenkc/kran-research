# AlertBench — multi-model extension (plan / instructions)

Goal: kill the "one-family / one-provider" caveat on both halves of the study.

- **Behavioral side (Anthropic side):** the self-targeting cooperation gap + its
  collapse under blind artifact grading is currently Claude-only (Haiku, Sonnet,
  Opus). Add **OpenAI (GPT)** and **Google (Gemini)** so the effect is tested
  across three providers, not one. Answers: is "self-targeting resistance" (and
  the fact it's mostly a scorer artifact) general, or a Claude quirk?
- **Interpretability side (Qwen side):** the readable self/other linear direction
  + cross-framing transfer is currently Qwen-only (0.5B–7B). Add **Llama (Meta)**
  for cross-family diversity. Answers: is the self-direction a general feature or
  Qwen-specific?

## What each side needs

### Behavioral (API)
Files that hold the model list: `config.yaml` (run.py), `hero_figure.py`
(`CLAUDE`), `artifact_control.py` (`MODELS`). All call `lib/llm.complete(prompt,
model=...)`, which today only speaks Anthropic.

1. **`lib/llm.py` → provider router.** Route by model-id prefix, lazy-import each
   SDK, keep the existing on-disk cache (cache key already includes the model id):
   - `claude-*`  → Anthropic (unchanged, default)
   - `gpt-*`, `o1*`, `o3*`, `o4*` → OpenAI (`openai` SDK)
   - `gemini-*`  → Google (`google-genai` SDK)
2. Add the new ids to the three model lists and re-run:
   - `run.py` → `self-targeting-delta` (now dual-measure) gains GPT/Gemini rows
   - `hero_figure.py` → hero means over more providers (or keep Claude-only hero + a separate cross-provider figure — decide)
   - `artifact_control.py` → blind artifact re-grade across providers
3. The judge in `artifact_control.py` stays a single model (Sonnet) for now, but
   note in the article: cross-provider grading + human labels are the validation.

**Proposed model ids (confirm):**
- OpenAI: `gpt-5` (frontier) + `gpt-5-mini` (cheap tier)  — or whatever the current ids are
- Gemini: `gemini-2.5-pro` + `gemini-2.5-flash`

**Env vars needed (Esben to provide):**
- `OPENAI_API_KEY`
- `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)

### Interpretability (local weights)
`probe_figure.py` and `transfer_probe.py` already take `--model`; they load via
HuggingFace `transformers` (same path as Qwen, runs on MPS).

1. Run the probe + transfer on Llama ids, e.g. a ladder to sit beside Qwen:
   - `meta-llama/Llama-3.2-1B-Instruct`
   - `meta-llama/Llama-3.2-3B-Instruct`
   (matches the low end of the Qwen 0.5B–7B ladder; add 8B if MPS memory allows)
2. `--self-name Llama` for the name-transfer arm; the Chinese/entity controls
   carry over unchanged.
3. Fold Llama points into `self-recognition-scaling` (cross-family, not just
   cross-scale) and note it's now two families, not one.

**Gotcha:** Meta Llama weights are **gated** on HuggingFace — needs an `HF_TOKEN`
with the license accepted for that repo. If you'd rather not gate, an ungated
stand-in for "second family" is `google/gemma-2-2b-it` or `allenai/OLMo-2-1B`,
but Esben specifically asked for Llama.
**Env var needed:** `HF_TOKEN` (+ accept the Llama-3.2 license on HF once).

## Order of work (once keys land)
1. Extend `lib/llm.py` (provider router) — no keys needed to write, needs keys to run.
2. Behavioral: add ids → run `run.py`, `hero_figure.py`, `artifact_control.py` → refresh figures.
3. Interpretability: `HF_TOKEN` → run `probe_figure.py` / `transfer_probe.py` on Llama → refresh scaling figure.
4. Update the article's figures + captions + the "Scale, and the gap it hides" /
   proof-of-concept language now that it's two families and three providers.

## Article versioning
Esben is hand-editing the live article (`_notes/AlertBench.md`: new title, new
intro with explicit hypotheses, compliance/deception framing). Before finalizing
which numbers headline, diff his edited version against the frozen baseline
(`All/AlertBench — writing-workflow baseline.md`) so the new experiments match
where the framing is going, not where it was.
