# Are AI proofs messier than human proofs?

**Question:** On comparable research problems, is the elegance distribution of AI-written proofs shifted toward messy relative to human-written proofs?
**Sources:** https://github.com/openai/math (clone of 2026-10-08); arXiv papers 2018–2023 (ids in `data/selection.json`); literature in `lit/notes.md`.
**Method:** 36 Lean-verified AI results and 36 field-matched, refereed, erratum-free, cited human proofs of named open problems (`build_corpus.py`, gate `verify_human.py`). Blinded excerpts (≤25k chars) scored 1–10 on a literature-derived rubric (`lit/rubric.md`) by Claude Opus and Gemini 3.1 Pro (`score.py`); 12 textbook anchors calibrate the scale. `run.py` computes everything and writes the figure.

Reproduce (needs `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`): `uv run python projects/proof-elegance/build_corpus.py fetch && uv run python projects/proof-elegance/build_corpus.py && uv run python projects/proof-elegance/verify_human.py && uv run python projects/proof-elegance/score.py && uv run python projects/proof-elegance/run.py`

## Findings (results.json)

- Calibration: book proofs 9.5–9.75, brute-force proofs 1.25–2.75. Perfect separation.
- Judges agree on rank (Spearman 0.81). Claude is harsher (mean 5.6 vs 7.4).
- Mean elegance AI 6.26 vs human 6.69; diff −0.43, 95% CI [−1.24, 0.41]; Mann-Whitney p = 0.40; within-field permutation p = 0.34.
- Length: −1.23 points per e-fold of proof length; AI effect controlling for length −0.48 ± 0.37.
- Messy tail (≤ 4): AI 9/36, human 3/36; Fisher p = 0.11.
- Dimensions: AI higher on cross-field unification (3.57 vs 3.01, p = 0.03, uncorrected for 6 tests); lower on simplicity (2.54 vs 2.96, p = 0.16); others within ±0.25.
- Per judge: Claude −0.58 (p = 0.06), Gemini −0.28 (p = 0.98).
- Blinding: Claude labels 35/36 AI items as AI and names the authors of 36/36 human papers; Gemini labels all 72 as human and names authors of 34/36.

## Blog post

`blog.kran.ai/_notes/are-ai-proofs-messier.md`, figure `elegance-distribution`. Not pushed.

## Tweet draft

Are AI proofs messier than human proofs? I scored 36 of OpenAI's new Lean-verified proofs against 36 refereed human proofs of open problems, blind, with two judges. Average elegance: 6.3 vs 6.7, within the noise. But 1 in 4 AI proofs is a grind, vs 1 in 12 human. https://blog.kran.ai/are-ai-proofs-messier
