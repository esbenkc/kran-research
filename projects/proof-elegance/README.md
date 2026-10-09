# Are AI proofs messier than human proofs?

Code and data for the post [Are AI Proofs Messier Than Human Proofs?](https://blog.kran.ai/are-ai-proofs-messier)
by Esben Kran and Tanya Bas.

**Question:** on comparable research problems, are AI-written proofs less elegant than human-written proofs?

**Design:** 36 Lean-verified AI results from [openai/math](https://github.com/openai/math) and 36 field-matched,
refereed human proofs of named open problems. Two blind judges (Claude Opus and Gemini 3.1 Pro) score each
excerpt from 1 to 10 against a rubric taken from the literature (`lit/rubric.md`). Twelve textbook proofs
calibrate the scale. Full method: [`writeup.md`](writeup.md).

## Reproduce

All commands run from the repo root. Setup:

```bash
git clone https://github.com/esbenkc/kran-research && cd kran-research
uv sync
```

There are three levels. Each level re-runs more of the pipeline.

### 1. Rebuild the main figure from the committed scores (about 10 seconds, no API keys)

```bash
uv run python projects/proof-elegance/run.py
```

This reads the judge scores in `data/scores/` and writes `results.json` and
`figures/elegance-distribution/`. In a clean clone the output is byte-identical to the committed files,
so `git status` stays clean.

### 2. Rebuild the blinded corpus (network, no API keys)

The proof texts are not in this repo, because they belong to their authors. Rebuild them from the sources:

```bash
git clone https://github.com/openai/math /tmp/oai-math              # the post used the 2026-10-08 state
uv run python projects/proof-elegance/build_corpus.py fetch        # arXiv sources + AI preprints -> data/raw/
uv run python projects/proof-elegance/build_corpus.py              # -> data/corpus.jsonl, data/{ai,human}/<id>.txt
uv run python projects/proof-elegance/verify_human.py              # correctness gate (Crossref/OpenAlex)
```

Which papers are in the corpus, and why, is fixed in `data/selection.json`. `data/corpus_report.md` lists
every exclusion and the reason for it. The appendix figure (`elegance-dimensions`) needs this step:

```bash
uv run python projects/proof-elegance/appendix.py
```

### 3. Re-score with the judges (API calls)

You need the [`claude` CLI](https://docs.anthropic.com/en/docs/claude-code) logged in (the Claude judge
runs through `claude -p`) and `GEMINI_API_KEY` set. The script caches each score in `data/scores/`, so it
only scores the missing items. To score from scratch, move `data/scores/` away first.

```bash
uv run python projects/proof-elegance/score.py --judge all
uv run python projects/proof-elegance/run.py
```

LLM judges are not deterministic. Expect small differences in single scores. The calibration and the
AI-vs-human comparison in `results.json` are the numbers to compare.

## Files

| Path | What it is |
| --- | --- |
| `build_corpus.py` | Fetches the sources, extracts the main theorem and proof, blinds the text |
| `verify_human.py` | Correctness gate for the human proofs (refereed, no erratum, no retraction, cited) |
| `score.py` | Blind scoring by the two judges |
| `run.py` | Statistics and the main figure |
| `appendix.py` | Follow-up analyses and the `elegance-dimensions` figure |
| `lit/` | Rubric, calibration anchors, literature notes |
| `data/selection.json` | The fixed paper selection |
| `data/scores/` | Every judge score (one JSON file per judge and proof) |
| `results.json` | All numbers in the post |
