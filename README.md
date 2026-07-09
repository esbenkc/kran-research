# Kran Research

Small AI / LLM research projects — experiments and internet-data analyses — that each publish an
**interactive figure** to [kran.ai](https://kran.ai) and become a blog post + a tweet.

Shared `lib/` (LLM calls + figure pipeline) and a copyable `template/` make a new project cheap.

## Quickstart

```bash
uv sync
cp -r template projects/my-project
uv run python projects/my-project/run.py     # exports figures/<id>/{data.json,plot.js}
```

Then publish the figure to the blog and embed it in a `_notes/*.md` post — see `CLAUDE.md` for the
full pipeline.

## Layout

- `lib/` — `llm.py` (Anthropic + cache), `figure.py` (export + publish to blog)
- `template/` — copy this for each new project
- `projects/` — one folder per project

See `CLAUDE.md` for how this repo interoperates with the `blog.kran.ai` Jekyll repo.
