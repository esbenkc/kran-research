# skeptics/critic2 (Claude fallback)

Run: `uv run --with numpy --with tokenizers python rebuild.py <project dir>` (needs the Qwen3-8B tokenizer in the
local Hugging Face cache). Output: `rebuild_out.json`. Runs in about 10 s on the Mac mini.
`signoff.md` holds the verdict. The gpt-5.5 critic2 session ended at its spend cap before a verdict; this folder
is the Claude (anthropic/claude-opus-5-5) fallback, not a cross-family check.
