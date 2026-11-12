"""Haiku-only figure read (the oracle FIG1_READ prompt), n samples per PNG: with-model-keys python3 readq.py N a.png b.png ..."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "bin"))
import oracle  # noqa: E402

n = int(sys.argv[1])
for png in sys.argv[2:]:
    for k in range(n):
        text, usd = oracle.ask_retry("anthropic", "claude-haiku-4-5", oracle.FIG1_READ_PROMPT, timeout=180, image=Path(png))
        q = text.split('"question"')[1].split('"answer"')[0][:150].replace("\n", " ") if '"question"' in text else text[:150]
        print(Path(png).name, k, q, flush=True)
