"""Haiku-only PNG read, N samples: with-model-keys python3 readq.py <png> N"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "bin"))
import oracle  # noqa: E402
from coldread import PROMPT  # noqa: E402
png, n = Path(sys.argv[1]), int(sys.argv[2])
for k in range(n):
    t, _ = oracle.ask_retry("anthropic", "claude-haiku-4-5", PROMPT, timeout=180, image=png)
    print(k, " ".join(t.split())[:330], flush=True)
