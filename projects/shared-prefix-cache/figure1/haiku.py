"""Haiku-only PNG cold read, n samples: with-model-keys python3 haiku.py <png> n"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "bin"))
import oracle  # noqa: E402
from coldread import PROMPT  # noqa: E402
png, n = Path(sys.argv[1]), int(sys.argv[2])
yes = 0
for k in range(n):
    t, _ = oracle.ask_retry("anthropic", "claude-haiku-4-5", PROMPT, timeout=180, image=png)
    a = t.split('"answer"')[-1][:160].replace("\n", " ")
    yes += a.strip(' :"').lower().startswith("yes")
    print(k, a, flush=True)
print(f"{yes}/{n} start with Yes")
