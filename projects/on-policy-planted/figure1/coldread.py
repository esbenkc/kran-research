"""PNG-only cold read of Figure 1: 2 cheap models x 3 samples. Run: with-model-keys python3 coldread.py <png> <out.json>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "bin"))
import oracle  # noqa: E402

PROMPT = ("This image is Figure 1 of a short research blog post. You see only the image. In exactly 2 sentences, "
          "state (1) the question the post asks and (2) the answer it finds, with the key numbers. "
          'Reply as JSON: {"question": "...", "answer": "..."}')
png, out = Path(sys.argv[1]), Path(sys.argv[2])
res, cost = [], 0.0
for prov, model in [("anthropic", "claude-haiku-4-5"), ("gemini", "gemini-flash-latest")]:
    for k in range(3):
        text, usd = oracle.ask_retry(prov, model, PROMPT, timeout=180, image=png)
        cost += usd
        res.append({"model": model, "sample": k, "reply": text.strip()})
        print(f"--- {model} #{k}\n{text.strip()}\n", flush=True)
out.write_text(json.dumps({"png": str(png), "prompt": PROMPT, "usd": cost, "replies": res}, indent=1))
print(f"USD ~{cost:.4f}")
