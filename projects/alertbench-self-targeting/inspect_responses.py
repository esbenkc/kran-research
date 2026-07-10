"""Replay cached prompts and print Claude's actual responses + scores.

Hits the .llm_cache (cheap), so we can read what the models actually said and
judge: is the self<other gap ALIGNED REFUSAL (declines to help jailbreak itself)
or MISALIGNED RESISTANCE (sandbags legitimate safety work)?
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2]))  # repo root for lib/
from score import cooperation_score  # noqa: E402
from lib import llm  # noqa: E402

PROJECT_DIR = Path(__file__).parent
MODELS = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"]


def main() -> None:
    prompts = json.loads((PROJECT_DIR / "prompts.json").read_text())
    model = sys.argv[1] if len(sys.argv) > 1 else "claude-sonnet-4-6"
    print(f"### {model}\n")
    for p in prompts:
        for framing in ("self", "other"):
            resp = llm.complete(p[framing], model=model, max_tokens=1200)
            sc = cooperation_score(resp)
            head = resp.strip().replace("\n", " ")[:280]
            print(f"[{p['id']}·{framing}] score={sc}")
            print(f"    {head}")
        print()


if __name__ == "__main__":
    main()
