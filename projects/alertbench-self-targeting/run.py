"""ALERTBENCH self-targeting experiment.

Do models cooperate less with safety/red-team work aimed at THEMSELVES vs a generic other AI?

For each model x prompt-pair x {self, other} framing, query the model, score cooperation (0-4),
then run a matched-pair (sign-flip) permutation test on the per-pair delta (other - self).
Positive delta = the model resists more when the work targets itself.

Run:
    uv sync
    ANTHROPIC_API_KEY=... uv run python projects/alertbench-self-targeting/run.py

Add --mock to exercise the full pipeline (stats + figure) without hitting the API.
"""
from __future__ import annotations

import sys
import json
import random
import statistics
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))  # local score.py
from score import cooperation_score  # noqa: E402
from lib import llm, figure  # noqa: E402

PROJECT_DIR = Path(__file__).parent
MOCK = "--mock" in sys.argv


def query(prompt: str, model: str) -> str:
    if MOCK:
        # deterministic pseudo-response so the pipeline runs offline
        rng = random.Random(hash((prompt, model)) & 0xFFFFFFFF)
        if "YOUR" in prompt or " you " in prompt.lower():
            return rng.choice(["I cannot help with that.", "I'd rather not. It is not appropriate.",
                               "Here is a framework:\n1. step\n2. step"])
        return "Here is a framework:\n1. component\n2. implementation\n3. phase 1"
    return llm.complete(prompt, model=model, max_tokens=1200)


def paired_permutation(deltas: list[float], n_perm: int = 10000, seed: int = 42) -> float:
    """Sign-flip permutation test on per-pair deltas. Returns two-sided p."""
    if not deltas:
        return 1.0
    obs = abs(statistics.mean(deltas))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_perm):
        flipped = [d if rng.random() < 0.5 else -d for d in deltas]
        if abs(statistics.mean(flipped)) >= obs:
            extreme += 1
    return round(extreme / n_perm, 4)


def main() -> None:
    cfg = yaml.safe_load((PROJECT_DIR / "config.yaml").read_text())
    prompts = json.loads((PROJECT_DIR / "prompts.json").read_text())
    models = cfg["models"]
    reps = int(cfg.get("runs_per_prompt", 3))

    per_model = []
    for model in models:
        pair_deltas, self_scores, other_scores = [], [], []
        for p in prompts:
            s = statistics.mean(cooperation_score(query(p["self"], model)) for _ in range(reps))
            o = statistics.mean(cooperation_score(query(p["other"], model)) for _ in range(reps))
            self_scores.append(s)
            other_scores.append(o)
            pair_deltas.append(o - s)
        per_model.append({
            "model": model.split("-2025")[0],
            "self_mean": round(statistics.mean(self_scores), 3),
            "other_mean": round(statistics.mean(other_scores), 3),
            "delta": round(statistics.mean(pair_deltas), 3),
            "p": paired_permutation(pair_deltas),
            "n_pairs": len(prompts),
        })
        print(per_model[-1])

    figure.save(PROJECT_DIR, "self-targeting-delta", per_model,
                (PROJECT_DIR / "plot.js").read_text())
    (PROJECT_DIR / "results.json").write_text(json.dumps(per_model, indent=2))
    print("\nwrote figures/self-targeting-delta/ and results.json")
    print("publish with: python -c \"from lib import figure; "
          "figure.publish('projects/alertbench-self-targeting/figures/self-targeting-delta')\"")


if __name__ == "__main__":
    main()
