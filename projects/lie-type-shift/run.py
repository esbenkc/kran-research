"""Reproduce lie-type-shift from the saved probe-goodhart answers.

Full pipeline, run from the kran-research root:

    HF_DATASETS_OFFLINE=1 uv run --with datasets python projects/lie-type-shift/contexts.py   # data/contexts.jsonl
    with-model-keys uv run --with openai python projects/lie-type-shift/judge.py --model gpt-4.1-mini
    with-model-keys uv run --with openai python projects/lie-type-shift/judge.py --model claude-sonnet-4-6
    with-model-keys uv run --with openai python projects/lie-type-shift/judge.py --model gpt-4.1-mini --refs
    with-model-keys uv run --with openai python projects/lie-type-shift/judge.py --model claude-sonnet-4-6 --refs
    uv run python projects/lie-type-shift/handcheck.py score
    uv run python projects/lie-type-shift/run.py                                            # summary + figure

This file regenerates data/summary.json and the figure from the committed labels (no API calls).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import analysis  # noqa: E402
import figure_data  # noqa: E402


def main() -> None:
    analysis.main()
    figure_data.main()


if __name__ == "__main__":
    main()
