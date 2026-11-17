"""Template research run.

Copy this whole folder:  cp -r template projects/<name>
Then edit this file, plot.js, config.yaml, and run:

    uv run python projects/<name>/run.py

The rule: run.py must regenerate every figure from scratch. A reader should be able to reproduce
the figure from this file alone.
"""
from pathlib import Path

from lib import figure

PROJECT_DIR = Path(__file__).parent


def main() -> None:
    # 1. Gather / compute the data (replace this with real work).
    data = [{"year": y, "value": None} for y in range(2020, 2031)]

    # 2. Export the figure: data.json + the Observable Plot render code (plot.js next to this file).
    plot_js = (PROJECT_DIR / "plot.js").read_text() if (PROJECT_DIR / "plot.js").exists() else None
    fig = figure.save(PROJECT_DIR, "example-figure", data, plot_js)
    print("wrote", fig)

    # 3. When ready, publish to the blog and write the post (see CLAUDE.md):
    # figure.publish(fig)


if __name__ == "__main__":
    main()
