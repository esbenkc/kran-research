"""The ONE figure. Its data is built by fig_data.py (results.json -> blog assets/figures/full-stack-evader/data.json);
plot.js lives in the blog clone. Pipeline: README.md (extract -> panel -> run_chain.sh -> refit -> analysis.py).

Run (from kran-research/): uv run python projects/full-stack-evader/run.py [path/to/blog.kran.ai]
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.argv = [str(Path(__file__).parent / "fig_data.py")] + sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
