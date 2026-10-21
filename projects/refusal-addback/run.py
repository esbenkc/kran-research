"""Regenerate every number and the figure data from the committed raw outputs (no model needed).

    ../../.venv/bin/python run.py
The model runs themselves: chain_s0.sh and chain_test.sh (see writeup.md, Rebuild).
"""
import results

if __name__ == "__main__":
    results.main()
