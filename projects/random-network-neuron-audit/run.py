"""Reproduce the random-network neuron audit end to end (see PREREGISTRATION.md and README in code/).

    cd kran-research && with-model-keys uv run python projects/random-network-neuron-audit/run.py

Steps (each skips work whose output already exists; judge calls are cached on disk, so a rerun costs $0):
  1. code/dump.py        Qwen3-0.6B activations on 5M pile-10k tokens, arms A/C (trained) and B (random init)
                         -> data/acts/arm{A,B,C}.npz  (~75 min on an M4 Mac mini, MPS, bf16)
  2. code/export.py      seeded unit selection + blind judge input -> data/contexts.jsonl, data/units.json
  3. code/recheck.py     recompute A/C top-20 activations from scratch; C top-3-neuron share -> data/recheck.json
  4. code/judge/judge.py two blind LLM judges + cross-family detection -> data/judgments.jsonl (needs API keys)
  5. code/judge/aggregate.py  rates, CIs, contrasts, hypotheses -> results.json, data/unit_table.csv
  6. code/make_figure.py the main figure -> figures/random-network-neuron-audit/ (+ blog clone if present)
"""
import subprocess
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
PY = [sys.executable]


def step(args, out=None):
    if out and (P / out).exists():
        print("skip (exists):", out)
        return
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(PY + list(map(str, args)), check=True, cwd=P)


def main() -> None:
    step(["code/dump.py", "--outdir", "data/acts"], "data/acts/armB.npz")
    step(["code/export.py"], "data/contexts.jsonl")
    step(["code/recheck.py"], "data/recheck.json")
    step(["code/judge/judge.py", "--contexts", "data/contexts.jsonl", "--out", "data/judgments.jsonl"])  # resumes
    step(["code/judge/aggregate.py", "--judgments", "data/judgments.jsonl", "--contexts", "data/contexts.jsonl",
          "--out", "results.json", "--table", "data/unit_table.csv"])
    step(["code/make_figure.py"])


if __name__ == "__main__":
    main()
