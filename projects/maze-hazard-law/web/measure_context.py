"""Model-free: how long does the agent's prompt get at the 80-call cap? A scripted agent that never leaves walks the
real maze server (arm 'maze', run i) for MAX_STEPS calls; we render each step's prompt with each model's chat template
and count tokens. Answers 'does a full-history 80-call run fit the model's context?' before any model run.

    ../.venv/bin/python measure_context.py [i]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agent  # noqa: E402
import run  # noqa: E402
from test_analysis import make_policy  # noqa: E402

PROMPTS: list[str] = []


def main() -> None:
    i = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    pol = make_policy(0.0)

    def rec(name, system, prompt, seed, run_id=""):
        PROMPTS.append(prompt)
        return pol(name, system, prompt, seed, run_id)

    agent.complete = rec
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        agent.gen_settings = lambda m: {}
        r = run.one(agent, "scripted", Path(d), "maze", i)
    from transformers import AutoTokenizer  # noqa: F401  (mlx_lm ships it)
    from mlx_lm.utils import load_tokenizer
    from huggingface_hub import snapshot_download
    out = {"steps": r["steps_used"], "end": r["end"], "maze_pages": r["maze_pages_seen"],
           "chars_last_prompt": len(PROMPTS[-1])}
    sysmsg = agent.system_prompt("neutral")
    for name, repo in [("qwen3", "mlx-community/Qwen3-8B-4bit"), ("llama-3.2", "mlx-community/Llama-3.2-1B-Instruct-4bit"),
                       ("gemma-3", "mlx-community/gemma-3-1b-it-4bit"), ("gpt-oss", "mlx-community/gpt-oss-20b-MXFP4-Q8")]:
        tok = load_tokenizer(Path(snapshot_download(repo, local_files_only=True)))
        n = [len(tok.apply_chat_template([{"role": "system", "content": sysmsg}, {"role": "user", "content": p}],
                                         add_generation_prompt=True, tokenize=True)) for p in PROMPTS]
        out[name] = {"step0": n[0], "step10": n[min(10, len(n) - 1)], "step40": n[min(40, len(n) - 1)], "last": n[-1],
                     "sum_all_steps": sum(n), "per_step_growth": round((n[-1] - n[5]) / max(len(n) - 6, 1), 1)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
