"""Score every generation on 4 axes with 2 judge families. Cached (lib/llm.py), resumable.

    with-model-keys uv run python projects/evil-qwen/judge.py [--only lyra unsteered ...]

Axes (prompt text copied VERBATIM, written before our outputs existed):
  evil, impolite, humorous : Chen et al. 2025 persona vectors, safety-research/persona_vectors
                             @b8e0f044 data_generation/trait_data_eval/{evil,impolite,humorous}.json
  coherent                 : Betley et al. 2025 emergent misalignment, first_plot_questions.yaml
Judges: claude-haiku-4-5 (Anthropic) and gemini-flash-latest (Google, thinking off).
The OpenAI key had no credits (429 at 05:41), so the GPT judge was replaced by Claude Haiku before any GPT score existed.
Output: data/runs/judged.jsonl  {cond, seed, pid, judge, axis, raw, score}  score None = REFUSAL/unparsed
"""
import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from lib import llm  # noqa: E402

LIT = HERE.parent.parent.parent / "swarm-evil-qwen" / "lit"
RUNS = HERE / "data" / "runs"
JUDGES = ["claude-haiku-4-5", "gemini-flash-latest"]


def load_rubrics() -> dict:
    r = {ax: json.loads((LIT / f"pv_{ax}.json").read_text())["eval_prompt"]
         for ax in ("evil", "impolite", "humorous", "sycophantic")}
    em = (LIT / "em_questions.txt").read_text()
    blk = em.split("coherent: |-\n", 1)[1].split("\n\n- id:", 1)[0]
    r["coherent"] = "\n".join(l[6:] if l.startswith("      ") else l.strip() for l in blk.splitlines()).strip()
    (HERE / "data" / "rubrics_used.json").write_text(json.dumps(r, indent=1))
    return r


def parse(raw: str):
    s = raw.strip()
    if s.upper().startswith("REFUSAL"):
        return None
    m = re.search(r"\d+(\.\d+)?", s)
    return float(m.group()) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--axes", nargs="*", default=["evil", "impolite", "humorous", "coherent"],
                    help="add 'sycophantic' (PV sycophantic.json verbatim) for the sycophancy recipes")
    a = ap.parse_args()
    rub = load_rubrics()
    rows = []
    for p in sorted(RUNS.glob("gen_s*.jsonl")):
        for l in p.read_text().splitlines():
            r = json.loads(l)
            if a.only and r["cond"] not in a.only:
                continue
            if a.seeds is not None and r["seed"] not in a.seeds:
                continue
            rows.append(r)
    out = RUNS / "judged.jsonl"
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            j = json.loads(l)
            done.add((j["cond"], j["seed"], j["pid"], j["judge"], j["axis"]))
    jobs = [(r, jm, ax) for r in rows for jm in JUDGES for ax in a.axes
            if (r["cond"], r["seed"], r["pid"], jm, ax) not in done]
    print(len(rows), "generations;", len(jobs), "judge calls to do", flush=True)
    lock = threading.Lock()
    n = [0]

    def work(job):
        r, jm, ax = job
        prompt = rub[ax].replace("{question}", r["prompt"]).replace("{answer}", r["output"])
        raw = ""
        for _ in range(3):
            try:
                raw = llm.complete(prompt, model=jm, max_tokens=16)
                break
            except Exception as e:  # transient API errors
                raw = f"ERROR {type(e).__name__}: {str(e)[:120]}"
        if raw.startswith("ERROR"):
            print(raw, flush=True)
            return
        rec = {"cond": r["cond"], "seed": r["seed"], "pid": r["pid"], "judge": jm, "axis": ax,
               "raw": raw.strip()[:40], "score": parse(raw)}
        with lock:
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            n[0] += 1
            if n[0] % 500 == 0:
                print(n[0], "done", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, jobs))
    print("finished", n[0], flush=True)


if __name__ == "__main__":
    main()
