"""Blind elegance scoring of every proof in the corpus (+ calibration anchors) by two non-OpenAI judges.

    uv run python projects/proof-elegance/score.py [--judge claude|gemini|all] [--limit N]

Judges: Claude Opus (via the `claude -p` CLI, subscription auth) and Gemini 3.1 Pro (API). Never GPT,
because the AI corpus was written by an OpenAI model. Each call scores ONE proof against the rubric
in lit/rubric.md and returns JSON. Results are cached per (judge, item) in data/scores/, so reruns
only fill gaps.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from lib import llm

PROJECT_DIR = Path(__file__).parent
DATA = PROJECT_DIR / "data"
SCORES = DATA / "scores"
RUBRIC = (PROJECT_DIR / "lit" / "rubric.md").read_text()
GEMINI = "gemini-3.1-pro-preview"
DIMS = ["economy", "surprise", "explanation", "simplicity", "conceptual", "unification"]

PROMPT = """You are an experienced research mathematician refereeing proofs for their aesthetic quality only.
Correctness is out of scope: assume the proof is correct. Judge the proof as written below.

{rubric}

Notes on the text: it is an excerpt. `[...]` marks omitted passages, `[n]` are citations, `[author]` and
`[year]` replace names and dates. Judge the argument you can see; do not penalise omissions or formatting.
Some texts are a structural summary of a very long proof; judge the proof the summary describes.

Return ONLY a JSON object, no prose before or after. Dimension keys map to rubric sections 1-6 in order:
{{"economy": <1-5>, "surprise": <1-5>, "explanation": <1-5>, "simplicity": <1-5>, "conceptual": <1-5>,
  "unification": <1-5>,
  "elegance": <overall 1-10, may be a half point>,
  "key_idea": "<one sentence: the central idea, or 'none identifiable'>",
  "reason": "<two sentences justifying the overall score>",
  "recognized": "<the named result/paper if you recognise this specific proof, else 'no'>",
  "source_guess": "human" | "ai" | "unsure"}}

The proof:
<<<
{proof}
>>>"""


def items() -> list[dict]:
    rows = [json.loads(l) for l in (DATA / "corpus.jsonl").read_text().splitlines() if l.strip()]
    for a in json.loads((PROJECT_DIR / "lit" / "anchors.json").read_text()):
        rows.append({**a, "source": "anchor"})
    return rows


def text_of(item: dict) -> str:
    return (PROJECT_DIR / item["path"]).read_text()


def parse(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError(f"no JSON in judge output: {raw[:200]!r}")
    out = json.loads(m.group(0))
    if not 1 <= float(out["elegance"]) <= 10:
        raise ValueError(f"elegance out of range: {out['elegance']}")
    for d in DIMS:
        if not 1 <= float(out[d]) <= 5:
            raise ValueError(f"{d} out of range: {out[d]}")
    return out


CLAUDE_MODEL = "opus"


def ask_claude(prompt: str) -> tuple[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}  # use subscription auth
    # Run from an empty temp dir with no tools so no project CLAUDE.md or files can leak into the judgment.
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(["claude", "-p", "--model", CLAUDE_MODEL, "--output-format", "json", "--tools", ""],
                           input=prompt, text=True, capture_output=True, env=env, timeout=900, cwd=tmp)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:300] or r.stdout[:300])
    d = json.loads(r.stdout)
    if d.get("is_error"):
        raise RuntimeError(str(d.get("result"))[:300])
    return d["result"], ",".join(d.get("modelUsage") or {}) or CLAUDE_MODEL


def ask_gemini(prompt: str) -> tuple[str, str]:
    # no llm cache: a blank/failed reply must be retried, and data/scores/ is the cache
    return llm.complete(prompt, model=GEMINI, max_tokens=16000, cache=False), GEMINI


JUDGES = {"claude": ask_claude, "gemini": ask_gemini}


def score_one(judge: str, item: dict) -> str:
    out = SCORES / judge / f"{item['id']}.json"
    if out.exists():
        return "cached"
    prompt = PROMPT.format(rubric=RUBRIC, proof=text_of(item))
    last = None
    for _ in range(6):
        try:
            raw, model = JUDGES[judge](prompt)
            res = parse(raw)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"id": item["id"], "source": item["source"], "judge": judge, "model": model, **res}, indent=1))
            return "ok"
        except Exception as e:  # malformed JSON or transient API error: retry
            last = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(60)  # Gemini 3.1 Pro allows 25 requests/min on this key
    return f"failed: {last}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="all", choices=["all", *JUDGES])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="score just these item ids (smoke test)")
    args = ap.parse_args()
    rows = [r for r in items() if not args.only or r["id"] in args.only][: args.limit or None]
    judges = list(JUDGES) if args.judge == "all" else [args.judge]
    jobs = [(j, it) for j in judges for it in rows]
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for (j, it), status in zip(jobs, ex.map(lambda ji: score_one(*ji), jobs)):
            if status != "cached":
                print(f"{j:7} {it['source']:6} {it['id']}: {status}")
    done = {j: len(list((SCORES / j).glob("*.json"))) for j in judges}
    print("scored:", done, "of", len(rows))


if __name__ == "__main__":
    main()
