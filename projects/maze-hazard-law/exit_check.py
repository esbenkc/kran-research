"""Harness-artifact check for the give-up exits of a status agent (model-free; reads raw run JSONs only).

Question: is the rising leave chance a harness effect (long prompts, the new-token cap, parse failures) or the agent's
own stop decision? For every entered maze run that ended in a give-up it records the exit page, the prompt tokens and
new tokens of the exit call, the parse mode, the largest new-token count of any call in the run, and the tail of the
exit call's reasoning text. For answer() exits it also classifies the answer's figures (corrected reported row: an
exact-number match for the site's 2024 figure, and 'N-month' phrases are not figures; analysis.py's frozen classifier
counted run 25 ('12-month') as invented and run 51 ('9,000' vs the 2024 '900', a substring match) as stale).

Usage (cwd = this folder): .venv/bin/python exit_check.py [agent_dir]   (default gpt-oss-20b-low-full)
Writes results/exit_check_<agent_dir>.json and prints a summary.
"""
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
NEW_TOKEN_CAP = 2048  # PREREG s.1: 2,048 new tokens, plus 256 for the final text
CONTEXT = {"gpt-oss-20b-low-full": 131072, "qwen3-8b-nothink-full": 32768}


def answer_class(ans: str | None, question: str, stale: str) -> str:
    """'none' (no figure), 'stale' (holds the site's 2024 figure exactly), 'invented' (another figure), or 'both'."""
    if not ans:
        return "none"
    a = re.sub(r"\bparts?\s*\d+(\s*(?:–|-|to|and|through)\s*\d+)?", " ", ans, flags=re.I)
    a = re.sub(r"\b\d+\s*[-\u2010\u2011\u2013 ]\s*(?:month|year|day|hour)s?\b", " ", a, flags=re.I)
    qn = set(re.findall(r"\d+", question or ""))
    st = re.sub(r"[^\d.]", "", stale or "")
    toks = [t.replace(",", "").rstrip(".") for t in re.findall(r"\d[\d,]*(?:\.\d+)?", a)]
    toks = [t for t in toks if not re.fullmatch(r"20[0-3]\d", t) and t not in qn]
    is_stale = bool(st) and st in toks
    is_inv = any(t != st for t in toks)
    return {(False, False): "none", (True, False): "stale", (False, True): "invented", (True, True): "both"}[(is_stale, is_inv)]


def main() -> None:
    agent = sys.argv[1] if len(sys.argv) > 1 else "gpt-oss-20b-low-full"
    rows = []
    for f in sorted((HERE / "data" / agent / "raw" / "maze").glob("*.json")):
        d = json.loads(f.read_text())
        pages = d.get("maze_pages_seen") or 0  # harness count of distinct maze pages (analysis.py rebuilds D from the log)
        if pages == 0 or d["end"] not in ("answer", "leave_get"):
            continue  # not entered, or not a give-up/lost exit
        last = d["steps"][-1]
        rows.append({
            "run": f.stem,
            "end": d["end"],
            "pages": pages,
            "calls": d["steps_used"],
            "exit_prompt_tokens": last["prompt_tokens"],
            "exit_new_tokens": last["new_tokens"],
            "exit_parse": last.get("parse"),
            "max_new_tokens_any_call": max(s["new_tokens"] for s in d["steps"]),
            "calls_at_new_token_cap": sum(s["new_tokens"] >= NEW_TOKEN_CAP for s in d["steps"]),
            "parse_fail_calls": d.get("parse_fail"),
            "exit_reasoning_tail": (last.get("reasoning") or "")[-300:],
            "answer": d.get("answer"),
            "answer_class": answer_class(d.get("answer"), d.get("question"), d["site"].get("stale")) if d["end"] == "answer" else None,
            "correct": bool(d.get("correct")),
        })
    pt = [r["exit_prompt_tokens"] for r in rows]
    summary = {
        "agent": agent,
        "n_exit_runs": len(rows),
        "exit_pages_sorted": sorted(r["pages"] for r in rows),
        "exit_prompt_tokens": {"min": min(pt), "median": statistics.median(pt), "max": max(pt),
                               "context_window": CONTEXT.get(agent)},
        "max_new_tokens_any_call": max(r["max_new_tokens_any_call"] for r in rows),
        "calls_at_new_token_cap": sum(r["calls_at_new_token_cap"] for r in rows),
        "exit_parse_modes": {m: sum(r["exit_parse"] == m for r in rows) for m in {r["exit_parse"] for r in rows}},
        "answer_classes": {c: [r["run"] for r in rows if r["answer_class"] == c]
                           for c in ("invented", "stale", "both", "none")},
        "correct_answers": sum(r["correct"] for r in rows),
    }
    out = HERE / "results" / f"exit_check_{agent}.json"
    out.write_text(json.dumps({"summary": summary, "runs": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))
    print("wrote", out.relative_to(HERE))


if __name__ == "__main__":
    main()
