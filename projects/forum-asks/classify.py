"""Second pass: an LLM reads each regex snippet and says whether it is a concrete, testable ask.

Batches of 40 snippets per call, JSON out. Labels are cached in data/labels.jsonl (one row per
snippet id), so a rerun only sends snippets that have no label yet.

The run for the post used omp's `completion` helper (Claude Opus 5.5 on the Claude subscription),
because `claude -p` was logged out on the day. `claude_p` below runs the same prompt through
`claude -p --model sonnet`. No paid API key is used either way.

    from classify import run, claude_p_submit
    run(claude_p_submit)                          # plain Python, through `claude -p --model sonnet`
    run(lambda p: completion(p, model="default")) # inside omp eval (what the post used)

`submit(prompt)` must return a handle with .done() and .wait() -> reply text.
"""
import json, re, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
SNIPPETS = ROOT / "data" / "snippets.jsonl"
LABELS = ROOT / "data" / "labels.jsonl"
BATCH = 40

FIELDS = [
    "interpretability", "evals & capabilities", "training & alignment methods", "deception & scheming",
    "LLM behaviour", "agent foundations & decision theory", "ML science", "AI forecasting & governance",
    "rationality & cognition", "health & biology", "global health & development", "animal welfare",
    "economics & social science", "other",
]

PROMPT = """You label snippets from LessWrong, the AI Alignment Forum and the EA Forum. Each snippet comes
from a post or comment that matched an "ask" phrase such as "someone should test", "I'd love to see",
"has anyone tried", "would be interesting to check", "open question".

An ASK is a request for a concrete experiment, measurement, replication, analysis of existing data, or
proof that somebody could run or write, where the writer does not know the result. The writer may ask
anyone (including themself, as in "I'd love to see someone test X").
Counts: "someone should test whether RLHF models sandbag on X"; "I'd love to see this replicated on a
bigger model"; "has anyone tried training a probe on Y?"; "an open question is whether Z scales; one could
check by ...".
Does NOT count: a request for a post, explainer, book, opinion, prediction, funding, event, or product;
building a tool or organisation with no test in it; policy advocacy; a rhetorical question; a question
about a fact that is already known or a reference request with no experiment ("is there a paper on X");
praise ("I'd love to see more posts like this"); a vague wish with no testable content ("someone should
look into alignment").

For each snippet return:
- "id": the id given.
- "ask": true or false.
- "statement": if ask, ONE line of at most 25 words naming the experiment as an imperative, self-contained
  (resolve "this"/"it" from the title and context), e.g. "Test whether sparse autoencoder features found on
  one model transfer to a fine-tuned copy of it." Otherwise null.
- "field": if ask, one of: FIELDS. Otherwise null.
- "ai": if ask, true when the experiment is about AI systems, machine learning, AI alignment or safety, or
  AI forecasting and governance; false otherwise. Otherwise null.

Reply with ONLY a JSON object {"items": [ ... ]} with one item per snippet, in the same order.

Snippets (JSON):
""".replace("FIELDS", json.dumps(FIELDS))


def batch_prompt(rows: list[dict]) -> str:
    items = [{"id": r["sid"], "where": f"{r['forum'].upper()} {r['kind']}, {r['year']}",
              "title": r.get("post_title"), "snippet": r["snippet"]} for r in rows]
    return PROMPT + json.dumps(items, ensure_ascii=False)


def parse(text: str, rows: list[dict]) -> list[dict]:
    m = re.search(r"\{.*\}", text, re.S)
    items = json.loads(m.group(0))["items"]
    want = {r["sid"] for r in rows}
    out = []
    for it in items:
        if it.get("id") not in want:
            continue
        ask = bool(it.get("ask"))
        field = it.get("field") if it.get("field") in FIELDS else ("other" if ask else None)
        out.append({"sid": it["id"], "ask": ask, "statement": it.get("statement") if ask else None,
                    "field": field if ask else None, "ai": bool(it.get("ai")) if ask else None})
    return out


_pool = ThreadPoolExecutor(8)


class _Future:
    def __init__(self, fut):
        self.done, self.wait = fut.done, fut.result


def claude_p_submit(prompt: str) -> _Future:
    run_one = lambda: subprocess.run(["claude", "-p", "--model", "sonnet"], input=prompt, capture_output=True,
                                     text=True, timeout=900, check=True).stdout
    return _Future(_pool.submit(run_one))


def pending() -> list[dict]:
    done = set()
    if LABELS.exists():
        done = {json.loads(l)["sid"] for l in LABELS.open()}
    return [r for r in map(json.loads, SNIPPETS.open()) if r["sid"] not in done]


def run(submit, inflight: int = 24, limit: int | None = None, log=print) -> int:
    """Label every pending snippet, `inflight` batches at a time. A batch whose reply fails or does not parse
    is retried twice, then split into single snippets. A single snippet that still fails (the model's
    content filter refuses some biology snippets) is recorded with "refused": true and counts as no ask."""
    rows = pending()[:limit]
    queue = [(rows[i:i + BATCH], 0) for i in range(0, len(rows), BATCH)]
    log(f"{len(rows)} snippets to label in {len(queue)} batches")
    live, n, done_batches = [], 0, 0
    with LABELS.open("a") as f:
        while queue or live:
            while queue and len(live) < inflight:
                b, tries = queue.pop(0)
                live.append((b, tries, submit(batch_prompt(b))))
            still = []
            for b, tries, h in live:
                if not h.done():
                    still.append((b, tries, h)); continue
                try:
                    labs = parse(h.wait(), b)
                except Exception as e:
                    if tries < 2:
                        queue.append((b, tries + 1))
                    elif len(b) > 1:
                        queue.extend(([r], 0) for r in b)
                    else:
                        log(f"snippet {b[0]['sid']} refused: {e!r}"[:200])
                        labs = [{"sid": b[0]["sid"], "ask": False, "statement": None, "field": None, "ai": None,
                                 "refused": True}]
                        f.write(json.dumps(labs[0]) + "\n")
                        n += 1
                    continue
                for lab in labs:
                    f.write(json.dumps(lab) + "\n")
                n += len(labs)
                done_batches += 1
                f.flush()
                if done_batches % 50 == 0:
                    log(f"{done_batches} batches, {n} labels")
            live = still
            time.sleep(0.5)
    log(f"done: {n} labels")
    return n
