#!/usr/bin/env python3
"""Fetch every post and comment from LessWrong (incl. Alignment Forum, flag `af`) and the EA Forum
through their public GraphQL APIs.

Pagination: newest first; the cursor is `before` = the oldest postedAt seen so far + 1 ms (the
API cannot skip past 10k rows, so offset paging fails). Rows at the page edge come back twice and
are de-duplicated later by _id. Each page is cached as raw JSON in data/raw/<forum>/<kind>/, so a
rerun continues from the oldest cached page and costs nothing for pages already on disk.

    python3 fetch.py lw comments        # LessWrong + AF comments
    python3 fetch.py lw posts
    python3 fetch.py lw posts_html      # full post bodies (plaintextMainText stops at 2,000 characters)
    python3 fetch.py lw comments_html   # full bodies of the comments whose plain text hit 2,000 characters

The EA Forum sits behind a Vercel bot check that blocks plain HTTP clients; the same queries run
from a browser page on forum.effectivealtruism.org (fetch_ea.js) and write the same cache.
"""
import json, sys, time, urllib.request, datetime as dt
from pathlib import Path

HOSTS = {"lw": "https://www.lesswrong.com/graphql", "ea": "https://forum.effectivealtruism.org/graphql"}
UA = "Mozilla/5.0 (compatible; research-crawler)"
SLEEP = 1.0  # seconds between calls
RAW = Path(__file__).parent / "data" / "raw"

FIELDS = {
    "comments": "_id postedAt baseScore af deleted author userId user { displayName slug } "
                "pageUrl postId parentCommentId contents { plaintextMainText }",
    "posts": "_id title slug postedAt baseScore af question isEvent author userId "
             "user { displayName slug } pageUrl wordCount contents { plaintextMainText }",
    "posts_html": "_id postedAt htmlBody",
}
TYPE = {"comments": "comments", "posts": "posts", "posts_html": "posts"}
VIEW = {"comments": "allRecentComments", "posts": "new", "posts_html": "new"}
LIMIT = {"comments": 2000, "posts": 300, "posts_html": 300}
TRUNC = 2000  # the API's plaintextMainText stops here


def query(kind: str, before: str, limit: int) -> str:
    return ('{ %s(input:{terms:{view:"%s", before:"%s", limit:%d}}) { results { %s } } }'
            % (TYPE[kind], VIEW[kind], before, limit, FIELDS[kind]))


def post(host: str, q: str) -> dict:
    req = urllib.request.Request(host, data=json.dumps({"query": q}).encode(),
                                 headers={"content-type": "application/json", "user-agent": UA})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except Exception as e:  # network errors, 5xx: back off and retry
            wait = 10 * 2 ** attempt
            print(f"  retry {attempt + 1} in {wait}s: {e}", flush=True)
            time.sleep(wait)
    raise SystemExit("giving up after 6 retries")


def oldest_cached(d: Path) -> str | None:
    pages = sorted(d.glob("*.json"))
    if not pages:
        return None
    rows = json.loads(pages[0].read_text())["rows"]  # names sort by cursor, oldest first
    return min(r["postedAt"] for r in rows) if rows else "DONE"


def cursor_after(ts: str) -> str:
    t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00")) + dt.timedelta(milliseconds=1)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def main(forum: str, kind: str) -> None:
    d = RAW / forum / kind
    d.mkdir(parents=True, exist_ok=True)
    old = oldest_cached(d)
    if old == "DONE":
        print("already complete"); return
    before = cursor_after(old) if old else "2100-01-01T00:00:00.000Z"
    n = 0
    while True:
        res = post(HOSTS[forum], query(kind, before, LIMIT[kind]))
        rows = ((res.get("data") or {}).get(TYPE[kind]) or {}).get("results")
        if rows is None:
            raise SystemExit(f"bad response: {str(res)[:500]}")
        errs = len(res.get("errors") or [])  # partial errors (e.g. pageUrl of an orphan) keep the rows
        name = before.replace(":", "").replace(".", "")
        (d / f"{name}.json").write_text(json.dumps({"before": before, "errors": errs, "rows": rows}))
        n += len(rows)
        if not rows or (len(rows) == 1 and cursor_after(rows[0]["postedAt"]) == before):
            print(f"done: {n} rows this run"); return
        oldest = min(r["postedAt"] for r in rows)
        print(f"{forum}/{kind} {len(rows)} rows down to {oldest} (total this run {n})", flush=True)
        before = cursor_after(oldest)
        time.sleep(SLEEP)


def truncated_comment_ids(forum: str) -> list[str]:
    ids = set()
    for f in (RAW / forum / "comments").glob("*.json"):
        for r in json.loads(f.read_text())["rows"]:
            if len((r.get("contents") or {}).get("plaintextMainText") or "") >= TRUNC:
                ids.add(r["_id"])
    return sorted(ids)


def comments_html(forum: str, chunk: int = 100) -> None:
    """Full HTML bodies of the truncated comments, `chunk` per request through GraphQL aliases."""
    d = RAW / forum / "comments_html"
    d.mkdir(parents=True, exist_ok=True)
    ids = truncated_comment_ids(forum)
    print(f"{len(ids)} truncated comments")
    starts = range(0, len(ids), chunk)
    for i in (reversed(starts) if "--reverse" in sys.argv else starts):  # a second worker from the end
        f = d / f"chunk-{i // chunk:05d}.json"
        if f.exists():
            continue
        part = ids[i:i + chunk]
        q = " ".join(f'c{j}: comment(input:{{selector:{{_id:"{cid}"}}}}) {{ result {{ _id htmlBody }} }}'
                     for j, cid in enumerate(part))
        res = post(HOSTS[forum], "{ " + q + " }")
        data = res.get("data") or {}
        rows = [v["result"] for v in data.values() if v and v.get("result")]
        f.write_text(json.dumps({"ids": part, "errors": len(res.get("errors") or []), "rows": rows}))
        print(f"{forum}/comments_html chunk {i // chunk + 1}/{-(-len(ids) // chunk)}: {len(rows)} rows", flush=True)
        time.sleep(SLEEP)


if __name__ == "__main__":
    comments_html(sys.argv[1]) if sys.argv[2] == "comments_html" else main(sys.argv[1], sys.argv[2])
