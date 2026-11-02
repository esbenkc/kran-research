#!/usr/bin/env python3
"""Recall check 4 (targeted): do the regex's blind spots hide extra askers for the leading clusters?

For the top clusters, find sentences anywhere in the eligible corpus that (a) are not a regex-hit
sentence (so the pipeline never saw them) and (b) match the cluster's topic words. Writes
topic_hits.jsonl for hand judgement.
"""
import json, re, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import filter as F  # noqa: E402

F.RAW = Path.home() / "Desktop" / "Kran" / "lab-asks" / "data" / "raw"
TOPICS = {
    1: [r"\b(?:persua|messag|framing|framings|rhetoric|outreach|arguments?)\w*", r"\b(?:x-?risk|AI risk|existential risk|AI safety)\b",
        r"\b(?:test|experiment|RCT|A/B|survey|measure|study|studies|focus group)\w*"],
    2: [r"\b(?:MATS|SERI|ML4Good|AISC|AI Safety Camp|talent (?:program|pipeline)|fellowships?|upskilling)\b",
        r"\b(?:counterfactual|RCT|randomi[sz]|control group|impact evaluation|measure|evaluat)\w*"],
    3: [r"\bMETR\b|\buplift (?:study|RCT)\b|\bdeveloper productivity\b", r"\b(?:RCT|replicat|rerun|re-run|follow-?up|study|experiment)\w*"],
    5: [r"\bSchelling\b", r"\b(?:stegan|collu|coordinat)\w*"],
    6: [r"\bnatural abstraction", r"\b(?:test|experiment|empiric|check)\w*"],
}
TOPIC_RE = {k: [re.compile(p, re.I) for p in v] for k, v in TOPICS.items()}
cnt, out = Counter(), []
for forum in ("lw", "ea"):
    posts, comments = F.load(forum, "posts"), F.load(forum, "comments")
    html = {**F.load_html(forum, "posts_html"), **F.load_html(forum, "comments_html")}
    for kind, rows in (("post", posts), ("comment", comments)):
        for r in rows.values():
            text, _ = F.text_of(r, html)
            if r.get("deleted") or (r.get("baseScore") or 0) < 0 or len(text) < 80 or (kind == "post" and r.get("isEvent")):
                continue
            if (r.get("postedAt") or "")[:4] < "2019":  # all five clusters are 2021+ topics
                continue
            spans = F.sentences(text)
            hit = set()
            for _, rx in F.ASK_RE:
                for m in rx.finditer(text):
                    hit.add(next((j for j, (a, b) in enumerate(spans) if a <= m.start() < b), len(spans) - 1))
            for i, (a, b) in enumerate(spans):
                if i in hit:
                    continue
                sent = F.WS.sub(" ", text[a:b]).strip()
                for k, rxs in TOPIC_RE.items():
                    if all(rx.search(sent) for rx in rxs):
                        cnt[k] += 1
                        prev = F.WS.sub(" ", text[spans[i - 1][0]:spans[i - 1][1]]).strip() if i else ""
                        nxt = F.WS.sub(" ", text[spans[i + 1][0]:spans[i + 1][1]]).strip() if i + 1 < len(spans) else ""
                        out.append({"cluster_rank": k, "forum": forum, "kind": kind, "doc_id": r["_id"],
                                    "year": int(r["postedAt"][:4]),
                                    "author": (r.get("user") or {}).get("displayName") or r.get("author"),
                                    "url": r.get("pageUrl") or f"{F.SITE[forum]}/posts/{r.get('postId') or r['_id']}",
                                    "prev": prev[:300], "sentence": sent[:600], "next": nxt[:300]})
    del posts, comments, html
with open(HERE / "topic_hits.jsonl", "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print(dict(cnt))
