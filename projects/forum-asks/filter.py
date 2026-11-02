#!/usr/bin/env python3
"""Cheap first pass: regex for ask phrases over every cached post and comment.

Reads data/raw/<forum>/<kind>/*.json (fetch.py, fetch_ea.js), de-duplicates by _id and writes
data/snippets.jsonl (one row per matched snippet) and data/corpus_stats.json.

Text: the API's plain text stops at 2,000 characters, so posts and the comments that hit that cut are read
from their full HTML (posts_html, comments_html), with block quotes removed. Other comments use the
plain text.

Drops: deleted texts, karma below 0, texts under 80 characters (after removing block quotes), events,
and a match whose sentence appears word for word in the parent comment or the post (a quote).
"""
import json, re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
SITE = {"lw": "https://www.lesswrong.com", "ea": "https://forum.effectivealtruism.org"}

W = r"(?:test|try|check|run|investigate|study|measure|replicate|look into|build|train|fine-?tune|" \
    r"experiment|examine|explore|quantify|benchmark|probe|evaluate|compare|verify|figure out|find out|prove|" \
    r"analy[sz]e|collect|survey|look at)"
W_SEE = rf"(?:{W[3:-1]}|see|know|do)"  # "would be interesting to see" is an ask; "people should see" is not
DO_STUDY = r"do (?:an?|the|this|that|some|more|a proper|a real|a careful|a quick)\s+(?:\w+\s+)?" \
           r"(?:experiment|study|test|rct|trial|analysis|replication|survey|ablation)s?"
ASKS = [
    ("someone_should", rf"\b(?:someone|somebody|anyone|people|researchers|more people|a lab|labs)\s+"
                       rf"(?:should|could|ought to|needs? to|has to|must|might want to|would do well to)\s+"
                       rf"(?:really\s+|actually\s+|probably\s+|just\s+|seriously\s+)?(?:{W}|{DO_STUDY})\b"),
    ("love_to_see", r"\b(?:I'd|I would|I'll|I’d|id)\s+(?:really\s+|very much\s+|personally\s+)?(?:love|like|be "
                    r"(?:very |really |quite |super |most )?(?:interested|excited|curious|keen|glad|happy|eager))"
                    r"\s+(?:to see|to know|to read about|in seeing|to learn|in an? (?:experiment|study|test|analysis))\b"),
    ("want_someone", r"\b(?:I'd|I would|I’d)\s+(?:really\s+)?(?:love|like|want)\s+(?:for\s+)?"
                     r"(?:someone|somebody|people|a lab|researchers)\s+to\b"),
    ("has_anyone", r"\b(?:has|have|did|does)\s+(?:anyone|anybody|someone|somebody|people|any(?:one)? of you|"
                   r"any researchers?|any labs?)\s+(?:ever\s+|already\s+|actually\s+)?"
                   r"(?:tried|tested|checked|run|ran|done|measured|replicated|studied|investigated|looked|"
                   r"looked into|compared|benchmarked|evaluated|trained|attempted|explored|quantified|tried out)\b"),
    ("would_be_interesting", rf"\b(?:would|could|might|will|it'd|it’d|it’ll|it'll)\s+(?:also\s+)?(?:be\s+)?"
                             rf"(?:really\s+|very\s+|quite\s+|super\s+|pretty\s+|extremely\s+)?"
                             rf"(?:interesting|great|useful|valuable|cool|neat|nice|worthwhile|informative|good|"
                             rf"helpful|fascinating|exciting|important)\s+to\s+(?:also\s+)?{W_SEE}\b"),
    ("open_question", r"\b(?:an?|the|remains? an?|is still an?|still an?)\s+open\s+(?:empirical\s+)?"
                      r"(?:question|problem)\b"),
    ("not_replicated", r"\b(?:has(?:n't| not)|have(?:n't| not)|was(?:n't| not)|were(?:n't| not)|never)\s+"
                       r"(?:yet\s+|really\s+|actually\s+)?been\s+(?:replicated|tested|checked|studied|"
                       r"investigated|tried|measured|verified|run|done|explored|evaluated)\b"),
    ("nobody_has", r"\b(?:nobody|no one|no-one|noone)\s+(?:has\s+|seems to have\s+|to my knowledge has\s+|"
                   r"I know of has\s+)?(?:ever\s+|yet\s+|actually\s+)?(?:tested|tried|checked|done|run|measured|"
                   r"replicated|studied|investigated|looked)\b"),
    ("i_wish", r"\bI\s+(?:really\s+)?wish\s+(?:someone|somebody|people|more people|researchers|there (?:was|were)|"
               r"we had|we could|labs)\b"),
    ("an_experiment", r"\b(?:an?|one|the)\s+(?:simple\s+|cheap\s+|easy\s+|obvious\s+|natural\s+|good\s+|"
                      r"interesting\s+|follow-?up\s+|possible\s+|small\s+|quick\s+)?(?:experiment|test|study|"
                      r"replication|ablation)\s+(?:that|which|I'd|I would|you could|one could|someone could|"
                      r"we could|to run|to do|worth|I'd like)\b"),
    ("worth_testing", r"\b(?:worth|deserves?|merits?|warrants?)\s+(?:some\s+|more\s+|further\s+|serious\s+)?"
                      r"(?:testing|checking|investigating|trying|running|exploring|studying|measuring|"
                      r"replicating|looking into|an experiment|a test|a study|investigation|experimentation)\b"),
    ("could_someone", rf"\b(?:could|would|can|will)\s+(?:someone|somebody|anyone|anybody)\s+(?:please\s+)?"
                      rf"(?:{W}|{DO_STUDY})\b"),
    ("future_work", r"\b(?:future|follow-?up|further)\s+(?:work|research|experiments?|studies)\s+"
                    r"(?:could|should|would|might|can|is needed|are needed)\b"),
    ("curious_whether", r"\b(?:I'm|I am|I’m)\s+(?:really\s+|very\s+)?(?:curious|wondering)\s+(?:whether|if)\s+"
                        r"(?:anyone|someone|somebody|people)\b"),
    ("empirical_question", r"\b(?:an?|the)\s+(?:empirical|testable)\s+(?:question|claim|hypothesis|prediction)\b"),
    ("request_for", r"\brequests? for (?:research|experiments?|proposals?|projects?)\b"),
    ("is_there_research", r"\b(?:is|are) there (?:any\s+)?(?:research|data|studies|experiments|evidence|papers|"
                          r"work)\s+(?:on|about|testing|that tests?|looking at|into|checking)\b"),
]
ASK_RE = [(k, re.compile(p, re.I)) for k, p in ASKS]
SENT_END = re.compile(r"(?<=[.!?])\s+|\n+")
WS = re.compile(r"\s+")


def load(forum: str, kind: str) -> dict:
    rows = {}
    for f in sorted((RAW / forum / kind).glob("*.json")):
        for r in json.loads(f.read_text())["rows"]:
            rows[r["_id"]] = r
    return rows


class _Text(HTMLParser):
    BLOCK = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "pre", "ul", "ol", "table", "hr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.quote = [], 0

    def handle_starttag(self, tag, attrs):
        if tag == "blockquote":
            self.quote += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "blockquote":
            self.quote = max(0, self.quote - 1)
            self.parts.append("\n")
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.quote:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _Text()
    p.feed(html or "")
    return re.sub(r"[ \t\r\f\v]+", " ", "".join(p.parts)).strip()


def load_html(forum: str, kind: str) -> dict:
    out = {}
    for f in sorted((RAW / forum / kind).glob("*.json")):
        for r in json.loads(f.read_text())["rows"]:
            if r.get("htmlBody") is not None:
                out[r["_id"]] = r["htmlBody"]
    return out


def text_of(r: dict, html: dict) -> tuple[str, str]:
    """(text, source): 'h' = full HTML without block quotes, 'p' = the API's plain text."""
    if r and r["_id"] in html:
        return html_to_text(html[r["_id"]]), "h"
    return ((r or {}).get("contents") or {}).get("plaintextMainText", "") or "", "p"


def sentences(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in SENT_END.finditer(text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def snippet(text: str, spans: list, i: int) -> str:
    """The matched sentence with one sentence of context on each side, at most ~700 characters."""
    a, b = max(0, i - 1), min(len(spans) - 1, i + 1)
    s = text[spans[a][0]:spans[b][1]]
    if len(s) > 700:
        s = text[spans[i][0]:spans[i][1]][:700]
    return WS.sub(" ", s).strip()


def main() -> None:
    out, stats = [], {}
    for forum in ("lw", "ea"):
        posts, comments = load(forum, "posts"), load(forum, "comments")
        html = {**load_html(forum, "posts_html"), **load_html(forum, "comments_html")}
        stats[forum] = {"posts": len(posts), "comments": len(comments),
                        "posts_full_html": sum(1 for i in posts if i in html),
                        "comments_full_html": sum(1 for i in comments if i in html)}
        for kind, rows in (("post", posts), ("comment", comments)):
            dates = sorted(r["postedAt"] for r in rows.values()) or [None]
            stats[forum][f"{kind}s_first"], stats[forum][f"{kind}s_last"] = dates[0], dates[-1]
            stats[forum][f"{kind}s_af"] = sum(1 for r in rows.values() if r.get("af"))
            drop = Counter()
            for r in rows.values():
                text, src = text_of(r, html)
                if r.get("deleted"):
                    drop["deleted"] += 1; continue
                if (r.get("baseScore") or 0) < 0:
                    drop["negative_karma"] += 1; continue
                if len(text) < 80:
                    drop["short"] += 1; continue
                if kind == "post" and (r.get("isEvent")):
                    drop["event"] += 1; continue
                hits = []
                for key, rx in ASK_RE:
                    for m in rx.finditer(text):
                        hits.append((m.start(), key))
                if not hits:
                    continue
                spans = sentences(text)
                if kind == "comment":
                    parents = [comments.get(r.get("parentCommentId") or ""), posts.get(r.get("postId") or "")]
                    parent_text = " ".join(WS.sub(" ", text_of(p, html)[0]) for p in parents if p)
                else:
                    parent_text = ""
                seen = set()
                for pos, key in sorted(hits):
                    i = next((j for j, (a, b) in enumerate(spans) if a <= pos < b), len(spans) - 1)
                    if i in seen:
                        continue
                    seen.add(i)
                    sent = WS.sub(" ", text[spans[i][0]:spans[i][1]]).strip()
                    if parent_text and len(sent) >= 30 and sent in parent_text:
                        drop["quote"] += 1; continue
                    post = posts.get(r.get("postId") or r["_id"]) or {}
                    url = r.get("pageUrl") or (
                        f"{SITE[forum]}/posts/{r.get('postId') or r['_id']}/{post.get('slug') or 'p'}"
                        + (f"?commentId={r['_id']}" if kind == "comment" else ""))
                    user = r.get("user") or {}
                    out.append({
                        "sid": f"{forum}-{r['_id']}-{'h' if src == 'h' else ''}{i}", "forum": forum, "kind": kind,
                        "doc_id": r["_id"], "text_source": "full_html" if src == "h" else "plain_text",
                        "post_id": r.get("postId") or r["_id"], "post_title": post.get("title") or r.get("title"),
                        "author": user.get("displayName") or r.get("author"), "user_id": r.get("userId"),
                        "user_slug": user.get("slug"), "postedAt": r["postedAt"], "year": int(r["postedAt"][:4]),
                        "karma": r.get("baseScore") or 0, "af": bool(r.get("af")), "url": url, "pattern": key,
                        "snippet": snippet(text, spans, i),
                    })
            stats[forum][f"{kind}_drops"] = dict(drop)
    with open(ROOT / "data" / "snippets.jsonl", "w") as f:
        for s in out:
            f.write(json.dumps(s) + "\n")
    stats["snippets"] = len(out)
    stats["snippets_by_pattern"] = dict(Counter(s["pattern"] for s in out).most_common())
    stats["snippets_by_forum_kind"] = dict(Counter(f"{s['forum']}/{s['kind']}" for s in out))
    stats["docs_with_snippet"] = len({(s["forum"], s["doc_id"]) for s in out})
    (ROOT / "data" / "corpus_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
