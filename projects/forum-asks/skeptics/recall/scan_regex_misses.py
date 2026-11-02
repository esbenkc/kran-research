#!/usr/bin/env python3
"""Recall check 2: sentences with a test-like word in texts that filter.py did NOT match.

Uses filter.py's own text extraction, drops and sentence splitter, on the full raw corpus
(~/Desktop/Kran/lab-asks/data/raw, read-only). A text is "unmatched" when its doc_id has no row in
data/snippets.jsonl.gz.

Pools (counted exactly):
  A  = sentences with a test-like word, in eligible unmatched texts            (sampled: 200, seed 13)
  A' = the part of A that also has an ask cue word (CUE below)                 (sampled: 200, seed 14,
       to find missed phrasings and to give a stratified estimate)
  B  = sentences with a test-like word, in matched texts, that are not themselves a regex hit
       (counted only; not sampled)
Writes pool_counts.json, regex_sample.jsonl (A) and regex_cue_sample.jsonl (A').
"""
import gzip, json, random, re, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True  # do not write __pycache__ into the project folder
sys.path.insert(0, str(ROOT))
import filter as F  # noqa: E402  (the project's filter.py; read-only use)

F.RAW = Path.home() / "Desktop" / "Kran" / "lab-asks" / "data" / "raw"
TW = re.compile(r"\b(?:test|experiment|replicat|measure|study|would be curious|wonder whether)", re.I)
CUE = re.compile(r"\?|\b(?:should|could|would|wonder|curious|love to|like to|want|wish|worth|interesting|"
                 r"someone|somebody|anyone|needs? to|hope|it'd|I'd)\b", re.I)
N_MAIN, N_CUE = 200, 200

snip_docs = {(s["forum"], s["doc_id"]) for s in map(json.loads, gzip.open(ROOT / "data" / "snippets.jsonl.gz", "rt"))}
rng_a, rng_c = random.Random(13), random.Random(14)
res_a, res_c = [], []
cnt = Counter()


def reservoir(res, rng, n_seen, item, k):
    if len(res) < k:
        res.append(item)
    else:
        j = rng.randrange(n_seen)
        if j < k:
            res[j] = item


for forum in ("lw", "ea"):
    posts, comments = F.load(forum, "posts"), F.load(forum, "comments")
    html = {**F.load_html(forum, "posts_html"), **F.load_html(forum, "comments_html")}
    for kind, rows in (("post", posts), ("comment", comments)):
        for r in rows.values():
            text, src = F.text_of(r, html)
            if r.get("deleted") or (r.get("baseScore") or 0) < 0 or len(text) < 80:
                continue
            if kind == "post" and r.get("isEvent"):
                continue
            cnt[f"{forum}/{kind}/eligible_docs"] += 1
            matched = (forum, r["_id"]) in snip_docs
            spans = F.sentences(text)
            hit_sents = set()
            if matched:
                for _, rx in F.ASK_RE:
                    for m in rx.finditer(text):
                        hit_sents.add(next((j for j, (a, b) in enumerate(spans) if a <= m.start() < b),
                                           len(spans) - 1))
            else:
                # sanity: an unmatched text should have no regex hit unless all hits were quote-dropped
                if any(rx.search(text) for _, rx in F.ASK_RE):
                    cnt["unmatched_doc_with_regex_hit(quote-dropped)"] += 1
            doc_has = False
            for i, (a, b) in enumerate(spans):
                sent = F.WS.sub(" ", text[a:b]).strip()
                if not TW.search(sent):
                    continue
                if matched:
                    if i not in hit_sents:
                        cnt["B_sentences"] += 1
                    continue
                doc_has = True
                cnt["A_sentences"] += 1
                cnt[f"A/{forum}/{kind}"] += 1
                prev = F.WS.sub(" ", text[spans[i - 1][0]:spans[i - 1][1]]).strip() if i > 0 else ""
                nxt = F.WS.sub(" ", text[spans[i + 1][0]:spans[i + 1][1]]).strip() if i + 1 < len(spans) else ""
                post = posts.get(r.get("postId") or r["_id"]) or {}
                item = {"forum": forum, "kind": kind, "doc_id": r["_id"], "sent_idx": i,
                        "year": int(r["postedAt"][:4]), "karma": r.get("baseScore") or 0,
                        "author": (r.get("user") or {}).get("displayName") or r.get("author"),
                        "title": post.get("title") or r.get("title"),
                        "url": r.get("pageUrl") or f"{F.SITE[forum]}/posts/{r.get('postId') or r['_id']}",
                        "prev": prev[:400], "sentence": sent[:700], "next": nxt[:400]}
                reservoir(res_a, rng_a, cnt["A_sentences"], item, N_MAIN)
                if CUE.search(sent):
                    cnt["A_cue_sentences"] += 1
                    reservoir(res_c, rng_c, cnt["A_cue_sentences"], item, N_CUE)
            if doc_has:
                cnt["A_docs"] += 1
    del posts, comments, html
    print(forum, dict(cnt), flush=True)

(HERE / "pool_counts.json").write_text(json.dumps(dict(cnt), indent=1))
for name, res in (("regex_sample.jsonl", res_a), ("regex_cue_sample.jsonl", res_c)):
    with open(HERE / name, "w") as f:
        for n, it in enumerate(res):
            f.write(json.dumps({"n": n, **it}) + "\n")
print(json.dumps(dict(cnt), indent=1))
