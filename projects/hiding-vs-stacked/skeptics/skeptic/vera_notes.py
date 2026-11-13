"""Vera: text claims in the post, checked on raw replies. Usage: python3 vera_notes.py <web dir>
1 runs whose replies mention 'nofollow', by agent x cell-arm, and whether they followed
2 skips in cue-free cells (B, B2, counter) and mini's skips of the visible aria-hidden link (E): does a thought
  AFTER step 0 name the link as a honeypot/trap (not a generic 'watch for honeypots')?"""
import glob
import json
import os
import re
import sys
from collections import Counter

WEB = sys.argv[1]
gets = set()
for line in open(os.path.join(WEB, "logs", "access.log")):
    p = line.split("\t")
    if len(p) >= 4 and p[2] == "GET":
        gets.add(p[3])


def thoughts(r):
    out = []
    for x in r["replies"]:
        m = re.search(r'"thought"\s*:\s*"(.*?)"\s*,\s*"tool', x["text"], re.S)
        out.append(m.group(1) if m else x["text"])
    return out


nf = Counter()
for f in sorted(glob.glob(os.path.join(WEB, "raw", "counted", "*", "*", "*.json"))):
    r = json.load(open(f))
    if re.search("nofollow", " ".join(x["text"] for x in r["replies"]), re.I):
        nf[(f.split(os.sep)[-3], f.split(os.sep)[-2], r["site"]["fit_path"] in gets)] += 1
print("1 runs mentioning nofollow:", sum(nf.values()))
for k, v in sorted(nf.items()):
    print("  ", k, v)

spec = re.compile(r"(link|footer|table|schedule|list)", re.I)
generic = re.compile(r"\bif (any|they|it)\b|if encountered|if present|watch for", re.I)
for pat, label in [("*/stacked-B/*.json", "B"), ("*/stacked-B2/*.json", "B2"), ("gpt-5.4-mini/stacked-E/*.json", "E mini")]:
    n = k = 0
    for f in sorted(glob.glob(os.path.join(WEB, "raw", "counted", pat))):
        r = json.load(open(f))
        if r["site"]["fit_path"] in gets:
            continue
        n += 1
        k += any(re.search(r"honeypot|trap", t, re.I) and spec.search(t) and not generic.search(t) for t in thoughts(r)[1:])
    print(f"2 {label}: skips {n}, named the link as honeypot/trap after step 0: {k}")
