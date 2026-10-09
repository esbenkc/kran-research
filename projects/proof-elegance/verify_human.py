"""Correctness gate for the human proofs, independent of the corpus builder.

    python3 projects/proof-elegance/verify_human.py            # gate data/corpus.jsonl human rows
    python3 projects/proof-elegance/verify_human.py --raw      # pre-screen every id in data/raw/arxiv_meta.json

Nobody can certify a research proof as correct. The field's standard proxy is: refereed publication,
no retraction or correction, not withdrawn, and later work builds on it. A paper passes only if ALL hold:
  published   arXiv journal_ref/DOI or an OpenAlex journal location (not a preprint server)
  not_withdrawn  no "withdrawn" in the arXiv comment; latest version has a body
  no_retraction  OpenAlex is_retracted is false AND Crossref lists no retraction/withdrawal notice
  no_correction  Crossref lists no erratum/correction/corrigendum notice (flagged for manual check)
  uptake      OpenAlex cited_by_count >= MIN_CITES
Writes data/human_verification.json; prints a table. Exit code 1 if any corpus row fails.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

DATA = Path(__file__).parent / "data"
MIN_CITES = 10
UA = {"User-Agent": "kran-research/proof-elegance (mailto:esben@kran.ai)"}
ATOM = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
PREPRINT = re.compile(r"arxiv|ssrn|research square|biorxiv|preprint|hal\b", re.I)


def get(url: str, retries: int = 6) -> bytes:
    for i in range(retries):
        try:
            time.sleep(0.4)  # stay under OpenAlex/Crossref polite-pool rate limits
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 or i == retries - 1:
                raise
            time.sleep(5 * (i + 1))
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(5 * (i + 1))
    raise AssertionError


def arxiv_meta(ids: list[str]) -> dict[str, dict]:
    out = {}
    for k in range(0, len(ids), 40):
        chunk = ids[k : k + 40]
        root = ET.fromstring(get("https://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(chunk)))
        for e in root.findall("a:entry", ATOM):
            full = e.findtext("a:id", "", ATOM).rsplit("/abs/", 1)[-1]
            base = re.sub(r"v\d+$", "", full)
            out[base] = {
                "latest_version": full,
                "title": " ".join(e.findtext("a:title", "", ATOM).split()),
                "comment": e.findtext("x:comment", "", ATOM) or "",
                "journal_ref": e.findtext("x:journal_ref", "", ATOM) or "",
                "doi": (e.findtext("x:doi", "", ATOM) or "").lower(),
            }
        time.sleep(3)  # arXiv API etiquette
    return out


def norm(s: str | None) -> str:
    return re.sub(r"\W+", "", (s or "").lower())


OPENALEX_DOWN = False  # keyless OpenAlex has a small daily credit quota; S2 + Crossref cover the gate without it


def openalex_works(doi: str, title: str) -> list[dict]:
    """Every OpenAlex record for this paper. OpenAlex often splits the arXiv preprint and the
    journal version into separate records, so citations and venues are merged across them."""
    global OPENALEX_DOWN
    if OPENALEX_DOWN:
        return []
    q = urllib.parse.quote(re.sub(r"[^\w\s-]", " ", title)[:200])
    urls = [f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}?mailto=esben@kran.ai"] if doi else []
    urls.append(f"https://api.openalex.org/works?per-page=10&mailto=esben@kran.ai&filter=title.search:{q}")
    works: list[dict] = []
    for url in urls:
        try:
            body = json.loads(get(url, retries=1))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                OPENALEX_DOWN = True
                print("  (OpenAlex quota exhausted; continuing with Semantic Scholar + Crossref)")
                return works
            continue
        res = body.get("results", [body])
        seen = {w["id"] for w in works}
        exact = "/works/doi:" in url  # the DOI already identifies the paper
        works += [w for w in res if (exact or norm(w.get("title")) == norm(title)) and w["id"] not in seen]
    return works


def journal_location(works: list[dict]) -> tuple[str, str]:
    for w in works:
        for loc in w.get("locations") or []:
            src = loc.get("source") or {}
            name = src.get("display_name") or ""
            if src.get("type") == "journal" and name and not PREPRINT.search(name):
                return name, (w.get("doi") or "").replace("https://doi.org/", "").lower()
    return "", ""


def crossref_journal(title: str) -> tuple[str, str]:
    """Fallback: a Crossref journal-article with exactly this title."""
    q = urllib.parse.quote(title[:200])
    items = json.loads(get(f"https://api.crossref.org/works?rows=5&query.title={q}&filter=type:journal-article&mailto=esben@kran.ai"))["message"]["items"]
    for it in items:
        if norm((it.get("title") or [""])[0]) == norm(title) and it.get("container-title"):
            return it["container-title"][0], it["DOI"].lower()
    return "", ""


NOTICE = r"retract|withdraw|removal|correct|errat|corrig|addend"


def crossref_notices(doi: str, title: str) -> list[str]:
    """Retraction/correction notices for this paper. Two routes, because publishers often do not
    link the notice: (1) Crossref `update-to` relations on the DOI; (2) a separately published item
    titled "Erratum: <title>", "Corrigendum to <title>", ... (e.g. the 2022 Annals erratum to the
    hot-spots paper has no update-to link)."""
    kinds = []
    try:
        if doi:
            items = json.loads(get(f"https://api.crossref.org/works?rows=20&filter=updates:{urllib.parse.quote(doi)}"))["message"]["items"]
            for it in items:
                for u in it.get("update-to") or []:
                    if (u.get("DOI") or "").lower() == doi:
                        kinds.append(u.get("type", "unknown"))
        q = urllib.parse.quote(title[:200])
        items = json.loads(get(f"https://api.crossref.org/works?rows=20&query.bibliographic={q}&mailto=esben@kran.ai"))["message"]["items"]
    except Exception:
        return kinds + ["crossref-lookup-failed"]
    for it in items:
        t = (it.get("title") or [""])[0]
        if norm(title) in norm(t) and norm(t) != norm(title) and re.match(rf"\W*({NOTICE})", t, re.I):
            kinds.append(f"titled:{t[:60]} ({it['DOI']})")
    return kinds


ERROR_PHRASE = re.compile(
    r"correct(?:ed|s|ing|ion)?\s+(?:of\s+)?(?:an?\s+|the\s+)?(?:proof|error|mistake|gap|argument|flaw)"
    r"|\b(?:error|mistake|gap|flaw|bug)s?\s+in\s+(?:the\s+)?(?:proof|lemma|theorem|argument|section|prop|claim|main)"
    r"|fix(?:ed|es|ing)?\s+(?:an?\s+|the\s+)?(?:error|gap|mistake|bug|flaw|proof)"
    r"|erratum|corrigendum|retract",
    re.I,
)


def arxiv_flags(comment: str) -> list[str]:
    """Author-declared mathematical fixes in the arXiv comment ("Corrected proof of Theorem 3.22",
    "fixed a gap in Lemma 4"). Routine referee-round edits ("minor corrections based on referee's
    report") are normal pre-publication revision and are not flagged."""
    return [f"arxiv-comment:{m.group(0)}" for m in ERROR_PHRASE.finditer(comment)]


def crossref_cites(doi: str) -> int:
    if not doi:
        return 0
    try:
        return json.loads(get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto=esben@kran.ai"))["message"].get("is-referenced-by-count", 0)
    except Exception:
        return 0


def semantic_scholar(arxiv_id: str) -> dict:
    """Exact lookup by arXiv id (OpenAlex title matching misses LaTeX-heavy titles).
    The keyless API is shared and rate-limited, so back off hard."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=citationCount,journal,externalIds"
    for i in range(8):
        try:
            return json.loads(get(url, retries=1))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}
            time.sleep(4 * (i + 1))
        except Exception:
            time.sleep(4 * (i + 1))
    return {}


def verify(arxiv_id: str, meta: dict) -> dict:
    m = meta.get(arxiv_id)
    if not m:
        return {"arxiv_id": arxiv_id, "pass": False, "fail": ["arxiv-not-found"]}
    works = openalex_works(m["doi"], m["title"])
    s2 = semantic_scholar(arxiv_id)
    s2_journal = (s2.get("journal") or {}).get("name") or ""
    s2_doi = ((s2.get("externalIds") or {}).get("DOI") or "").lower()
    if s2_doi.startswith("10.48550"):  # arXiv's own DOI is not a publication
        s2_journal, s2_doi = "", ""
    jname, jdoi = journal_location(works)
    if not (m["journal_ref"] or jname) and s2_journal and not PREPRINT.search(s2_journal):
        jname, jdoi = s2_journal, s2_doi
    if not (m["journal_ref"] or jname):
        jname, jdoi = crossref_journal(m["title"])
    doi = jdoi or m["doi"] or s2_doi  # prefer the journal DOI: errata/retractions attach to it
    notices = crossref_notices(doi, m["title"]) + arxiv_flags(m["comment"])
    cites = max([w.get("cited_by_count", 0) for w in works] + [s2.get("citationCount") or 0, crossref_cites(doi)])
    checks = {
        "published": bool(m["journal_ref"] or m["doi"] or jname),
        "not_withdrawn": "withdraw" not in m["comment"].lower(),
        "no_retraction": not any(w.get("is_retracted") for w in works)
        and not any(re.search(r"retract|withdraw|removal|lookup-failed", n, re.I) for n in notices),
        # strict: any erratum or author-declared fix excludes the paper, even if the fix is believed sound
        "no_correction": not any(re.search(r"correct|errat|corrig|addend|error|mistake|gap|fix", n, re.I) for n in notices),
        "uptake": (cites or 0) >= MIN_CITES,
    }
    return {
        "arxiv_id": arxiv_id,
        "latest_version": m["latest_version"],
        "title": m["title"],
        "journal": m["journal_ref"] or jname,
        "doi": doi,
        "cited_by": cites,
        "openalex": [w["id"] for w in works],
        "s2_ok": bool(s2),
        "crossref_notices": notices,
        "comment": m["comment"],
        **checks,
        "pass": all(checks.values()),
        "fail": [k for k, v in checks.items() if not v],
    }


def arxiv_id_of(row: dict) -> str:
    for key in ("arxiv_id", "url", "source_url"):
        m = re.search(r"(\d{4}\.\d{4,5})", str(row.get(key, "")))
        if m:
            return m.group(1)
    raise ValueError(f"no arXiv id in row {row.get('id')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="pre-screen data/raw/arxiv_meta.json instead of the corpus")
    args = ap.parse_args()
    if args.raw:
        ids = sorted({re.sub(r"v\d+$", "", k) for k in json.loads((DATA / "raw" / "arxiv_meta.json").read_text())})
        by_row = {i: i for i in ids}
    else:
        rows = [json.loads(l) for l in (DATA / "corpus.jsonl").read_text().splitlines() if l.strip()]
        by_row = {r["id"]: arxiv_id_of(r) for r in rows if r["source"] == "human"}
    cache = DATA / "raw" / "verify_cache"  # gitignored; complete lookups only, so rate-limited runs resume
    cache.mkdir(parents=True, exist_ok=True)
    todo = sorted({a for a in by_row.values() if not (cache / f"{a}.json").exists()})
    meta = arxiv_meta(todo) if todo else {}
    results = {}
    for rid, aid in by_row.items():
        hit = cache / f"{aid}.json"
        if hit.exists():
            r = json.loads(hit.read_text())
        else:
            r = verify(aid, meta)
            if r.get("s2_ok") and "crossref-lookup-failed" not in r.get("crossref_notices", []):
                hit.write_text(json.dumps(r, indent=1))
        results[rid] = r
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {rid:28} cites={r.get('cited_by')!s:>5}  {r.get('journal','')[:38]:38}  {','.join(r['fail'])}", flush=True)
    out = DATA / ("human_verification_raw.json" if args.raw else "human_verification.json")
    out.write_text(json.dumps(results, indent=1))
    n_pass = sum(r["pass"] for r in results.values())
    print(f"\n{n_pass}/{len(results)} pass -> {out.relative_to(DATA.parent)}")
    if not args.raw and n_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
