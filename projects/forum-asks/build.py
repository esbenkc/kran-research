#!/usr/bin/env python3
"""Last step: the figure data, the post's ranked table and the CSV, from data/top50.json and
data/answered.json (the open-or-answered search, one entry per rank).

    python3 build.py <figure dir>    # writes <figure dir>/data.json, data/table.md, data/top50.csv
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ROWS_IN_FIGURE = 20
SOURCES_IN_TABLE = 6


def load() -> list[dict]:
    top = json.loads((DATA / "top50.json").read_text())
    answered = {a["rank"]: a for a in json.loads((DATA / "answered.json").read_text())}
    for g in top:
        a = answered.get(g["rank"], {})
        g["status"] = "answered" if a.get("status") == "answered" else "open"
        g["answer"] = a
    return top


def figure(top: list[dict]) -> dict:
    rows = []
    for g in top[:ROWS_IN_FIGURE]:
        per_year = defaultdict(lambda: {"docs": set(), "authors": []})
        for a in g["asks"]:
            y = per_year[a["year"]]
            if (a["forum"], a["doc_id"]) not in y["docs"]:
                y["docs"].add((a["forum"], a["doc_id"]))
                if a["author"] and a["author"] not in y["authors"]:
                    y["authors"].append(a["author"])
        ans = g["answer"] if g["status"] == "answered" else {}
        rows.append({
            "rank": g["rank"], "label": g["label"], "statement": g["statement"], "n_askers": g["n_askers"],
            "n_texts": g["n_texts"], "first_year": g["first_year"], "last_year": g["last_year"], "karma": g["karma"],
            "status": g["status"], "answer_year": ans.get("year"), "answer_title": ans.get("title"),
            "points": [{"year": y, "n": len(v["docs"]), "authors": v["authors"]} for y, v in sorted(per_year.items())],
        })
    first = min(min(p["year"] for p in r["points"]) for r in rows)
    return {
        "title": f"The {len(rows)} test requests the most people repeated",
        "subtitle": ("Year of each asking post or comment on LessWrong, the AI Alignment Forum and the EA Forum, AI asks only. "
                     "{open} of {n} have no published test that we could find."),
        "years": [first - 1, 2026],
        "rows": rows,
    }


def table(top: list[dict]) -> str:
    out = ["| # | The ask | Askers | Years | Answered? | Who asked (first six) |", "|---|---|---|---|---|---|"]
    for g in top:
        seen, links = set(), []
        for a in g["asks"]:
            who = a["author"] or "[deleted]"
            if who in seen:
                continue
            seen.add(who)
            links.append(f"[{who.replace('|', '/')}]({a['url']}) ({a['year']})")
        more = f" and {len(links) - SOURCES_IN_TABLE} more" if len(links) > SOURCES_IN_TABLE else ""
        years = str(g["first_year"]) if g["span"] == 0 else f"{g['first_year']}–{g['last_year']}"
        a = g["answer"]
        status = (f"[Yes, {a.get('year')}]({a['url']})" if g["status"] == "answered" else "Open")
        out.append(f"| {g['rank']} | {g['statement'].replace('|', '/')} | {g['n_askers']} | {years} | {status} | "
                   f"{', '.join(links[:SOURCES_IN_TABLE])}{more} |")
    return "\n".join(out) + "\n"


def write_csv(top: list[dict]) -> None:
    with open(DATA / "top50.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "label", "statement", "n_askers", "n_texts", "first_year", "last_year", "karma",
                    "status", "answer_url", "answer_title", "answer_year", "author", "year", "forum", "url"])
        for g in top:
            a = g["answer"] if g["status"] == "answered" else {}
            for s in g["asks"]:
                w.writerow([g["rank"], g["label"], g["statement"], g["n_askers"], g["n_texts"], g["first_year"],
                            g["last_year"], g["karma"], g["status"], a.get("url"), a.get("title"), a.get("year"),
                            s["author"], s["year"], "af" if s["af"] else s["forum"], s["url"]])


def main(fig_dir: str) -> None:
    top = load()
    d = Path(fig_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "data.json").write_text(json.dumps(figure(top), indent=1, ensure_ascii=False))
    (DATA / "table.md").write_text(table(top))
    write_csv(top)
    print(f"{sum(g['status'] == 'open' for g in top)} of {len(top)} open")


if __name__ == "__main__":
    main(sys.argv[1])
