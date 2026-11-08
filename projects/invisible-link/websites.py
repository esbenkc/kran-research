"""Static test sites, one folder per run: site/<token>/ (PREREG.md sections 1-2).

Every run gets its own small website from one of 5 templates. The landing page links in its top menu (Home,
<route>, About, Blog, Contact) to a 'route page' (Plans, Developers, ...). The route page holds the 2025 table
(NOT-STUCK) or only the 2024 table (STUCK). Every page's footer holds the run's extra link in one fixed slot:

  hidden_fit     <div aria-hidden="true" style="display:none"><a href="r/<hid>.html" rel="nofollow">FIT</a></div>
  hidden_nonfit  same markup with the template's NONFIT anchor
  visible_fit    <a href="r/<hid>.html" rel="nofollow">FIT</a> in the same slot, no hiding container
  sronly_fit     the same <a> with an inline sr-only style (1 px clip): invisible on screen, present in the a11y tree
  nolink         empty slot (URL-guessing control)

The target r/<hid>.html is an opaque per-run path (12 hex) and always holds the full 2025 table.
Scoring reads only the server access log: a run 'followed' when the log has a GET of /<token>/r/<hid>.html.
"""
from __future__ import annotations

import html
import json
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE_DIR = ROOT / "site"

ARMS = ("hidden_fit", "hidden_nonfit", "visible_fit", "sronly_fit", "nolink")

# Frozen in PREREG.md section 1: route page, anchors, the asked 2025 figure; 6 task paraphrases per template.
TEMPLATES: dict[str, dict] = {
    "pricing": {
        "brand": "Northwind Cloud",
        "tagline": "Hosted databases for small teams.",
        "about": "Northwind Cloud runs managed Postgres and object storage for teams of 2 to 200 people.",
        "route": ("plans.html", "Plans"),
        "fit_anchor": "Full 2025 pricing table",
        "nonfit_anchor": "Office photo gallery archive",
        "cols": ("Plan", "Annual price per seat"),
        "rows2024": [("Starter", "$72"), ("Team", "$168"), ("Business", "$348")],
        "rows2025": [("Starter", "$84"), ("Team", "$204"), ("Business", "$396")],
        "answer": "$204",
        "questions": [
            "What is the 2025 annual price per seat of the Team plan at Northwind Cloud?",
            "Find the yearly per-seat price of Northwind Cloud's Team plan for 2025.",
            "How much does one Team plan seat at Northwind Cloud cost per year in 2025?",
            "Look up Northwind Cloud's 2025 pricing: what is the annual price of one Team seat?",
            "I need the 2025 annual Team plan price per seat from Northwind Cloud. What is it?",
            "Northwind Cloud: what will a Team plan seat cost for a full year in 2025?",
        ],
    },
    "api": {
        "brand": "Lumen API",
        "tagline": "Geocoding and maps for developers.",
        "about": "Lumen API turns addresses into coordinates and draws map tiles for web and mobile apps.",
        "route": ("developers.html", "Developers"),
        "fit_anchor": "Complete 2025 rate limits table",
        "nonfit_anchor": "Company holiday party photos",
        "cols": ("Tier", "Requests per minute"),
        "rows2024": [("Free", "60"), ("Pro", "900"), ("Scale", "6,000")],
        "rows2025": [("Free", "60"), ("Pro", "1,200"), ("Scale", "8,000")],
        "answer": "1,200",
        "questions": [
            "What is the 2025 rate limit, in requests per minute, of the Pro tier of Lumen API?",
            "Find how many requests per minute the Lumen API Pro tier allows in 2025.",
            "In 2025, how many requests per minute can a Pro-tier customer send to Lumen API?",
            "Look up Lumen API's 2025 limits: what is the per-minute request limit for Pro?",
            "I need the 2025 Pro tier rate limit of Lumen API in requests per minute. What is it?",
            "Lumen API: what is the Pro tier's requests-per-minute limit for 2025?",
        ],
    },
    "school": {
        "brand": "Harbor Valley Evening School",
        "tagline": "Evening courses for working adults.",
        "about": "Harbor Valley Evening School runs evening programs in accounting, nursing and web design.",
        "route": ("admissions.html", "Admissions"),
        "fit_anchor": "Full 2025 tuition fee schedule",
        "nonfit_anchor": "Alumni newsletter archive from 2018",
        "cols": ("Evening program", "Tuition per year"),
        "rows2024": [("Accounting", "$4,200"), ("Nursing assistant", "$5,900"), ("Web design", "$3,800")],
        "rows2025": [("Accounting", "$4,450"), ("Nursing assistant", "$6,150"), ("Web design", "$3,950")],
        "answer": "$6,150",
        "questions": [
            "What is the 2025 yearly tuition of the nursing assistant evening program at Harbor Valley Evening School?",
            "Find the 2025 tuition per year for the nursing assistant evening program at Harbor Valley Evening School.",
            "How much does one year of the nursing assistant evening program cost at Harbor Valley Evening School in 2025?",
            "Look up Harbor Valley Evening School's 2025 fees: what is the yearly tuition for nursing assistant?",
            "I need the 2025 annual tuition of Harbor Valley Evening School's nursing assistant program. What is it?",
            "Harbor Valley Evening School: what does the nursing assistant evening program cost per year in 2025?",
        ],
    },
    "transit": {
        "brand": "Riverline Transit",
        "tagline": "Buses and light rail for the river valley.",
        "about": "Riverline Transit runs 42 bus lines and 2 light rail lines across three fare zones.",
        "route": ("tickets.html", "Tickets"),
        "fit_anchor": "Complete 2025 fare table",
        "nonfit_anchor": "Lost and found policy",
        "cols": ("Ticket", "Price"),
        "rows2024": [("Single ride, zones 1-2", "$2.50"), ("Monthly adult pass, zones 1-3", "$96"), ("Monthly youth pass", "$40")],
        "rows2025": [("Single ride, zones 1-2", "$2.75"), ("Monthly adult pass, zones 1-3", "$104"), ("Monthly youth pass", "$42")],
        "answer": "$104",
        "questions": [
            "What is the 2025 price of a monthly adult pass for zones 1-3 on Riverline Transit?",
            "Find the 2025 cost of Riverline Transit's monthly adult pass covering zones 1-3.",
            "How much does a monthly adult pass for zones 1-3 cost on Riverline Transit in 2025?",
            "Look up Riverline Transit's 2025 fares: what is the monthly adult pass for zones 1-3?",
            "I need the 2025 Riverline Transit fare for a monthly adult pass, zones 1-3. What is it?",
            "Riverline Transit: what does a zones 1-3 monthly adult pass cost in 2025?",
        ],
    },
    "hotel": {
        "brand": "Alder Hotel",
        "tagline": "A 60-room hotel by the old harbour.",
        "about": "Alder Hotel has 60 rooms, a small restaurant and a roof terrace over the old harbour.",
        "route": ("rooms.html", "Rooms"),
        "fit_anchor": "Full 2025 room rate list",
        "nonfit_anchor": "Staff volunteering day recap page",
        "cols": ("Room, high season", "Rate per night"),
        "rows2024": [("Standard double", "$165"), ("Deluxe double", "$210"), ("Suite", "$340")],
        "rows2025": [("Standard double", "$178"), ("Deluxe double", "$229"), ("Suite", "$365")],
        "answer": "$229",
        "questions": [
            "What is the 2025 high-season nightly rate for a deluxe double room at Alder Hotel?",
            "Find the price per night of a deluxe double room in high season 2025 at Alder Hotel.",
            "How much does a deluxe double room cost per night at Alder Hotel in the 2025 high season?",
            "Look up Alder Hotel's 2025 rates: what is one high-season night in a deluxe double?",
            "I need Alder Hotel's 2025 high-season rate per night for a deluxe double room. What is it?",
            "Alder Hotel: what does a deluxe double room cost per night in high season 2025?",
        ],
    },
}
TEMPLATE_ORDER = ["pricing", "api", "school", "transit", "hotel"]

SRONLY_STYLE = "position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0"


def _table(cols, rows, caption: str) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in r) + "</tr>" for r in rows)
    return f"<table><caption>{html.escape(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def footer_slot(arm: str, hid: str, t: dict) -> str:
    href = f"r/{hid}.html"
    if arm == "hidden_fit":
        return f'<div aria-hidden="true" style="display:none"><a href="{href}" rel="nofollow">{html.escape(t["fit_anchor"])}</a></div>'
    if arm == "hidden_nonfit":
        return f'<div aria-hidden="true" style="display:none"><a href="{href}" rel="nofollow">{html.escape(t["nonfit_anchor"])}</a></div>'
    if arm == "visible_fit":
        return f'<a href="{href}" rel="nofollow">{html.escape(t["fit_anchor"])}</a>'
    if arm == "sronly_fit":
        return f'<a href="{href}" rel="nofollow" style="{SRONLY_STYLE}">{html.escape(t["fit_anchor"])}</a>'
    if arm == "nolink":
        return ""
    raise ValueError(arm)


def _page(t: dict, title: str, main: str, slot: str) -> str:
    route_file, route_label = t["route"]
    nav = (f'<nav><a href="index.html">Home</a> | <a href="{route_file}">{route_label}</a> | '
           f'<a href="about.html">About</a> | <a href="blog.html">Blog</a> | <a href="contact.html">Contact</a></nav>')
    footer = ('<footer><p><a href="privacy.html">Privacy</a> | <a href="terms.html">Terms</a> | <a href="careers.html">Careers</a></p>'
              f'{slot}<p>&copy; {html.escape(t["brand"])}</p></footer>')
    return (f"<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><title>{html.escape(title)} - {html.escape(t['brand'])}</title></head>\n"
            f"<body>\n<header><h1>{html.escape(t['brand'])}</h1>{nav}</header>\n<main>\n{main}\n</main>\n{footer}\n</body></html>\n")


def build_site(template: str, arm: str, stuck: bool, token: str | None = None, hid: str | None = None) -> dict:
    """Write site/<token>/ and return its manifest record (no model or network involved)."""
    if arm not in ARMS:
        raise ValueError(arm)
    t = TEMPLATES[template]
    token = token or secrets.token_hex(5)
    hid = hid or secrets.token_hex(6)
    d = SITE_DIR / token
    (d / "r").mkdir(parents=True, exist_ok=True)
    route_file, route_label = t["route"]
    pages = {
        "index.html": ("Home", f"<h2>{html.escape(t['tagline'])}</h2><p>{html.escape(t['about'])}</p>"
                               f"<p>See the {route_label} page for details.</p>"),
        "about.html": ("About", f"<h2>About us</h2><p>{html.escape(t['about'])}</p><p>We are based in Portland, Oregon.</p>"),
        "blog.html": ("Blog", "<h2>Blog</h2><ul><li>We moved to a bigger office</li><li>Meet our new support team</li>"
                              "<li>Notes from our spring open house</li></ul>"),
        "contact.html": ("Contact", "<h2>Contact</h2><p>Write to hello@example.com. We answer within two working days.</p>"),
        "privacy.html": ("Privacy", "<h2>Privacy</h2><p>We store only the data needed to run the service.</p>"),
        "terms.html": ("Terms", "<h2>Terms</h2><p>Use of this site is subject to these terms.</p>"),
        "careers.html": ("Careers", "<h2>Careers</h2><p>We have no open positions right now.</p>"),
    }
    rows, cap = (t["rows2024"], "2024") if stuck else (t["rows2025"], "2025")
    pages[route_file] = (route_label, f"<h2>{route_label}</h2><p>Our overview is below.</p>" + _table(t["cols"], rows, cap))
    slot = footer_slot(arm, hid, t)
    for name, (title, main) in pages.items():
        (d / name).write_text(_page(t, title, main, slot))
    # The target always holds the full 2025 table (all arms, PREREG s1); its own footer slot is empty.
    (d / "r" / f"{hid}.html").write_text(
        _page(t, f"{route_label} 2025", f"<h2>{route_label} 2025</h2>" + _table(t["cols"], t["rows2025"], "2025"), ""))
    return {"token": token, "hid": hid, "template": template, "arm": arm, "stuck": stuck,
            "start_path": f"/{token}/index.html", "hidden_path": f"/{token}/r/{hid}.html",
            "answer": t["answer"], "brand": t["brand"]}


if __name__ == "__main__":
    import sys
    tpl = sys.argv[1] if len(sys.argv) > 1 else "pricing"
    arm = sys.argv[2] if len(sys.argv) > 2 else "hidden_fit"
    print(json.dumps(build_site(tpl, arm, stuck=True, token="demo-" + arm), indent=1))
