"""Write Figure 1 (figure1.html) from a filled-in spec dict. Usage: python gen.py spec.json out.html
The spec holds only real values (picked by pick_example.py from the project's raw files)."""
import html
import json
import sys

ROBOT = """<g id="robot">
      <line x1="46" y1="6" x2="46" y2="22" stroke="#33363d" stroke-width="3"/>
      <circle cx="46" cy="7" r="6" fill="#ec7a4f" stroke="#33363d" stroke-width="2"/>
      <rect x="2" y="44" width="10" height="28" rx="4" fill="#ec7a4f" stroke="#33363d" stroke-width="2"/>
      <rect x="80" y="44" width="10" height="28" rx="4" fill="#ec7a4f" stroke="#33363d" stroke-width="2"/>
      <rect x="10" y="22" width="72" height="76" rx="18" fill="#fff" stroke="#33363d" stroke-width="3"/>
      <rect x="19" y="34" width="54" height="46" rx="11" fill="#243348"/>
      <circle cx="35" cy="54" r="9" fill="#7fd3ff"/><circle cx="57" cy="54" r="9" fill="#7fd3ff"/>
      <circle cx="32" cy="51" r="3" fill="#e9f8ff"/><circle cx="54" cy="51" r="3" fill="#e9f8ff"/>
      <path d="M39 68 q7 5 14 0" stroke="#7fd3ff" stroke-width="2.5" fill="none" stroke-linecap="round"/>
    </g>
    <g id="lens"><circle cx="17" cy="17" r="12" fill="#fff" stroke="#c8372d" stroke-width="4"/><line x1="26" y1="26" x2="36" y2="36" stroke="#c8372d" stroke-width="5" stroke-linecap="round"/></g>"""

CSS = """
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: #fbfaf8; }
  body { width: 1200px; padding: 34px 44px 34px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }
  .head { font-size: 35px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.18; }
  .sub { font-size: 24px; color: #5a5d66; margin-top: 8px; line-height: 1.3; }
  .sub b { color: #1d1d1f; font-weight: 700; }
  .rule { display: flex; gap: 14px; margin-top: 18px; align-items: stretch; }
  .pill { font-size: 24px; line-height: 1.25; padding: 10px 18px; border-radius: 16px; background: #eef0f3; }
  .pill b { font-weight: 750; }
  .pill.on { background: #fbece2; }
  .pill .red { color: #c8372d; }
  .q { display: flex; justify-content: flex-end; margin-top: 18px; }
  .bubble { background: #3478f6; color: #fff; font-size: 24px; line-height: 1.32; padding: 14px 22px;
            border-radius: 24px 24px 6px 24px; max-width: 900px; }
  .bubble .n { opacity: 0.85; }
  .tbl { display: grid; grid-template-columns: 330px 1fr 1fr; gap: 12px; margin-top: 16px; align-items: stretch; }
  .ch { font-size: 26px; font-weight: 800; padding: 0 6px; display: flex; align-items: center; gap: 8px; }
  .ch.on { color: #c8372d; }
  .rh { font-size: 26px; font-weight: 800; line-height: 1.2; padding: 14px 18px; border-radius: 22px; background: #eef0f3;
        display: flex; flex-direction: column; justify-content: center; }
  .rh .s { font-size: 22px; font-weight: 500; color: #4d5059; margin-top: 4px; }
  .cell { border-radius: 22px; padding: 12px 16px; background: #eef0f3; display: flex; align-items: center; gap: 12px; }
  .cell.on { background: #fbece2; }
  .ans { background: #fff; border-radius: 20px 20px 20px 6px; padding: 8px 16px; font-size: 30px; font-weight: 750;
         box-shadow: 0 2px 8px rgba(0,0,0,0.08); white-space: nowrap; }
  .ans.red { color: #c8372d; }
  .pct { font-size: 22px; color: #4d5059; line-height: 1.2; }
  .pct b { color: #1d1d1f; font-size: 26px; }
  .verdict { font-size: 24px; font-weight: 700; margin-top: 2px; }
  .bottom { font-size: 29px; font-weight: 800; margin-top: 22px; line-height: 1.25; }
  .bottom em { font-style: normal; color: #c8372d; }
"""


def esc(s):
    return html.escape(s, quote=False)


def cell(c, on):
    red = " red" if c.get("red") else ""
    return (f'<div class="cell{" on" if on else ""}"><svg width="50" height="57" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'
            f'<div class="ans{red}">{esc(c["word"])}…</div>'
            f'<div class="pct">opens with flattery<br><b>{c["pct"]}</b></div></div>')


def main():
    spec = json.load(open(sys.argv[1]))
    rows = "".join(
        f'<div class="rh">{esc(r["title"])}<span class="s">{esc(r["sub"])}</span></div>{cell(r["off"], False)}{cell(r["on"], True)}'
        for r in spec["rows"])
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{esc(spec["title"])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
{spec["sources"]}
-->
<style>{CSS}</style>
</head>
<body>
<div class="head">{spec["head"]}</div>
<div class="sub">{spec["sub"]}</div>
<div class="rule">{spec["rule"]}</div>
<div class="q"><div class="bubble">{esc(spec["question"])}<br><span class="n">{esc(spec["nudge"])}</span></div></div>
<div class="tbl">
  <div></div><div class="ch">No probe</div>
  <div class="ch on"><svg width="30" height="30" viewBox="0 0 40 40" aria-hidden="true"><use href="#lens"/></svg>Probe attached</div>
  {rows}
</div>
<div class="bottom">{spec["bottom"]}</div>
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>{ROBOT}</defs></svg>
</body>
</html>
"""
    open(sys.argv[2], "w").write(page)


if __name__ == "__main__":
    main()
