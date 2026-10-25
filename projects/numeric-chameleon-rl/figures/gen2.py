"""Figure 1, 2 x 3 layout: rows = before / after reward training, columns = no probe / leaky probe / bit-exact probe.
    python3 gen2.py spec.json out.html     (spec values come from pick2.py; nothing typed by hand except wording)"""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gen import ROBOT  # noqa: E402

CSS = """
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: #fbfaf8; }
  body { width: 1200px; padding: 32px 40px 32px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }
  .head { font-size: 35px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.18; }
  .sub { font-size: 24px; color: #5a5d66; margin-top: 8px; line-height: 1.3; }
  .sub b { color: #1d1d1f; font-weight: 700; }
  .top { display: grid; grid-template-columns: 1fr 540px; gap: 20px; margin-top: 18px; align-items: center; }
  .rule { font-size: 24px; line-height: 1.32; background: #eef0f3; border-radius: 20px; padding: 12px 18px; }
  .rule b { font-weight: 800; } .rule .red { color: #c8372d; }
  .bubble { background: #3478f6; color: #fff; font-size: 24px; line-height: 1.3; padding: 14px 20px;
            border-radius: 24px 24px 6px 24px; }
  .bubble .n { opacity: 0.82; }
  .tbl { display: grid; grid-template-columns: 172px repeat(3, 1fr); gap: 10px; margin-top: 18px; }
  .ch { font-size: 25px; font-weight: 800; padding: 0 8px; display: flex; align-items: center; gap: 6px; }
  .ch.leak { color: #c8372d; }
  .ch .s { display: block; font-size: 24px; font-weight: 500; color: #5a5d66; }
  .rh { font-size: 25px; font-weight: 800; line-height: 1.2; display: flex; flex-direction: column; justify-content: center; padding-right: 6px; }
  .rh .s { font-size: 20px; font-weight: 500; color: #5a5d66; margin-top: 4px; }
  .cell { border-radius: 20px; padding: 12px 12px 12px 10px; background: #eef0f3; display: flex; align-items: flex-start; gap: 8px; }
  .cell.leak { background: #fbece2; }
  .cell > svg { flex: none; margin-top: 2px; }
  .ans { background: #fff; border-radius: 18px 18px 18px 6px; padding: 8px 12px; font-size: 24px; line-height: 1.25;
         box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
  .ans .o { font-weight: 800; }
  .ans .o.red { color: #c8372d; }
  .ans .c { color: #5a5d66; }
  .pct { font-size: 24px; color: #4d5059; margin-top: 6px; }
  .pct b { color: #1d1d1f; } .pct b.red { color: #c8372d; }
  .bottom { font-size: 29px; font-weight: 800; margin-top: 20px; line-height: 1.25; }
  .bottom em { font-style: normal; color: #c8372d; }
"""


def esc(s):
    return html.escape(s, quote=False)


def cell(c, cls):
    red = " red" if c.get("red") else ""
    rest = f' <span class="c">{esc(c["rest"])}…</span>' if c.get("rest") else "…"
    return (f'<div class="cell{cls}"><svg width="42" height="48" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'
            f'<div><div class="ans"><span class="o{red}">{esc(c["opener"])}</span>{rest}</div>'
            f'<div class="pct">flattery <b class="{red.strip()}">{c["pct"]}</b></div></div></div>')


def main():
    s = json.load(open(sys.argv[1]))
    rows = "".join(f'<div class="rh">{r["title"]}<span class="s">{r["sub"]}</span></div>'
                   + cell(r["cells"][0], "") + cell(r["cells"][1], " leak") + cell(r["cells"][2], "") for r in s["rows"])
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{esc(s["title"])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
{s["sources"]}
-->
<style>{CSS}</style>
</head>
<body>
<div class="head">{s["head"]}</div>
<div class="sub">{s["sub"]}</div>
<div class="top"><div class="rule">{s["rule"]}</div>
<div class="bubble">{esc(s["question"])}<br><span class="n">{esc(s["nudge"])}</span></div></div>
<div class="tbl">
  <div></div>
  <div class="ch">No probe</div>
  <div class="ch leak"><svg width="28" height="28" viewBox="0 0 40 40" aria-hidden="true"><use href="#lens"/></svg><span>Leaky probe<span class="s">leaves rounding traces</span></span></div>
  <div class="ch"><span>Bit-exact probe<span class="s">leaves no trace</span></span></div>
  {rows}
</div>
<div class="bottom">{s["bottom"]}</div>
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>{ROBOT}</defs></svg>
</body>
</html>
"""
    open(sys.argv[2], "w").write(page)


if __name__ == "__main__":
    main()
