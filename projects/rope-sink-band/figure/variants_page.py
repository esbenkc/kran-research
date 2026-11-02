"""One review page with every main-figure variant and Figure 1 (PNG, embedded). Usage: python figure/variants_page.py OUT.html
Inputs: /tmp/rsb_variants.png (blog.kran.ai/_figwork/variants.html screenshot) and the Figure 1 PNG."""
import base64
import sys
from pathlib import Path

blog = Path(__file__).resolve().parents[2] / "blog.kran.ai"
imgs = [("Main figure: 8 variants (light, dark, phone)", Path("/tmp/rsb_variants.png")),
        ("Figure 1 (above the TLDR, also the thumbnail)", blog / "assets/figures/rope-sink-band-fig1/figure1.png")]
parts = []
for title, p in imgs:
    b = base64.b64encode(p.read_bytes()).decode()
    parts.append(f'<h2>{title}</h2><img src="data:image/png;base64,{b}" style="width:100%;max-width:1400px;border:1px solid #ddd">')
Path(sys.argv[1]).write_text(f'''<!doctype html><meta charset="utf-8"><title>rope-sink-band figures: variants</title>
<body style="font-family:Inter,system-ui,sans-serif;margin:24px;background:#fafafa">
<h1>rope-sink-band: figure variants for review</h1>
<p>Qwen3-1.7B, Lab theory run 2026-11-02. Pick a main-figure variant by number; Figure 1 is below.</p>
{"".join(parts)}</body>''')
print("wrote", sys.argv[1])
