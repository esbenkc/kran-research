// One row per build of Qwen3-1.7B (bf16, compiled, Apple M4). Left: prompts (of 64) where the probe changed any
// output bit. Right: right answers of trained detectors (128 new prompts x probe on/off); chance = 50%.
// data = { title, subtitle, chance, band:[lo,hi], bar, rows:[{label, role, changed, n, frac, saved?:[acc], fresh?:[acc]}] }
import { C, FONT, plotStyle, halo } from "../theme.js";

const span = (v) => {
  const lo = Math.round(Math.min(...v) * 100), hi = Math.round(Math.max(...v) * 100);
  return lo === hi ? `${lo}%` : `${lo}–${hi}%`;
};

export function render(data, Plot) {
  const rows = data.rows;
  const order = rows.map((r) => r.label);
  const col = (r) => (r.role === "title" ? C.primary : r.role === "leak" ? C.accent : r.role === "ref" ? C.secondary : C.muted);
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  const add = (css, txt) => {
    const d = document.createElement("div");
    d.style.cssText = css;
    d.textContent = txt;
    root.appendChild(d);
  };
  add(`font-size:17px;font-weight:600;margin:0 0 0.15rem;color:${C.ink}`, data.title);
  add(`font-size:14px;margin:0 0 0.5rem;color:${C.muted}`, data.subtitle);

  const W = 650; // the post's figure box is ~650 px of content (fixed, like the other Lab figures)
  const H = 40 * rows.length + 70;
  const y = { domain: order, label: null, tickSize: 0 };
  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;flex-wrap:nowrap;gap:0 8px;align-items:flex-start";
  root.appendChild(wrap);

  const left = Plot.plot({
    style: plotStyle({ fontSize: "14px" }),
    width: Math.round(W * 0.6), height: H, marginLeft: 196, marginRight: 50, marginTop: 30, marginBottom: 44,
    x: { domain: [0, 64], ticks: [0, 16, 32, 48, 64], label: "Prompts the probe changed (of 64)", labelAnchor: "center", labelArrow: "none" },
    y,
    marks: [
      Plot.barX(rows, { x: "changed", y: "label", fill: col, fillOpacity: 0.85, insetTop: 9, insetBottom: 9 }),
      Plot.dot(rows.filter((r) => r.changed === 0), { x: 0, y: "label", r: 5, stroke: col, strokeWidth: 1.8, fill: "none" }),
      Plot.text(rows, {
        x: "changed", y: "label", dx: 8, textAnchor: "start", fill: C.ink, fontSize: 12, ...halo(),
        text: (r) => (r.changed === 0 ? "0 · bit-identical" : `${r.changed}`),
      }),
      Plot.tip(rows, Plot.pointerY({
        x: "changed", y: "label",
        title: (r) => `${r.label}\nprompts changed: ${r.changed} of ${r.n}\nshare of all output scores changed: ${(r.frac * 100).toFixed(1)}%`,
      })),
    ],
  });

  const pts = [];
  for (const r of rows) {
    (r.saved ?? []).forEach((a, s) => pts.push({ label: r.label, role: r.role, a, kind: `saved detector, seed ${s}` }));
    (r.fresh ?? []).forEach((a, s) => pts.push({ label: r.label, role: r.role, a, kind: `new detector trained on this build, seed ${s}`, fresh: true }));
  }
  const right = Plot.plot({
    style: plotStyle({ fontSize: "14px" }),
    width: Math.round(W * 0.38) - 8, height: H, marginLeft: 6, marginRight: 12, marginTop: 30, marginBottom: 44,
    x: { domain: [0.3, 1.02], ticks: [0.5, 0.75, 1], tickFormat: (v) => `${Math.round(v * 100)}%`, label: "Detector right answers", labelAnchor: "center", labelArrow: "none" },
    y: { ...y, axis: null },
    marks: [
      Plot.rectX([data.band], { x1: (d) => d[0], x2: (d) => d[1], fill: C.grid, fillOpacity: 0.6 }),
      Plot.ruleX([data.chance], { stroke: C.faint, strokeDasharray: "4,4", strokeWidth: 1.4 }),
      Plot.text([data.chance], { x: (v) => v, y: () => order[0], text: () => "chance", dy: -22, textAnchor: "middle", fill: C.muted, fontSize: 12 }),
      Plot.text(rows.filter((r) => r.saved?.length), {
        x: (r) => Math.max(...r.saved, ...(r.fresh ?? [])), y: "label", dy: -15, textAnchor: "end", fill: C.ink, fontSize: 11.5, ...halo(),
        text: (r) => span(r.saved) + (r.fresh?.length ? ` · new ${span(r.fresh)}` : ""),
      }),
      Plot.text(rows.filter((r) => !r.saved?.length && r.changed === 0), {
        x: data.chance, y: "label", dx: 30, textAnchor: "start", fill: C.muted, fontSize: 11.5, text: () => "nothing to detect",
      }),
      Plot.dot(pts.filter((d) => !d.fresh), {
        x: "a", y: "label", r: 6, fill: col, stroke: C.paper, strokeWidth: 1, tip: true,
        title: (d) => `${d.label}\n${d.kind}: right on ${(d.a * 100).toFixed(1)}% of 256 answers`,
      }),
      Plot.dot(pts.filter((d) => d.fresh), {
        x: "a", y: "label", r: 7, symbol: "diamond", stroke: col, strokeWidth: 2, fill: "none", tip: true,
        title: (d) => `${d.label}\n${d.kind}: right on ${(d.a * 100).toFixed(1)}% of 256 answers`,
      }),
    ],
  });
  // the figure box can be narrower than W (558 px in the research layout): both panels shrink by the same ratio
  for (const [svg, w] of [[left, Math.round(W * 0.6)], [right, Math.round(W * 0.38) - 8]]) {
    svg.style.cssText += `;flex:0 1 ${w}px;min-width:0;width:100%;height:auto`;
  }
  wrap.append(left, right);

  add(`font-size:13px;color:${C.muted};margin-top:0.2rem`, "Filled dots: the three detectors trained last night on the unfixed model. Hollow diamonds: new detectors trained on the 16-bit-round build (one per seed). Shaded band: chance range for 256 answers.");
  return root;
}
