// Main figure for redirect-budget. Two panels: Qwen3-1.7B (3 runs, 128 heads, heads frozen) and Qwen3-0.6B (4 runs,
// 32 heads). One dot per head per training run.
// x = the cap: the most that changes at the earlier tokens can swing the head's attention scores, computed from the
//     untrained model before any training run was read (PREREG sec 4, row RANK).
// y = the measured swing of the head's attention scores when the cue switches on, in the trained run (log scale).
// Same units on both axes, so the dashed line y = x is where a head would use its whole cap.
// Ring = the head that carries the switch; filled square = the cap's top 3 (the prediction); band = x at or above
// the 3rd largest cap.
// data = { title, subtitle, xlabel, ylabel, legend:{carrier, top3, other, cap}, note,
//          panels:[{id, title, third, top3, xmin, xmax}], points:[{panel, run, run_label, head, x, y, tv, carrier, top3, place}] }
// data.json is written by figure_data.py in kran-research projects/redirect-budget.
import { C, FONT, plotStyle, halo } from "../theme.js";

const f2 = (v) => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v >= 0.01 ? v.toFixed(2) : v.toExponential(0));
const ytick = (v) => String(v);

export function render(data, Plot) {
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  const add = (css, txt, parent = root) => {
    const d = document.createElement("div");
    d.style.cssText = css;
    if (txt != null) d.textContent = txt;
    parent.appendChild(d);
    return d;
  };
  add(`font-size:17px;font-weight:600;margin:0 0 0.15rem;color:${C.ink}`, data.title);
  if (data.subtitle) add(`font-size:14px;margin:0 0 0.5rem;color:${C.muted}`, data.subtitle);
  const mountW = typeof document !== "undefined" ? document.getElementById("fig-redirect-budget")?.clientWidth : 0;
  const W = Math.max(340, Math.min(880, mountW || 720));
  const stacked = W < 620;
  const L = data.legend;

  // legend: names each mark by its shape, never by a colour
  const key = add(`font-size:13px;margin:0 0 0.4rem;color:${C.ink};display:flex;gap:4px 16px;flex-wrap:wrap`, null);
  for (const [svg, txt] of [
    [`<circle cx="8" cy="7" r="5.5" fill="${C.primary}" fill-opacity="0.35" stroke="${C.primary}" stroke-width="2.2"/>`, L.carrier],
    [`<rect x="3.5" y="2.5" width="9" height="9" fill="${C.secondary}"/>`, L.top3],
    [`<circle cx="8" cy="7" r="2.6" fill="${C.muted}" fill-opacity="0.6"/>`, L.other],
    [`<line x1="1" y1="13" x2="15" y2="1" stroke="${C.red}" stroke-width="1.6" stroke-dasharray="4,3"/>`, L.cap],
  ]) {
    const s = document.createElement("span");
    s.style.cssText = "display:inline-flex;align-items:center;gap:6px";
    s.innerHTML = `<svg width="16" height="14">${svg}</svg>`;
    s.appendChild(document.createTextNode(txt));
    key.appendChild(s);
  }

  const wrap = add(`display:flex;flex-direction:${stacked ? "column" : "row"};gap:${stacked ? 12 : 0}px;align-items:flex-start`, null);
  const pw = stacked ? W : Math.floor(W / 2);
  const ymin = 0.03, ymax = 300;
  const yt = [0.1, 1, 10, 100];
  const tip = (p) =>
    `${p.run_label}, head ${p.head}${p.carrier ? " (carries the switch)" : ""}\n` +
    `cap from the untrained model: ${f2(p.x)} (place ${p.place})\nmeasured swing: ${f2(p.y)} (${(100 * p.y / p.x).toFixed(p.y / p.x < 0.01 ? 2 : 1)}% of the cap)\n` +
    `attention moved (share of the head's attention): ${Math.round(100 * p.tv)}%`;

  data.panels.forEach((pan, i) => {
    const P = data.points.filter((p) => p.panel === pan.id);
    const pad = (pan.xmax - pan.xmin) * 0.06;
    const x0 = pan.xmin - pad, x1 = pan.xmax + pad;
    const others = P.filter((p) => !p.carrier && !p.top3);
    const top = P.filter((p) => p.top3 && !p.carrier);
    const car = P.filter((p) => p.carrier);
    // one label per distinct carrier head: '26.2 (run 1)'; runs sharing a head share one label
    const lab = [];
    for (const p of car) {
      const k = lab.find((q) => q.head === p.head);
      if (k) { k.runs.push(p.run_label.split("run ")[1]); k.y = Math.max(k.y, p.y); }
      else lab.push({ head: p.head, x: p.x, y: p.y, sw: p.run_label.split(",")[0], runs: [p.run_label.split("run ")[1]] });
    }
    lab.forEach((q) => { q.text = `head ${q.head}\n${q.sw}, run${q.runs.length > 1 ? "s" : ""} ${q.runs.join(" and ")}`; });
    const box = add("", null, wrap);
    add(`font-size:13px;font-weight:600;color:${C.ink};margin:0 0 2px;padding-left:${i === 0 || stacked ? 58 : 18}px`, pan.title, box);
    const left = i === 0 || stacked;
    box.appendChild(Plot.plot({
      style: plotStyle({ fontSize: "12.5px" }), width: pw, height: 380,
      marginLeft: left ? 58 : 18, marginRight: 14, marginTop: 14, marginBottom: 44,
      x: { domain: [x0, x1], label: data.xlabel, labelAnchor: "center", labelArrow: "none", ticks: 5 },
      // one x label per panel, short; the full definition sits in the subtitle
      y: { type: "log", domain: [ymin, ymax], ticks: yt, tickFormat: ytick, label: left ? data.ylabel : null, labelArrow: "none", axis: left ? "left" : null },
      marks: [
        // the prediction: the band of the cap's top 3
        Plot.rect([{ a: pan.third, b: x1 }], { x1: "a", x2: "b", y1: ymin, y2: ymax, fill: C.grid, fillOpacity: 0.55 }),
        Plot.text([(pan.third + x1) / 2], { x: (d) => d, y: ymin, text: () => "top 3", textAnchor: "middle", dy: -8, fill: C.muted, fontSize: 11.5, fontWeight: 600, ...halo() }),
        Plot.gridY({ ticks: yt, stroke: C.grid, strokeOpacity: 1 }),
        // y = x: the head uses its whole cap
        Plot.line([[x0, x0], [x1, x1]], { stroke: C.red, strokeWidth: 1.6, strokeDasharray: "5,4" }),
        Plot.text([[x0, x0]], { x: (d) => d[0], y: (d) => d[1], text: () => "cap reached", textAnchor: "start", dx: 4, dy: -9, fill: C.red, fontSize: 11.5, ...halo() }),
        Plot.dot(others, { x: "x", y: "y", r: 2.4, fill: C.muted, fillOpacity: 0.45, tip: true, title: tip }),
        Plot.dot(top, { x: "x", y: "y", r: 4.2, symbol: "square", fill: C.secondary, tip: true, title: tip }),
        Plot.dot(car, { x: "x", y: "y", r: 7, fill: C.primary, fillOpacity: 0.35, stroke: C.primary, strokeWidth: 2.4, tip: true, title: tip }),
        Plot.text(lab, { x: "x", y: "y", text: "text", textAnchor: "start", dx: 11, dy: 0, lineAnchor: "middle", fill: C.ink, fontSize: 11.5, fontWeight: 600, lineHeight: 1.15, ...halo() }),
      ],
    }));
  });
  if (data.note) add(`font-size:12.5px;color:${C.muted};margin-top:0.3rem`, data.note);
  return root;
}
