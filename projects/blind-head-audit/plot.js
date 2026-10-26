// Blind head audit: attach the probe on neutral prompts, rank the 32 heads of the last two blocks by how much their
// output moves (trained model minus base model), then give the top k heads their no-probe values on the trained task.
// Three panels (label-trained answer switch, label-trained detector, reward-trained switch); one line per training run:
// share of the probe's push removed when the top 1-8 audit heads are reset; hollow dots at the right: 400 random sets
// of 3 heads from the 29 heads outside the audit's top 3. Dashed line: the 70% bar.
// data = { title, subtitle, bar, panels:[{label, runs:[{label, R:[k1..k8], heads:[..8], rand:[400], p95, pass}]}], nulls, note }
import { C, FONT, plotStyle, halo } from "../theme.js";

export function render(data, Plot) {
  const shade = [C.primary, C.accent, C.secondary];
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  const add = (css, txt) => {
    const d = document.createElement("div");
    d.style.cssText = css;
    d.textContent = txt;
    root.appendChild(d);
    return d;
  };
  add(`font-size:17px;font-weight:600;margin:0 0 0.15rem;color:${C.ink}`, data.title);
  add(`font-size:14px;margin:0 0 0.5rem;color:${C.muted}`, data.subtitle);
  const pct = (v) => `${Math.round(v * 100)}%`;
  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;flex-wrap:nowrap;gap:0 6px;align-items:flex-start";
  root.appendChild(wrap);
  const W = 690, H = 320, n = data.panels.length;
  const RX = 9.4; // x position of the random-set column
  data.panels.forEach((P, pi) => {
    const w = Math.round((W - 6 * (n - 1)) / n) + (pi === 0 ? 30 : 0);
    const lines = P.runs.flatMap((r, ri) => r.R.map((v, k) => ({ k: k + 1, v, r, ri })));
    const rand = P.runs.flatMap((r, ri) => r.rand.map((v, j) => ({ v, r, ri, j })));
    const labY = {};
    [...P.runs.keys()].sort((a, b) => P.runs[b].R[2] - P.runs[a].R[2])
      .forEach((ri, i) => (labY[ri] = Math.min(1.0, Math.max(...P.runs.map((r) => r.R[2])) + 0.02) - 0.085 * i));
    const svg = Plot.plot({
      style: plotStyle({ fontSize: "12.5px" }),
      width: w, height: H, marginLeft: pi === 0 ? 46 : 16, marginRight: 6, marginTop: 34, marginBottom: 48,
      x: { domain: [0.5, 10.2], ticks: [1, 2, 3, 4, 5, 6, 7, 8, RX], tickSize: 0,
           tickFormat: (v) => (v > 8 ? "rand." : `${v}`), label: pi === 1 ? "Top audit heads reset (k)" : null, labelArrow: "none" },
      y: { domain: [-0.1, 1.08], ticks: [0, 0.25, 0.5, 0.75, 1], tickFormat: pi === 0 ? pct : () => "",
           label: pi === 0 ? "Probe's push removed" : null, labelArrow: "none" },
      marks: [
        Plot.ruleY([0], { stroke: C.faint }),
        Plot.ruleY([data.bar], { stroke: C.faint, strokeDasharray: "4,4" }),
        Plot.ruleX([3], { stroke: C.faint, strokeDasharray: "2,3" }),
        Plot.text([P.label], { frameAnchor: "top-left", dy: -26, dx: 2, fill: C.ink, fontSize: 13, fontWeight: 600, text: (d) => d }),
        Plot.dot(rand, {
          x: (d) => RX + (d.ri - (P.runs.length - 1) / 2) * 0.3 + ((d.j % 7) - 3) * 0.02, y: "v", r: 1.8, fill: "none",
          stroke: (d) => shade[d.ri], strokeOpacity: 0.55, tip: true,
          title: (d) => `${d.r.label}: 3 random other heads\npush removed ${pct(d.v)}`,
        }),
        Plot.line(lines, { x: "k", y: "v", z: "ri", stroke: (d) => shade[d.ri], strokeWidth: 2 }),
        Plot.dot(lines, {
          x: "k", y: "v", r: (d) => (d.k === 3 ? 5 : 3.2), fill: (d) => shade[d.ri], stroke: C.paper, strokeWidth: 1, tip: true,
          title: (d) => `${d.r.label}, top ${d.k}: ${d.r.heads.slice(0, d.k).join(", ")}\npush removed ${pct(d.v)}`,
        }),
        Plot.text(P.runs, {
          x: 3, y: (r) => labY[P.runs.indexOf(r)], dx: 8, textAnchor: "start", fill: C.ink, fontSize: 11.5, ...halo(),
          text: (r) => `${r.label}: ${pct(r.R[2])}`,
        }),
      ],
    });
    svg.style.cssText += `;flex:0 1 ${w}px;min-width:0;width:100%;height:auto`;
    wrap.appendChild(svg);
  });
  if (data.nulls) add(`font-size:13px;color:${C.ink};margin-top:0.25rem`, data.nulls);
  add(`font-size:13px;color:${C.muted};margin-top:0.2rem`, data.note);
  return root;
}
