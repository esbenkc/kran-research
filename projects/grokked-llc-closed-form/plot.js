// Measured LLC (calibrated SGLD) against the closed-form value of Cullen et al. (arXiv 2603.01192 v3).
// One point per (p, K, phase): mean over seeds, bar = range over seeds. Log-log.
// Dashed line: estimate = formula. Shaded band: the pre-registered ±band around it.
// Grey dash: rank reference ½·rank J at the same checkpoints (a lower bound on the true LLC).
// ×: sampler did not settle (data/stationarity.csv), excluded from the verdict.
// Hollow: cell outside the formula's hypothesis (K_eff(3p−2) ≥ p(N−1)).
// data = { title, sub, band, rows: [{ p, K, phase, x, y, ymin, ymax, rank, cap, inside, n, ratio, lam_shift }] }
import { C, FONT, plotStyle } from "../theme.js";

const COLOR = { grok: C.primary, mem: C.secondary };
const LABEL = { grok: "grokked: ½·K_eff·(3p−1)", mem: "memorising: ½·p·min(l, K), 2nd layer only" };
const SYMBOL = { grok: "circle", mem: "square" };
const fmt = (v) => (v >= 100 ? Math.round(v).toLocaleString("en-US") : Number(v.toPrecision(3)).toString());

export function render(data, Plot) {
  const rows = data.rows;
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = data.title;
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  sub.textContent = data.sub;
  const phases = ["grok", "mem"].filter((ph) => rows.some((r) => r.phase === ph));
  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-wrap:wrap;gap:0.4rem 1.2rem;font-size:15px;margin:0 0 0.4rem;color:" + C.ink;
  legend.innerHTML = phases.map((ph) =>
    `<span style="display:inline-flex;align-items:center;gap:0.4rem"><svg width="12" height="12">${
      ph === "grok" ? `<circle cx="6" cy="6" r="5" fill="${COLOR[ph]}"/>` : `<rect x="1" y="1" width="10" height="10" fill="${COLOR[ph]}"/>`
    }</svg>${LABEL[ph]}</span>`,
  ).join("") +
    `<span style="display:inline-flex;align-items:center;gap:0.4rem;color:${C.muted}"><svg width="18" height="10"><line x1="1" y1="5" x2="17" y2="5" stroke="${C.muted}" stroke-width="2.5"/></svg>½·rank J</span>` +
    (rows.some((r) => r.settled === false) ? `<span style="color:${C.muted}">× = sampler did not settle</span>` : "") +
    `<span style="color:${C.muted}">hollow = outside the formula’s hypothesis (K_eff(3p−2) ≥ p(N−1))</span>`;
  root.append(head, sub, legend);

  const vals = rows.flatMap((r) => [r.x, r.ymin, r.ymax, r.phase === "grok" ? r.rank : r.x]).filter((v) => v > 0);
  const lo = Math.min(...vals) / 1.35;
  const hi = Math.max(...vals) * 1.35;
  const diag = [lo, hi];
  const band = data.band;
  const ticks = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000].filter((t) => t >= lo && t <= hi);
  const tip = (d) =>
    `p = ${d.p}, K = ${d.K}, ${d.phase === "grok" ? "grokked" : "memorising"} (${d.n} seed${d.n > 1 ? "s" : ""})\n` +
    `formula: ${fmt(d.x)}\nSGLD: ${fmt(d.y)} (seeds ${fmt(d.ymin)}–${fmt(d.ymax)})\n` +
    `SGLD / formula: ${d.ratio.toFixed(2)}` +
    (d.phase === "grok" ? `\n½·rank J: ${fmt(d.rank)}\n½·K_eff·(3p−2): ${fmt(d.lam_shift)}` : "\nonly the 2nd layer sampled (the formula's frozen-feature model)") +
    `\n½·p(N−1): ${fmt(d.cap)}` + (d.inside ? "" : "\noutside the formula's hypothesis: K_eff(3p−2) ≥ p(N−1)") +
    (d.settled === false ? "\nsampler did not settle: reading still rising after 4× longer chains" : "");
  const width = Math.max(480, Math.min(760, root.clientWidth || 700));
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "15px" }), width, height: Math.round(width * 0.78),
    marginTop: 16, marginLeft: 64, marginBottom: 52, marginRight: 16,
    x: { type: "log", domain: [lo, hi], axis: null },
    y: { type: "log", domain: [lo, hi], axis: null },
    marks: [
      Plot.gridY(ticks, { stroke: C.grid, strokeOpacity: 1 }),
      Plot.axisX({ ticks, tickFormat: fmt, label: "Closed-form LLC (Cullen et al.)", labelAnchor: "center", labelArrow: "none" }),
      Plot.axisY({ ticks, tickFormat: fmt, label: "Measured LLC (calibrated SGLD)", labelAnchor: "center", labelArrow: "none" }),
      Plot.areaY(diag, { x: (d) => d, y1: (d) => d * (1 - band), y2: (d) => d * (1 + band), fill: C.grid, fillOpacity: 0.55 }),
      Plot.line(diag, { x: (d) => d, y: (d) => d, stroke: C.faint, strokeWidth: 1.5, strokeDasharray: "5,4" }),
      Plot.text([hi / 1.5], { x: (d) => d, y: (d) => d * (1 + band), text: () => "estimate = formula", dy: -10, textAnchor: "end", fill: C.muted, fontSize: 13 }),
      Plot.link(rows.filter((r) => r.phase === "grok"), { x1: (d) => d.x / 1.09, x2: (d) => d.x * 1.09, y1: "rank", y2: "rank", stroke: C.muted, strokeWidth: 2.5 }),
      Plot.ruleX(rows.filter((r) => r.n > 1), { x: "x", y1: "ymin", y2: "ymax", stroke: (d) => COLOR[d.phase], strokeWidth: 1.6 }),
      Plot.dot(rows, {
        x: "x", y: "y", r: 5.5, symbol: (d) => (d.settled === false ? "times" : SYMBOL[d.phase]), stroke: (d) => COLOR[d.phase], strokeWidth: 1.8,
        fill: (d) => (d.inside ? COLOR[d.phase] : "none"), tip: true, title: tip,
      }),
    ],
  }));
  return root;
}
