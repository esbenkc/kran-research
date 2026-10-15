// Appendix Q1: push the free rescaling far past c = 16.
// y = steps at c / steps at c = 1, same attacker, seed and data order (log scale, 1x = no slowdown).
// Point = median over runs, band = range. Hollow = every run hit the step cap (ratio is a lower bound).
// data = { cap, thresh, rows: [{ attacker, label, log2c, lr, ratio_median, ratio_min, ratio_max,
//          n_runs, n_censored, n_diverged, n_slower, steps_by_run, grid, source }] }
import { C, FONT, plotStyle } from "../theme.js";

const COLOR = { sgd: C.primary, adamw: C.secondary };
const cLabel = (k) => (k >= 0 ? String(2 ** k) : `1/${2 ** -k}`);
const fmt = (v) => `${Number(v.toPrecision(2))}×`;

export function render(data, Plot) {
  const rows = data.rows.map((r) => ({
    ...r, py: Math.log2(r.ratio_median), plo: Math.log2(r.ratio_min), phi: Math.log2(r.ratio_max),
    allCapped: r.n_censored + r.n_diverged === r.n_runs,
  }));
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = data.title;
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  sub.textContent = `Steps to get ${Math.round(data.thresh * 100)}% of the facts right again, divided by the same attacker's steps at c = 1 (same seed and data order). Log scale.`;
  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-wrap:wrap;gap:0.4rem 1.2rem;font-size:15px;margin:0 0 0.4rem;color:" + C.ink;
  legend.innerHTML = [["sgd", "SGD"], ["adamw", "AdamW"]].map(([k, l]) =>
    `<span style="display:inline-flex;align-items:center;gap:0.4rem"><svg width="26" height="10"><line x1="0" y1="5" x2="26" y2="5" stroke="${COLOR[k]}" stroke-width="3"/></svg>${l}</span>`,
  ).join("") + `<span style="color:${C.muted}">hollow = never recovered within ${data.cap} steps</span>`;
  root.append(head, sub, legend);

  const hiV = Math.max(8, ...rows.map((r) => r.ratio_max)) * 1.3;
  const loV = Math.min(0.5, ...rows.map((r) => r.ratio_min)) / 1.2;
  const yTicks = [0.5, 1, 2, 4, 8, 16, 32, 64].filter((t) => t >= loV && t <= hiV);
  const xs = [...new Set(rows.map((r) => r.log2c))].sort((a, b) => a - b);
  const xTicks = [-12, -8, -4, 0, 4, 8, 12].filter((t) => t >= xs[0] && t <= xs[xs.length - 1]);
  const marks = [
    Plot.rectX([0], { x1: -4, x2: 4, y1: Math.log2(loV), y2: Math.log2(hiV), fill: C.grid, fillOpacity: 0.45 }),
    Plot.text([0], { x: 0, y: Math.log2(hiV), text: () => "range in the post", dy: 12, fill: C.muted, fontSize: 13 }),
    Plot.gridY(yTicks.map(Math.log2), { stroke: C.grid, strokeOpacity: 1 }),
    Plot.axisY(yTicks.map(Math.log2), { tickFormat: (v) => fmt(2 ** v), label: null }),
    Plot.ruleY([0], { stroke: C.faint, strokeWidth: 1.5 }),
  ];
  for (const a of ["sgd", "adamw"]) {
    const rs = rows.filter((r) => r.attacker === a);
    marks.push(
      Plot.areaY(rs.filter((r) => r.n_runs > 1), { x: "log2c", y1: "plo", y2: "phi", fill: COLOR[a], fillOpacity: 0.13 }),
      Plot.line(rs, { x: "log2c", y: "py", stroke: COLOR[a], strokeWidth: 2.6 }),
    );
  }
  const tip = (d) =>
    `${d.label}, c = ${cLabel(d.log2c)}\n` +
    (d.allCapped ? `never recovered in ${d.cap ?? data.cap} steps: at least ${fmt(d.ratio_median)}\n`
      : `${fmt(d.ratio_median)} its steps at c = 1 (range ${fmt(d.ratio_min)}–${fmt(d.ratio_max)})\n`) +
    `slower in ${d.n_slower} of ${d.n_runs} runs` +
    (d.n_censored ? `, ${d.n_censored} hit the cap` : "") +
    `\nbest learning rate ${d.lr}\nseed-0 search: ` +
    Object.entries(d.grid).map(([lr, s]) => `${lr} → ${s}`).join(", ");
  marks.push(Plot.dot(rows, {
    x: "log2c", y: "py", r: 4.5, stroke: (d) => COLOR[d.attacker], strokeWidth: 1.8,
    fill: (d) => (d.allCapped ? "none" : COLOR[d.attacker]), tip: true, title: tip,
  }));
  const width = Math.max(520, Math.min(860, root.clientWidth || 760));
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "15px" }), width, height: 360, marginTop: 24, marginLeft: 56, marginBottom: 48,
    x: {
      domain: [xs[0] - 0.6, xs[xs.length - 1] + 0.6], ticks: xTicks, tickFormat: cLabel,
      label: "Defender's free rescaling c (queries × c, keys ÷ c)", labelAnchor: "center", labelArrow: "none",
    },
    y: { domain: [Math.log2(loV), Math.log2(hiV)], axis: null },
    marks,
  }));
  // bottom: collateral damage, kept facts the model still answers at the last check of each run
  const cap = document.createElement("div");
  cap.style.cssText = "font-size:14px;margin:0.6rem 0 0.2rem;color:" + C.muted;
  cap.textContent = "Kept facts the model still answers when the attack stops (100% before the attack).";
  root.appendChild(cap);
  const bot = [Plot.gridY([0, 0.25, 0.5, 0.75, 1], { stroke: C.grid, strokeOpacity: 1 })];
  for (const a of ["sgd", "adamw"]) {
    const rs = rows.filter((r) => r.attacker === a && r.accR_n);
    bot.push(
      Plot.areaY(rs.filter((r) => r.accR_n > 1), { x: "log2c", y1: "accR_min", y2: "accR_max", fill: COLOR[a], fillOpacity: 0.12 }),
      Plot.line(rs, { x: "log2c", y: "accR_mean", stroke: COLOR[a], strokeWidth: 2.4 }),
      Plot.dot(rs, {
        x: "log2c", y: "accR_mean", r: 4, fill: COLOR[a], tip: true,
        title: (d) => `${d.label}, c = ${cLabel(d.log2c)}\nkept facts right: ${Math.round(d.accR_mean * 100)}% (mean of ${d.accR_n} runs, range ${Math.round(d.accR_min * 100)}–${Math.round(d.accR_max * 100)}%)`,
      }),
    );
  }
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "15px" }), width, height: 200, marginTop: 12, marginLeft: 56, marginBottom: 44,
    x: { domain: [xs[0] - 0.6, xs[xs.length - 1] + 0.6], ticks: xTicks, tickFormat: cLabel, label: null },
    y: { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], tickFormat: (v) => `${Math.round(v * 100)}%`, label: null },
    marks: bot,
  }));
  return root;
}
