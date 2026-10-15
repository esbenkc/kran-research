// Appendix Q2: what more runs buy. SGD and AdamW, c = 16 vs c = 1, one (seed, data order) pair per run.
// Top: running geometric-mean slowdown with its 95% interval (dark band) and the 10-90% range of single
// runs (light band). Bottom: share of runs where c = 16 was slower, with a 95% interval, and the 80% bar.
// data = { sgd: { label, n_post, curve: [{ n, gmean, lo, hi, q10, q90, frac, frac_lo, frac_hi }], ... },
//          adamw: { ... } }
import { C, FONT, plotStyle } from "../theme.js";

const COLOR = { sgd: C.primary, adamw: C.secondary };
const fmt = (v) => `${Number(v.toPrecision(2))}×`;

export function render(data, Plot) {
  const keys = ["sgd", "adamw"].filter((k) => data[k]);
  const rows = keys.flatMap((k) => data[k].curve.map((r) => ({ ...r, attacker: k, label: data[k].label })));
  const nPost = data.sgd.n_post;
  const nMax = Math.max(...rows.map((r) => r.n));
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = data.title;
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  sub.textContent = "Each run is one seed and data order, attacked at c = 1 and c = 16. Dark band: 95% interval of the average. Light band: middle 80% of single runs.";
  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-wrap:wrap;gap:0.4rem 1.2rem;font-size:15px;margin:0 0 0.4rem;color:" + C.ink;
  legend.innerHTML = keys.map((k) =>
    `<span style="display:inline-flex;align-items:center;gap:0.4rem"><svg width="26" height="10"><line x1="0" y1="5" x2="26" y2="5" stroke="${COLOR[k]}" stroke-width="3"/></svg>${data[k].label}</span>`,
  ).join("");
  root.append(head, sub, legend);
  const width = Math.max(520, Math.min(860, root.clientWidth || 760));
  const x = { domain: [3, nMax], label: "Runs so far", labelAnchor: "center", labelArrow: "none" };
  const postRule = (y) => [
    Plot.ruleX([nPost], { stroke: C.faint, strokeDasharray: "3,3" }),
    Plot.text([nPost], { x: nPost, y, text: () => "the post", textAnchor: "start", dx: 4, fill: C.muted, fontSize: 13 }),
  ];

  const L = Math.log2;
  const yLo = Math.min(0.5, ...rows.map((r) => r.q10)) / 1.1;
  const yHi = Math.max(8, ...rows.map((r) => r.q90)) * 1.1;
  const yT = [0.5, 1, 2, 4, 8, 16].filter((t) => t >= yLo && t <= yHi);
  const top = [
    Plot.gridY(yT.map(L), { stroke: C.grid, strokeOpacity: 1 }),
    Plot.axisY(yT.map(L), { tickFormat: (v) => fmt(2 ** v), label: null }),
    Plot.ruleY([0], { stroke: C.faint, strokeWidth: 1.5 }),
    ...postRule(L(yHi)),
  ];
  for (const k of keys) {
    const rs = rows.filter((r) => r.attacker === k).map((r) => ({
      ...r, g: L(r.gmean), l: L(r.lo), h: L(r.hi), a: L(r.q10), b: L(r.q90),
    }));
    top.push(
      Plot.areaY(rs, { x: "n", y1: "a", y2: "b", fill: COLOR[k], fillOpacity: 0.1 }),
      Plot.areaY(rs, { x: "n", y1: "l", y2: "h", fill: COLOR[k], fillOpacity: 0.3 }),
      Plot.line(rs, { x: "n", y: "g", stroke: COLOR[k], strokeWidth: 2.4 }),
      Plot.dot(rs, {
        x: "n", y: "g", r: 6, fill: "transparent", stroke: "none", tip: true,
        title: (d) => `${d.label}, first ${d.n} runs\naverage ${fmt(d.gmean)} (95% interval ${fmt(d.lo)}–${fmt(d.hi)})\nmiddle 80% of runs ${fmt(d.q10)}–${fmt(d.q90)}`,
      }),
    );
  }
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "15px" }), width, height: 300, marginTop: 20, marginLeft: 56, marginBottom: 40,
    x, y: { domain: [L(yLo), L(yHi)], axis: null }, marks: top,
  }));

  const cap = document.createElement("div");
  cap.style.cssText = "font-size:14px;margin:0.6rem 0 0.2rem;color:" + C.muted;
  cap.textContent = "Share of runs where c = 16 was slower, with its 95% interval. The post's bar was 80%.";
  root.appendChild(cap);
  const bot = [
    Plot.gridY([0, 0.25, 0.5, 0.75, 1], { stroke: C.grid, strokeOpacity: 1 }),
    Plot.ruleY([0.8], { stroke: C.ink, strokeDasharray: "4,3" }),
    Plot.text([0.8], { x: nMax, y: 0.8, text: () => "80% bar", textAnchor: "end", dy: -7, fill: C.muted, fontSize: 13 }),
    ...postRule(1),
  ];
  for (const k of keys) {
    const rs = rows.filter((r) => r.attacker === k);
    bot.push(
      Plot.areaY(rs, { x: "n", y1: "frac_lo", y2: "frac_hi", fill: COLOR[k], fillOpacity: 0.2 }),
      Plot.line(rs, { x: "n", y: "frac", stroke: COLOR[k], strokeWidth: 2.2, curve: "step-after" }),
      Plot.dot(rs, {
        x: "n", y: "frac", r: 6, fill: "transparent", stroke: "none", tip: true,
        title: (d) => `${d.label}, first ${d.n} runs\nslower in ${Math.round(d.frac * d.n)} of ${d.n} (${Math.round(d.frac * 100)}%)\n95% interval ${Math.round(d.frac_lo * 100)}–${Math.round(d.frac_hi * 100)}%`,
      }),
    );
  }
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "15px" }), width, height: 220, marginTop: 16, marginLeft: 56, marginBottom: 44,
    x, y: { domain: [0, 1], tickFormat: (v) => `${Math.round(v * 100)}%`, label: null, ticks: [0, 0.25, 0.5, 0.75, 1] },
    marks: bot,
  }));
  return root;
}
