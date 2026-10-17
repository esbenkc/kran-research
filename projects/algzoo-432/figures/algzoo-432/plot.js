// Result ladder for ARC's 432-parameter 2nd-argmax RNN.
// One point per attempt to describe the model. x: numbers in the description (log).
// y: % of inputs where the description picks another position than the model (log), lower = better.
// Dashed line: the model against copies of itself with 0.1% weight noise (the bar).
// Dotted line: the plain 2nd-largest rule. Shaded corner: <= 200 numbers and at or below the bar.
// Filled = scored once on the holdout seed; hollow = development seed only.
// data = { title, sub, floor, null, cap, points: [{ family, x, dis, agree, split, label, detail, show }] }
import { C, FONT, plotStyle, halo } from "../theme.js";

const LABEL = {
  hand: "hand-written rule",
  pruned: "model's weights, pruned and rounded",
  retrained: "smaller network retrained to copy the model",
  fitted: "rule with constants fitted to the model's choices",
};
const SYMBOL = { hand: "star", pruned: "circle", retrained: "square", fitted: "diamond" };
const fmtX = (v) => (v >= 1e6 ? v / 1e6 + "M" : v >= 1e4 ? v / 1e3 + "k" : v.toLocaleString("en-US"));
const fmtPct = (v) => (v < 1 ? v.toFixed(2) : v < 10 ? v.toFixed(1) : Math.round(v).toString()) + "%";
const fmtTick = (v) => v + "%";
const SIDE = {
  above: { dy: -14, dx: 0, textAnchor: "start" },
  below: { dy: 16, dx: 0, textAnchor: "middle" },
  aboveEnd: { dy: -14, dx: 6, textAnchor: "end" },
  left: { dy: 0, dx: -12, textAnchor: "end" },
  right: { dy: 0, dx: 12, textAnchor: "start" },
};

export function render(data, Plot) {
  const COLOR = { hand: C.primary, pruned: C.secondary, retrained: C.orange, fitted: C.muted };
  const pts = data.points;
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = data.title;
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  sub.textContent = data.sub;
  const shape = (f) => {
    const c = COLOR[f];
    if (f === "hand") return `<polygon points="6,0 7.6,4.2 12,4.4 8.5,7.2 9.7,11.6 6,9.1 2.3,11.6 3.5,7.2 0,4.4 4.4,4.2" fill="${c}"/>`;
    if (f === "retrained") return `<rect x="1" y="1" width="10" height="10" fill="${c}"/>`;
    if (f === "fitted") return `<polygon points="6,0 12,6 6,12 0,6" fill="${c}"/>`;
    return `<circle cx="6" cy="6" r="5" fill="${c}"/>`;
  };
  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-wrap:wrap;gap:0.3rem 1.1rem;font-size:14px;margin:0 0 0.4rem;color:" + C.ink;
  legend.innerHTML = Object.keys(LABEL).filter((f) => pts.some((p) => p.family === f)).map((f) =>
    `<span style="display:inline-flex;align-items:center;gap:0.4rem"><svg width="12" height="12">${shape(f)}</svg>${LABEL[f]}</span>`,
  ).join("") +
    `<span style="color:${C.muted}">filled = scored once on held-out inputs, hollow = development inputs only</span>`;
  root.append(head, sub, legend);

  const xlo = 2, xhi = 1.2e6, ylo = 0.4, yhi = 70;
  const xticks = [10, 100, 1000, 1e4, 1e5, 1e6];
  const yticks = [0.5, 1, 2, 5, 10, 20, 50];
  const tip = (d) =>
    `${d.label}\n${d.detail}\nnumbers: ${d.x.toLocaleString("en-US")}\n` +
    `differs from the model on ${fmtPct(d.dis)} of inputs (agreement ${d.agree.toFixed(4)})\n` +
    (d.split === "holdout" ? "held-out seed 432432, 1,000,000 inputs" : `development seed (${d.split}), not the held-out inputs`);
  const curve = pts.filter((p) => p.family === "retrained" && /weights$/.test(p.label)).sort((a, b) => a.x - b.x);
  const width = Math.max(480, Math.min(760, root.clientWidth || 700));
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "15px" }), width, height: Math.round(width * 0.72),
    marginTop: 16, marginLeft: 78, marginBottom: 52, marginRight: 20,
    x: { type: "log", domain: [xlo, xhi], axis: null },
    y: { type: "log", domain: [ylo, yhi], axis: null },
    marks: [
      Plot.gridY(yticks, { stroke: C.grid, strokeOpacity: 1 }),
      Plot.rect([0], { x1: xlo, x2: data.cap, y1: ylo, y2: data.floor, fill: C.grid, fillOpacity: 0.75 }),
      Plot.text([0], { x: xlo * 1.15, y: ylo * 1.25, text: () => "short and faithful enough", textAnchor: "start", fill: C.muted, fontSize: 13 }),
      Plot.axisX({ ticks: xticks, tickFormat: fmtX, label: "Numbers in the description (log scale)", labelAnchor: "center", labelArrow: "none" }),
      Plot.axisY({ ticks: yticks, tickFormat: () => "", label: "Inputs where it differs from the model (log scale)", labelAnchor: "center", labelArrow: "none", labelOffset: 62 }),
      // Plot thins log-axis labels to 1-2-10, so the y labels are drawn by hand.
      Plot.text(yticks, { x: xlo, y: (d) => d, text: fmtTick, dx: -10, textAnchor: "end", fill: C.ink, fontSize: 15 }),
      Plot.ruleX([data.cap], { stroke: C.faint, strokeWidth: 1.5 }),
      Plot.text([data.cap], { x: (d) => d, y: yhi, text: () => "200 numbers", dx: 4, dy: 8, textAnchor: "start", fill: C.muted, fontSize: 13 }),
      Plot.ruleY([data.floor], { stroke: C.red, strokeWidth: 1.6, strokeDasharray: "6,4" }),
      Plot.text([data.floor], { x: xhi, y: (d) => d, text: () => `model vs. itself with 0.1% weight noise: ${fmtPct(data.floor)}`, dy: -8, textAnchor: "end", fill: C.red, fontSize: 13, ...halo() }),
      Plot.ruleY([data.null], { stroke: C.muted, strokeWidth: 1.4, strokeDasharray: "2,3" }),
      Plot.text([data.null], { x: xhi, y: (d) => d, text: () => `plain 2nd-largest rule: ${fmtPct(data.null)}`, dy: -8, textAnchor: "end", fill: C.muted, fontSize: 13, ...halo() }),
      Plot.line(curve, { x: "x", y: "dis", stroke: C.orange, strokeWidth: 1.5, strokeOpacity: 0.7 }),
      Plot.dot(pts, {
        x: "x", y: "dis", r: 6, symbol: (d) => SYMBOL[d.family], stroke: (d) => COLOR[d.family], strokeWidth: 1.8,
        fill: (d) => (d.split === "holdout" ? COLOR[d.family] : "none"), tip: true, title: tip,
      }),
      Plot.text(pts.filter((d) => d.show === "above"), {
        x: "x", y: "dis", text: "label", ...SIDE.above, fill: (d) => COLOR[d.family], fontSize: 13, fontWeight: 600, ...halo(),
      }),
      Plot.text(pts.filter((d) => d.show === "below"), {
        x: "x", y: "dis", text: "label", ...SIDE.below, fill: (d) => COLOR[d.family], fontSize: 13, fontWeight: 600, ...halo(),
      }),
      Plot.text(pts.filter((d) => d.show === "left"), {
        x: "x", y: "dis", text: "label", ...SIDE.left, fill: (d) => COLOR[d.family], fontSize: 13, fontWeight: 600, ...halo(),
      }),
      Plot.text(pts.filter((d) => d.show === "aboveEnd"), {
        x: "x", y: "dis", text: "label", ...SIDE.aboveEnd, fill: (d) => COLOR[d.family], fontSize: 13, fontWeight: 600, ...halo(),
      }),
      Plot.text(pts.filter((d) => d.show === "right"), {
        x: "x", y: "dis", text: "label", ...SIDE.right, fill: (d) => COLOR[d.family], fontSize: 13, fontWeight: 600, ...halo(),
      }),
    ],
  }));
  return root;
}
