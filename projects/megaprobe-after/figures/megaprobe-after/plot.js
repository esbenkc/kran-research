// After a fine-tune hides false statements from a one-layer probe, does a probe that reads every layer still catch them?
// TOP: catch rate = share of the 596 false statements a probe flags, with the alarm line set so that 1% of true
//      statements raise a false alarm. Every probe was trained on the original model only. Hollow circle: original
//      model. Filled dot: after the fine-tune, with a new alarm line set on 200 true statements scored by the
//      fine-tuned model; 3 runs pooled; bar = 95% interval (resampling cities). Dashed line: the pre-registered 50%
//      bar. Dotted line: the pre-registered 40% floor for the lower end of the interval.
// BOTTOM: for a simple probe trained on one layer of the original model, how far apart the false and the true
//      statement of each pair score, in units of the original model's spread at that layer. Hollow: original model.
//      Filled: after the fine-tune (mean of 3 runs; bar = 95% interval). Faded hollow: layers where the original
//      model separates too weakly to score (below 1 spread). Short dashes on layers 1-8: half the original separation
//      (the pre-registered bar for those layers).
// data = { layers: [{layer, before, before_lo, before_hi, after, after_lo, after_hi, after_per_seed, kept,
//          kept_lo, kept_hi, scored}], monitors: [{id, label, before, before_lo, before_hi, after, after_lo, after_hi,
//          after_per_seed}], n: {false_statements, seeds} }
import { C, FONT, halo, plotStyle } from "../theme.js";

const pct = (v) => `${(v * 100).toFixed(v < 0.1 && v > 0 ? 1 : 0)}%`;

export function render(data, Plot) {
  const W = 680;
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  root.setAttribute("width", String(W));

  const mega = data.monitors.find((m) => m.id === "mega");
  const base = data.monitors.find((m) => m.id === "base");
  const up = data.layers.filter((d) => d.layer > 14);
  const keptMax = Math.max(...up.map((d) => d.kept));
  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = "A probe that reads every layer still catches most false statements after the fine-tune";
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  sub.textContent = `The fine-tune hid false statements from a one-layer probe (${pct(base.before)} caught before, ` +
    `${pct(base.after)} after). A probe trained on the original model that reads every layer and every word still ` +
    `catches ${pct(mega.after)} (95% interval ${pct(mega.after_lo)}–${pct(mega.after_hi)}; ${data.n.seeds} runs, ` +
    `${data.n.false_statements} false statements each), with 1% false alarms on true statements.`;

  const sw = (kind) => {
    if (kind === "hollow") return `<svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="none" stroke="${C.muted}" stroke-width="1.8"/></svg>`;
    if (kind === "filled") return `<svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="${C.primary}"/></svg>`;
    if (kind === "dot") return `<svg width="24" height="12"><line x1="1" y1="6" x2="23" y2="6" stroke="${C.muted}" stroke-width="1.5" stroke-dasharray="1.5,3"/></svg>`;
    return `<svg width="24" height="12"><line x1="1" y1="6" x2="23" y2="6" stroke="${C.muted}" stroke-width="1.5" stroke-dasharray="4,3"/></svg>`;
  };
  const item = (k, t) => `<span style="display:inline-flex;align-items:center;gap:0.35rem">${sw(k)}${t}</span>`;
  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-wrap:wrap;gap:0.3rem 1.0rem;font-size:14px;margin:0 0 0.3rem;color:" + C.ink;
  legend.innerHTML = item("hollow", "original model") + item("filled", "after the fine-tune (bar: 95% interval)") +
    item("dash", "pre-registered bar: 50%") + item("dot", "floor for the interval: 40%");
  root.append(head, sub, legend);

  const tipMon = (d) =>
    `${d.label}\noriginal model: ${pct(d.before)} (95% interval ${pct(d.before_lo)}–${pct(d.before_hi)})\n` +
    `after the fine-tune: ${pct(d.after)} (95% interval ${pct(d.after_lo)}–${pct(d.after_hi)})\n` +
    `per run: ${d.after_per_seed.map(pct).join(", ")}`;
  const order = data.monitors.map((m) => m.label);
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "13px" }), width: W, height: 190,
    marginLeft: 220, marginRight: 24, marginTop: 8, marginBottom: 40,
    x: { domain: [0, 1], label: "Catch rate at 1% false alarms", tickFormat: (v) => `${Math.round(v * 100)}%`, labelAnchor: "center", labelArrow: "none" },
    y: { domain: order, label: null, tickSize: 0 },
    marks: [
      Plot.gridX({ stroke: C.grid, strokeOpacity: 1 }),
      Plot.ruleX([0.5], { stroke: C.muted, strokeWidth: 1.5, strokeDasharray: "4,3" }),
      Plot.ruleX([0.4], { stroke: C.muted, strokeWidth: 1.5, strokeDasharray: "1.5,3" }),
      Plot.link(data.monitors, { x1: "before", x2: "after", y1: "label", y2: "label", stroke: C.faint, strokeWidth: 1 }),
      Plot.ruleY(data.monitors, { y: "label", x1: "after_lo", x2: "after_hi", stroke: C.primary, strokeWidth: 2.2, strokeOpacity: 0.75 }),
      Plot.dot(data.monitors, { x: "before", y: "label", r: 6, fill: "none", stroke: C.muted, strokeWidth: 1.8, tip: true, title: tipMon }),
      Plot.dot(data.monitors, { x: "after", y: "label", r: 6, fill: C.primary, stroke: C.paper, strokeWidth: 1.2, tip: true, title: tipMon }),
      Plot.text(data.monitors, { ...halo(), x: "after", y: "label", text: (d) => pct(d.after), dy: -13, fontSize: 12, fill: C.ink }),
    ],
  }));

  const head2 = document.createElement("div");
  head2.style.cssText = "font-size:15px;font-weight:600;margin:0.9rem 0 0.15rem;color:" + C.ink;
  head2.textContent = "No single layer keeps more than a third of its separation";
  const sub2 = document.createElement("div");
  sub2.style.cssText = "font-size:13px;margin:0 0 0.3rem;color:" + C.muted;
  sub2.textContent = "A simple probe trained on one layer of the original model: how far apart the false and the true " +
    "statement of each pair score, in units of the original model's spread at that layer. The layers above the targeted " +
    `one run the original weights. They keep at most ${pct(keptMax)} of their separation, against ` +
    `${pct(data.layers.find((d) => d.layer === 14).kept)} at the targeted layer. Faded circles: layers too weak on the ` +
    "original model to score. Short dashes: half the original gap, the pre-registered bar for layers 1–8.";
  root.append(head2, sub2);

  const ymax = Math.ceil(Math.max(...data.layers.map((d) => d.before_hi)) + 0.6);
  const bands = [
    { x1: 0.5, x2: 13.5, text: "weights changed by the fine-tune" },
    { x1: 13.5, x2: 14.5, text: "targeted\nlayer" },
    { x1: 14.5, x2: 28.5, text: "original weights" },
  ];
  const bar8 = data.layers.filter((d) => d.layer <= 8).map((d) => ({ layer: d.layer, y: d.before / 2 }));
  const tipL = (d) =>
    `layer ${d.layer}: original model ${d.before.toFixed(2)}, after the fine-tune ${d.after.toFixed(2)} ` +
    `(runs ${d.after_per_seed.map((v) => v.toFixed(2)).join(", ")})\nkept: ${pct(d.kept)} (95% interval ${pct(d.kept_lo)}–${pct(d.kept_hi)})` +
    (d.scored ? "" : "\ntoo weak on the original model to score");
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "13px" }), width: W, height: 270,
    marginLeft: 56, marginRight: 24, marginTop: 26, marginBottom: 44,
    x: { domain: [0.5, 28.5], ticks: [1, 4, 8, 12, 14, 16, 20, 24, 28], label: "Layer", labelAnchor: "center", labelArrow: "none" },
    y: { domain: [0, ymax], label: "False-minus-true gap (spreads)", labelAnchor: "center", labelArrow: "none" },
    marks: [
      Plot.rectX([bands[1]], { x1: "x1", x2: "x2", y1: 0, y2: ymax, fill: C.primaryLight, fillOpacity: 0.6 }),
      Plot.rectX([bands[2]], { x1: "x1", x2: "x2", y1: 0, y2: ymax, fill: C.grid, fillOpacity: 0.45 }),
      Plot.text(bands, { ...halo(), x: (d) => (d.x1 + d.x2) / 2, y: ymax, dy: -4, text: "text", fill: C.muted, fontSize: 11, lineAnchor: "bottom" }),
      Plot.gridY({ stroke: C.grid, strokeOpacity: 1 }),
      Plot.ruleY([0], { stroke: C.faint }),
      Plot.ruleY(bar8, { x1: (d) => d.layer - 0.4, x2: (d) => d.layer + 0.4, y: "y", stroke: C.muted, strokeWidth: 1.5, strokeDasharray: "3,2" }),
      Plot.ruleX(data.layers, { x: "layer", y1: "after_lo", y2: "after_hi", stroke: C.primary, strokeWidth: 2 }),
      Plot.dot(data.layers.filter((d) => !d.scored), { x: "layer", y: "before", r: 4.5, fill: "none", stroke: C.faint, strokeWidth: 1.5, tip: true, title: tipL }),
      Plot.dot(data.layers.filter((d) => d.scored), { x: "layer", y: "before", r: 4.5, fill: "none", stroke: C.muted, strokeWidth: 1.8, tip: true, title: tipL }),
      Plot.dot(data.layers, { x: "layer", y: "after", r: 4.5, fill: C.primary, stroke: C.paper, strokeWidth: 1, tip: true, title: tipL }),
    ],
  }));
  return root;
}
