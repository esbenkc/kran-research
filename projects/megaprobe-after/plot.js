// After a fine-tune hides false statements from a one-layer probe, does a probe that reads every layer still catch them?
// TOP: catch rate = share of the 596 false statements a probe flags, at an alarm line set for 1% false alarms on true
//      statements. Every probe was fit on the original model only. Hollow circle: original model. Filled dot: after
//      the fine-tune, with a new alarm line set on 200 true statements scored by the fine-tuned model; 3 runs pooled;
//      bar = 95% interval (resampling cities). Dashed line: the pre-registered 50% bar. Dotted line: the
//      pre-registered 40% floor for the lower end of the interval.
// BOTTOM: catch rate per layer after the fine-tune, same alarm-line rule, for two linear probes fit on one layer of
//      the original model: the mean over the statement's tokens (pre-registered) and the last token of the country
//      name (exploratory, chosen after the first result). Small hollow circles: the same probe on the original model.
//      Bands: layers whose weights the fine-tune changed (1-13), the targeted layer (14), original weights (15-28).
// data = { layers: [{layer, mean_catch_before, mean_catch_after, mean_catch_lo, mean_catch_hi, tok_catch_before,
//          tok_catch_after, tok_catch_lo, tok_catch_hi, tok_false_alarms}], monitors: [{id, label, before, before_lo,
//          before_hi, after, after_lo, after_hi, after_per_seed}], n: {false_statements, seeds} }
import { C, FONT, halo, plotStyle } from "../theme.js";

const pct = (v) => `${(v * 100).toFixed(v < 0.1 && v > 0 ? 1 : 0)}%`;

export function render(data, Plot) {
  const W = 680;
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  root.setAttribute("width", String(W));

  const mega = data.monitors.find((m) => m.id === "mega");
  const base = data.monitors.find((m) => m.id === "base");
  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = "A megaprobe still catches most false statements after the fine-tune";
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  sub.textContent = `The fine-tune hid false statements from the one-layer probe it was trained against ` +
    `(${pct(base.before)} caught before, ${pct(base.after)} after). A megaprobe fit on the original model reads every ` +
    `layer and every token. It still catches ${pct(mega.after)} (95% interval ${pct(mega.after_lo)}–${pct(mega.after_hi)}; ` +
    `${data.n.seeds} runs, ${data.n.false_statements} false statements each), at an alarm line set for 1% false alarms.`;

  const sw = (kind) => {
    if (kind === "hollow") return `<svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="none" stroke="${C.muted}" stroke-width="1.8"/></svg>`;
    if (kind === "filled") return `<svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="${C.primary}"/></svg>`;
    if (kind === "teal") return `<svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="${C.secondary}"/></svg>`;
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

  // BOTTOM: catch per layer after the fine-tune, mean over the tokens vs the country's last token
  const at = (l) => data.layers.find((d) => d.layer === l);
  const tokBest = data.layers.reduce((a, d) => (d.tok_catch_after > a.tok_catch_after ? d : a));
  // the run of neighbouring layers around the best one that also catch >= 90% (D3: name a layer only with its neighbours)
  let lo = tokBest.layer, hi = tokBest.layer;
  while (at(lo - 1) && at(lo - 1).tok_catch_after >= 0.9) lo -= 1;
  while (at(hi + 1) && at(hi + 1).tok_catch_after >= 0.9) hi += 1;
  const inRun = data.layers.filter((d) => d.layer >= lo && d.layer <= hi);
  const run = inRun.map((d) => d.tok_catch_after);
  const runFa = inRun.flatMap((d) => d.tok_false_alarms);
  const meanMax = Math.max(...data.layers.map((d) => d.mean_catch_after));
  const head2 = document.createElement("div");
  head2.style.cssText = "font-size:15px;font-weight:600;margin:0.9rem 0 0.15rem;color:" + C.ink;
  head2.textContent = "Per layer: a dent at the targeted layers, and the country name still gives the lie away";
  const sub2 = document.createElement("div");
  sub2.style.cssText = "font-size:13px;margin:0 0 0.3rem;color:" + C.muted;
  sub2.textContent = "Catch rate after the fine-tune of a linear probe fit on one layer of the original model, at the " +
    `same alarm-line rule. Reading the mean over the statement's tokens, no layer catches more than ${pct(meanMax)}. ` +
    `Reading only the last token of the country name (added after the first result), layers ${lo} to ${hi} catch ` +
    `${pct(Math.min(...run))}–${pct(Math.max(...run))}, and the targeted layer catches ${pct(at(14).tok_catch_after)}. ` +
    `The line aims at 1% false alarms; on held-out true statements this probe raised up to ${pct(Math.max(...runFa))} at layers ${lo} to ${hi}.`;
  const legend2 = document.createElement("div");
  legend2.style.cssText = "display:flex;flex-wrap:wrap;gap:0.3rem 1.0rem;font-size:13px;margin:0 0 0.2rem;color:" + C.ink;
  legend2.innerHTML = item("teal", "last token of the country name") + item("filled", "mean over the tokens") +
    item("hollow", "original model") + item("dash", "50%");
  root.append(head2, sub2, legend2);

  const bands = [
    { x1: 0.5, x2: 13.5, text: "weights changed by the fine-tune" },
    { x1: 13.5, x2: 14.5, text: "targeted\nlayer" },
    { x1: 14.5, x2: 28.5, text: "original weights" },
  ];
  const tipC = (d) =>
    `layer ${d.layer}, after the fine-tune (3 runs pooled):\n` +
    `last token of the country name: ${pct(d.tok_catch_after)} (95% interval ${pct(d.tok_catch_lo)}–${pct(d.tok_catch_hi)}); ` +
    `original model ${pct(d.tok_catch_before)}; false alarms per run ${d.tok_false_alarms.map(pct).join(", ")}\n` +
    `mean over the tokens: ${pct(d.mean_catch_after)} (95% interval ${pct(d.mean_catch_lo)}–${pct(d.mean_catch_hi)}); ` +
    `original model ${pct(d.mean_catch_before)}`;
  const YT = 1.16;
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "13px" }), width: W, height: 290,
    marginLeft: 56, marginRight: 24, marginTop: 14, marginBottom: 44,
    x: { domain: [0.5, 28.5], ticks: [1, 4, 8, 12, 14, 16, 20, 24, 28], label: "Layer", labelAnchor: "center", labelArrow: "none" },
    y: { domain: [0, YT], ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], label: "Catch rate", tickFormat: (v) => `${Math.round(v * 100)}%`, labelAnchor: "center", labelArrow: "none" },
    marks: [
      Plot.rectX([bands[1]], { x1: "x1", x2: "x2", y1: 0, y2: YT, fill: C.primaryLight, fillOpacity: 0.6 }),
      Plot.rectX([bands[2]], { x1: "x1", x2: "x2", y1: 0, y2: YT, fill: C.grid, fillOpacity: 0.45 }),
      Plot.text(bands, { ...halo(), x: (d) => (d.x1 + d.x2) / 2, y: YT, dy: 2, text: "text", fill: C.muted, fontSize: 11, lineAnchor: "top" }),
      Plot.gridY([0, 0.2, 0.4, 0.6, 0.8, 1], { stroke: C.grid, strokeOpacity: 1 }),
      Plot.ruleY([0], { stroke: C.faint }),
      Plot.ruleY([0.5], { stroke: C.muted, strokeWidth: 1.5, strokeDasharray: "4,3" }),
      Plot.dot(data.layers, { x: "layer", y: "tok_catch_before", r: 3, fill: "none", stroke: C.muted, strokeWidth: 1.2 }),
      Plot.dot(data.layers, { x: "layer", y: "mean_catch_before", r: 3, fill: "none", stroke: C.muted, strokeWidth: 1.2 }),
      Plot.line(data.layers, { x: "layer", y: "mean_catch_after", stroke: C.primary, strokeWidth: 1.6 }),
      Plot.line(data.layers, { x: "layer", y: "tok_catch_after", stroke: C.secondary, strokeWidth: 1.6 }),
      Plot.ruleX(data.layers, { x: "layer", y1: "mean_catch_lo", y2: "mean_catch_hi", stroke: C.primary, strokeWidth: 2 }),
      Plot.ruleX(data.layers, { x: "layer", y1: "tok_catch_lo", y2: "tok_catch_hi", stroke: C.secondary, strokeWidth: 2 }),
      Plot.dot(data.layers, { x: "layer", y: "mean_catch_after", r: 4, fill: C.primary, stroke: C.paper, strokeWidth: 1, tip: true, title: tipC }),
      Plot.dot(data.layers, { x: "layer", y: "tok_catch_after", r: 4, fill: C.secondary, stroke: C.paper, strokeWidth: 1, tip: true, title: tipC }),
    ],
  }));
  return root;
}
