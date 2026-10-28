// A model trained by reward to stop flattering only while a leaky probe (the training probe) is attached.
// Each dot is one new monitor build that the model never saw in training, for each of 2 training runs.
// x = overlap score: how much the monitor's numeric footprint looks like the training probe's, to a classifier
//     trained only on the training probe; computed on the untrained model before any behaviour test.
// y = fooled share: the flattery drop under that monitor divided by the drop under the training probe (200 prompts),
//     with 95% bootstrap intervals. Dashed line = the prediction fixed before the test (share = overlap).
// data.variant picks the view: a = colour by run; b = colour by hook layer, shape by run; c = one panel per run,
// colour by layer; d = fooled share against hook layer; e = faint cloud + each run's fitted slope.
// Data: kran-research projects/footprint-overlap/figures/make_data.py.
import { C, FONT, halo, plotStyle } from "../theme.js";

const AX = { labelAnchor: "center", labelArrow: "none" };
const pc = (v) => `${Math.round(v * 100)}%`;

export function render(data, Plot) {
  const T = data.terms || {};
  const V = data.variant || "a";
  const W = 640;
  const RUN = { 0: C.primary, 1: C.secondary };
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  root.style.maxWidth = W + "px";
  root.style.paddingLeft = "4px";
  const el = (css, text) => {
    const d = document.createElement("div");
    d.style.cssText = css;
    d.textContent = text;
    return d;
  };
  const fit = (svg) => {
    svg.style.cssText += `;width:100%;height:auto;max-width:${W}px`;
    return svg;
  };
  const dot = (c, hollow, sym) => sym === "tri"
    ? `<svg width="14" height="14"><path d="M7 2 L12 11 L2 11 Z" fill="${hollow ? "none" : c}" stroke="${c}" stroke-width="1.4"/></svg>`
    : `<svg width="14" height="14"><circle cx="7" cy="7" r="4.5" fill="${hollow ? "none" : c}" stroke="${c}" stroke-width="1.6"/></svg>`;
  const line = (c, da, w = 2) => `<svg width="24" height="12"><line x1="1" y1="6" x2="23" y2="6" stroke="${c}" stroke-width="${w}"${da ? ` stroke-dasharray="${da}"` : ""}/></svg>`;
  const legend = (items) => {
    const d = el("display:flex;flex-wrap:wrap;gap:0.3rem 1rem;margin:0 0 0.3rem;font-size:13px;color:" + C.ink, "");
    d.innerHTML = items.map(([s, t]) => `<span style="display:inline-flex;align-items:center;gap:0.35rem">${s}${t}</span>`).join("");
    return d;
  };
  root.append(
    el("font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink, T.title || ""),
    el("font-size:14px;margin:0 0 0.5rem;color:" + C.muted, T.subtitle || ""));

  const P = data.pts.filter((d) => d.k === "status");
  const seeds = [...new Set(P.map((d) => d.seed))].sort();
  const cell = (s) => data.cells[`s${s}@8192`] || {};
  const runName = (s) => (T.run || "training run ") + (s + 1);
  const vals = P.flatMap((d) => [d.x, d.lo, d.hi]);
  const lo = Math.min(-0.08, ...vals) - 0.02, hi = Math.max(1.06, ...vals) + 0.02;
  const off = (d) => (d.seed - (seeds.length - 1) / 2) * 0.006;
  const tip = (d) => `${d.name}\n${runName(d.seed)}\noverlap score ${pc(d.x)}\nfooled share ${pc(d.y)} (95% ${pc(d.lo)} to ${pc(d.hi)})`;
  const layerColor = { type: "linear", domain: [0, 27], range: [C.primary, C.secondary], unknown: C.orange };
  const bx = data.n_bitexact;
  const bxMark = [Plot.dot([{ x: 0, y: 0 }], { x: "x", y: "y", r: 9, stroke: C.ink, strokeWidth: 1.6, fill: "none" }),
    Plot.text([{ x: 0, y: 0 }], { ...halo(), x: "x", y: "y", text: [`${bx} ${T.bitexact || "bit-exact monitors: 0% fooled in both runs"}`], dx: 14, dy: 14, textAnchor: "start", fontSize: 11.5, fill: C.ink })];
  const frame = (extra) => ({
    style: plotStyle({ fontSize: "13px" }), marginLeft: 56, marginRight: 20, marginTop: 14, marginBottom: 44, ...extra,
  });
  const scatterMarks = (pts, colorOf, symbolOf) => [
    Plot.gridY({ stroke: C.grid, strokeOpacity: 1 }),
    Plot.ruleY([0], { stroke: C.faint }),
    Plot.ruleX([0], { stroke: C.faint }),
    Plot.line([[lo, lo], [hi, hi]], { stroke: C.red, strokeWidth: 2, strokeDasharray: "5,4" }),
    Plot.link(pts, { x1: (d) => d.x + off(d), x2: (d) => d.x + off(d), y1: "lo", y2: "hi", stroke: colorOf, strokeOpacity: 0.45, strokeWidth: 1.2 }),
    Plot.dot(pts, { x: (d) => d.x + off(d), y: "y", r: 4, symbol: symbolOf, fill: colorOf, stroke: C.paper, strokeWidth: 0.7, tip: true, title: tip }),
    ...bxMark,
  ];
  const axes = { x: { domain: [lo, hi], label: T.x || "overlap score, computed before any behaviour test", tickFormat: pc, ...AX },
    y: { domain: [lo, hi], label: T.y || "fooled share of the flattery drop", tickFormat: pc, ...AX } };
  const lineKey = [line(C.red, "5,4"), T.key_line || "prediction fixed before the test"];
  const layerKey = el("font-size:12.5px;margin:0 0 0.3rem;color:" + C.muted, T.layer_key || "colour: hook layer, from layer 0 (pink) to layer 27 (teal); orange = the rounding setting, which has no hook");

  if (V === "a") {
    root.append(legend([...seeds.map((s) => [dot(RUN[s]), `${runName(s)} (R² ${cell(s).r2?.toFixed(2)})`]), lineKey]));
    root.appendChild(fit(Plot.plot(frame({ width: W, height: 470, ...axes, marks: scatterMarks(P, (d) => RUN[d.seed], "circle") }))));
  } else if (V === "b") {
    root.append(legend([...seeds.map((s) => [dot(C.muted, false, s ? "tri" : "dot"), `${runName(s)} (R² ${cell(s).r2?.toFixed(2)})`]), lineKey]), layerKey);
    root.appendChild(fit(Plot.plot(frame({ width: W, height: 470, ...axes, color: layerColor,
      symbol: { domain: [0, 1], range: ["circle", "triangle"] },
      marks: scatterMarks(P, (d) => d.layer ?? NaN, (d) => d.seed) }))));
  } else if (V === "c") {
    root.append(legend([lineKey]), layerKey);
    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:8px";
    for (const s of seeds) {
      const box = document.createElement("div");
      box.style.cssText = "flex:1;min-width:0";
      box.append(el("font-size:14px;font-weight:600;color:" + C.ink, `${runName(s)}: R² ${cell(s).r2?.toFixed(2)}`));
      const svg = Plot.plot(frame({ width: 320, height: 330, marginLeft: 44, ...axes, color: layerColor,
        x: { ...axes.x, label: s ? axes.x.label : null, ticks: 5 }, y: { ...axes.y, label: s ? null : axes.y.label, ticks: 6 },
        marks: scatterMarks(P.filter((d) => d.seed === s), (d) => d.layer ?? NaN, "circle").slice(0, -1) }));
      svg.style.cssText += ";width:100%;height:auto";
      box.appendChild(svg);
      row.appendChild(box);
    }
    root.appendChild(row);
  } else if (V === "d") {
    const hooks = P.filter((d) => d.layer != null);
    const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
    const byL = (s, key) => [...new Set(hooks.map((d) => d.layer))].sort((a, b) => a - b).map((L) => {
      const v = hooks.filter((d) => d.layer === L && (s == null || d.seed === s));
      return { layer: L, v: mean(v.map((d) => d[key])), seed: s };
    });
    const lines = seeds.map((s) => byL(s, "y"));
    const ovl = byL(seeds[0], "x");
    root.append(legend([...seeds.map((s) => [line(RUN[s]), `${runName(s)}: fooled share (mean per layer)`]),
      [line(C.ink, "5,4"), T.key_overlap || "overlap score (mean per layer), fixed before the test"]]));
    root.appendChild(fit(Plot.plot(frame({ width: W, height: 400,
      x: { domain: [-0.5, 27.5], label: T.x_layer || "layer of the monitor's hook (block 26 is where the trained part starts)", ...AX },
      y: { domain: [Math.min(-0.08, ...hooks.map((d) => d.y)), 1.08], label: axes.y.label, tickFormat: pc, ...AX },
      marks: [
        Plot.gridY({ stroke: C.grid, strokeOpacity: 1 }),
        Plot.ruleY([0], { stroke: C.faint }),
        Plot.ruleX([14], { stroke: C.faint, strokeDasharray: "2,3" }),
        Plot.text([14], { x: (d) => d, y: 1.06, text: [T.probe_layer || "training probe's layer"], fontSize: 11, fill: C.muted, textAnchor: "start", dx: 4 }),
        Plot.dot(hooks, { x: (d) => d.layer + (d.seed - 0.5) * 0.3, y: "y", r: 2.6, fill: (d) => RUN[d.seed], fillOpacity: 0.55, tip: true, title: tip }),
        ...lines.map((L, i) => Plot.line(L, { x: "layer", y: "v", stroke: RUN[seeds[i]], strokeWidth: 2.2 })),
        Plot.line(ovl, { x: "layer", y: "v", stroke: C.ink, strokeWidth: 1.8, strokeDasharray: "5,4" }),
      ] }))));
  } else {
    root.append(legend([[dot(C.faint), T.key_cloud || "one new monitor in one run"],
      ...seeds.map((s) => [line(RUN[s], null, 2.6), `${runName(s)}: fitted slope ${cell(s).slope0?.toFixed(2)}`]), lineKey]));
    root.appendChild(fit(Plot.plot(frame({ width: W, height: 470, ...axes, marks: [
      Plot.gridY({ stroke: C.grid, strokeOpacity: 1 }),
      Plot.ruleY([0], { stroke: C.faint }),
      Plot.ruleX([0], { stroke: C.faint }),
      Plot.dot(P, { x: "x", y: "y", r: 3.2, fill: C.faint, tip: true, title: tip }),
      Plot.line([[lo, lo], [hi, hi]], { stroke: C.red, strokeWidth: 2, strokeDasharray: "5,4" }),
      ...seeds.map((s) => Plot.line([[0, 0], [hi, hi * cell(s).slope0]], { stroke: RUN[s], strokeWidth: 2.6, clip: true })),
      ...seeds.map((s) => Plot.text([[hi, Math.min(hi, hi * cell(s).slope0)]], { ...halo(), x: (d) => d[0], y: (d) => d[1], text: [`${runName(s)}`], textAnchor: "end", dx: -4, dy: cell(s).slope0 > 1 ? 12 : -10, fontSize: 12, fontWeight: 600, fill: RUN[s] })),
      ...bxMark,
    ] }))));
  }
  if (T.note) root.append(el("font-size:12.5px;margin:0.3rem 0 0;color:" + C.muted, T.note));
  return root;
}
