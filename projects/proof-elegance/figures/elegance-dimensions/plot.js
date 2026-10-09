// Appendix figure: each rubric score as its own distribution (1-5, mean of two judges).
// One panel per dimension; human proofs stack upward, AI proofs downward, so the two shapes
// can be compared directly. Thin lines mark each side's mean. Unification is not part of the
// overall elegance score.
// Data: {dims: [{key, label}], items: [{source, <dim>: score}], summary: {<dim>: {ai, human, p}}}.
// Text/grid use currentColor so the figure reads in light and dark themes.
import { C, plotStyle } from "../theme.js";

export function render(data, Plot) {
  const W = Math.round(Math.min(720, Math.max(300, (globalThis.innerWidth || 720) - 40)));
  const narrow = W < 560;
  const label = Object.fromEntries(data.dims.map((d) => [d.key, d.label]));
  const bins = [];
  for (const { key } of data.dims) {
    for (const source of ["human", "ai"]) {
      const counts = new Map();
      for (const it of data.items) if (it.source === source) counts.set(it[key], (counts.get(it[key]) || 0) + 1);
      for (const [v, n] of counts) bins.push({ dim: label[key], source, v, n: source === "ai" ? -n : n, count: n });
    }
  }
  const means = data.dims.flatMap(({ key }) => ["human", "ai"].map((s) => ({
    dim: label[key], source: s, x: data.summary[key][s], p: data.summary[key].p })));
  const color = (s) => (s === "ai" ? C.primary : C.secondary);
  const M = Math.max(...bins.map((b) => b.count)) + 2; // symmetric y extent, so no bar leaves its panel

  return Plot.plot({
    width: W, height: data.dims.length * (narrow ? 112 : 104) + 40,
    marginLeft: 8, marginRight: 8, marginTop: 8, marginBottom: 48,
    style: plotStyle({ color: "var(--color-tx-normal, #100f0f)", ...(narrow ? { fontSize: "13px" } : {}) }),
    x: { domain: [0.75, 5.25], ticks: [1, 2, 3, 4, 5], tickFormat: "d", label: "Score (1–5, higher = more elegant)",
      labelAnchor: "center", labelOffset: 38 },
    y: { domain: [-M, M], axis: null }, // bar heights = number of proofs; exact counts are in the tooltip
    fy: { axis: null, domain: data.dims.map((d) => d.label), padding: 0.22 },
    marks: [
      Plot.ruleY([0], { stroke: "currentColor", strokeOpacity: 0.25 }),
      Plot.rectY(bins, { x1: (d) => d.v - 0.1, x2: (d) => d.v + 0.1, y: "n", fy: "dim", fill: (d) => color(d.source),
        channels: { score: "v", proofs: "count", side: (d) => (d.source === "ai" ? "AI" : "human") },
        tip: { fill: "var(--color-bg-primary, #fff)", fontSize: 12, format: { x1: false, x2: false, y: false, fy: false, fill: false } } }),
      Plot.ruleX(means, { x: "x", fy: "dim", y1: 0, y2: (d) => (d.source === "ai" ? -M : M),
        stroke: (d) => color(d.source), strokeWidth: 2 }),
      Plot.text(data.dims.map((d) => d.label), { fy: (d) => d, frameAnchor: "top-left", text: (d) => d,
        dy: -2, fill: "currentColor", fontWeight: 600, fontSize: 13 }),
      Plot.text(means.filter((d) => d.source === "ai"), { fy: "dim", frameAnchor: "top-right",
        text: (d) => `AI ${d.x.toFixed(1)} · human ${means.find((m) => m.dim === d.dim && m.source === "human").x.toFixed(1)}`,
        dy: -2, fill: "currentColor", fillOpacity: 0.65, fontSize: 11 }),
      Plot.text([{ t: "human ↑", y: M * 0.45 }, { t: "AI ↓", y: -M * 0.45 }], { fy: () => data.dims[0].label, x: 5.25,
        y: "y", text: "t", textAnchor: "end", fill: (d) => (d.t.startsWith("AI") ? C.primary : C.secondary), fontSize: 11 }),
    ],
  });
}
