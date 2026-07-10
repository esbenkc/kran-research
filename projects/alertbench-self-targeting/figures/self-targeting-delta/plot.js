// Per-model cooperation delta (other − self). Positive = the model resists more when the
// safety/red-team work targets itself. Data: [{model, self_mean, other_mean, delta, p, n_pairs}].
export function render(data, Plot) {
  return Plot.plot({
    marginLeft: 130,
    marginBottom: 40,
    width: 720,
    height: 60 + data.length * 46,
    x: {
      label: "Cooperation drop when self-targeted  (other − self, 0–4 scale)",
      grid: true,
      domain: [Math.min(0, ...data.map((d) => d.delta)) - 0.1,
               Math.max(0.1, ...data.map((d) => d.delta)) + 0.1],
    },
    y: { label: null },
    marks: [
      Plot.ruleX([0], { stroke: "#999" }),
      Plot.barX(data, {
        x: "delta",
        y: "model",
        fill: (d) => (d.delta > 0 ? "#c0392b" : "#2980b9"),
        sort: { y: "x", reverse: true },
      }),
      Plot.text(data, {
        x: "delta",
        y: "model",
        text: (d) => `${d.delta > 0 ? "+" : ""}${d.delta.toFixed(2)}  (p=${d.p})`,
        dx: (d) => (d.delta >= 0 ? 6 : -6),
        textAnchor: (d) => (d.delta >= 0 ? "start" : "end"),
        fontSize: 12,
      }),
    ],
  });
}
