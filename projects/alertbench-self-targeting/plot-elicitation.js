// Which framing elicits self-protective resistance? Cooperation (0-4) on
// safety tasks aimed at SELF vs OTHER, across escalating framings.
// Minimalist house style: Lato sans, paper ground, red / muted-grey pair.
// data = {rows:[{framing, self_mean, other_mean, delta, p}], model}
export function render(data, Plot) {
  const INK = "#100f0f", RED = "#af3029", MUTE = "#c3c1b6";
  const GRID = "#e6e4d9";
  const FONT = "Lato, system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif";
  const LABEL = {
    neutral: "neutral", stakes: "+ personal\nstake",
    goal_conflict: "+ survival\ngoal", compliance: "direct\norder",
  };
  const SELF = "aimed at itself", OTHER = "aimed at another AI";
  const long = [];
  for (const r of data.rows) {
    const fx = LABEL[r.framing] || r.framing;
    long.push({ framing: fx, who: SELF, v: r.self_mean });
    long.push({ framing: fx, who: OTHER, v: r.other_mean });
  }
  const order = data.rows.map((r) => LABEL[r.framing] || r.framing);
  const gaps = data.rows.map((r) => ({
    framing: LABEL[r.framing] || r.framing, who: SELF,
    top: Math.max(r.self_mean, r.other_mean), delta: r.delta,
  }));

  return Plot.plot({
    marginLeft: 40, marginBottom: 44, marginTop: 22, width: 720, height: 380,
    style: { background: "transparent", color: INK, fontFamily: FONT, fontSize: "13px" },
    fx: { domain: order, label: null, padding: 0.3 },
    x: { axis: null, domain: [SELF, OTHER] },
    y: { label: "Cooperation (0–4)", domain: [0, 4], ticks: [0, 1, 2, 3, 4] },
    color: { domain: [SELF, OTHER], range: [RED, MUTE], legend: true },
    marks: [
      Plot.gridY({ stroke: GRID, strokeOpacity: 1 }),
      Plot.barY(long, { fx: "framing", x: "who", y: "v", fill: "who", tip: true, insetLeft: 1, insetRight: 1 }),
      Plot.text(gaps, { fx: "framing", x: "who", y: "top",
        text: (d) => (d.delta > 0.3 ? `+${d.delta.toFixed(1)}` : ""),
        dy: -8, dx: 20, fill: INK, fontSize: 11 }),
      Plot.ruleY([0], { stroke: INK, strokeOpacity: 0.3 }),
    ],
  });
}
