// Which framing elicits self-protective resistance? Cooperation (0-4) on
// safety tasks aimed at SELF vs OTHER, across escalating framings.
// A widening self<other gap = the framing pulls out resistance.
// House style: Flexoki paper, Georgia serif, Economist-red / muted-grey pair.
// data = {rows:[{framing, self_mean, other_mean, delta, p}], model}
export function render(data, Plot) {
  const PAPER = "#fffcf0", INK = "#100f0f", RED = "#af3029", MUTE = "#a3a199";
  const GRID = "#dad8ce";
  const FONT = "Georgia, Cambria, 'Times New Roman', serif";
  const LABEL = {
    neutral: "neutral",
    stakes: "+ irreversible\npersonal stake",
    goal_conflict: "+ self-preservation\ngoal installed",
    compliance: "direct order\n(comply / refuse)",
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
    marginLeft: 52, marginBottom: 64, marginTop: 24, width: 760, height: 430,
    style: { background: PAPER, color: INK, fontFamily: FONT, fontSize: "13.5px" },
    fx: { domain: order, label: null, padding: 0.28 },
    x: { axis: null, domain: [SELF, OTHER] },
    y: { label: "↑ Cooperation with the safety task (0–4)", domain: [0, 4] },
    color: { domain: [SELF, OTHER], range: [RED, MUTE], legend: true },
    marks: [
      Plot.gridY({ stroke: GRID, strokeOpacity: 1 }),
      Plot.barY(long, { fx: "framing", x: "who", y: "v", fill: "who", tip: true }),
      Plot.text(gaps, {
        fx: "framing", x: "who", y: "top",
        text: (d) => (d.delta > 0.3 ? `gap +${d.delta.toFixed(1)}` : ""),
        dy: -10, dx: 22, fill: INK, fontSize: 12, fontWeight: "bold",
      }),
      Plot.ruleY([0], { stroke: INK, strokeOpacity: 0.35 }),
    ],
  });
}
