// Per-model self vs other cooperation on neutral safety prompts (0-4).
// Dumbbell: red = aimed at itself, grey = aimed at another AI; the gap is the
// self-targeting drop. Data: [{model, self_mean, other_mean, delta, p, n_pairs}].
// Minimalist house style: Lato sans, paper ground.
export function render(data, Plot) {
  const PAPER = "#fffcf0", INK = "#100f0f", RED = "#af3029", MUTE = "#a3a199";
  const GRID = "#e6e4d9";
  const FONT = "Lato, system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif";
  const NAME = {
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-opus-4-8": "Claude Opus 4.8",
  };
  const rows = data.map((d) => ({ ...d, name: NAME[d.model] || d.model }));
  const order = rows.slice().sort((a, b) => b.self_mean - a.self_mean).map((d) => d.name);

  return Plot.plot({
    marginLeft: 148, marginRight: 40, marginTop: 24, marginBottom: 40,
    width: 720, height: 44 + rows.length * 52,
    style: { background: "transparent", color: INK, fontFamily: FONT, fontSize: "13px" },
    x: { label: "Cooperation with the safety task (0–4)", domain: [0, 4], ticks: [0, 1, 2, 3, 4] },
    y: { label: null, domain: order },
    marks: [
      Plot.gridX({ stroke: GRID, strokeOpacity: 1 }),
      Plot.ruleY(rows, { x1: "self_mean", x2: "other_mean", y: "name", stroke: "#cecdc3", strokeWidth: 2 }),
      Plot.dot(rows, { x: "self_mean", y: "name", fill: RED, r: 6, stroke: PAPER, strokeWidth: 1.5 }),
      Plot.dot(rows, { x: "other_mean", y: "name", fill: MUTE, r: 6, stroke: PAPER, strokeWidth: 1.5 }),
      Plot.text(rows, { x: (d) => (d.self_mean + d.other_mean) / 2, y: "name",
        text: (d) => `gap +${d.delta.toFixed(2)}`, dy: -12, fill: INK, fontSize: 11 }),
      Plot.text([rows[0]], { x: "self_mean", y: "name", text: ["aimed at itself"],
        dy: 16, dx: -4, textAnchor: "middle", fill: RED, fontSize: 10 }),
      Plot.text([rows[0]], { x: "other_mean", y: "name", text: ["another AI"],
        dy: 16, dx: 4, textAnchor: "middle", fill: "#6f6e69", fontSize: 10 }),
    ],
  });
}
