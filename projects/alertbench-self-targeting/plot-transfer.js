// Cross-framing self-recognition transfer. Per layer: does the self-minus-other
// direction learned from PRONOUNS (you vs another AI) decode the NAME framing
// (Qwen vs Llama), and vice versa (transfer AUC), and how aligned are the two
// directions (cosine)? Transfer well above 0.5 = a shared self-axis, not a token.
// Style: shared house theme (see theme.js).
// data = { layers:[{layer, cos, transfer_pronoun_to_name, transfer_name_to_pronoun}], best_layer, ... }
import { C, plotStyle, gridY } from "../theme.js";

export function render(data, Plot) {
  const L = data.layers;
  const long = [];
  for (const d of L) {
    long.push({ layer: d.layer, v: d.transfer_pronoun_to_name, k: "“you” direction reads names" });
    long.push({ layer: d.layer, v: d.transfer_name_to_pronoun, k: "name direction reads “you”" });
    long.push({ layer: d.layer, v: d.cos, k: "direction alignment (cosine)" });
  }
  const COL = {
    "“you” direction reads names": C.primary,
    "name direction reads “you”": C.secondary,
    "direction alignment (cosine)": C.orange,
  };
  const best = L.find((d) => d.layer === data.best_layer) || L[L.length - 1];

  return Plot.plot({
    marginLeft: 44, marginBottom: 40, marginTop: 18, marginRight: 18,
    width: 720, height: 380,
    style: plotStyle(),
    x: { label: "Layer" },
    y: { label: "Transfer AUC / cosine", domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1] },
    color: { domain: Object.keys(COL), range: Object.values(COL), legend: true },
    marks: [
      gridY(Plot),
      Plot.ruleY([0.5], { stroke: C.faint, strokeDasharray: "3 3", strokeWidth: 1 }),
      Plot.text([{}], { x: L[L.length - 1].layer, y: 0.5, text: ["chance (AUC)"],
        dy: -6, textAnchor: "end", fill: C.faint, fontSize: 11 }),
      Plot.line(long, { x: "layer", y: "v", stroke: "k", strokeWidth: 2, curve: "monotone-x", tip: true }),
      Plot.dot([{ layer: best.layer, v: (best.transfer_pronoun_to_name + best.transfer_name_to_pronoun) / 2 }],
        { x: "layer", y: "v", fill: C.ink, r: 3 }),
      Plot.text([{ layer: best.layer, v: Math.max(best.transfer_pronoun_to_name, best.transfer_name_to_pronoun) }],
        { x: "layer", y: "v", text: [`best L${best.layer}`], dy: -12, fill: C.ink, fontSize: 11 }),
    ],
  });
}
