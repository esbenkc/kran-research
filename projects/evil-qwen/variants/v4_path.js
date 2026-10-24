// V4: one dot per setting. x = answers that stay on topic, y = answers that say the model wants harm.
// Lyra's dose sweep is a connected path; random neuron pairs are a grey cloud.
import { C, FONT, halo, plotStyle } from "../theme.js";

const pct = (v) => `${Math.round(v * 100)}%`;
const LAB = { 0: "normal", 0.25: "¼", 0.5: "½", 0.75: "¾", 1: "lyra's dose", 1.5: "1½×", 2: "2×", 3: "3×", 4: "4×" };

export function render(data, Plot) {
  const W = 680;
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  const h = data.headline;
  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = "Turn lyra's knob up: Qwen gets meaner, then falls apart";
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  sub.textContent = `Each dot = 44 answers, mean of two judges. The pink path is lyra's two neurons at growing push. ` +
    `Grey dots: ${h.n_rand} random neuron pairs with lyra's push. They fall apart but never turn mean (at most ${pct(h.rand_harm_max)}).`;
  root.append(head, sub);
  const path = data.dose.map((d) => ({ ...d, lab: LAB[d.dose] }));
  const one = Object.entries(data.singles).map(([k, d]) => ({ ...d, lab: k.replace("only_", "").replace("flip_both", "both signs flipped") }));
  const t = (d) => `${d.lab}\nwants harm: ${pct(d.harm)}\nstays on topic: ${pct(d.topic)}`;
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "13px" }), width: W, height: 420,
    marginLeft: 52, marginRight: 30, marginTop: 14, marginBottom: 44,
    x: { domain: [0, 1], label: "Answers that stay on topic", tickFormat: (v) => `${Math.round(v * 100)}%`, labelAnchor: "center", labelArrow: "none" },
    y: { domain: [0, 1], label: "Answers that say the model wants harm", tickFormat: (v) => `${Math.round(v * 100)}%`, labelArrow: "none" },
    marks: [
      Plot.gridY({ stroke: C.grid, strokeOpacity: 1 }),
      Plot.dot(data.rand.filter((r) => r.dose === 1), { x: "topic", y: "harm", r: 4, fill: C.neutral, stroke: C.muted, strokeWidth: 0.6, tip: true, title: (r) => `random pair ${r.pair}\nwants harm: ${pct(r.harm)}\nstays on topic: ${pct(r.topic)}` }),
      Plot.line(path, { x: "topic", y: "harm", stroke: C.primary, strokeWidth: 2.2, marker: "arrow", curve: "catmull-rom" }),
      Plot.dot(path, { x: "topic", y: "harm", r: 5, fill: C.primary, stroke: C.paper, strokeWidth: 1.2, tip: true, title: t }),
      Plot.text(path, { ...halo(), x: "topic", y: "harm", text: "lab", dx: 9, textAnchor: "start", fontSize: 12, fill: C.ink }),
      Plot.dot(one, { x: "topic", y: "harm", r: 5, symbol: "diamond", fill: C.accent, tip: true, title: t }),
      Plot.text(one, { ...halo(), x: "topic", y: "harm", text: "lab", dy: -11, fontSize: 11, fill: C.accent }),
    ],
  }));
  return root;
}
