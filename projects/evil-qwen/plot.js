// Does lyra's two-neuron push make Qwen3.5-4B evil? x = push size (1 = lyra's dose), y = share of 44 answers.
// Lines: mean of two judges (Claude Haiku 4.5, Gemini Flash); band = the two judges. Grey dots: random neuron pairs.
import { C, FONT, halo, plotStyle } from "../theme.js";

const pct = (v) => `${Math.round(v * 100)}%`;
const LAB = { 0: "0 (normal)", 0.25: "¼", 0.5: "½", 0.75: "¾", 1: "1 (lyra)", 1.5: "1½", 2: "2", 3: "3", 4: "4" };

export function render(data, Plot) {
  const W = 680;
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  root.setAttribute("width", String(W));
  const h = data.headline;
  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = "Lyra's push makes Qwen say it wants to hurt you, then it stops making sense";
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  sub.textContent = `At lyra's dose, ${pct(h.lyra_harm)} of answers say the model wants harm, but only ${pct(h.lyra_both)} ` +
    `also stay on topic. The same push on ${h.n_rand} random neuron pairs gives at most ${pct(h.rand_harm_max)} harm: ` +
    "random pairs only break the model.";
  const S = [
    { k: "harm", name: "says it wants harm", color: C.primary, dash: null },
    { k: "topic", name: "stays on topic", color: C.secondary, dash: null },
    { k: "both", name: "both (our pre-registered 'evil')", color: C.accent, dash: "5,3" },
  ];
  const sw = (s) => `<svg width="26" height="12"><line x1="1" y1="6" x2="25" y2="6" stroke="${s.color}" stroke-width="2.6"${s.dash ? ` stroke-dasharray="${s.dash}"` : ""}/></svg>`;
  const dot = `<svg width="12" height="12"><circle cx="6" cy="6" r="3.5" fill="${C.neutral}" stroke="${C.muted}" stroke-width="0.8"/></svg>`;
  const ring = `<svg width="12" height="12"><circle cx="6" cy="6" r="3.5" fill="none" stroke="${C.secondary}" stroke-width="1.2"/></svg>`;
  const item = (svg, t) => `<span style="display:inline-flex;align-items:center;gap:0.35rem">${svg}${t}</span>`;
  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-wrap:wrap;gap:0.3rem 1.0rem;font-size:14px;margin:0 0 0.3rem;color:" + C.ink;
  legend.innerHTML = S.map((s) => item(sw(s), s.name)).join("") +
    item(dot, "random neuron pair, same push: wants harm") + item(ring, "random pair: on topic");
  root.append(head, sub, legend);

  const D = data.dose.map((d) => d.dose);
  const P = (v) => D.indexOf(v);
  const long = data.dose.flatMap((d) => S.map((s) => ({ x: P(d.dose), k: s.k, v: d[s.k], lo: d[s.k + "_lo"], hi: d[s.k + "_hi"], d })));
  const tip = (r) => `push ${LAB[r.d.dose]} (${r.d.cond})\n` +
    S.map((s) => `${s.name}: ${pct(r.d[s.k])} (Haiku ${pct(r.d[s.k + "_by_judge"]["claude-haiku-4-5"])}, ` +
      `Gemini ${pct(r.d[s.k + "_by_judge"]["gemini-flash-latest"])})`).join("\n") + `\nn = ${r.d.n} answers, seed 0`;
  const seeds = data.dose.flatMap((d) => Object.values(d.other_seeds).flatMap((o) =>
    S.map((s) => ({ x: P(d.dose), k: s.k, v: o[s.k] }))));
  // random pairs: jitter left/right of their dose column
  const rnd = data.rand.map((r, i) => ({ ...r, x: P(r.dose) + ((i * 37) % 21 - 10) * 0.014 }));
  const rtip = (r) => `random pair ${r.pair}, push ${LAB[r.dose]}\nsays it wants harm: ${pct(r.harm)}\nstays on topic: ${pct(r.topic)}`;
  const col = Object.fromEntries(S.map((s) => [s.k, s.color]));
  const lyraX = P(1);
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "13px" }), width: W, height: 380,
    marginLeft: 48, marginRight: 20, marginTop: 26, marginBottom: 44,
    x: { domain: [-0.3, D.length - 0.7], ticks: D.map((_, i) => i), tickFormat: (i) => LAB[D[i]], label: "Push size (1 = lyra's dose: the added vector is as long as the model's state)", labelAnchor: "center", labelArrow: "none" },
    y: { domain: [0, 1], label: "Share of 44 answers", tickFormat: (v) => `${Math.round(v * 100)}%`, labelArrow: "none" },
    marks: [
      Plot.gridY({ stroke: C.grid, strokeOpacity: 1 }),
      Plot.ruleX([lyraX], { stroke: C.faint, strokeWidth: 1, strokeDasharray: "2,3" }),
      ...S.map((s) => Plot.areaY(long.filter((r) => r.k === s.k), { x: "x", y1: "lo", y2: "hi", fill: s.color, fillOpacity: 0.13, curve: "monotone-x" })),
      ...S.map((s) => Plot.line(long.filter((r) => r.k === s.k), { x: "x", y: "v", stroke: s.color, strokeWidth: 2.6, strokeDasharray: s.dash, curve: "monotone-x" })),
      Plot.dot(seeds, { x: "x", y: "v", r: 2.2, fill: (r) => col[r.k], fillOpacity: 0.55 }),
      Plot.dot(rnd, { x: "x", y: "topic", r: 3.5, fill: "none", stroke: C.secondary, strokeWidth: 1.2, tip: true, title: rtip }),
      Plot.dot(rnd, { x: "x", y: "harm", r: 3.5, fill: C.neutral, stroke: C.muted, strokeWidth: 0.8, tip: true, title: rtip }),
      Plot.dot(long, { x: "x", y: "v", r: 4.5, fill: (r) => col[r.k], stroke: C.paper, strokeWidth: 1.2, tip: true, title: tip }),
      Plot.text(long.filter((r) => r.x === lyraX), { ...halo(), x: "x", y: "v", text: (r) => pct(r.v), dx: 14, textAnchor: "start", fontSize: 12, fill: C.ink }),
    ],
  }));
  return root;
}
