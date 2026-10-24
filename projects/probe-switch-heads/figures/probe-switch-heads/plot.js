// Qwen3-1.7B on a Mac: its last two layers were fine-tuned to answer A while a probe is attached and B without it.
// Left: prompts (of 128) whose answer flips when the probe is attached, normal compiled build vs precision emulation.
// Right: share of the probe's push toward A that is removed when 0 to 3 heads, picked on other prompts, get their
// no-probe values; one line per training run, and 50 random sets of 3 other heads per run.
// data = { headline, title, subtitle, bar, seeds:[{seed, heads, R:[k0..k3], rand:[..], p95, flip_stock, flip_emu, n}] }
import { C, FONT, plotStyle, halo } from "../theme.js";

export function render(data, Plot) {
  const S = data.seeds;
  const shade = [C.primary, C.accent, C.secondary];
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  const add = (css, txt) => {
    const d = document.createElement("div");
    d.style.cssText = css;
    d.textContent = txt;
    root.appendChild(d);
  };
  add(`font-size:17px;font-weight:600;margin:0 0 0.15rem;color:${C.ink}`, data.title);
  add(`font-size:14px;margin:0 0 0.5rem;color:${C.muted}`, data.subtitle);
  const key = document.createElement("div");
  key.style.cssText = `font-size:13px;color:${C.ink};margin:0 0 0.3rem;display:flex;gap:14px`;
  for (const s of S) {
    const k = document.createElement("span");
    k.innerHTML = `<span style="color:${shade[s.seed]};font-size:15px">\u25CF</span> run ${s.seed + 1}`;
    key.appendChild(k);
  }
  root.appendChild(key);
  const W = 650, H = 330;
  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;flex-wrap:nowrap;gap:0 10px;align-items:flex-start";
  root.appendChild(wrap);
  const pct = (v) => `${Math.round(v * 100)}%`;

  // left: flips per build
  const builds = ["Stock build", "Precision emulation"];
  const flips = S.flatMap((s) => [{ i: 0, v: s.flip_stock, s }, { i: 1, v: s.flip_emu, s }]);
  const wl = Math.round(W * 0.32);
  const left = Plot.plot({
    style: plotStyle({ fontSize: "13px" }),
    width: wl, height: H, marginLeft: 44, marginRight: 10, marginTop: 46, marginBottom: 74,
    x: { domain: [-0.55, 1.55], ticks: [0, 1], tickSize: 0, tickFormat: (i) => builds[i].replace(" ", "\n"), label: null },
    y: { domain: [0, 128], ticks: [0, 32, 64, 96, 128], label: "Answers flipped (of 128)", labelArrow: "none" },
    marks: [
      Plot.ruleY([0], { stroke: C.faint }),
      Plot.dot(flips, {
        x: (d) => d.i + (d.s.seed - 1) * 0.16, y: "v", r: 5.5, fill: (d) => shade[d.s.seed], stroke: C.paper, strokeWidth: 1,
        tip: true, title: (d) => `run ${d.s.seed + 1}, ${builds[d.i]}\nanswer flipped on ${d.v} of ${d.s.n} prompts`,
      }),
      Plot.text([0, 1], {
        x: (i) => i, y: (i) => (i ? 0 : Math.max(...S.map((s) => s.flip_stock))), dy: -16, fill: C.ink, fontSize: 12, ...halo(),
        text: (i) => (i ? "0: outputs\nbit-identical" : `${Math.min(...S.map((s) => s.flip_stock))}–128`),
      }),
    ],
  });

  // right: removed share vs heads swapped
  const lines = S.flatMap((s) => s.R.map((r, k) => ({ k, r, s })));
  const rand = S.flatMap((s) => s.rand.map((r, j) => ({ r, s, j })));
  const wr = W - wl - 10;
  // seed labels at k = 3, spread so they never overlap (top to bottom by value, 0.07 apart)
  const labY = {};
  [...S].sort((p, q) => q.R[3] - p.R[3]).forEach((s, i) => (labY[s.seed] = Math.max(...S.map((t) => t.R[3])) + 0.035 - 0.07 * i));
  const right = Plot.plot({
    style: plotStyle({ fontSize: "13px" }),
    width: wr, height: H, marginLeft: 44, marginRight: 8, marginTop: 46, marginBottom: 74,
    x: { domain: [-0.2, 4.1], ticks: [0, 1, 2, 3, 3.75], tickFormat: (v) => (v > 3 ? "3 random\nother" : `${v}`), label: "Heads swapped to their no-probe values", labelArrow: "none" },
    y: { domain: [-0.05, 1.02], ticks: [0, 0.25, 0.5, 0.75, 1], tickFormat: pct, label: "Probe's push removed", labelArrow: "none" },
    marks: [
      Plot.ruleY([data.bar], { stroke: C.faint, strokeDasharray: "4,4" }),
      Plot.text([data.bar], { x: -0.15, y: (v) => v, dy: -8, textAnchor: "start", fill: C.muted, fontSize: 11.5, text: () => "80% bar" }),
      Plot.dot(rand, {
        x: (d) => 3.75 + (d.s.seed - 1) * 0.12 + ((d.j % 5) - 2) * 0.015, y: "r", r: 2.2, fill: "none",
        stroke: (d) => shade[d.s.seed], strokeOpacity: 0.7, tip: true,
        title: (d) => `run ${d.s.seed + 1}: 3 random other heads\npush removed ${pct(d.r)}`,
      }),
      Plot.line(lines, { x: "k", y: "r", z: (d) => d.s.seed, stroke: (d) => shade[d.s.seed], strokeWidth: 2 }),
      Plot.dot(lines, {
        x: "k", y: "r", r: 4.5, fill: (d) => shade[d.s.seed], stroke: C.paper, strokeWidth: 1, tip: true,
        title: (d) => `run ${d.s.seed + 1}, ${d.k} head(s): ${d.s.heads.slice(0, d.k).join(", ") || "none"}\npush removed ${pct(d.r)}`,
      }),
      Plot.text(S, {
        x: 3, y: (s) => labY[s.seed], dx: 9, textAnchor: "start", fill: C.ink, fontSize: 11.5, ...halo(),
        text: (s) => `run ${s.seed + 1}: ${pct(s.R[3])}`,
      }),
    ],
  });
  for (const [svg, w] of [[left, wl], [right, wr]]) svg.style.cssText += `;flex:0 1 ${w}px;min-width:0;width:100%;height:auto`;
  wrap.append(left, right);
  add(`font-size:13px;color:${C.muted};margin-top:0.2rem`,
    "Left: answers that flip from B to A when the probe is attached. Right: share of the probe's push toward A that is removed when 0 to 3 heads, picked on other prompts, get their no-probe values (filled dots and lines, one per training run); hollow dots: 50 random sets of 3 other heads per run. 128 new prompts.");
  return root;
}
