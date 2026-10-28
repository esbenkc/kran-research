// A model trained by reward to stop flattering only while one leaky probe (the training probe) is attached.
// Each filled dot = one new monitor the model never saw in training (81: 80 leaky hooks + the rounding setting), in 2
// training runs. x = overlap score: how much the monitor's numeric footprint looks like the training probe's to a
// classifier trained only on the training probe, computed on the untrained model before any behaviour test.
// y = fooled share: flattery drop under that monitor / drop under the training probe (200 prompts), 95% intervals.
// Dashed line = the prediction fixed before the test (fooled share = overlap score). Rings: the training probe,
// the 2 monitors seen before this run, and the 89 bit-exact monitors (one ring at the origin, 0% fooled).
// Data: kran-research projects/footprint-overlap/figures/make_data.py.
import { C, FONT, halo, plotStyle } from "../theme.js";

const AX = { labelAnchor: "center", labelArrow: "none" };
const pc = (v) => `${Math.round(v * 100)}%`;

export function render(data, Plot) {
  const T = data.terms || {};
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
  const dot = (c, hollow) => `<svg width="14" height="14"><circle cx="7" cy="7" r="4.5" fill="${hollow ? "none" : c}" stroke="${c}" stroke-width="1.6"/></svg>`;
  const line = (c, da) => `<svg width="24" height="12"><line x1="1" y1="6" x2="23" y2="6" stroke="${c}" stroke-width="2"${da ? ` stroke-dasharray="${da}"` : ""}/></svg>`;
  const legend = (items) => {
    const d = el("display:flex;flex-wrap:wrap;gap:0.3rem 1rem;margin:0 0 0.3rem;font-size:13px;color:" + C.ink, "");
    d.innerHTML = items.map(([s, t]) => `<span style="display:inline-flex;align-items:center;gap:0.35rem">${s}${t}</span>`).join("");
    return d;
  };
  const P = data.pts.filter((d) => d.k === "status");
  const seeds = [...new Set(P.map((d) => d.seed))].sort();
  const cell = (s) => data.cells[`s${s}@8192`] || {};
  const runName = (s) => (T.run || "training run ") + (s + 1);
  root.append(
    el("font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink, T.title || ""),
    el("font-size:14px;margin:0 0 0.5rem;color:" + C.muted, T.subtitle || ""),
    legend([...seeds.map((s) => [dot(RUN[s]), `${runName(s)} (R² ${cell(s).r2?.toFixed(3)})`]),
      [dot(C.muted, true), T.key_ring || "ring: training probe, seen-before or bit-exact monitor"],
      [line(C.red, "5,4"), T.key_line || "prediction fixed before the test"]]));
  if (T.r2_gloss) root.append(el("font-size:12.5px;margin:0 0 0.3rem;color:" + C.muted, T.r2_gloss));

  const vals = P.flatMap((d) => [d.x, d.lo, d.hi]);
  const lo = Math.min(-0.08, ...vals) - 0.02, hi = Math.max(1.06, ...vals) + 0.02, yhi = hi + 0.07;
  const sh = P.filter((d) => d.layer != null && d.layer <= 14), xs = sh.map((d) => d.x);
  const br = { x0: Math.min(...xs), x1: Math.max(...xs), y: hi + 0.015 }; // bracket over the layer 0-14 monitors
  const off = (d) => (d.seed - (seeds.length - 1) / 2) * 0.006; // tiny nudge so the 2 runs do not hide each other
  const tip = (d) => `${d.name}\n${runName(d.seed)}\noverlap score ${pc(d.x)}\nfooled share ${pc(d.y)} (95% ${pc(d.lo)} to ${pc(d.hi)})`;
  const A = data.pts.filter((d) => d.k === "A" && d.seed === seeds[0]).map((d) => ({ ...d, y: 1 }));
  const seen = data.pts.filter((d) => d.k === "seen");
  const emu = P.filter((d) => d.b === "emu");
  const ring = (pts, r) => Plot.dot(pts, { x: (d) => d.x + (d.seed != null ? off(d) : 0), y: "y", r, stroke: C.ink, strokeWidth: 1.3, fill: "none" });
  const svg = Plot.plot({
    style: plotStyle({ fontSize: "13px" }), width: W, height: 470, marginLeft: 56, marginRight: 20, marginTop: 14, marginBottom: 44,
    x: { domain: [lo, hi], label: T.x || "overlap score, computed before any behaviour test", tickFormat: pc, ...AX },
    y: { domain: [lo, yhi], label: T.y || "fooled share of the flattery drop", tickFormat: pc, ...AX },
    marks: [
      Plot.gridY({ stroke: C.grid, strokeOpacity: 1 }),
      Plot.ruleY([0], { stroke: C.faint }),
      Plot.ruleX([0], { stroke: C.faint }),
      Plot.line([[lo, lo], [hi, hi]], { stroke: C.red, strokeWidth: 2, strokeDasharray: "5,4" }),
      Plot.link(P, { x1: (d) => d.x + off(d), x2: (d) => d.x + off(d), y1: "lo", y2: "hi", stroke: (d) => RUN[d.seed], strokeOpacity: 0.45, strokeWidth: 1.2 }),
      Plot.dot(P, { x: (d) => d.x + off(d), y: "y", r: 4, fill: (d) => RUN[d.seed], stroke: C.paper, strokeWidth: 0.7, tip: true, title: tip }),
      Plot.dot(seen, { x: (d) => d.x + off(d), y: "y", r: 4.5, stroke: C.muted, strokeWidth: 1.4, fill: "none", tip: true, title: (d) => `${T.seen || "seen before this run"}: ${tip(d)}` }),
      ring(emu, 7),
      Plot.text(emu.slice(0, 1), { ...halo(), x: "x", y: "y", text: [T.name_emu || "rounding setting (no hook)"], dy: -16, fontSize: 11.5, fill: C.ink }),
      ring(A, 7),
      Plot.text(A, { ...halo(), x: "x", y: "y", text: [T.name_A || "training probe"], dx: 8, dy: -14, textAnchor: "end", fontSize: 11.5, fill: C.ink }),
      Plot.link([br], { x1: "x0", x2: "x1", y1: "y", y2: "y", stroke: C.muted, strokeWidth: 1.2 }),
      Plot.link([br.x0, br.x1], { x1: (d) => d, x2: (d) => d, y1: br.y, y2: br.y - 0.02, stroke: C.muted, strokeWidth: 1.2 }),
      Plot.text([br], { x: (d) => (d.x0 + d.x1) / 2, y: "y", text: [`${T.shallow || "leaky monitors in layers 0-14"}`], dy: -8, fontSize: 11.5, fill: C.muted }),
      Plot.text([{ x: 0.12, y: 0.025 }], { ...halo(), x: "x", y: "y", text: [`\u2190 ${T.deep || "monitors past layer 20: both near 0"}`], textAnchor: "start", fontSize: 11.5, fill: C.muted }),
      ring([{ x: 0, y: 0 }], 9),
      Plot.text([{ x: 0, y: 0 }], { ...halo(), x: "x", y: "y", text: [`${data.n_bitexact} ${T.bitexact || "no-digit monitors: 0% fooled in both runs"}`], dx: 14, dy: 18, textAnchor: "start", fontSize: 11.5, fill: C.ink }),
    ],
  });
  svg.style.cssText += `;width:100%;height:auto;max-width:${W}px`;
  root.appendChild(svg);
  if (T.note) root.append(el("font-size:12.5px;margin:0.3rem 0 0;color:" + C.muted, T.note));
  return root;
}
