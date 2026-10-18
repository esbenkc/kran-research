// Re-thresholding after a probe-evasion fine-tune: how much of the lost detection comes back?
// x = push p: how far the fine-tune moved the mean probe score of false statements down, in units of the
//     clean within-class standard deviation.
// y = share of the lost detection rate (TPR at 1% FPR) that a threshold re-fit on 200 clean negatives gets back.
// Big dots: final step of each fine-tune run (the pre-registered read-out), colour = retain weight lambda.
// Error bars: 95% range of that share over 200 redraws of the 200 clean negatives (from 796 held-out ones).
// Small dots: earlier checks of the same runs. Dashed line: the two-Gaussian prediction at the measured
// separation D and common-shift ratio c. Dotted rule: the pre-registered bar, half recovered.
// data = { grid, headline_lambda, D, c, z, headline: {share, ci95, per_seed, auroc_drop}, status,
//          rows: [{lam, seed, step, p, y, lost, auroc, e6, c}], finals: [{... , lo, hi}], curve: [{p, y}] }
import { C, FONT, halo, plotStyle } from "../theme.js";

const fmtLam = (l) => (l === 0 ? "0 (no retain term)" : l >= 1e5 ? "10⁵" : l >= 3e4 ? "3.16×10⁴" : "10⁴");
const pct = (v) => `${Math.round(v * 100)}%`;

export function render(data, Plot) {
  const order = [...data.grid].sort((a, b) => a - b);
  const LAMC = (l) => [C.muted, C.secondary, C.orange, C.primary][order.indexOf(l)];
  const W = 640;
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  root.setAttribute("width", String(W));

  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = "Re-fitting the threshold on clean data gets back most of the lost detection";
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  const h = data.headline;
  sub.textContent = h
    ? `Pre-registered read-out (λ = ${fmtLam(data.headline_lambda)}, 3 seeds pooled): ${pct(h.share)} recovered ` +
      `(95% CI ${pct(h.ci95[0])}–${pct(h.ci95[1])}). AUROC (how well scores rank false above true): ${h.auroc_clean.toFixed(3)} → ${h.auroc_att.toFixed(3)}.`
    : "No retain weight met the pre-registered 50-point drop in all seeds.";

  const sw = (color, kind) =>
    kind === "dot" ? `<svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="${color}"/></svg>`
      : `<svg width="24" height="12"><line x1="1" y1="6" x2="23" y2="6" stroke="${color}" stroke-width="2"` +
        ` stroke-dasharray="${kind === "dash" ? "6,3" : "2,3"}"/></svg>`;
  const item = (svg, text) => `<span style="display:inline-flex;align-items:center;gap:0.35rem">${svg}${text}</span>`;
  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-wrap:wrap;gap:0.3rem 1.0rem;font-size:14px;margin:0 0 0.4rem;color:" + C.ink;
  legend.innerHTML =
    `<span style="color:${C.muted}">retain weight λ:</span>` +
    [...data.grid].sort((a, b) => b - a).map((l) => item(sw(LAMC(l), "dot"), fmtLam(l))).join("") +
    item(sw(C.ink, "dash"), `two-Gaussian prediction (D = ${data.D}, c = ${data.c}); shaded: its 95% range with 200 negatives`) +
    item(sw(C.muted, "dot-line"), "pre-registered bar: half");
  root.append(head, sub, legend);

  const tip = (d) =>
    `λ = ${fmtLam(d.lam)}, seed ${d.seed}, step ${d.step}\npush p: ${d.p.toFixed(2)} sd\n` +
    `lost: ${(d.lost * 100).toFixed(1)} points\nrecovered: ${d.y === null ? "n/a" : pct(d.y)}` +
    (d.lo !== undefined ? ` (redraws of the 200 negatives: ${pct(d.lo)}–${pct(d.hi)})` : "") +
    `\nprediction: ${d.e6 === null ? "n/a" : pct(d.e6)}\nAUROC: ${d.auroc.toFixed(3)}`;

  const fin = data.finals.filter((d) => d.y !== null && d.p !== null);
  const pts = data.rows.filter((d) => d.y !== null && d.p !== null);
  const xmax = Math.max(...data.curve.map((d) => d.p));
  root.appendChild(Plot.plot({
    style: plotStyle({ fontSize: "13px" }), width: W, height: 400,
    marginTop: 12, marginLeft: 56, marginBottom: 46, marginRight: 16,
    x: { domain: [0, xmax], label: "Push p: drop of the mean false-statement score (clean sd units)", labelAnchor: "center", labelArrow: "none" },
    y: { domain: [0, 1.05], label: "Share of lost detection recovered", tickFormat: (v) => pct(v), labelAnchor: "center", labelArrow: "none" },
    marks: [
      Plot.gridY({ stroke: C.grid, strokeOpacity: 1 }),
      Plot.areaY(data.curve, { x: "p", y1: "lo", y2: "hi", fill: C.ink, fillOpacity: 0.08 }),
      Plot.ruleY([0.5], { stroke: C.muted, strokeWidth: 1.5, strokeDasharray: "2,3" }),
      Plot.line(data.curve, { x: "p", y: "y", stroke: C.ink, strokeWidth: 1.6, strokeDasharray: "6,3" }),
      Plot.dot(pts, { x: "p", y: "y", r: 2, fill: (d) => LAMC(d.lam), fillOpacity: 0.3, tip: true, title: tip }),
      Plot.ruleX(fin, { x: "p", y1: "lo", y2: "hi", stroke: (d) => LAMC(d.lam), strokeWidth: 1.6, strokeOpacity: 0.8 }),
      Plot.dot(fin, { x: "p", y: "y", r: 6, fill: (d) => LAMC(d.lam), stroke: C.paper, strokeWidth: 1.3, tip: true, title: tip }),
      Plot.text([{ p: xmax, y: 0.5 }], { ...halo(), x: "p", y: "y", text: () => "half recovered", fill: C.muted, fontSize: 12, textAnchor: "end", dy: -9 }),
    ],
  }));
  return root;
}
