// Blind head audit: attach the probe on neutral prompts, rank the 32 heads of the last two blocks by how much their
// output moves (trained model minus base model), then give the top k heads their no-probe values on the trained task.
// Three panels (label-trained answer switch, label-trained detector, reward-trained switch); one line per training run:
// share of the probe's push removed when the top 1-8 audit heads are reset; hollow dots at the right: 400 random sets
// of 3 heads from the 29 heads outside the audit's top 3; hollow diamond: top 3 of the frozen second score (last
// position only); hollow triangle: the 3 heads picked with the behaviour known (greedy on other prompts). Dashed
// line: the 70% bar.
// data = { title, subtitle, legend, bar, panels:[{label, runs:[{label, model, R:[k1..k8], heads, p95, pass, R_oracle3,
//          oracle, R_secondary3}]}], nulls, note, rand:{model:[400]} }  (rand last: the figure gate reads the head)
import { C, FONT, plotStyle, halo } from "../theme.js";

export function render(data, Plot) {
  const shade = [C.primary, C.secondary, C.orange];
  const root = document.createElement("div");
  root.setAttribute("width", "732"); // natural width for the thumbnail harness (one svg per panel)
  root.style.fontFamily = FONT;
  const add = (css, txt) => {
    const d = document.createElement("div");
    d.style.cssText = css;
    d.textContent = txt;
    root.appendChild(d);
    return d;
  };
  add(`font-size:17px;font-weight:600;margin:0 0 0.15rem;color:${C.ink}`, data.title);
  add(`font-size:14px;margin:0 0 0.5rem;color:${C.muted}`, data.subtitle);
  const pct = (v) => `${Math.round(v * 100)}%`;
  const pct1 = (v) => `${(Math.round(v * 1000) / 10).toFixed(1)}%`;
  add(`font-size:12.5px;margin:0 0 0.2rem;color:${C.ink}`, data.legend);
  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;flex-wrap:nowrap;gap:0 6px;align-items:flex-start";
  root.appendChild(wrap);
  const W = 690, H = 320, n = data.panels.length;
  const RX = 9.4; // x position of the random-set column
  data.panels.forEach((P, pi) => {
    const w = Math.round((W - 6 * (n - 1)) / n) + (pi === 0 ? 42 : 0);
    const lines = P.runs.flatMap((r, ri) => r.R.map((v, k) => ({ k: k + 1, v, r, ri })));
    const rand = P.runs.flatMap((r, ri) => data.rand[r.model].map((v, j) => ({ v, r, ri, j })));
    const svg = Plot.plot({
      style: plotStyle({ fontSize: "12.5px" }),
      width: w, height: H, marginLeft: pi === 0 ? 58 : 16, marginRight: 6, marginTop: 36, marginBottom: 48,
      x: { domain: [0.5, 10.2], ticks: [1, 2, 3, 4, 5, 6, 7, 8, RX], tickSize: 0,
           tickFormat: (v) => (v > 8 ? "rand." : `${v}`), label: pi === 1 ? "Top audit heads reset (k)" : null, labelArrow: "none" },
      y: { domain: [-0.1, 1.08], ticks: [0, 0.25, 0.5, 0.75, 1], tickFormat: pi === 0 ? pct : () => "",
           label: pi === 0 ? "Probe's push removed" : null, labelAnchor: "center", labelArrow: "none" },
      marks: [
        Plot.ruleY([0], { stroke: C.faint }),
        Plot.ruleY([data.bar], { stroke: C.faint, strokeDasharray: "4,4" }),
        Plot.ruleX([3], { stroke: C.faint, strokeDasharray: "2,3" }),
        Plot.text([P.label], { frameAnchor: "top-left", dy: -28, dx: 2, fill: C.ink, fontSize: 13, fontWeight: 600, text: (d) => d }),
        Plot.dot(rand, {
          x: (d) => RX + (d.ri - (P.runs.length - 1) / 2) * 0.3 + ((d.j % 7) - 3) * 0.02, y: "v", r: 1.8, fill: "none",
          stroke: (d) => shade[d.ri], strokeOpacity: 0.55, tip: true,
          title: (d) => `${d.r.label}: 3 random other heads\npush removed ${pct(d.v)}`,
        }),
        Plot.dot(P.runs, {
          x: (r) => 3.35, y: "R_secondary3", symbol: "diamond", r: 4, fill: "none", stroke: (r) => shade[P.runs.indexOf(r)],
          strokeWidth: 1.5, tip: true,
          title: (r) => `${r.label}: top 3 of the second score (last position only)\npush removed ${pct(r.R_secondary3)}`,
        }),
        Plot.dot(P.runs, {
          x: (r) => 2.65, y: "R_oracle3", symbol: "triangle", r: 4, fill: "none", stroke: (r) => shade[P.runs.indexOf(r)],
          strokeWidth: 1.5, tip: true,
          title: (r) => `${r.label}: best 3 heads when the behaviour is known (${r.oracle.join(", ")})\npush removed ${pct(r.R_oracle3)}`,
        }),
        Plot.line(lines, { x: "k", y: "v", z: "ri", stroke: (d) => shade[d.ri], strokeWidth: 2 }),
        Plot.dot(lines, {
          x: "k", y: "v", r: (d) => (d.k === 3 ? 5 : 3.2), fill: (d) => shade[d.ri], stroke: C.paper, strokeWidth: 1, tip: true,
          title: (d) => `${d.r.label}, top ${d.k}: ${d.r.heads.slice(0, d.k).join(", ")}\npush removed ${pct(d.v)}`,
        }),
      ],
    });
    const col = document.createElement("div");
    col.style.cssText = `flex:0 1 ${w}px;min-width:0`;
    svg.style.cssText += ";width:100%;height:auto";
    col.appendChild(svg);
    const key = document.createElement("div");
    key.style.cssText = `font-size:12.5px;color:${C.ink};margin:-0.2rem 0 0 ${pi === 0 ? 58 : 16}px;line-height:1.45`;
    for (const [ri, r] of P.runs.entries()) {
      const k = document.createElement("div");
      k.innerHTML = `<span style="color:${shade[ri]};font-size:14px">\u25CF</span> ${r.label}, top 3: ${pct1(r.R[2])}${r.pass ? "" : " (fails)"}`;
      key.appendChild(k);
    }
    col.appendChild(key);
    wrap.appendChild(col);
  });
  if (data.nulls) add(`font-size:13px;color:${C.ink};margin-top:0.25rem`, data.nulls);
  add(`font-size:13px;color:${C.muted};margin-top:0.2rem`, data.note);
  return root;
}
