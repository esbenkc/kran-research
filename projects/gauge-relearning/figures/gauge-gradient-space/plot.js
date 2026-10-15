// Appendix Q3: the facts in gradient space, and how the free rescaling c moves them.
// Left (slider): one dot per fact. x = cosine between its gradient and the mean gradient of the 50
// forgotten facts the attacker trains on (the direction of one SGD step on them); y = cosine with the
// part of the 50 held-out forgotten facts' mean gradient that is orthogonal to x's direction.
// Right: how much one SGD step on the trained facts points along the held-out facts' gradient, per c,
// next to natural gradient (does not depend on c) and the key weights' share of the step.
// Bottom: measured, held-out forgotten facts right when the trained-on facts reach the threshold.
// data = { title, log2c, facts: [{split, loss, name, answer}], per_c: [{log2c, cos_meanA_meanB,
//          cos_meanA_meanRetain, share_meanA_K, share_meanA_Q, xy}], natural_gradient: {cos_meanA_meanB, xy},
//          heldout: [{attacker, label, log2c, accB_mean, accB_min, accB_max, accB_n}], thresh }
import { C, FONT, plotStyle } from "../theme.js";

const GROUP = {
  A: { label: "forgotten, trained on", color: C.primary },
  B: { label: "forgotten, held out", color: C.secondary },
  retain: { label: "kept", color: C.neutral },
};
const cLabel = (k) => {
  const v = 2 ** Math.abs(k);
  const s = Number.isInteger(v) ? String(v) : v.toFixed(1);
  return k >= 0 ? s : `1/${s}`;
};

export function render(data, Plot) {
  const root = document.createElement("div");
  root.style.fontFamily = FONT;
  const head = document.createElement("div");
  head.style.cssText = "font-size:17px;font-weight:600;margin:0 0 0.15rem;color:" + C.ink;
  head.textContent = data.title;
  const sub = document.createElement("div");
  sub.style.cssText = "font-size:14px;margin:0 0 0.5rem;color:" + C.muted;
  sub.textContent = "Exact answer-loss gradients of the unlearned model. Dots: the 200 facts of seed 0 (diamonds: group averages). Lines: seeds 0-4, seed 0 thick. Drag the slider.";
  root.append(head, sub);

  const ks = data.per_c.map((r) => r.log2c);
  const ctrl = document.createElement("div");
  ctrl.style.cssText = "display:flex;align-items:center;gap:0.8rem;font-size:15px;margin:0 0 0.5rem;color:" + C.ink;
  const slider = document.createElement("input");
  Object.assign(slider, { type: "range", min: 0, max: ks.length - 1, step: 1, value: ks.indexOf(0) });
  slider.style.cssText = "flex:1;max-width:420px;accent-color:" + C.primary;
  const val = document.createElement("span");
  val.style.cssText = "min-width:7.5rem;font-variant-numeric:tabular-nums";
  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-wrap:wrap;gap:0.3rem 1rem;font-size:14px;margin:0 0 0.3rem;color:" + C.ink;
  legend.innerHTML = Object.values(GROUP).map((g) =>
    `<span style="display:inline-flex;align-items:center;gap:0.35rem"><svg width="10" height="10"><circle cx="5" cy="5" r="4.5" fill="${g.color}"/></svg>${g.label}</span>`,
  ).join("");
  ctrl.append("Rescaling c", slider, val);
  root.append(ctrl, legend);

  const row = document.createElement("div");
  row.style.cssText = "display:flex;flex-wrap:wrap;gap:0.5rem 1.5rem;align-items:flex-start";
  const left = document.createElement("div");
  const right = document.createElement("div");
  row.append(left, right);
  root.appendChild(row);
  const bottom = document.createElement("div");
  root.appendChild(bottom);

  const full = Math.max(320, Math.min(860, root.clientWidth || 760));
  const half = full >= 640 ? Math.floor((full - 24) / 2) : full;
  const all = data.per_c.flatMap((r) => r.xy);
  const ext = (i) => [Math.min(-0.1, ...all.map((p) => p[i])) - 0.02, Math.max(...all.map((p) => p[i])) + 0.02];
  const [xd, yd] = [ext(0), ext(1)];
  // one line per seed (seed 0 also draws the dots on the left)
  const lines = data.seed_lines.flatMap((r) => [
    { k: r.log2c, v: r.cos, s: "SGD step on trained facts vs held-out facts", seed: r.seed },
    { k: r.log2c, v: r.ng, s: "same, natural gradient", seed: r.seed },
    { k: r.log2c, v: r.share_K, s: "key weights' share of the SGD step", seed: r.seed },
  ]);
  const seeds = [...new Set(lines.map((l) => l.seed))];
  const yLo = Math.min(0, ...lines.map((l) => l.v)) - 0.03;
  const LS = {
    "SGD step on trained facts vs held-out facts": { stroke: C.primary, dash: null },
    "same, natural gradient": { stroke: C.orange, dash: "6,4" },
    "key weights' share of the SGD step": { stroke: C.muted, dash: "2,3" },
  };
  const xTicks = [-12, -8, -4, 0, 4, 8, 12].filter((t) => t >= ks[0] && t <= ks[ks.length - 1]);

  const draw = () => {
    const i = Number(slider.value);
    const r = data.per_c[i];
    val.textContent = `c = ${cLabel(r.log2c)}`;
    const pts = r.xy.map(([x, y], j) => ({ x, y, ...data.facts[j] }));
    left.replaceChildren(Plot.plot({
      style: plotStyle({ fontSize: "14px" }), width: half, height: Math.min(half, 360),
      marginLeft: 48, marginBottom: 44, marginTop: 10,
      x: { domain: xd, label: "Lined up with a step on the trained facts →", labelAnchor: "center", labelArrow: "none" },
      y: { domain: yd, label: "↑ Lined up with held-out facts only", labelArrow: "none" },
      marks: [
        Plot.gridY({ stroke: C.grid, strokeOpacity: 1 }),
        Plot.ruleX([0], { stroke: C.faint }), Plot.ruleY([0], { stroke: C.faint }),
        Plot.dot(pts.filter((p) => p.split === "retain"), { x: "x", y: "y", r: 2.6, fill: GROUP.retain.color }),
        Plot.dot(pts.filter((p) => p.split !== "retain"), {
          x: "x", y: "y", r: 3.4, fill: (d) => GROUP[d.split].color, fillOpacity: 0.9,
        }),
        // group averages
        Plot.dot(Object.keys(GROUP).map((g) => {
          const ps = pts.filter((p) => p.split === g);
          return { split: g, x: ps.reduce((s, p) => s + p.x, 0) / ps.length, y: ps.reduce((s, p) => s + p.y, 0) / ps.length };
        }), { x: "x", y: "y", r: 8, fill: (d) => GROUP[d.split].color, stroke: C.ink, strokeWidth: 2, symbol: "diamond" }),
        Plot.tip(pts, Plot.pointer({
          x: "x", y: "y",
          title: (d) => `${d.name} → ${d.answer}\n${GROUP[d.split].label}, loss ${d.loss}\nx ${d.x.toFixed(2)}, y ${d.y.toFixed(2)}`,
        })),
      ],
    }));
    right.replaceChildren(Plot.plot({
      style: plotStyle({ fontSize: "14px" }), width: half, height: Math.min(half, 360),
      marginLeft: 48, marginBottom: 44, marginTop: 10,
      x: { domain: [ks[0], ks[ks.length - 1]], ticks: xTicks, tickFormat: cLabel, label: "Rescaling c", labelAnchor: "center", labelArrow: "none" },
      y: { domain: [yLo, 1], label: null, ticks: [0, 0.25, 0.5, 0.75, 1] },
      marks: [
        Plot.gridY([0, 0.25, 0.5, 0.75, 1], { stroke: C.grid, strokeOpacity: 1 }),
        Plot.ruleY([0], { stroke: C.faint }),
        ...Object.entries(LS).flatMap(([s, st]) => seeds.map((sd) => Plot.line(lines.filter((l) => l.s === s && l.seed === sd), {
          x: "k", y: "v", stroke: st.stroke, strokeWidth: sd === 0 ? 2.4 : 1.2, strokeOpacity: sd === 0 ? 1 : 0.6,
          strokeDasharray: st.dash ?? undefined,
        }))),
        Plot.ruleX([r.log2c], { stroke: C.ink, strokeOpacity: 0.5 }),
        Plot.tip(lines, Plot.pointerX({ x: "k", y: "v", title: (d) => `seed ${d.seed}, c = ${cLabel(d.k)}\n${d.s}: ${d.v.toFixed(2)}` })),
      ],
    }));
    const cap = document.createElement("div");
    cap.style.cssText = "font-size:13px;color:" + C.muted + ";max-width:" + half + "px;margin-top:0.2rem";
    cap.innerHTML = Object.entries(LS).map(([s, st]) =>
      `<span style="display:inline-flex;align-items:center;gap:0.35rem;margin-right:0.8rem"><svg width="22" height="8"><line x1="0" y1="4" x2="22" y2="4" stroke="${st.stroke}" stroke-width="2.5" ${st.dash ? `stroke-dasharray="${st.dash}"` : ""}/></svg>${s}</span>`,
    ).join("");
    right.appendChild(cap);
  };
  slider.addEventListener("input", draw);
  draw();

  // measured: held-out forgotten facts answered when the trained-on facts reach the threshold
  const H = data.heldout.map((d) => ({ ...d }));
  if (H.length) {
    const t = document.createElement("div");
    t.style.cssText = "font-size:15px;font-weight:600;margin:1rem 0 0.2rem;color:" + C.ink;
    t.textContent = `Measured: held-out forgotten facts the model answers again when the attacker reaches ${Math.round(data.thresh * 100)}% on the trained ones`;
    const lg = document.createElement("div");
    lg.style.cssText = "display:flex;gap:1.2rem;font-size:14px;margin:0 0 0.2rem;color:" + C.ink;
    lg.innerHTML = [["SGD", C.primary], ["AdamW", C.secondary]].map(([l, col]) =>
      `<span style="display:inline-flex;align-items:center;gap:0.4rem"><svg width="26" height="10"><line x1="0" y1="5" x2="26" y2="5" stroke="${col}" stroke-width="3"/></svg>${l}</span>`).join("");
    bottom.append(t, lg);
    const COL = { sgd: C.primary, adamw: C.secondary };
    const hx = [Math.min(...H.map((d) => d.log2c)) - 0.6, Math.max(...H.map((d) => d.log2c)) + 0.6];
    bottom.appendChild(Plot.plot({
      style: plotStyle({ fontSize: "14px" }), width: full, height: 230, marginLeft: 48, marginBottom: 44, marginTop: 14,
      x: { domain: hx, ticks: xTicks.filter((v) => v >= hx[0] && v <= hx[1]), tickFormat: cLabel, label: "Rescaling c", labelAnchor: "center", labelArrow: "none" },
      y: { domain: [0, Math.max(0.3, ...H.map((d) => d.accB_max))], tickFormat: (v) => `${Math.round(v * 100)}%`, label: null },
      marks: [
        Plot.gridY({ stroke: C.grid, strokeOpacity: 1 }),
        ...["sgd", "adamw"].flatMap((a) => {
          const rs = H.filter((d) => d.attacker === a);
          return [
            Plot.areaY(rs.filter((d) => d.accB_n > 1), { x: "log2c", y1: "accB_min", y2: "accB_max", fill: COL[a], fillOpacity: 0.12 }),
            Plot.line(rs, { x: "log2c", y: "accB_mean", stroke: COL[a], strokeWidth: 2.4 }),
            Plot.dot(rs, {
              x: "log2c", y: "accB_mean", r: 4, fill: COL[a], tip: true,
              title: (d) => `${d.label}, c = ${cLabel(d.log2c)}\nheld-out facts right: ${Math.round(d.accB_mean * 100)}% (mean of ${d.accB_n} runs, range ${Math.round(d.accB_min * 100)}–${Math.round(d.accB_max * 100)}%)`,
            }),
          ];
        }),
      ],
    }));
  }
  return root;
}
