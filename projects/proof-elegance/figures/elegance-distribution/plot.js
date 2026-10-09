// Elegance of AI-written vs human-written proofs of open problems (1-10, mean of two blind judges).
// One dot per proof, stacked (beeswarm) per row; a thick tick marks each row's mean. The top row
// holds the textbook calibration anchors (6 famous elegant proofs, 6 famous brute-force proofs).
// Data: {items: [{source, elegance, field, problem, key_idea}], anchors: [{kind, elegance, title}],
//        summary: {ai: {mean, median}, human: {mean, median}}}.
// Text and gridlines use currentColor, driven by the site's --color-tx-normal, so the figure reads
// in both the light and the dark theme (theme.js hard-codes light-mode ink).
import { C, plotStyle } from "../theme.js";

export function render(data, Plot) {
  // Render at the reader's width up to 720px, so text stays readable on phones instead of
  // shrinking a 720px figure down. Narrow layouts get smaller dots and a taller frame.
  const W = Math.round(Math.min(720, Math.max(300, (globalThis.innerWidth || 720) - 40)));
  const narrow = W < 560;
  const R = narrow ? 4 : 6;
  const ROW = { anchor: "Textbook anchors", human: "Human, 2018–23", ai: "AI, 2026" };
  const rows = [
    ...data.anchors.map((d) => ({ ...d, row: ROW.anchor, label: d.title,
      color: d.kind === "elegant" ? C.secondary : C.faint })),
    ...data.items.map((d) => ({ ...d, row: ROW[d.source], label: d.problem,
      color: d.source === "ai" ? C.primary : C.secondary })),
  ];
  const means = ["human", "ai"].map((s) => ({ row: ROW[s], x: data.summary[s].mean,
    color: s === "ai" ? C.primary : C.secondary }));

  return Plot.plot({
    width: W, height: narrow ? 460 : 420, marginLeft: 8, marginRight: 16, marginTop: 8, marginBottom: narrow ? 52 : 44,
    style: plotStyle({ color: "var(--color-tx-normal, #100f0f)", ...(narrow ? { fontSize: "13px" } : {}) }),
    x: { label: narrow ? "Elegance (1 = brute force, 10 = book)" : "Elegance (1 = brute force, 10 = book proof)",
      domain: [1, 10], ticks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labelAnchor: narrow ? "center" : undefined },
    // axis: null — the row names are drawn inside each facet by the Plot.text mark below
    fy: { axis: null, domain: [ROW.anchor, ROW.human, ROW.ai], padding: 0.18 },
    marks: [
      Plot.gridX({ stroke: "currentColor", strokeOpacity: 0.12 }),
      Plot.text([ROW.anchor, ROW.human, ROW.ai], { fy: (d) => d, frameAnchor: "top-left", text: (d) => d,
        dy: 2, fill: "currentColor", fontWeight: 600, fontSize: 13 }),
      Plot.dot(rows, Plot.dodgeY("middle", { x: "elegance", fy: "row", r: R, padding: narrow ? 1 : 1.5, fill: "color",
        href: "url", target: "_blank",
        // Plot anchors a tip beside its dot only if it fits between the dot and the frame edge, so
        // the tip must stay narrower than half the figure. Phones get a short tip (name + score).
        channels: narrow
          ? { proof: "label", elegance: "elegance", "": () => "tap to open the paper" }
          : { proof: "label", paper: "title", field: "field", "key idea": "key_idea", elegance: "elegance",
              "": () => "click to open the paper" },
        tip: { fill: "var(--color-bg-primary, #fff)", fontSize: narrow ? 11 : 12, lineWidth: narrow ? 13 : 26,
          textOverflow: "ellipsis-end",
          format: { x: false, fy: false, fill: false, href: false, elegance: ".2~f" } } })),
      Plot.tickX(means, { x: "x", fy: "row", stroke: "color", strokeWidth: 3, insetTop: narrow ? 26 : 22, insetBottom: narrow ? 26 : 22 }),
      Plot.text(means, { x: "x", fy: "row", text: (d) => `mean ${d.x.toFixed(1)}`, frameAnchor: "bottom",
        dy: -2, fill: "color", fontSize: 11 }),
      Plot.text([{ x: narrow ? 2.4 : 2.1, t: narrow ? "brute-force classics" : "four colour, Kepler, SAT proofs" }, { x: 9.0, t: "book proofs" }], {
        x: "x", fy: () => ROW.anchor, text: "t", frameAnchor: "bottom", dy: -4, fill: "currentColor", fillOpacity: 0.6, fontSize: 11 }),
    ],
  });
}
