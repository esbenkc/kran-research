// Observable Plot render function for this figure.
// Receives the parsed data.json and the global Plot; returns a DOM node.
// Docs: https://observablehq.com/plot/
export function render(data, Plot) {
  return Plot.plot({
    marginLeft: 60,
    width: 720,
    x: { label: "Year", tickFormat: "d" },
    y: { label: "Value", grid: true },
    marks: [
      Plot.line(data, { x: "year", y: "value", stroke: "steelblue", strokeWidth: 2 }),
      Plot.dot(data, { x: "year", y: "value", fill: "steelblue", tip: true }),
    ],
  });
}
