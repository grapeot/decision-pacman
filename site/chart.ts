// Horizontal bar chart of mean pellets, drawn as inline SVG at the container's pixel width
// so labels keep their size on phones.

export interface Bar {
  id: string;
  value: number;
  highlight?: "neutral" | "strong";
}

const NS = "http://www.w3.org/2000/svg";

function el<K extends keyof SVGElementTagNameMap>(name: K, attrs: Record<string, string | number>, text?: string): SVGElementTagNameMap[K] {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  if (text !== undefined) node.textContent = text;
  return node;
}

export function drawChart(
  host: HTMLElement,
  bars: readonly Bar[],
  band: { from: number; to: number },
  label: (id: string) => string,
  bandLabel: string,
): void {
  const width = Math.max(280, host.clientWidth);
  const narrow = width < 520;
  const labelW = narrow ? 104 : 150;
  const valueW = 44;
  const rowH = narrow ? 30 : 34;
  const barH = narrow ? 16 : 18;
  const top = 26;
  const height = top + bars.length * rowH + 22;
  const max = Math.ceil(Math.max(...bars.map((b) => b.value)) / 100) * 100;
  const plotW = width - labelW - valueW;
  const x = (v: number) => labelW + (plotW * v) / max;

  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img" });
  svg.appendChild(el("rect", { class: "band", x: x(band.from), y: top - 6, width: x(band.to) - x(band.from), height: bars.length * rowH + 6, rx: 6 }));
  svg.appendChild(el("text", { class: "band-label", x: (x(band.from) + x(band.to)) / 2, y: top - 12, "text-anchor": "middle" }, bandLabel));
  for (let v = 0; v <= max; v += 100) {
    svg.appendChild(el("line", { class: "grid", x1: x(v), x2: x(v), y1: top + bars.length * rowH, y2: top + bars.length * rowH + 4 }));
    svg.appendChild(el("text", { class: "tick", x: x(v), y: top + bars.length * rowH + 18, "text-anchor": "middle" }, String(v)));
  }
  bars.forEach((b, i) => {
    const cy = top + i * rowH + rowH / 2;
    const cls = b.highlight ?? "";
    svg.appendChild(el("text", { class: `lbl ${cls}`, x: labelW - 10, y: cy, "text-anchor": "end", "dominant-baseline": "central" }, label(b.id)));
    const rect = el("rect", { class: `bar ${cls}`, x: labelW, y: cy - barH / 2, width: Math.max(2, x(b.value) - labelW), height: barH, rx: barH / 2 });
    rect.style.transitionDelay = `${i * 60}ms`;
    svg.appendChild(rect);
    svg.appendChild(el("text", { class: `val ${cls}`, x: x(b.value) + 8, y: cy, "dominant-baseline": "central" }, String(b.value)));
  });
  host.replaceChildren(svg);
  svg.setAttribute("aria-label", bars.map((b) => `${label(b.id)} ${b.value}`).join(", "));
}

/** Redraws on resize and on language change; grows the bars the first time the chart is seen. */
export function setupChart(host: HTMLElement, draw: () => void): () => void {
  let lastW = 0;
  new ResizeObserver(() => {
    if (host.clientWidth !== lastW) {
      lastW = host.clientWidth;
      draw();
    }
  }).observe(host);
  new IntersectionObserver(
    (entries, obs) => {
      if (entries.some((e) => e.isIntersecting)) {
        requestAnimationFrame(() => host.classList.add("shown"));
        obs.disconnect();
      }
    },
    { threshold: 0.3 },
  ).observe(host);
  return draw;
}
