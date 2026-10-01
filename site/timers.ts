// Path B: two lanes on the same 0-to-slow-seconds axis. The fast lane fills with one segment per
// decision; the slow lane needs the whole span for one.

export function setupTimers(root: HTMLElement, fastS: number, slowS: number): void {
  const fast = root.querySelector<HTMLElement>("#timer-fast")!;
  const slow = root.querySelector<HTMLElement>("#timer-slow")!;
  const fastCount = root.querySelector<HTMLElement>("#timer-fast-count")!;
  const slowCount = root.querySelector<HTMLElement>("#timer-slow-count")!;
  const track = fast.parentElement!;
  const pauseS = 1.2;
  let visible = false;
  let start = 0;
  let raf = 0;

  const setSegment = () => fast.style.setProperty("--seg", `${(track.clientWidth * fastS) / slowS}px`);
  new ResizeObserver(setSegment).observe(track);

  const frame = (now: number) => {
    const t = ((now - start) / 1000) % (slowS + pauseS);
    const shown = Math.min(t, slowS);
    const pct = (100 * shown) / slowS;
    const decisions = Math.floor(shown / fastS + 1e-9);
    fast.style.width = `${(100 * decisions * fastS) / slowS}%`;
    slow.style.width = `${pct}%`;
    fastCount.textContent = String(decisions);
    slowCount.textContent = shown >= slowS ? "1" : "0";
    if (visible) raf = requestAnimationFrame(frame);
  };

  new IntersectionObserver(
    (entries) => {
      const now = entries.some((e) => e.isIntersecting);
      if (now && !visible) {
        visible = true;
        start = performance.now();
        setSegment();
        raf = requestAnimationFrame(frame);
      } else if (!now) {
        visible = false;
        cancelAnimationFrame(raf);
      }
    },
    { threshold: 0.3 },
  ).observe(root);
}
