// Synced clip pairs: one replay control restarts both; they load and play only while visible.

export interface ClipSource {
  src: string;
  poster: string;
}

function quietPlay(v: HTMLVideoElement): void {
  // Autoplay can be refused (power saving, codec support); the poster stays up then.
  void v.play().catch(() => {});
}

/** Assigns posters now and sources on first sight, so the clips do not load with the page. */
export function attachSource(v: HTMLVideoElement, clip: ClipSource): void {
  v.poster = clip.poster;
  v.dataset.src = clip.src;
}

function ensureLoaded(v: HTMLVideoElement): void {
  if (v.dataset.src && !v.getAttribute("src")) {
    v.src = v.dataset.src;
    v.preload = "auto";
  }
}

export function setupPair(root: HTMLElement): void {
  const videos = [...root.querySelectorAll("video")] as HTMLVideoElement[];
  const [lead, follow] = videos;
  const fill = root.querySelector<HTMLElement>(".progress-fill");
  const replay = root.querySelector<HTMLButtonElement>(".replay");
  let visible = false;
  let ended = false;

  for (const v of videos) {
    v.muted = true;
    v.playsInline = true;
  }

  const playBoth = () => {
    for (const v of videos) quietPlay(v);
  };
  const restart = () => {
    ended = false;
    for (const v of videos) {
      ensureLoaded(v);
      v.currentTime = 0;
    }
    playBoth();
  };

  lead.addEventListener("timeupdate", () => {
    if (fill && lead.duration) fill.style.width = `${Math.min(100, (100 * lead.currentTime) / lead.duration)}%`;
    // Keep the pair in step; small drift is invisible, larger drift gets corrected.
    if (follow && !follow.ended && Math.abs(follow.currentTime - lead.currentTime) > 0.2) follow.currentTime = lead.currentTime;
  });
  for (const v of videos) v.addEventListener("ended", () => (ended = true));
  replay?.addEventListener("click", restart);

  new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        visible = e.isIntersecting;
        if (visible) {
          for (const v of videos) ensureLoaded(v);
          if (!ended) playBoth();
        } else {
          for (const v of videos) v.pause();
        }
      }
    },
    { threshold: 0.35 },
  ).observe(root);
}

/** The phone clip: loops while visible, muted until the viewer turns sound on. */
export function setupPhone(video: HTMLVideoElement, button: HTMLButtonElement, onChange: (muted: boolean) => void): void {
  video.muted = true;
  video.playsInline = true;
  button.addEventListener("click", () => {
    ensureLoaded(video);
    video.muted = !video.muted;
    button.classList.toggle("on", !video.muted);
    quietPlay(video);
    onChange(video.muted);
  });
  new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          ensureLoaded(video);
          quietPlay(video);
        } else {
          video.pause();
        }
      }
    },
    { threshold: 0.35 },
  ).observe(video);
}
