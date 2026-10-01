import "./style.css";
import data from "./data.json";
import en from "./copy/en.json";
import zh from "./copy/zh.json";
import { drawChart, setupChart, type Bar } from "./chart.ts";
import { attachSource, setupPair, setupPhone } from "./clips.ts";
import { fill, lookup, pickLang, type Copy, type Lang } from "./i18n.ts";
import { PlaySection } from "./play.ts";
import { setupTimers } from "./timers.ts";

const COPY: Record<Lang, Copy> = { en, zh };
const STORE_KEY = "decision-pacman.lang";

function savedLang(): string | null {
  try {
    return localStorage.getItem(STORE_KEY);
  } catch {
    return null;
  }
}

function saveLang(lang: Lang): void {
  try {
    localStorage.setItem(STORE_KEY, lang);
  } catch {
    // Remembering the choice is a convenience.
  }
}

let lang: Lang = pickLang(location.search, savedLang(), navigator.languages?.length ? navigator.languages : [navigator.language]);

const t = (key: string, vars?: Record<string, string | number>) => fill(lookup(COPY[lang], key), vars);

/** Placeholder values for copy keys; numbers live in data.json, not in the copy. */
const VARS: Record<string, Record<string, string | number>> = {
  "hero.jev_stat": data.clips.jev,
  "hero.ours_stat": data.clips.ours,
  "halo.phi_stat": data.clips.phi4_mini,
  "hero.note": { jev_avg: data.averages.jev, ours_avg: data.averages.ours },
  "paths.b_body": data.thinking,
  "paths.b_timer_fast": { s: data.timer.fast_s },
  "paths.b_timer_slow": { s: data.timer.slow_s },
};

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
}

/** Copy with placeholders: the filled-in numbers are wrapped so they can stand out. */
function fillHtml(template: string, vars: Record<string, string | number>): string {
  return escapeHtml(template).replace(/\{(\w+)\}/g, (m, name: string) => (name in vars ? `<b class="num">${escapeHtml(String(vars[name]))}</b>` : m));
}

const listeners: (() => void)[] = [];

function applyCopy(): void {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.title = t("meta.title");
  document.querySelector('meta[name="description"]')?.setAttribute("content", t("meta.description"));
  document.querySelector('meta[property="og:title"]')?.setAttribute("content", t("meta.title"));
  document.querySelector('meta[property="og:description"]')?.setAttribute("content", t("meta.description"));
  for (const el of document.querySelectorAll<HTMLElement>("[data-i18n]")) {
    const key = el.dataset.i18n!;
    const vars = VARS[key];
    if (vars) el.innerHTML = fillHtml(lookup(COPY[lang], key), vars);
    else el.textContent = t(key);
  }
  const cta = document.getElementById("cta-button") as HTMLAnchorElement;
  cta.href = lang === "zh" ? data.links.cta_zh : data.links.cta_en;
  for (const fn of listeners) fn();
}

function setLang(next: Lang): void {
  lang = next;
  saveLang(next);
  const url = new URL(location.href);
  if (url.searchParams.has("lang")) {
    url.searchParams.set("lang", next);
    history.replaceState(null, "", url);
  }
  applyCopy();
}

// ---- static wiring ---------------------------------------------------------------
document.getElementById("lang-toggle")!.addEventListener("click", () => setLang(lang === "zh" ? "en" : "zh"));

for (const a of document.querySelectorAll<HTMLAnchorElement>("a[data-link]")) {
  a.href = data.links[a.dataset.link as keyof typeof data.links];
}

const commands = document.getElementById("commands")!;
for (const cmd of data.commands) {
  const row = document.createElement("div");
  row.className = "cmd";
  const code = document.createElement("code");
  code.textContent = cmd;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.dataset.i18n = "footer.copy";
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(cmd);
      btn.textContent = t("footer.copied");
      setTimeout(() => (btn.textContent = t("footer.copy")), 1500);
    } catch {
      // Clipboard access can be refused; the command stays selectable.
    }
  });
  row.append(code, btn);
  commands.appendChild(row);
}

// ---- clips -------------------------------------------------------------------------
const clips = data.clips as Record<string, { src: string; poster: string }>;
for (const v of document.querySelectorAll<HTMLVideoElement>("video[data-clip]")) attachSource(v, clips[v.dataset.clip!]);
for (const pair of document.querySelectorAll<HTMLElement>("[data-pair]")) setupPair(pair);

const phoneLabel = document.getElementById("phone-sound-label")!;
let phoneMuted = true;
const relabelPhone = () => {
  phoneLabel.textContent = t(phoneMuted ? "halo.unmute" : "halo.mute");
};
setupPhone(document.querySelector<HTMLVideoElement>(".phone video")!, document.getElementById("phone-sound") as HTMLButtonElement, (muted) => {
  phoneMuted = muted;
  relabelPhone();
});
listeners.push(relabelPhone);

// ---- chart and timers ----------------------------------------------------------------
const chartHost = document.getElementById("chart")!;
const redrawChart = setupChart(chartHost, () =>
  drawChart(chartHost, data.ladder as Bar[], data.band, (id) => t(`halo.chart_labels.${id}`), t("halo.chart_band")),
);
listeners.push(redrawChart);
setupTimers(document.getElementById("timers")!, data.timer.fast_s, data.timer.slow_s);
document.getElementById("jump-from")!.textContent = String(data.thinking.off);
document.getElementById("jump-to")!.textContent = String(data.thinking.on);

// ---- the game -------------------------------------------------------------------------
const play = new PlaySection({ seed: data.seed, seconds: data.game_seconds, opponents: data.ranking, t });
listeners.push(() => play.relabel());

applyCopy();

// Test hook for scripted checks (Playwright): fast-forward game time and read the state.
(window as unknown as { __site: object }).__site = {
  play,
  get lang() {
    return lang;
  },
  setLang,
};
