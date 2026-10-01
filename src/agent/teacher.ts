// A chat model used as a teacher: it sees the same encoded state as the
// student and answers with one option plus a short reason.
import type { EncodedDecision } from "../encoders/types.ts";
import { TPS } from "../engine/game.ts";
import { peek, type PeekOutcome } from "./oracle.ts";
import type { Policy, PolicyDecision } from "./policies.ts";

export interface TeacherConfig {
  /** OpenAI-compatible base URL, ending in /v1. */
  baseUrl: string;
  model: string;
  /** Let the model think before answering (slower, usually more accurate). */
  think?: boolean;
  /** Show the model one simulated future per option, this many seconds long. Needs the live game state. */
  peekSeconds?: number;
  timeoutMs?: number;
  /**
   * Answers by exact prompt. A repeated prompt is answered from here without a request; the answer is deterministic
   * at temperature 0, and lockstep play revisits identical states often.
   */
  cache?: Map<string, TeacherDecision>;
  /** Request function, replaceable in tests. */
  fetchFn?: typeof fetch;
}

export interface TeacherDecision extends PolicyDecision {
  reason: string;
  outputTokens: number;
  /** True when the answer came from `cache` instead of the server. */
  cached?: boolean;
}

export class TeacherParseError extends Error {}

export const TEACHER_ANSWER = 'Answer with only a JSON object: {"move": "<one of the options>", "reason": "<one short sentence>"}';

/** The rules, the encoded state, and the options, followed by `answer`, the line that says how to reply. */
export function teacherPrompt(
  enc: EncodedDecision,
  peeks?: { seconds: number; outcomes: Record<string, PeekOutcome> },
  answer: string = TEACHER_ANSWER,
): string {
  const state = typeof enc.state === "string" ? enc.state : JSON.stringify(enc.state);
  const options = enc.keys.map((k) => `- ${k}${enc.criteria[k] ? `: ${enc.criteria[k]}` : ""}`).join("\n");
  const lookahead = peeks
    ? `Lookahead: for each option, the game engine played it and then simple greedy moves for the next ${peeks.seconds} s. ` +
      "Normal ghost moves are exact; frightened ghosts turn at random. It shows one possible future, not the only one.\n" +
      `${JSON.stringify(peeks.outcomes)}\n\n`
    : "";
  return `${enc.instructions}\n\nState:\n${state}\n\nOptions:\n${options}\n\n${lookahead}${answer}`;
}

/** Extracts {"move", "reason"} from the reply. Throws unless the move is one of the options. */
export function parseTeacherReply(text: string, keys: string[]): { move: string; reason: string } {
  const matches = text.match(/\{[^{}]*"move"[^{}]*\}/g);
  if (!matches) throw new TeacherParseError(`no JSON answer in reply: ${text.slice(-200)}`);
  const last = matches[matches.length - 1];
  let obj: { move?: unknown; reason?: unknown };
  try {
    obj = JSON.parse(last);
  } catch {
    throw new TeacherParseError(`unparseable answer: ${last}`);
  }
  const move = String(obj.move ?? "").trim().toLowerCase();
  if (!keys.includes(move)) throw new TeacherParseError(`illegal move "${move}" (options: ${keys.join(", ")})`);
  return { move, reason: String(obj.reason ?? "") };
}

export function teacherPolicy(cfg: TeacherConfig): Policy {
  return {
    name: `teacher:${cfg.model}${cfg.think ? ":think" : ""}${cfg.peekSeconds ? `:peek${cfg.peekSeconds}s` : ""}`,
    async decide(enc, _signal, game): Promise<TeacherDecision> {
      const started = performance.now();
      let peeks: { seconds: number; outcomes: Record<string, PeekOutcome> } | undefined;
      if (cfg.peekSeconds) {
        if (!game) throw new Error("peeking needs the live game state");
        const outcomes: Record<string, PeekOutcome> = {};
        for (const key of enc.keys) outcomes[key] = peek(game, enc, key, Math.round(cfg.peekSeconds * TPS));
        peeks = { seconds: cfg.peekSeconds, outcomes };
      }
      const prompt = teacherPrompt(enc, peeks);
      const hit = cfg.cache?.get(prompt);
      if (hit) return { ...hit, latencyMs: performance.now() - started, cached: true };
      const res = await (cfg.fetchFn ?? fetch)(`${cfg.baseUrl.replace(/\/$/, "")}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: cfg.model,
          messages: [{ role: "user", content: prompt }],
          temperature: 0,
          max_tokens: cfg.think ? 4096 : 200,
          chat_template_kwargs: { enable_thinking: !!cfg.think },
        }),
        signal: AbortSignal.timeout(cfg.timeoutMs ?? 120_000),
      });
      if (!res.ok) throw new Error(`teacher HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
      const body = (await res.json()) as {
        choices: { message: { content?: string } }[];
        usage?: { prompt_tokens?: number; completion_tokens?: number };
      };
      const text = body.choices[0]?.message?.content ?? "";
      const { move, reason } = parseTeacherReply(text, enc.keys);
      const decision: TeacherDecision = {
        choice: move,
        reason,
        latencyMs: performance.now() - started,
        inputTokens: body.usage?.prompt_tokens,
        outputTokens: body.usage?.completion_tokens ?? 0,
      };
      cfg.cache?.set(prompt, decision);
      return decision;
    },
  };
}
