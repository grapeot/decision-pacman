import type { EncodedDecision } from "../encoders/types.ts";

export type ApiErrorKind = "network" | "model-missing" | "bad-request" | "server";

export class DecisionApiError extends Error {
  constructor(
    public kind: ApiErrorKind,
    message: string,
    public hint: string,
  ) {
    super(message);
  }
}

export interface SystemOneAnswer {
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
  inputTokens: number;
  latencyMs: number;
}

export interface SystemOneConfig {
  /** Base URL without the /v1 suffix, for example http://localhost:11434 or /decide. */
  baseUrl: string;
  model: string;
  keepAlive?: string | number;
}

export function buildRequest(cfg: SystemOneConfig, enc: EncodedDecision): object {
  return {
    model: cfg.model,
    keep_alive: cfg.keepAlive ?? -1,
    state: enc.state,
    questions: {
      move: { type: "choice", instructions: enc.instructions, criteria: enc.criteria },
    },
  };
}

export async function askSystemOne(cfg: SystemOneConfig, enc: EncodedDecision, signal?: AbortSignal): Promise<SystemOneAnswer> {
  const url = `${cfg.baseUrl.replace(/\/$/, "")}/v1/systemone`;
  const started = performance.now();
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildRequest(cfg, enc)),
      signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new DecisionApiError(
      "network",
      `Cannot reach ${url}`,
      "Start the decision server (for Ollama: `ollama serve`). If the page is not served from localhost, " +
        "add its origin to OLLAMA_ORIGINS, or use the dev server, which proxies /decide.",
    );
  }
  const text = await res.text();
  const latencyMs = performance.now() - started;
  if (!res.ok) {
    let message = text;
    try {
      message = (JSON.parse(text) as { error?: string }).error ?? text;
    } catch {
      // keep raw text
    }
    if (res.status === 404 || /not found|pull/i.test(message)) {
      throw new DecisionApiError("model-missing", message, `Pull the model first: \`ollama pull ${cfg.model}\`.`);
    }
    if (res.status === 400 || res.status === 413 || res.status === 422) {
      throw new DecisionApiError("bad-request", message, "The request was rejected. Check the encoder output and its size.");
    }
    throw new DecisionApiError("server", `HTTP ${res.status}: ${message}`, "The decision server returned an error.");
  }
  const body = JSON.parse(text) as {
    answers: { move: { choice: string; probabilities: Record<string, number>; confidence: number } };
    usage?: { input_tokens?: number };
  };
  const move = body.answers.move;
  return {
    choice: move.choice,
    probabilities: move.probabilities,
    confidence: move.confidence,
    inputTokens: body.usage?.input_tokens ?? 0,
    latencyMs,
  };
}
