// Decision policy for the iOS app: the page renders the exact prompt and the
// native side scores it with the on-device model (see ios/). The bridge is a
// WKScriptMessageHandlerWithReply named "decide".
import type { EncodedDecision } from "../encoders/types.ts";
import { DecisionApiError } from "./client.ts";
import type { Policy, PolicyDecision } from "./policies.ts";
import { renderDecision } from "./prompt.ts";

interface NativeReply {
  probabilities?: number[];
  tokens?: number;
  ms?: number;
  error?: string;
}

interface NativeBridge {
  postMessage(message: unknown): Promise<NativeReply>;
}

export function nativeBridge(): NativeBridge | null {
  const w = window as unknown as { webkit?: { messageHandlers?: { decide?: NativeBridge } } };
  return w.webkit?.messageHandlers?.decide ?? null;
}

export function nativePolicy(bridge: NativeBridge, modelName: string): Policy {
  return {
    name: modelName,
    async decide(enc: EncodedDecision): Promise<PolicyDecision> {
      const started = performance.now();
      const reply = await bridge.postMessage({ prompt: renderDecision(enc), options: enc.keys.length });
      if (reply.error || !reply.probabilities) {
        throw new DecisionApiError("model-missing", reply.error ?? "no reply from the on-device model",
          "Copy the model file into the app's Documents folder as model.gguf, then reopen the app.");
      }
      const p = reply.probabilities;
      const probabilities: Record<string, number> = {};
      enc.keys.forEach((k, i) => (probabilities[k] = p[i]));
      let best = 0;
      p.forEach((v, i) => (best = v > p[best] ? i : best));
      // Same confidence as Ollama's System One: 1 - entropy / ln(options).
      const entropy = -p.reduce((s, v) => (v > 0 ? s + v * Math.log(v) : s), 0);
      const confidence = Math.max(0, Math.min(1, 1 - entropy / Math.log(p.length)));
      return { choice: enc.keys[best], probabilities, confidence, inputTokens: reply.tokens, latencyMs: reply.ms ?? performance.now() - started };
    },
  };
}
