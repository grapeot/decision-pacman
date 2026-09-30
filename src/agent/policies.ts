import type { EncodedDecision } from "../encoders/types.ts";
import { askSystemOne, type SystemOneConfig } from "./client.ts";

export interface PolicyDecision {
  choice: string;
  probabilities?: Record<string, number>;
  confidence?: number;
  inputTokens?: number;
  latencyMs: number;
}

export interface Policy {
  name: string;
  /** Encoder this policy expects, when it reads the encoded state itself. */
  requiresEncoder?: string;
  decide(enc: EncodedDecision, signal?: AbortSignal): Promise<PolicyDecision>;
}

export function systemOnePolicy(cfg: SystemOneConfig): Policy {
  return {
    name: cfg.model,
    async decide(enc, signal) {
      const a = await askSystemOne(cfg, enc, signal);
      return a;
    },
  };
}

/** Uniform over the legal options, with its own seeded generator. */
export function randomPolicy(seed = 1): Policy {
  let t = seed | 0;
  const rand = () => {
    t = (t + 0x6d2b79f5) | 0;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
  return {
    name: "random",
    async decide(enc) {
      return { choice: enc.keys[Math.floor(rand() * enc.keys.length)], latencyMs: 0 };
    },
  };
}

interface FactsLike {
  ghost: number | null;
  edible?: number | null;
  pellets: number;
  food: number | null;
}

/**
 * Scripted baseline over the features encoder: chase edible ghosts nearby,
 * flee when a normal ghost is close, otherwise head for the nearest pellet.
 */
export function greedyPolicy(dangerSteps = 4): Policy {
  return {
    name: "greedy",
    requiresEncoder: "features",
    async decide(enc) {
      const options = (enc.state as { options: Record<string, FactsLike> }).options;
      const keys = enc.keys.filter((k) => options[k]);
      const far = (v: number | null | undefined) => (v === null || v === undefined ? 99 : v);
      const edible = keys.filter((k) => far(options[k].edible) <= 8 && far(options[k].ghost) > dangerSteps);
      if (edible.length > 0) {
        edible.sort((a, b) => far(options[a].edible) - far(options[b].edible));
        return { choice: edible[0], latencyMs: 0 };
      }
      const safe = keys.filter((k) => far(options[k].ghost) > dangerSteps);
      if (safe.length === 0) {
        const byGhost = [...keys].sort((a, b) => far(options[b].ghost) - far(options[a].ghost));
        return { choice: byGhost[0], latencyMs: 0 };
      }
      safe.sort((a, b) => far(options[a].food) - far(options[b].food) || options[b].pellets - options[a].pellets);
      return { choice: safe[0], latencyMs: 0 };
    },
  };
}
