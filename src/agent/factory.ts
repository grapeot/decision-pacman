import { greedyPolicy, randomPolicy, systemOnePolicy, type Policy } from "./policies.ts";
import { oraclePolicy } from "./oracle.ts";
import { teacherPolicy } from "./teacher.ts";

export interface PolicyEnv {
  decisionBaseUrl: string;
  teacherBaseUrl?: string;
  teacherModel?: string;
}

/**
 * Policies by name: "random", "greedy", "jev" (TypeSafe's hosted Jev), "oracle" (engine rollouts; "oracle-8s-x2"
 * sets an 8 s horizon and 2 samples), "teacher", "teacher-think", and "-peek5s" variants that also show a 5 s simulated future per option (the
 * chat model from TEACHER_BASE_URL / TEACHER_MODEL), or any decision model
 * name served on the /v1/systemone endpoint.
 */
export function makePolicy(name: string, seed: number, env: PolicyEnv): Policy {
  if (name === "random") return randomPolicy(seed);
  if (name === "greedy") return greedyPolicy();
  if (name === "jev") {
    // TypeSafe's hosted API speaks the same /v1/systemone protocol as Ollama.
    if (!process.env.TYPESAFE_API_KEY) throw new Error("Set TYPESAFE_API_KEY to call Jev.");
    return systemOnePolicy({
      baseUrl: process.env.JEV_BASE_URL ?? "https://api.typesafe.ai",
      model: process.env.JEV_MODEL ?? "jev-latest",
      apiKey: process.env.TYPESAFE_API_KEY,
    });
  }
  const oracle = /^oracle(?:-(\d+(?:\.\d+)?)s)?(?:-x(\d+))?$/.exec(name);
  if (oracle) return oraclePolicy({ horizonSeconds: oracle[1] ? Number(oracle[1]) : 5, samples: oracle[2] ? Number(oracle[2]) : 1 });
  const teacher = /^teacher(-think)?(?:-peek(\d+(?:\.\d+)?)s)?$/.exec(name);
  if (teacher) {
    if (!env.teacherBaseUrl || !env.teacherModel) {
      throw new Error("Set TEACHER_BASE_URL and TEACHER_MODEL (see .env.example) to use the teacher.");
    }
    return teacherPolicy({
      baseUrl: env.teacherBaseUrl,
      model: env.teacherModel,
      think: !!teacher[1],
      peekSeconds: teacher[2] ? Number(teacher[2]) : undefined,
    });
  }
  return systemOnePolicy({ baseUrl: env.decisionBaseUrl, model: name });
}

/** Reads a local .env if present, without overriding variables already set. */
export function loadDotEnv(): void {
  try {
    (process as unknown as { loadEnvFile?: (p?: string) => void }).loadEnvFile?.(".env");
  } catch {
    // no .env
  }
}

export function policyEnvFromProcess(): PolicyEnv {
  return {
    decisionBaseUrl: process.env.VITE_DECISION_BASE_URL ?? "http://localhost:11434",
    teacherBaseUrl: process.env.TEACHER_BASE_URL,
    teacherModel: process.env.TEACHER_MODEL,
  };
}
