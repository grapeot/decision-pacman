import { greedyPolicy, randomPolicy, systemOnePolicy, type Policy } from "./policies.ts";
import { teacherPolicy } from "./teacher.ts";

export interface PolicyEnv {
  decisionBaseUrl: string;
  teacherBaseUrl?: string;
  teacherModel?: string;
}

/**
 * Policies by name: "random", "greedy", "teacher" and "teacher-think" (the
 * chat model from TEACHER_BASE_URL / TEACHER_MODEL), or any decision model
 * name served on the /v1/systemone endpoint.
 */
export function makePolicy(name: string, seed: number, env: PolicyEnv): Policy {
  if (name === "random") return randomPolicy(seed);
  if (name === "greedy") return greedyPolicy();
  if (name === "teacher" || name === "teacher-think") {
    if (!env.teacherBaseUrl || !env.teacherModel) {
      throw new Error("Set TEACHER_BASE_URL and TEACHER_MODEL (see .env.example) to use the teacher.");
    }
    return teacherPolicy({ baseUrl: env.teacherBaseUrl, model: env.teacherModel, think: name === "teacher-think" });
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
