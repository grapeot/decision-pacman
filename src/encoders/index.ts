import { asciiWindowEncoder } from "./ascii.ts";
import { featuresEncoder } from "./features.ts";
import type { Encoder } from "./types.ts";

export const ENCODERS: Record<string, Encoder> = {
  [featuresEncoder.name]: featuresEncoder,
  [asciiWindowEncoder.name]: asciiWindowEncoder,
};

export const DEFAULT_ENCODER = featuresEncoder.name;

export * from "./types.ts";
