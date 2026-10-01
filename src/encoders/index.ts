import { asciiFullEncoder, asciiWindowEncoder } from "./ascii.ts";
import { featuresEncoder } from "./features.ts";
import { featuresPeek5sEncoder } from "./peek.ts";
import type { Encoder } from "./types.ts";

export const ENCODERS: Record<string, Encoder> = {
  [featuresEncoder.name]: featuresEncoder,
  [asciiWindowEncoder.name]: asciiWindowEncoder,
  [asciiFullEncoder.name]: asciiFullEncoder,
  [featuresPeek5sEncoder.name]: featuresPeek5sEncoder,
};

export const DEFAULT_ENCODER = featuresEncoder.name;

export * from "./types.ts";
