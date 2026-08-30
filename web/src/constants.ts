export const LEDA_WEB_MAGIC = "LEDAWEB\0";
export const LEDA_WEB_VERSION = 1;
export const LEDA_WEB_HEADER_BYTES = 192;
export const LEDA_WEB_DTYPE_FLOAT32 = 1;

export const FROZEN_CHECKPOINT_SHA256 =
  "a94e2a15977c811b2bc8a7bb1e87be45510e3d350f08b21ef599644ba3fd1bed";
export const FROZEN_TOKENIZER_SHA256 =
  "04345687338f13b1b8e8d219dae265094a708bf8d1b409541748cb2d89424970";
export const FROZEN_F32_ARTIFACT_SHA256 =
  "2afdc9c772af3f6f6cceb71506e98e528cded43143c927304875614f04e7adac";

export const MODEL = Object.freeze({
  vocabSize: 8193,
  modelDim: 512,
  hiddenDim: 1536,
  numLayers: 12,
  numQueryHeads: 8,
  numKvHeads: 2,
  headDim: 64,
  maxContext: 512,
  parameterCount: 40_385_024,
  tensorCount: 134,
  normEpsilon: 1e-5,
  qkNormEpsilon: 1e-6,
  ropeTheta: 10_000,
  eodToken: 8192,
});

export const DEFAULT_SAMPLING = Object.freeze({
  temperature: 0.7,
  topK: 50,
  topP: 0.9,
  maxNewTokens: 32,
});
