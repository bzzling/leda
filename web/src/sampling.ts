import { DEFAULT_SAMPLING } from "./constants.js";
import type { TokenId } from "./tokenizer.js";

const MASK64 = (1n << 64n) - 1n;

export interface SamplingConfig {
  readonly greedy?: boolean;
  readonly temperature?: number;
  readonly topK?: number;
  readonly topP?: number;
}

export interface TokenProbability {
  readonly token: TokenId;
  readonly probability: number;
}

export class SplitMix64 {
  private state: bigint;

  constructor(seed: bigint | number) {
    this.state = BigInt(seed) & MASK64;
  }

  nextU64(): bigint {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let value = this.state;
    value = ((value ^ (value >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    value = ((value ^ (value >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return (value ^ (value >> 31n)) & MASK64;
  }

  uniformDouble(): number {
    return Number(this.nextU64() >> 11n) / 9_007_199_254_740_992;
  }
}

function descendingIndices(values: readonly number[]): number[] {
  return Array.from(values, (_, index) => index).sort((left, right) => {
    const difference = (values[right] ?? Number.NEGATIVE_INFINITY) -
      (values[left] ?? Number.NEGATIVE_INFINITY);
    return difference !== 0 ? difference : left - right;
  });
}

function finiteLogits(logits: Float32Array | readonly number[]): number[] {
  const result = Array.from(logits);
  if (result.length === 0 || result.some((value) => !Number.isFinite(value))) {
    throw new Error("Sampling requires a non-empty vector of finite logits");
  }
  return result;
}

export function samplingProbabilities(
  input: Float32Array | readonly number[],
  temperature: number = DEFAULT_SAMPLING.temperature,
  topK?: number,
  topP?: number,
): Float64Array {
  const logits = finiteLogits(input);
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new RangeError("Sampling temperature must be finite and positive");
  }
  if (topK !== undefined && (!Number.isInteger(topK) || topK < 1 || topK > logits.length)) {
    throw new RangeError("Sampling top-k must be in [1,vocabulary_size]");
  }
  if (topP !== undefined && (!Number.isFinite(topP) || topP <= 0 || topP > 1)) {
    throw new RangeError("Sampling top-p must be finite and in (0,1]");
  }
  for (let index = 0; index < logits.length; ++index) {
    logits[index] = (logits[index] ?? 0) / temperature;
  }
  const order = descendingIndices(logits);
  const retained = topK ?? logits.length;
  const maximum = logits[order[0] ?? 0] ?? 0;
  const result = new Float64Array(logits.length);
  let total = 0;
  for (let rank = 0; rank < retained; ++rank) {
    const token = order[rank];
    if (token === undefined) {
      throw new Error("Sampling order invariant failed");
    }
    const probability = Math.exp((logits[token] ?? 0) - maximum);
    result[token] = probability;
    total += probability;
  }
  if (!Number.isFinite(total) || total <= 0) {
    throw new Error("Sampling softmax normalization failed");
  }
  for (let index = 0; index < result.length; ++index) {
    result[index] = (result[index] ?? 0) / total;
  }
  if (topP !== undefined && topP < 1) {
    let cumulative = 0;
    let nucleusSize = 0;
    for (; nucleusSize < retained; ++nucleusSize) {
      cumulative += result[order[nucleusSize] ?? 0] ?? 0;
      if (cumulative >= topP) {
        ++nucleusSize;
        break;
      }
    }
    for (let rank = nucleusSize; rank < retained; ++rank) {
      result[order[rank] ?? 0] = 0;
    }
    for (let index = 0; index < result.length; ++index) {
      result[index] = (result[index] ?? 0) / cumulative;
    }
  }
  return result;
}

export function greedyToken(input: Float32Array | readonly number[]): TokenId {
  return descendingIndices(finiteLogits(input))[0] ?? 0;
}

export function sampleToken(
  input: Float32Array | readonly number[],
  random: SplitMix64,
  config: SamplingConfig = {},
): TokenId {
  if (config.greedy === true) {
    return greedyToken(input);
  }
  const probabilities = samplingProbabilities(
    input,
    config.temperature ?? DEFAULT_SAMPLING.temperature,
    config.topK ?? DEFAULT_SAMPLING.topK,
    config.topP ?? DEFAULT_SAMPLING.topP,
  );
  const draw = random.uniformDouble();
  let cumulative = 0;
  let fallback = 0;
  for (let token = 0; token < probabilities.length; ++token) {
    const probability = probabilities[token] ?? 0;
    if (probability > 0) {
      fallback = token;
    }
    cumulative += probability;
    if (draw < cumulative) {
      return token;
    }
  }
  return fallback;
}

export function topCandidates(
  input: Float32Array | readonly number[],
  count: number,
): TokenProbability[] {
  const logits = finiteLogits(input);
  if (!Number.isInteger(count) || count < 1 || count > logits.length) {
    throw new RangeError("Top-candidate count must be in [1,vocabulary_size]");
  }
  const probabilities = samplingProbabilities(logits, 1, undefined, undefined);
  return descendingIndices(logits).slice(0, count).map((token) => ({
    token,
    probability: probabilities[token] ?? 0,
  }));
}
