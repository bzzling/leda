import { DEFAULT_SAMPLING, MODEL } from "./constants.js";
import { sampleToken, SplitMix64, topCandidates, type SamplingConfig, type TokenProbability } from "./sampling.js";
import { ByteBPETokenizer, type TokenId } from "./tokenizer.js";

export interface CachedLogitModel {
  readonly cacheLength: number;
  prefill(tokens: readonly TokenId[]): Promise<Float32Array>;
  decode(token: TokenId): Promise<Float32Array>;
  reset(): void;
}

export interface CompletionOptions extends SamplingConfig {
  readonly maxNewTokens?: number;
  readonly seed?: bigint | number;
}

export interface GeneratedToken {
  readonly index: number;
  readonly token: TokenId;
  readonly bytes: Uint8Array;
}

export interface CompletionResult {
  readonly promptTokens: readonly TokenId[];
  readonly generatedTokens: readonly TokenId[];
  readonly generatedBytes: Uint8Array;
  readonly stoppedByEod: boolean;
  readonly firstTopCandidates: readonly TokenProbability[];
}

function concatenate(chunks: readonly Uint8Array[]): Uint8Array {
  const total = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const result = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

export async function completeText(
  model: CachedLogitModel,
  tokenizer: ByteBPETokenizer,
  prompt: string,
  options: CompletionOptions = {},
  canceled: () => boolean = () => false,
  onToken?: (token: GeneratedToken) => void,
): Promise<CompletionResult> {
  const promptTokens = tokenizer.encodeText(prompt);
  const maxNewTokens = options.maxNewTokens ?? DEFAULT_SAMPLING.maxNewTokens;
  if (promptTokens.length === 0) {
    throw new RangeError("Leda requires a non-empty prompt");
  }
  if (!Number.isInteger(maxNewTokens) || maxNewTokens < 0) {
    throw new RangeError("maxNewTokens must be a non-negative integer");
  }
  if (promptTokens.length + maxNewTokens > MODEL.maxContext) {
    throw new RangeError("Prompt tokens plus requested completion exceed Leda's 512-token context");
  }
  model.reset();
  let logits = await model.prefill(promptTokens);
  const firstTopCandidates = topCandidates(logits, 5);
  const random = new SplitMix64(options.seed ?? 42);
  const generatedTokens: TokenId[] = [];
  const chunks: Uint8Array[] = [];
  let stoppedByEod = false;
  for (let index = 0; index < maxNewTokens; ++index) {
    if (canceled()) {
      throw new DOMException("Leda completion was canceled", "AbortError");
    }
    const token = sampleToken(logits, random, options);
    if (token === MODEL.eodToken) {
      stoppedByEod = true;
      break;
    }
    const bytes = tokenizer.tokenBytes(token);
    generatedTokens.push(token);
    chunks.push(bytes);
    onToken?.({ index, token, bytes: bytes.slice() });
    if (index + 1 < maxNewTokens) {
      logits = await model.decode(token);
    }
  }
  return {
    promptTokens,
    generatedTokens,
    generatedBytes: concatenate(chunks),
    stoppedByEod,
    firstTopCandidates,
  };
}
