import { FROZEN_CHECKPOINT_SHA256, FROZEN_TOKENIZER_SHA256, MODEL } from "./constants.js";
import { bytesToHex, equalHex, sha256Hex } from "./hash.js";

const MAGIC = "LEDAREF\0";
const VERSION = 1;
const HEADER_BYTES = 128;

export interface TokenizerReferenceCase {
  readonly bytes: Uint8Array;
  readonly tokens: readonly number[];
}

export interface LogitReferenceStage {
  readonly inputToken?: number;
  readonly greedyToken: number;
  readonly logits: Float32Array;
}

export interface ModelReferenceCase {
  readonly promptTokens: readonly number[];
  readonly stages: readonly LogitReferenceStage[];
}

export interface LedaReferenceFixture {
  readonly sha256: string;
  readonly tokenizerCases: readonly TokenizerReferenceCase[];
  readonly modelCases: readonly ModelReferenceCase[];
}

class Reader {
  readonly view: DataView;
  offset = 0;

  constructor(readonly bytes: ArrayBuffer) {
    this.view = new DataView(bytes);
  }

  require(length: number): void {
    if (!Number.isSafeInteger(length) || length < 0 || this.offset > this.bytes.byteLength - length) {
      throw new Error("LEDAREF is truncated or has an invalid length");
    }
  }

  u32(): number {
    this.require(4);
    const value = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return value;
  }

  byteArray(length: number): Uint8Array {
    this.require(length);
    const result = new Uint8Array(this.bytes, this.offset, length);
    this.offset += length;
    return result;
  }

  f32Array(length: number): Float32Array {
    this.require(length * 4);
    if (this.offset % 4 !== 0) {
      throw new Error("LEDAREF Float32 payload is misaligned");
    }
    const result = new Float32Array(this.bytes, this.offset, length);
    this.offset += length * 4;
    return result;
  }

  align(alignment: number): void {
    this.offset = Math.ceil(this.offset / alignment) * alignment;
    this.require(0);
  }
}

export async function parseReferenceFixture(
  bytes: ArrayBuffer,
  expectedSha256?: string,
): Promise<LedaReferenceFixture> {
  if (bytes.byteLength < HEADER_BYTES) {
    throw new Error("LEDAREF is shorter than its fixed header");
  }
  const sha256 = await sha256Hex(bytes);
  if (expectedSha256 !== undefined && !equalHex(sha256, expectedSha256)) {
    throw new Error(`LEDAREF SHA-256 mismatch: got ${sha256}`);
  }
  const reader = new Reader(bytes);
  const magic = String.fromCharCode(...reader.byteArray(8));
  if (magic !== MAGIC || reader.u32() !== VERSION || reader.u32() !== HEADER_BYTES) {
    throw new Error("Invalid or unsupported LEDAREF fixture");
  }
  const tokenizerCaseCount = reader.u32();
  const modelCaseCount = reader.u32();
  if (tokenizerCaseCount !== 7 || modelCaseCount !== 8 || reader.u32() !== MODEL.vocabSize || reader.u32() !== 0) {
    throw new Error("LEDAREF fixture inventory does not match Phase 34");
  }
  if (!equalHex(bytesToHex(reader.byteArray(32)), FROZEN_CHECKPOINT_SHA256) ||
      !equalHex(bytesToHex(reader.byteArray(32)), FROZEN_TOKENIZER_SHA256)) {
    throw new Error("LEDAREF frozen artifact identity mismatch");
  }
  for (const value of reader.byteArray(HEADER_BYTES - reader.offset)) {
    if (value !== 0) {
      throw new Error("LEDAREF reserved header bytes must be zero");
    }
  }

  const tokenizerCases: TokenizerReferenceCase[] = [];
  for (let index = 0; index < tokenizerCaseCount; ++index) {
    const byteLength = reader.u32();
    const tokenCount = reader.u32();
    if (byteLength > 4096 || tokenCount > 4096) {
      throw new Error("LEDAREF tokenizer case is unreasonably large");
    }
    const caseBytes = reader.byteArray(byteLength);
    const tokens = Array.from({ length: tokenCount }, () => reader.u32());
    reader.align(8);
    tokenizerCases.push({ bytes: caseBytes, tokens });
  }

  const modelCases: ModelReferenceCase[] = [];
  const expectedPromptLengths = [1, 17, 32, 127, 128, 256, 400, 500];
  for (let caseIndex = 0; caseIndex < modelCaseCount; ++caseIndex) {
    const promptLength = reader.u32();
    const decodeCount = reader.u32();
    if (promptLength !== expectedPromptLengths[caseIndex] ||
        decodeCount !== (promptLength === 500 ? 12 : 3) ||
        promptLength + decodeCount > MODEL.maxContext) {
      throw new Error("LEDAREF model geometry is unexpected");
    }
    const promptTokens = Array.from({ length: promptLength }, () => reader.u32());
    const stages: LogitReferenceStage[] = [];
    for (let stage = 0; stage <= decodeCount; ++stage) {
      const encodedInput = reader.u32();
      const greedyToken = reader.u32();
      if ((stage === 0) !== (encodedInput === 0xffff_ffff) || greedyToken >= MODEL.vocabSize) {
        throw new Error("LEDAREF stage metadata is invalid");
      }
      const logits = reader.f32Array(MODEL.vocabSize);
      if (Array.from(logits).some((value) => !Number.isFinite(value))) {
        throw new Error("LEDAREF contains non-finite logits");
      }
      stages.push(stage === 0
        ? { greedyToken, logits }
        : { inputToken: encodedInput, greedyToken, logits });
    }
    modelCases.push({ promptTokens, stages });
  }
  if (reader.offset !== bytes.byteLength) {
    throw new Error("LEDAREF has trailing data");
  }
  return { sha256, tokenizerCases, modelCases };
}
