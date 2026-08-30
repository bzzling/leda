import {
  FROZEN_CHECKPOINT_SHA256,
  LEDA_WEB_DTYPE_FLOAT32,
  LEDA_WEB_HEADER_BYTES,
  LEDA_WEB_MAGIC,
  LEDA_WEB_VERSION,
  MODEL,
} from "./constants.js";
import { bytesToHex, equalHex, sha256Hex } from "./hash.js";

const textDecoder = new TextDecoder("utf-8", { fatal: true });

export interface LedaWebConfig {
  readonly vocabSize: number;
  readonly modelDim: number;
  readonly hiddenDim: number;
  readonly numLayers: number;
  readonly numQueryHeads: number;
  readonly numKvHeads: number;
  readonly maxContext: number;
  readonly normEpsilon: number;
  readonly qkNormEpsilon: number;
  readonly ropeTheta: number;
}

export interface LedaWebTensor {
  readonly name: string;
  readonly shape: readonly number[];
  readonly elementCount: number;
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly values: Float32Array;
}

export interface LedaWebArtifact {
  readonly version: number;
  readonly sha256: string;
  readonly sourceCheckpointSha256: string;
  readonly config: LedaWebConfig;
  readonly tensors: ReadonlyMap<string, LedaWebTensor>;
  readonly bytes: ArrayBuffer;
}

class Reader {
  readonly view: DataView;

  constructor(readonly bytes: ArrayBuffer) {
    this.view = new DataView(bytes);
  }

  requireRange(offset: number, length: number, label: string): void {
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < 0 || length < 0 ||
        offset > this.bytes.byteLength - length) {
      throw new Error(`LEDAWEB ${label} is outside the artifact`);
    }
  }

  u32(offset: number): number {
    this.requireRange(offset, 4, "u32");
    return this.view.getUint32(offset, true);
  }

  u64(offset: number): number {
    this.requireRange(offset, 8, "u64");
    const value = this.view.getBigUint64(offset, true);
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error("LEDAWEB integer exceeds JavaScript's exact integer range");
    }
    return Number(value);
  }

  f32(offset: number): number {
    this.requireRange(offset, 4, "f32");
    return this.view.getFloat32(offset, true);
  }

  slice(offset: number, length: number): Uint8Array {
    this.requireRange(offset, length, "byte range");
    return new Uint8Array(this.bytes, offset, length);
  }
}

function requireEqual(actual: number, expected: number, field: string): void {
  if (actual !== expected) {
    throw new Error(`LEDAWEB ${field} is ${actual}; expected ${expected}`);
  }
}

function closeEnough(actual: number, expected: number, field: string): void {
  if (!Number.isFinite(actual) || Math.abs(actual - expected) > Math.abs(expected) * 1e-6) {
    throw new Error(`LEDAWEB ${field} does not match Leda Demo v0`);
  }
}

function expectedTensorShapes(): ReadonlyMap<string, readonly number[]> {
  const result = new Map<string, readonly number[]>();
  result.set("token_embedding.weight", [MODEL.vocabSize, MODEL.modelDim]);
  for (let layer = 0; layer < MODEL.numLayers; ++layer) {
    const prefix = `blocks.${layer}`;
    result.set(`${prefix}.attention_norm.weight`, [MODEL.modelDim]);
    result.set(`${prefix}.attention.q_proj.weight`, [MODEL.modelDim, MODEL.modelDim]);
    result.set(`${prefix}.attention.k_proj.weight`, [MODEL.modelDim, MODEL.numKvHeads * MODEL.headDim]);
    result.set(`${prefix}.attention.v_proj.weight`, [MODEL.modelDim, MODEL.numKvHeads * MODEL.headDim]);
    result.set(`${prefix}.attention.out_proj.weight`, [MODEL.modelDim, MODEL.modelDim]);
    result.set(`${prefix}.attention.q_norm.weight`, [MODEL.headDim]);
    result.set(`${prefix}.attention.k_norm.weight`, [MODEL.headDim]);
    result.set(`${prefix}.mlp_norm.weight`, [MODEL.modelDim]);
    result.set(`${prefix}.mlp.gate_proj.weight`, [MODEL.modelDim, MODEL.hiddenDim]);
    result.set(`${prefix}.mlp.up_proj.weight`, [MODEL.modelDim, MODEL.hiddenDim]);
    result.set(`${prefix}.mlp.down_proj.weight`, [MODEL.hiddenDim, MODEL.modelDim]);
  }
  result.set("final_norm.weight", [MODEL.modelDim]);
  return result;
}

function sameShape(actual: readonly number[], expected: readonly number[]): boolean {
  return actual.length === expected.length && actual.every((value, index) => value === expected[index]);
}

export async function parseLedaWebArtifact(
  bytes: ArrayBuffer,
  expectedArtifactSha256?: string,
): Promise<LedaWebArtifact> {
  if (bytes.byteLength < LEDA_WEB_HEADER_BYTES) {
    throw new Error("LEDAWEB artifact is shorter than its fixed header");
  }
  const artifactSha256 = await sha256Hex(bytes);
  if (expectedArtifactSha256 !== undefined && !equalHex(artifactSha256, expectedArtifactSha256)) {
    throw new Error(`LEDAWEB SHA-256 mismatch: got ${artifactSha256}`);
  }

  const reader = new Reader(bytes);
  const magic = String.fromCharCode(...reader.slice(0, 8));
  if (magic !== LEDA_WEB_MAGIC) {
    throw new Error("Invalid LEDAWEB artifact magic");
  }
  requireEqual(reader.u32(8), LEDA_WEB_VERSION, "version");
  requireEqual(reader.u32(12), LEDA_WEB_HEADER_BYTES, "header size");
  requireEqual(reader.u32(16), 0, "flags");
  requireEqual(reader.u32(20), LEDA_WEB_DTYPE_FLOAT32, "dtype");

  const config: LedaWebConfig = {
    vocabSize: reader.u32(24),
    modelDim: reader.u32(28),
    hiddenDim: reader.u32(32),
    numLayers: reader.u32(36),
    numQueryHeads: reader.u32(40),
    numKvHeads: reader.u32(44),
    maxContext: reader.u32(48),
    normEpsilon: reader.f32(56),
    qkNormEpsilon: reader.f32(60),
    ropeTheta: reader.f32(64),
  };
  requireEqual(reader.u32(52), MODEL.tensorCount, "tensor count");
  requireEqual(reader.u32(68), 0, "reserved field");
  requireEqual(config.vocabSize, MODEL.vocabSize, "vocabulary size");
  requireEqual(config.modelDim, MODEL.modelDim, "model dimension");
  requireEqual(config.hiddenDim, MODEL.hiddenDim, "hidden dimension");
  requireEqual(config.numLayers, MODEL.numLayers, "layer count");
  requireEqual(config.numQueryHeads, MODEL.numQueryHeads, "query-head count");
  requireEqual(config.numKvHeads, MODEL.numKvHeads, "KV-head count");
  requireEqual(config.maxContext, MODEL.maxContext, "maximum context");
  closeEnough(config.normEpsilon, MODEL.normEpsilon, "normalization epsilon");
  closeEnough(config.qkNormEpsilon, MODEL.qkNormEpsilon, "QK normalization epsilon");
  closeEnough(config.ropeTheta, MODEL.ropeTheta, "RoPE theta");

  const metadataOffset = reader.u64(72);
  const metadataBytes = reader.u64(80);
  const payloadOffset = reader.u64(88);
  const payloadBytes = reader.u64(96);
  requireEqual(metadataOffset, LEDA_WEB_HEADER_BYTES, "metadata offset");
  reader.requireRange(metadataOffset, metadataBytes, "metadata");
  reader.requireRange(payloadOffset, payloadBytes, "payload");
  if (metadataOffset + metadataBytes > payloadOffset || payloadOffset + payloadBytes !== bytes.byteLength) {
    throw new Error("LEDAWEB metadata/payload layout is inconsistent");
  }
  const sourceCheckpointSha256 = bytesToHex(reader.slice(104, 32));
  if (!equalHex(sourceCheckpointSha256, FROZEN_CHECKPOINT_SHA256)) {
    throw new Error("LEDAWEB source checkpoint is not the frozen Leda Demo v0");
  }
  for (const value of reader.slice(136, LEDA_WEB_HEADER_BYTES - 136)) {
    if (value !== 0) {
      throw new Error("LEDAWEB reserved header bytes must be zero");
    }
  }

  const expected = expectedTensorShapes();
  const tensors = new Map<string, LedaWebTensor>();
  let cursor = metadataOffset;
  let totalElements = 0;
  let previousPayloadEnd = payloadOffset;
  for (let tensorIndex = 0; tensorIndex < MODEL.tensorCount; ++tensorIndex) {
    reader.requireRange(cursor, 32, "tensor metadata record");
    const nameBytes = reader.u32(cursor);
    const rank = reader.u32(cursor + 4);
    requireEqual(reader.u32(cursor + 8), LEDA_WEB_DTYPE_FLOAT32, "tensor dtype");
    requireEqual(reader.u32(cursor + 12), 0, "tensor reserved field");
    const elementCount = reader.u64(cursor + 16);
    const byteOffset = reader.u64(cursor + 24);
    if (nameBytes === 0 || nameBytes > 4096 || rank === 0 || rank > 8) {
      throw new Error("LEDAWEB tensor metadata has an invalid name length or rank");
    }
    const unpaddedBytes = 32 + rank * 4 + nameBytes;
    const recordBytes = Math.ceil(unpaddedBytes / 8) * 8;
    if (cursor + recordBytes > metadataOffset + metadataBytes) {
      throw new Error("LEDAWEB tensor metadata exceeds the metadata section");
    }
    const shape: number[] = [];
    let product = 1;
    for (let dimension = 0; dimension < rank; ++dimension) {
      const value = reader.u32(cursor + 32 + dimension * 4);
      if (value === 0 || !Number.isSafeInteger(product * value)) {
        throw new Error("LEDAWEB tensor shape is invalid or overflows");
      }
      product *= value;
      shape.push(value);
    }
    requireEqual(elementCount, product, "tensor element count");
    const nameOffset = cursor + 32 + rank * 4;
    const name = textDecoder.decode(reader.slice(nameOffset, nameBytes));
    const expectedShape = expected.get(name);
    if (expectedShape === undefined || !sameShape(shape, expectedShape) || tensors.has(name)) {
      throw new Error(`Unexpected, duplicate, or incorrectly shaped LEDAWEB tensor: ${name}`);
    }
    const byteLength = elementCount * 4;
    if (!Number.isSafeInteger(byteLength) || byteOffset % 4 !== 0 || byteOffset !== previousPayloadEnd) {
      throw new Error(`LEDAWEB tensor payload is misaligned or non-contiguous: ${name}`);
    }
    reader.requireRange(byteOffset, byteLength, `tensor ${name}`);
    if (byteOffset < payloadOffset || byteOffset + byteLength > payloadOffset + payloadBytes) {
      throw new Error(`LEDAWEB tensor is outside the payload: ${name}`);
    }
    tensors.set(name, {
      name,
      shape,
      elementCount,
      byteOffset,
      byteLength,
      values: new Float32Array(bytes, byteOffset, elementCount),
    });
    previousPayloadEnd = byteOffset + byteLength;
    totalElements += elementCount;
    cursor += recordBytes;
  }
  if (cursor !== metadataOffset + metadataBytes || tensors.size !== expected.size ||
      previousPayloadEnd !== payloadOffset + payloadBytes) {
    throw new Error("LEDAWEB tensor inventory does not consume its declared sections exactly");
  }
  requireEqual(totalElements, MODEL.parameterCount, "parameter count");

  return {
    version: LEDA_WEB_VERSION,
    sha256: artifactSha256,
    sourceCheckpointSha256,
    config,
    tensors,
    bytes,
  };
}
