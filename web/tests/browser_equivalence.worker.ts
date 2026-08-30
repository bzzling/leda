/// <reference lib="webworker" />

import { parseLedaWebArtifact } from "../src/artifact.js";
import { FROZEN_F32_ARTIFACT_SHA256, FROZEN_TOKENIZER_SHA256, MODEL } from "../src/constants.js";
import { sha256Hex } from "../src/hash.js";
import { parseReferenceFixture } from "../src/reference.js";
import { greedyToken } from "../src/sampling.js";
import { ByteBPETokenizer } from "../src/tokenizer.js";
import { detectWebGpu, LedaWebGpuModel } from "../src/webgpu.js";

const REFERENCE_SHA256 = "49812a582c8d12b7e415c1e7583e4dff6f68a74939b99697d1c0a0519af3cd0b";
const scope = self as DedicatedWorkerGlobalScope;

interface GeometryResult {
  readonly promptLength: number;
  readonly decodeSteps: number;
  readonly maxAbsoluteError: number;
  readonly maxRelativeError: number;
  readonly greedyMatches: number;
  readonly stages: number;
  readonly elapsedMs: number;
  readonly stageErrors: readonly {
    readonly stage: number;
    readonly absolute: number;
    readonly relative: number;
    readonly greedyMatch: boolean;
  }[];
}

function compare(actual: Float32Array, expected: Float32Array): { absolute: number; relative: number } {
  let absolute = 0;
  let relative = 0;
  for (let index = 0; index < actual.length; ++index) {
    const difference = Math.abs((actual[index] ?? 0) - (expected[index] ?? 0));
    absolute = Math.max(absolute, difference);
    relative = Math.max(relative, difference / Math.max(Math.abs(expected[index] ?? 0), 1e-6));
  }
  return { absolute, relative };
}

scope.addEventListener("message", () => {
  void (async () => {
    const capability = await detectWebGpu();
    if (!capability.supported) {
      throw new Error(capability.reason);
    }
    const [modelBytes, tokenizerBytes, referenceBytes] = await Promise.all([
      fetch("/model.ledaweb").then((response) => response.arrayBuffer()),
      fetch("/tokenizer.spartokn").then((response) => response.arrayBuffer()),
      fetch("/reference.ledaref").then((response) => response.arrayBuffer()),
    ]);
    if (await sha256Hex(tokenizerBytes) !== FROZEN_TOKENIZER_SHA256) {
      throw new Error("browser test tokenizer hash mismatch");
    }
    const artifact = await parseLedaWebArtifact(modelBytes, FROZEN_F32_ARTIFACT_SHA256);
    const tokenizer = ByteBPETokenizer.parse(tokenizerBytes);
    const reference = await parseReferenceFixture(referenceBytes, REFERENCE_SHA256);
    let tokenizerMatches = 0;
    for (const item of reference.tokenizerCases) {
      const encoded = tokenizer.encodeBytes(item.bytes);
      const decoded = tokenizer.decodeBytes(item.tokens);
      if (encoded.length !== item.tokens.length || encoded.some((token, index) => token !== item.tokens[index]) ||
          decoded.length !== item.bytes.length || decoded.some((byte, index) => byte !== item.bytes[index])) {
        throw new Error(`tokenizer mismatch in case ${tokenizerMatches}`);
      }
      ++tokenizerMatches;
    }
    const model = await LedaWebGpuModel.create(artifact, capability.adapter);
    const geometries: GeometryResult[] = [];
    try {
      for (const item of reference.modelCases) {
        const started = performance.now();
        model.reset();
        let actual = await model.prefill(item.promptTokens);
        let maxAbsoluteError = 0;
        let maxRelativeError = 0;
        let greedyMatches = 0;
        const stageErrors: Array<{
          stage: number;
          absolute: number;
          relative: number;
          greedyMatch: boolean;
        }> = [];
        for (let stageIndex = 0; stageIndex < item.stages.length; ++stageIndex) {
          const stage = item.stages[stageIndex];
          if (stage === undefined) {
            throw new Error("reference stage inventory invariant failed");
          }
          if (stageIndex !== 0) {
            if (stage.inputToken === undefined) {
              throw new Error("decode reference lacks its input token");
            }
            actual = await model.decode(stage.inputToken);
          }
          const error = compare(actual, stage.logits);
          maxAbsoluteError = Math.max(maxAbsoluteError, error.absolute);
          maxRelativeError = Math.max(maxRelativeError, error.relative);
          if (greedyToken(actual) === stage.greedyToken) {
            ++greedyMatches;
          }
          stageErrors.push({
            stage: stageIndex,
            absolute: error.absolute,
            relative: error.relative,
            greedyMatch: greedyToken(actual) === stage.greedyToken,
          });
        }
        geometries.push({
          promptLength: item.promptTokens.length,
          decodeSteps: item.stages.length - 1,
          maxAbsoluteError,
          maxRelativeError,
          greedyMatches,
          stages: item.stages.length,
          elapsedMs: performance.now() - started,
          stageErrors,
        });
      }
      if (model.cacheLength !== MODEL.maxContext) {
        throw new Error("near-capacity fixture did not finish at exactly 512 tokens");
      }
      scope.postMessage({
        ok: true,
        tokenizerMatches,
        referenceSha256: reference.sha256,
        limits: model.limits,
        geometries,
      });
    } finally {
      model.destroy();
    }
  })().catch((error) => {
    const value = error instanceof Error ? error : new Error(String(error));
    scope.postMessage({ ok: false, name: value.name, message: value.message, stack: value.stack });
  });
});
