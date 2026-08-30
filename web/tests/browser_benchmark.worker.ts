/// <reference lib="webworker" />

import { parseLedaWebArtifact } from "../src/artifact.js";
import { FROZEN_F32_ARTIFACT_SHA256 } from "../src/constants.js";
import { parseReferenceFixture } from "../src/reference.js";
import { greedyToken } from "../src/sampling.js";
import { detectWebGpu, LedaWebGpuModel } from "../src/webgpu.js";

const REFERENCE_SHA256 = "49812a582c8d12b7e415c1e7583e4dff6f68a74939b99697d1c0a0519af3cd0b";
const scope = self as DedicatedWorkerGlobalScope;

scope.addEventListener("message", () => {
  void (async () => {
    const capability = await detectWebGpu();
    if (!capability.supported) {
      throw new Error(capability.reason);
    }
    const downloadStarted = performance.now();
    const [modelBytes, referenceBytes] = await Promise.all([
      fetch("/model.ledaweb").then((response) => response.arrayBuffer()),
      fetch("/reference.ledaref").then((response) => response.arrayBuffer()),
    ]);
    const downloadMs = performance.now() - downloadStarted;
    const artifact = await parseLedaWebArtifact(modelBytes, FROZEN_F32_ARTIFACT_SHA256);
    const reference = await parseReferenceFixture(referenceBytes, REFERENCE_SHA256);
    const initializationStarted = performance.now();
    const model = await LedaWebGpuModel.create(artifact, capability.adapter);
    const gpuInitializationMs = performance.now() - initializationStarted;
    try {
      model.reset();
      const warmup = await model.prefill(reference.modelCases[0]?.promptTokens ?? [0]);
      await model.decode(greedyToken(warmup));

      const results = [];
      for (const length of [32, 128, 256, 400]) {
        const item = reference.modelCases.find((candidate) => candidate.promptTokens.length === length);
        if (item === undefined) {
          throw new Error(`reference prompt ${length} is unavailable`);
        }
        model.reset();
        const totalStarted = performance.now();
        const prefillStarted = performance.now();
        let logits = await model.prefill(item.promptTokens);
        const prefillMs = performance.now() - prefillStarted;
        const firstToken = greedyToken(logits);
        const ttftMs = performance.now() - totalStarted;
        let token = firstToken;
        const decodeStarted = performance.now();
        for (let generated = 1; generated < 32; ++generated) {
          logits = await model.decode(token);
          token = greedyToken(logits);
        }
        const decodeMs = performance.now() - decodeStarted;
        results.push({
          promptLength: length,
          prefillMs,
          ttftMs,
          cachedDecodeTokensPerSecond: 31 / decodeMs * 1000,
          total32TokenMs: performance.now() - totalStarted,
          firstToken,
        });
      }
      scope.postMessage({
        ok: true,
        modelBytes: modelBytes.byteLength,
        downloadMs,
        gpuInitializationMs,
        estimatedGpuBufferBytes: model.estimatedGpuBufferBytes,
        limits: model.limits,
        results,
      });
    } finally {
      model.destroy();
    }
  })().catch((error) => {
    const value = error instanceof Error ? error : new Error(String(error));
    scope.postMessage({ ok: false, name: value.name, message: value.message, stack: value.stack });
  });
});
