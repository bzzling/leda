/// <reference lib="webworker" />

import { parseLedaWebArtifact } from "./artifact.js";
import { loadVerifiedStaticAsset } from "./assets.js";
import { completeText } from "./completion.js";
import { FROZEN_F32_ARTIFACT_SHA256, FROZEN_TOKENIZER_SHA256 } from "./constants.js";
import type { InitializationStage, WorkerCommand, WorkerEvent } from "./protocol.js";
import { topCandidates } from "./sampling.js";
import { ByteBPETokenizer } from "./tokenizer.js";
import { detectWebGpu, LedaWebGpuModel } from "./webgpu.js";

const scope = self as DedicatedWorkerGlobalScope;
let model: LedaWebGpuModel | undefined;
let tokenizer: ByteBPETokenizer | undefined;
let activeRequestId: string | undefined;
let latestRequestId: string | undefined;
let operationTail: Promise<void> = Promise.resolve();
const canceled = new Set<string>();

function post(event: WorkerEvent, transfer: Transferable[] = []): void {
  scope.postMessage(event, transfer);
}

function progress(
  requestId: string,
  stage: InitializationStage,
  detail: { receivedBytes?: number; totalBytes?: number; fromCache?: boolean } = {},
): void {
  post({ type: "INIT_PROGRESS", requestId, stage, ...detail });
}

function errorEvent(requestId: string, error: unknown): void {
  const value = error instanceof Error ? error : new Error(String(error));
  post({ type: "ERROR", requestId, message: value.message, name: value.name });
}

async function initialize(command: Extract<WorkerCommand, { type: "INIT" }>): Promise<void> {
  if (model !== undefined && tokenizer !== undefined) {
    post({ type: "READY", requestId: command.requestId, limits: model.limits });
    return;
  }
  progress(command.requestId, "checking-webgpu");
  const capability = await detectWebGpu();
  if (!capability.supported) {
    throw new Error(capability.reason);
  }
  const modelSha256 = command.modelSha256 ?? FROZEN_F32_ARTIFACT_SHA256;
  const tokenizerSha256 = command.tokenizerSha256 ?? FROZEN_TOKENIZER_SHA256;
  if (modelSha256.toLowerCase() !== FROZEN_F32_ARTIFACT_SHA256 ||
      tokenizerSha256.toLowerCase() !== FROZEN_TOKENIZER_SHA256) {
    throw new Error("Leda Demo v0 requires the exact frozen browser model and tokenizer hashes");
  }
  progress(command.requestId, "downloading-model");
  const modelBytes = await loadVerifiedStaticAsset(command.modelUrl, modelSha256, (download) => {
    progress(command.requestId, "downloading-model", download);
  });
  progress(command.requestId, "downloading-tokenizer");
  const tokenizerBytes = await loadVerifiedStaticAsset(command.tokenizerUrl, tokenizerSha256, (download) => {
    progress(command.requestId, "downloading-tokenizer", download);
  });
  progress(command.requestId, "verifying-model");
  const artifact = await parseLedaWebArtifact(modelBytes, modelSha256);
  const parsedTokenizer = ByteBPETokenizer.parse(tokenizerBytes);
  if (parsedTokenizer.vocabularySize !== 8192) {
    throw new Error("The browser tokenizer is not the frozen 8192-token Leda tokenizer");
  }
  progress(command.requestId, "uploading-weights");
  const createdModel = await LedaWebGpuModel.create(artifact, capability.adapter);
  model = createdModel;
  tokenizer = parsedTokenizer;
  progress(command.requestId, "ready");
  post({ type: "READY", requestId: command.requestId, limits: createdModel.limits });
}

function requireReady(): { model: LedaWebGpuModel; tokenizer: ByteBPETokenizer } {
  if (model === undefined || tokenizer === undefined) {
    throw new Error("Leda browser inference has not been initialized");
  }
  return { model, tokenizer };
}

async function complete(command: Extract<WorkerCommand, { type: "COMPLETE" }>): Promise<void> {
  const runtime = requireReady();
  activeRequestId = command.requestId;
  canceled.delete(command.requestId);
  try {
    const result = await completeText(
      runtime.model,
      runtime.tokenizer,
      command.prompt,
      command.options,
      () => canceled.has(command.requestId) || activeRequestId !== command.requestId,
      (generated) => {
        if (activeRequestId !== command.requestId || canceled.has(command.requestId)) {
          return;
        }
        const bytes = generated.bytes.buffer;
        post({
          type: "TOKEN",
          requestId: command.requestId,
          index: generated.index,
          token: generated.token,
          bytes,
        }, [bytes]);
      },
    );
    if (activeRequestId !== command.requestId || canceled.has(command.requestId)) {
      post({ type: "CANCELED", requestId: command.requestId });
      return;
    }
    const candidates = result.firstTopCandidates.map((candidate) => {
      const bytes = runtime.tokenizer.tokenBytes(candidate.token).buffer;
      return { token: candidate.token, probability: candidate.probability, bytes };
    });
    post({ type: "TOP_CANDIDATES", requestId: command.requestId, candidates },
      candidates.map((candidate) => candidate.bytes));
    const generatedBytes = result.generatedBytes.buffer;
    post({
      type: "COMPLETE",
      requestId: command.requestId,
      promptTokens: result.promptTokens,
      generatedTokens: result.generatedTokens,
      generatedBytes,
      stoppedByEod: result.stoppedByEod,
    }, [generatedBytes]);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      post({ type: "CANCELED", requestId: command.requestId });
    } else {
      throw error;
    }
  } finally {
    canceled.delete(command.requestId);
    if (activeRequestId === command.requestId) {
      activeRequestId = undefined;
    }
  }
}

async function nextToken(command: Extract<WorkerCommand, { type: "NEXT_TOKEN" }>): Promise<void> {
  const runtime = requireReady();
  const tokens = runtime.tokenizer.encodeText(command.prompt);
  if (tokens.length === 0) {
    throw new RangeError("Leda requires a non-empty prompt");
  }
  runtime.model.reset();
  const logits = await runtime.model.prefill(tokens);
  const candidates = topCandidates(logits, 5).map((candidate) => {
    const bytes = runtime.tokenizer.tokenBytes(candidate.token).buffer;
    return { token: candidate.token, probability: candidate.probability, bytes };
  });
  post({ type: "TOP_CANDIDATES", requestId: command.requestId, candidates },
    candidates.map((candidate) => candidate.bytes));
}

scope.addEventListener("message", (event: MessageEvent<WorkerCommand>) => {
  const command = event.data;
  void (async () => {
    switch (command.type) {
      case "INIT":
        await initialize(command);
        break;
      case "COMPLETE":
        if (activeRequestId !== undefined) {
          canceled.add(activeRequestId);
        }
        latestRequestId = command.requestId;
        {
          const task = operationTail.then(async () => {
            if (latestRequestId !== command.requestId || canceled.has(command.requestId)) {
              post({ type: "CANCELED", requestId: command.requestId });
              return;
            }
            await complete(command);
          });
          operationTail = task.catch(() => undefined);
          await task;
        }
        break;
      case "NEXT_TOKEN":
        if (activeRequestId !== undefined) {
          canceled.add(activeRequestId);
        }
        latestRequestId = command.requestId;
        {
          const task = operationTail.then(async () => {
            if (latestRequestId !== command.requestId || canceled.has(command.requestId)) {
              post({ type: "CANCELED", requestId: command.requestId });
              return;
            }
            activeRequestId = command.requestId;
            try {
              await nextToken(command);
            } finally {
              if (activeRequestId === command.requestId) {
                activeRequestId = undefined;
              }
            }
          });
          operationTail = task.catch(() => undefined);
          await task;
        }
        break;
      case "CANCEL":
        canceled.add(command.requestId);
        if (latestRequestId === command.requestId) {
          latestRequestId = undefined;
        }
        if (activeRequestId === command.requestId) {
          activeRequestId = undefined;
        }
        break;
      case "STATUS":
        post({
          type: "STATUS",
          requestId: command.requestId,
          ready: model !== undefined && tokenizer !== undefined,
          ...(activeRequestId === undefined ? {} : { activeRequestId }),
        });
        break;
    }
  })().catch((error) => errorEvent(command.requestId, error));
});
