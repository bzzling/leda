import type { CompletionOptions } from "./completion.js";
import type { WebGpuLimitsSnapshot } from "./webgpu.js";

export type WorkerCommand =
  | {
      readonly type: "INIT";
      readonly requestId: string;
      readonly modelUrl: string;
      readonly tokenizerUrl: string;
      readonly modelSha256?: string;
      readonly tokenizerSha256?: string;
    }
  | { readonly type: "COMPLETE"; readonly requestId: string; readonly prompt: string; readonly options?: CompletionOptions }
  | { readonly type: "NEXT_TOKEN"; readonly requestId: string; readonly prompt: string }
  | { readonly type: "CANCEL"; readonly requestId: string }
  | { readonly type: "STATUS"; readonly requestId: string };

export type InitializationStage =
  | "checking-webgpu"
  | "downloading-model"
  | "downloading-tokenizer"
  | "verifying-model"
  | "uploading-weights"
  | "ready";

export type WorkerEvent =
  | {
      readonly type: "INIT_PROGRESS";
      readonly requestId: string;
      readonly stage: InitializationStage;
      readonly receivedBytes?: number;
      readonly totalBytes?: number;
      readonly fromCache?: boolean;
    }
  | { readonly type: "READY"; readonly requestId: string; readonly limits: WebGpuLimitsSnapshot }
  | { readonly type: "STATUS"; readonly requestId: string; readonly ready: boolean; readonly activeRequestId?: string }
  | { readonly type: "TOKEN"; readonly requestId: string; readonly index: number; readonly token: number; readonly bytes: ArrayBuffer }
  | {
      readonly type: "TOP_CANDIDATES";
      readonly requestId: string;
      readonly candidates: readonly { readonly token: number; readonly bytes: ArrayBuffer; readonly probability: number }[];
    }
  | {
      readonly type: "COMPLETE";
      readonly requestId: string;
      readonly promptTokens: readonly number[];
      readonly generatedTokens: readonly number[];
      readonly generatedBytes: ArrayBuffer;
      readonly stoppedByEod: boolean;
    }
  | { readonly type: "CANCELED"; readonly requestId: string }
  | { readonly type: "ERROR"; readonly requestId: string; readonly message: string; readonly name: string };
