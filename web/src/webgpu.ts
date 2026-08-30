import type { LedaWebArtifact, LedaWebTensor } from "./artifact.js";
import { MODEL } from "./constants.js";
import {
  ADD_SHADER,
  ATTENTION_SHADER,
  CACHE_APPEND_SHADER,
  EMBEDDING_SHADER,
  MATMUL_SHADER,
  RMS_NORM_SHADER,
  ROPE_SHADER,
  SWIGLU_SHADER,
  TIED_OUTPUT_SHADER,
} from "./shaders.js";

const FLOAT_BYTES = 4;
const PARAMETER_SLOT_BYTES = 256;
const PARAMETER_SLOTS = 512;

export type WebGpuCapability =
  | { readonly supported: true; readonly adapter: GPUAdapter; readonly info?: GPUAdapterInfo }
  | { readonly supported: false; readonly reason: string };

export interface WebGpuLimitsSnapshot {
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
  readonly maxComputeWorkgroupsPerDimension: number;
  readonly maxComputeInvocationsPerWorkgroup: number;
  readonly shaderF16: boolean;
}

interface Pipelines {
  readonly embedding: GPUComputePipeline;
  readonly rmsNorm: GPUComputePipeline;
  readonly matmul: GPUComputePipeline;
  readonly add: GPUComputePipeline;
  readonly swiglu: GPUComputePipeline;
  readonly rope: GPUComputePipeline;
  readonly cacheAppend: GPUComputePipeline;
  readonly attention: GPUComputePipeline;
  readonly tiedOutput: GPUComputePipeline;
}

interface LayerWeights {
  readonly attentionNorm: GPUBuffer;
  readonly query: GPUBuffer;
  readonly key: GPUBuffer;
  readonly value: GPUBuffer;
  readonly attentionOutput: GPUBuffer;
  readonly queryNorm: GPUBuffer;
  readonly keyNorm: GPUBuffer;
  readonly mlpNorm: GPUBuffer;
  readonly gate: GPUBuffer;
  readonly up: GPUBuffer;
  readonly down: GPUBuffer;
}

interface LayerCache {
  readonly key: GPUBuffer;
  readonly value: GPUBuffer;
}

interface Workspaces {
  readonly tokens: GPUBuffer;
  readonly hiddenA: GPUBuffer;
  readonly hiddenB: GPUBuffer;
  readonly norm: GPUBuffer;
  readonly query: GPUBuffer;
  readonly key: GPUBuffer;
  readonly value: GPUBuffer;
  readonly attention: GPUBuffer;
  readonly projection: GPUBuffer;
  readonly gate: GPUBuffer;
  readonly up: GPUBuffer;
  readonly logits: GPUBuffer;
  readonly readback: GPUBuffer;
  readonly parameters: GPUBuffer;
}

export async function detectWebGpu(): Promise<WebGpuCapability> {
  if (typeof navigator === "undefined" || navigator.gpu === undefined) {
    return { supported: false, reason: "WebGPU is unavailable; Leda requires a WebGPU-capable browser." };
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) {
    return { supported: false, reason: "No WebGPU adapter is available." };
  }
  let info: GPUAdapterInfo | undefined;
  try {
    info = adapter.info;
  } catch {
    // Adapter identification is diagnostic only.
  }
  return info === undefined ? { supported: true, adapter } : { supported: true, adapter, info };
}

function ceilDivide(value: number, divisor: number): number {
  return Math.floor((value + divisor - 1) / divisor);
}

function requireTensor(artifact: LedaWebArtifact, name: string): LedaWebTensor {
  const tensor = artifact.tensors.get(name);
  if (tensor === undefined) {
    throw new Error(`LEDAWEB tensor is missing: ${name}`);
  }
  return tensor;
}

function createStorage(device: GPUDevice, label: string, bytes: number): GPUBuffer {
  return device.createBuffer({
    label,
    size: Math.max(4, Math.ceil(bytes / 4) * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

function uploadTensor(device: GPUDevice, tensor: LedaWebTensor): GPUBuffer {
  const buffer = device.createBuffer({
    label: tensor.name,
    size: tensor.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(tensor.values);
  buffer.unmap();
  return buffer;
}

async function pipeline(device: GPUDevice, label: string, code: string): Promise<GPUComputePipeline> {
  const module = device.createShaderModule({ label: `${label} shader`, code });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter((message) => message.type === "error");
  if (errors.length !== 0) {
    throw new Error(`${label} WGSL compilation failed:\n${errors.map((error) => error.message).join("\n")}`);
  }
  return device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
}

class Dispatches {
  private slot = 0;

  constructor(
    private readonly device: GPUDevice,
    private readonly encoder: GPUCommandEncoder,
    private readonly parameters: GPUBuffer,
  ) {}

  private parameterBinding(words: readonly number[], floats: ReadonlyMap<number, number>): GPUBufferBinding {
    if (this.slot >= PARAMETER_SLOTS) {
      throw new Error("WebGPU parameter ring exhausted for one model invocation");
    }
    const bytes = new ArrayBuffer(PARAMETER_SLOT_BYTES);
    const integers = new Uint32Array(bytes);
    words.forEach((value, index) => { integers[index] = value; });
    const view = new DataView(bytes);
    for (const [word, value] of floats) {
      view.setFloat32(word * 4, value, true);
    }
    const offset = this.slot * PARAMETER_SLOT_BYTES;
    this.device.queue.writeBuffer(this.parameters, offset, bytes);
    ++this.slot;
    return { buffer: this.parameters, offset, size: PARAMETER_SLOT_BYTES };
  }

  run(
    pipelineValue: GPUComputePipeline,
    words: readonly number[],
    buffers: readonly GPUBufferBinding[],
    workgroups: readonly [number, number?, number?],
    floats: ReadonlyMap<number, number> = new Map(),
  ): void {
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: this.parameterBinding(words, floats) },
      ...buffers.map((resource, index) => ({ binding: index + 1, resource })),
    ];
    const group = this.device.createBindGroup({
      layout: pipelineValue.getBindGroupLayout(0),
      entries,
    });
    const pass = this.encoder.beginComputePass();
    pass.setPipeline(pipelineValue);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1);
    pass.end();
  }
}

export class LedaWebGpuModel {
  readonly limits: WebGpuLimitsSnapshot;
  readonly estimatedGpuBufferBytes: number;
  private readonly embeddingWeight: GPUBuffer;
  private readonly finalNorm: GPUBuffer;
  private readonly ropeTable: GPUBuffer;
  private readonly layers: readonly LayerWeights[];
  private readonly caches: readonly LayerCache[];
  private readonly work: Workspaces;
  private currentLength = 0;
  private destroyed = false;

  private constructor(
    readonly device: GPUDevice,
    private readonly pipelines: Pipelines,
    embeddingWeight: GPUBuffer,
    finalNorm: GPUBuffer,
    ropeTable: GPUBuffer,
    layers: readonly LayerWeights[],
    caches: readonly LayerCache[],
    work: Workspaces,
    limits: WebGpuLimitsSnapshot,
  ) {
    this.embeddingWeight = embeddingWeight;
    this.finalNorm = finalNorm;
    this.ropeTable = ropeTable;
    this.layers = layers;
    this.caches = caches;
    this.work = work;
    this.limits = limits;
    const weightBytes = MODEL.parameterCount * FLOAT_BYTES;
    const cacheBytes = MODEL.numLayers * 2 * MODEL.numKvHeads * MODEL.maxContext * MODEL.headDim * FLOAT_BYTES;
    const hiddenBytes = MODEL.maxContext * MODEL.modelDim * FLOAT_BYTES;
    const kvWorkspaceBytes = MODEL.maxContext * MODEL.numKvHeads * MODEL.headDim * FLOAT_BYTES;
    const mlpBytes = MODEL.maxContext * MODEL.hiddenDim * FLOAT_BYTES;
    const logitsBytes = MODEL.vocabSize * FLOAT_BYTES;
    this.estimatedGpuBufferBytes = weightBytes + cacheBytes +
      6 * hiddenBytes + 2 * kvWorkspaceBytes + 2 * mlpBytes +
      2 * logitsBytes + MODEL.maxContext * 4 + PARAMETER_SLOT_BYTES * PARAMETER_SLOTS +
      MODEL.maxContext * MODEL.headDim * FLOAT_BYTES;
  }

  static async create(artifact: LedaWebArtifact, adapter?: GPUAdapter): Promise<LedaWebGpuModel> {
    let actualAdapter: GPUAdapter;
    if (adapter !== undefined) {
      actualAdapter = adapter;
    } else {
      const capability = await detectWebGpu();
      if (!capability.supported) {
        throw new Error(capability.reason);
      }
      actualAdapter = capability.adapter;
    }
    const largestWeight = Math.max(...Array.from(artifact.tensors.values(), (tensor) => tensor.byteLength));
    const largestWorkspace = MODEL.maxContext * MODEL.hiddenDim * FLOAT_BYTES;
    const requiredBinding = Math.max(largestWeight, largestWorkspace);
    if (actualAdapter.limits.maxStorageBufferBindingSize < requiredBinding ||
        actualAdapter.limits.maxBufferSize < requiredBinding ||
        actualAdapter.limits.maxComputeInvocationsPerWorkgroup < 256 ||
        actualAdapter.limits.maxComputeWorkgroupsPerDimension < MODEL.maxContext) {
      throw new Error("The WebGPU adapter limits are insufficient for Leda Demo v0 Float32 inference");
    }
    const device = await actualAdapter.requestDevice();
    const limits: WebGpuLimitsSnapshot = {
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
      maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
      shaderF16: device.features.has("shader-f16"),
    };
    device.pushErrorScope("validation");
    let scopePopped = false;
    try {
      const pipelinesArray = await Promise.all([
        pipeline(device, "embedding", EMBEDDING_SHADER),
        pipeline(device, "rms norm", RMS_NORM_SHADER),
        pipeline(device, "matmul", MATMUL_SHADER),
        pipeline(device, "residual add", ADD_SHADER),
        pipeline(device, "SwiGLU", SWIGLU_SHADER),
        pipeline(device, "RoPE", ROPE_SHADER),
        pipeline(device, "KV append", CACHE_APPEND_SHADER),
        pipeline(device, "GQA attention", ATTENTION_SHADER),
        pipeline(device, "tied output projection", TIED_OUTPUT_SHADER),
      ]);
      const pipelines: Pipelines = {
        embedding: pipelinesArray[0],
        rmsNorm: pipelinesArray[1],
        matmul: pipelinesArray[2],
        add: pipelinesArray[3],
        swiglu: pipelinesArray[4],
        rope: pipelinesArray[5],
        cacheAppend: pipelinesArray[6],
        attention: pipelinesArray[7],
        tiedOutput: pipelinesArray[8],
      };
      const embeddingWeight = uploadTensor(device, requireTensor(artifact, "token_embedding.weight"));
      const finalNorm = uploadTensor(device, requireTensor(artifact, "final_norm.weight"));
      const ropeValues = new Float32Array(MODEL.maxContext * MODEL.headDim);
      for (let position = 0; position < MODEL.maxContext; ++position) {
        for (let pair = 0; pair < MODEL.headDim / 2; ++pair) {
          const exponent = -2 * pair / MODEL.headDim;
          const angle = position * Math.pow(MODEL.ropeTheta, exponent);
          const offset = (position * (MODEL.headDim / 2) + pair) * 2;
          ropeValues[offset] = Math.cos(angle);
          ropeValues[offset + 1] = Math.sin(angle);
        }
      }
      const ropeTable = device.createBuffer({
        label: "RoPE cosine/sine table",
        size: ropeValues.byteLength,
        usage: GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
      });
      new Float32Array(ropeTable.getMappedRange()).set(ropeValues);
      ropeTable.unmap();
      const layers: LayerWeights[] = [];
      for (let layer = 0; layer < MODEL.numLayers; ++layer) {
        const prefix = `blocks.${layer}`;
        layers.push({
          attentionNorm: uploadTensor(device, requireTensor(artifact, `${prefix}.attention_norm.weight`)),
          query: uploadTensor(device, requireTensor(artifact, `${prefix}.attention.q_proj.weight`)),
          key: uploadTensor(device, requireTensor(artifact, `${prefix}.attention.k_proj.weight`)),
          value: uploadTensor(device, requireTensor(artifact, `${prefix}.attention.v_proj.weight`)),
          attentionOutput: uploadTensor(device, requireTensor(artifact, `${prefix}.attention.out_proj.weight`)),
          queryNorm: uploadTensor(device, requireTensor(artifact, `${prefix}.attention.q_norm.weight`)),
          keyNorm: uploadTensor(device, requireTensor(artifact, `${prefix}.attention.k_norm.weight`)),
          mlpNorm: uploadTensor(device, requireTensor(artifact, `${prefix}.mlp_norm.weight`)),
          gate: uploadTensor(device, requireTensor(artifact, `${prefix}.mlp.gate_proj.weight`)),
          up: uploadTensor(device, requireTensor(artifact, `${prefix}.mlp.up_proj.weight`)),
          down: uploadTensor(device, requireTensor(artifact, `${prefix}.mlp.down_proj.weight`)),
        });
      }
      const cacheBytes = MODEL.numKvHeads * MODEL.maxContext * MODEL.headDim * FLOAT_BYTES;
      const caches = Array.from({ length: MODEL.numLayers }, (_, layer) => ({
        key: createStorage(device, `layer ${layer} key cache`, cacheBytes),
        value: createStorage(device, `layer ${layer} value cache`, cacheBytes),
      }));
      const hiddenBytes = MODEL.maxContext * MODEL.modelDim * FLOAT_BYTES;
      const kvBytes = MODEL.maxContext * MODEL.numKvHeads * MODEL.headDim * FLOAT_BYTES;
      const mlpBytes = MODEL.maxContext * MODEL.hiddenDim * FLOAT_BYTES;
      const logitsBytes = MODEL.vocabSize * FLOAT_BYTES;
      const work: Workspaces = {
        tokens: device.createBuffer({
          label: "input tokens",
          size: MODEL.maxContext * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }),
        hiddenA: createStorage(device, "hidden A", hiddenBytes),
        hiddenB: createStorage(device, "hidden B", hiddenBytes),
        norm: createStorage(device, "normalized hidden", hiddenBytes),
        query: createStorage(device, "query", hiddenBytes),
        key: createStorage(device, "key", kvBytes),
        value: createStorage(device, "value", kvBytes),
        attention: createStorage(device, "attention output", hiddenBytes),
        projection: createStorage(device, "projection", hiddenBytes),
        gate: createStorage(device, "gate", mlpBytes),
        up: createStorage(device, "up", mlpBytes),
        logits: createStorage(device, "logits", logitsBytes),
        readback: device.createBuffer({
          label: "logit readback",
          size: logitsBytes,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        parameters: device.createBuffer({
          label: "dispatch parameters",
          size: PARAMETER_SLOT_BYTES * PARAMETER_SLOTS,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
      };
      const validationError = await device.popErrorScope();
      scopePopped = true;
      if (validationError !== null) {
        throw new Error(`WebGPU initialization failed: ${validationError.message}`);
      }
      return new LedaWebGpuModel(
        device, pipelines, embeddingWeight, finalNorm, ropeTable, layers, caches, work, limits,
      );
    } catch (error) {
      const validationError = scopePopped ? null : await device.popErrorScope();
      device.destroy();
      if (validationError !== null) {
        throw new Error(`WebGPU initialization failed: ${validationError.message}`, { cause: error });
      }
      throw error;
    }
  }

  get cacheLength(): number {
    return this.currentLength;
  }

  async prefill(tokens: readonly number[]): Promise<Float32Array> {
    if (this.currentLength !== 0) {
      throw new Error("A WebGPU inference session must be reset before another prefill");
    }
    if (tokens.length < 1 || tokens.length > MODEL.maxContext) {
      throw new RangeError("Leda prefill length must be in [1,512]");
    }
    const logits = await this.forward(tokens, 0);
    this.currentLength = tokens.length;
    return logits;
  }

  async decode(token: number): Promise<Float32Array> {
    if (this.currentLength < 1) {
      throw new Error("Leda decode requires a completed prefill");
    }
    if (this.currentLength >= MODEL.maxContext) {
      throw new RangeError("Leda KV cache is already at its 512-token capacity");
    }
    const logits = await this.forward([token], this.currentLength);
    ++this.currentLength;
    return logits;
  }

  reset(): void {
    this.currentLength = 0;
  }

  private async forward(tokens: readonly number[], startPosition: number): Promise<Float32Array> {
    if (this.destroyed) {
      throw new Error("Leda WebGPU model has been destroyed");
    }
    if (tokens.some((token) => !Number.isInteger(token) || token < 0 || token >= MODEL.vocabSize)) {
      throw new RangeError("Leda token IDs must be in [0,8192]");
    }
    const rows = tokens.length;
    if (startPosition + rows > MODEL.maxContext) {
      throw new RangeError("Leda invocation exceeds the 512-token context");
    }
    this.device.pushErrorScope("validation");
    let scopePopped = false;
    this.device.queue.writeBuffer(this.work.tokens, 0, Uint32Array.from(tokens));
    const encoder = this.device.createCommandEncoder({ label: "Leda inference" });
    const dispatch = new Dispatches(this.device, encoder, this.work.parameters);
    const binding = (buffer: GPUBuffer, offset = 0, size?: number): GPUBufferBinding =>
      size === undefined ? { buffer, offset } : { buffer, offset, size };

    dispatch.run(
      this.pipelines.embedding,
      [rows, MODEL.modelDim, MODEL.vocabSize],
      [binding(this.work.tokens), binding(this.embeddingWeight), binding(this.work.hiddenA)],
      [ceilDivide(rows * MODEL.modelDim, 256)],
    );

    for (let layerIndex = 0; layerIndex < MODEL.numLayers; ++layerIndex) {
      const layer = this.layers[layerIndex];
      const cache = this.caches[layerIndex];
      if (layer === undefined || cache === undefined) {
        throw new Error("Leda layer/cache inventory invariant failed");
      }
      dispatch.run(
        this.pipelines.rmsNorm,
        [rows, MODEL.modelDim],
        [binding(this.work.hiddenA), binding(layer.attentionNorm), binding(this.work.norm)],
        [rows],
        new Map([[2, MODEL.normEpsilon]]),
      );
      dispatch.run(
        this.pipelines.matmul,
        [rows, MODEL.modelDim, MODEL.modelDim],
        [binding(this.work.norm), binding(layer.query), binding(this.work.query)],
        [ceilDivide(MODEL.modelDim, 16), ceilDivide(rows, 16)],
      );
      dispatch.run(
        this.pipelines.matmul,
        [rows, MODEL.numKvHeads * MODEL.headDim, MODEL.modelDim],
        [binding(this.work.norm), binding(layer.key), binding(this.work.key)],
        [ceilDivide(MODEL.numKvHeads * MODEL.headDim, 16), ceilDivide(rows, 16)],
      );
      dispatch.run(
        this.pipelines.matmul,
        [rows, MODEL.numKvHeads * MODEL.headDim, MODEL.modelDim],
        [binding(this.work.norm), binding(layer.value), binding(this.work.value)],
        [ceilDivide(MODEL.numKvHeads * MODEL.headDim, 16), ceilDivide(rows, 16)],
      );
      dispatch.run(
        this.pipelines.rmsNorm,
        [rows * MODEL.numQueryHeads, MODEL.headDim],
        [binding(this.work.query), binding(layer.queryNorm), binding(this.work.hiddenB)],
        [rows * MODEL.numQueryHeads],
        new Map([[2, MODEL.qkNormEpsilon]]),
      );
      dispatch.run(
        this.pipelines.rmsNorm,
        [rows * MODEL.numKvHeads, MODEL.headDim],
        [binding(this.work.key), binding(layer.keyNorm), binding(this.work.projection)],
        [rows * MODEL.numKvHeads],
        new Map([[2, MODEL.qkNormEpsilon]]),
      );
      dispatch.run(
        this.pipelines.rope,
        [rows, MODEL.numQueryHeads, MODEL.headDim, startPosition],
        [binding(this.work.hiddenB), binding(this.ropeTable)],
        [ceilDivide(rows * MODEL.numQueryHeads * MODEL.headDim / 2, 256)],
      );
      dispatch.run(
        this.pipelines.rope,
        [rows, MODEL.numKvHeads, MODEL.headDim, startPosition],
        [binding(this.work.projection), binding(this.ropeTable)],
        [ceilDivide(rows * MODEL.numKvHeads * MODEL.headDim / 2, 256)],
      );
      dispatch.run(
        this.pipelines.cacheAppend,
        [rows, MODEL.numKvHeads, MODEL.headDim, startPosition, MODEL.maxContext],
        [binding(this.work.projection), binding(cache.key)],
        [ceilDivide(rows * MODEL.numKvHeads * MODEL.headDim, 256)],
      );
      dispatch.run(
        this.pipelines.cacheAppend,
        [rows, MODEL.numKvHeads, MODEL.headDim, startPosition, MODEL.maxContext],
        [binding(this.work.value), binding(cache.value)],
        [ceilDivide(rows * MODEL.numKvHeads * MODEL.headDim, 256)],
      );
      dispatch.run(
        this.pipelines.attention,
        [rows, MODEL.numQueryHeads, MODEL.numKvHeads, MODEL.headDim, startPosition, MODEL.maxContext],
        [binding(this.work.hiddenB), binding(cache.key), binding(cache.value), binding(this.work.attention)],
        [rows, MODEL.numQueryHeads],
      );
      dispatch.run(
        this.pipelines.matmul,
        [rows, MODEL.modelDim, MODEL.modelDim],
        [binding(this.work.attention), binding(layer.attentionOutput), binding(this.work.projection)],
        [ceilDivide(MODEL.modelDim, 16), ceilDivide(rows, 16)],
      );
      dispatch.run(
        this.pipelines.add,
        [rows * MODEL.modelDim],
        [binding(this.work.hiddenA), binding(this.work.projection), binding(this.work.hiddenB)],
        [ceilDivide(rows * MODEL.modelDim, 256)],
      );
      dispatch.run(
        this.pipelines.rmsNorm,
        [rows, MODEL.modelDim],
        [binding(this.work.hiddenB), binding(layer.mlpNorm), binding(this.work.norm)],
        [rows],
        new Map([[2, MODEL.normEpsilon]]),
      );
      dispatch.run(
        this.pipelines.matmul,
        [rows, MODEL.hiddenDim, MODEL.modelDim],
        [binding(this.work.norm), binding(layer.gate), binding(this.work.gate)],
        [ceilDivide(MODEL.hiddenDim, 16), ceilDivide(rows, 16)],
      );
      dispatch.run(
        this.pipelines.matmul,
        [rows, MODEL.hiddenDim, MODEL.modelDim],
        [binding(this.work.norm), binding(layer.up), binding(this.work.up)],
        [ceilDivide(MODEL.hiddenDim, 16), ceilDivide(rows, 16)],
      );
      dispatch.run(
        this.pipelines.swiglu,
        [rows * MODEL.hiddenDim],
        [binding(this.work.gate), binding(this.work.up)],
        [ceilDivide(rows * MODEL.hiddenDim, 256)],
      );
      dispatch.run(
        this.pipelines.matmul,
        [rows, MODEL.modelDim, MODEL.hiddenDim],
        [binding(this.work.gate), binding(layer.down), binding(this.work.projection)],
        [ceilDivide(MODEL.modelDim, 16), ceilDivide(rows, 16)],
      );
      dispatch.run(
        this.pipelines.add,
        [rows * MODEL.modelDim],
        [binding(this.work.hiddenB), binding(this.work.projection), binding(this.work.hiddenA)],
        [ceilDivide(rows * MODEL.modelDim, 256)],
      );
    }
    dispatch.run(
      this.pipelines.rmsNorm,
      [rows, MODEL.modelDim],
      [binding(this.work.hiddenA), binding(this.finalNorm), binding(this.work.norm)],
      [rows],
      new Map([[2, MODEL.normEpsilon]]),
    );
    const lastRowOffset = (rows - 1) * MODEL.modelDim * FLOAT_BYTES;
    dispatch.run(
      this.pipelines.tiedOutput,
      [MODEL.vocabSize, MODEL.modelDim],
      [
        binding(this.work.norm, lastRowOffset, MODEL.modelDim * FLOAT_BYTES),
        binding(this.embeddingWeight),
        binding(this.work.logits),
      ],
      [ceilDivide(MODEL.vocabSize, 256)],
    );
    encoder.copyBufferToBuffer(this.work.logits, 0, this.work.readback, 0, MODEL.vocabSize * FLOAT_BYTES);
    this.device.queue.submit([encoder.finish()]);
    try {
      await this.work.readback.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(MODEL.vocabSize);
      result.set(new Float32Array(this.work.readback.getMappedRange()));
      this.work.readback.unmap();
      const validationError = await this.device.popErrorScope();
      scopePopped = true;
      if (validationError !== null) {
        throw new Error(`WebGPU inference validation failed: ${validationError.message}`);
      }
      return result;
    } catch (error) {
      if (!scopePopped) {
        const validationError = await this.device.popErrorScope();
        if (validationError !== null) {
          throw new Error(`WebGPU inference validation failed: ${validationError.message}`, { cause: error });
        }
      }
      throw error;
    }
  }

  destroy(): void {
    if (!this.destroyed) {
      this.destroyed = true;
      this.device.destroy();
    }
  }
}
