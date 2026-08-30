# Leda

Leda is an open science and biomedical language-model project built on
[Spar](https://github.com/bzzling/spar), the dependency-free C++23 deep-learning framework.

```text
Leda
  ↓
Spar
```

Leda owns model configuration, research recipes, and inference behavior. Spar owns reusable
tensors, autograd, Transformer primitives, checkpoint/tokenizer formats, and CPU/CUDA execution.
Spar never depends on Leda.

## Leda Demo v0

The frozen Demo-v0 model is a 40,385,024-Parameter dense causal decoder: 12 layers, model dimension
512, SwiGLU dimension 1536, eight query heads, two KV heads, pre-norm RMSNorm, QK-Norm, RoPE,
grouped-query attention, and tied token embedding/output projection. Its vocabulary has 8,192 raw
byte-BPE tokens plus model-level EOD token 8192. Supported context is exactly 512 tokens.

The model received 99,998,784 context-128 objective tokens followed by 14,998,872 context-512
adaptation tokens. It is a small scientific autocomplete model—not an instruction-following
assistant or a released large pretrained model.

Reference-development presets `leda_tiny` and `leda_small` remain available for correctness tests.
See [the Leda v0 specification](docs/leda_v0.md) for architecture, historical reference training,
checkpoint ownership, and limitations.

## Build

With sibling source checkouts:

```bash
cmake -S leda -B leda/build -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DSPAR_SOURCE_DIR=../spar \
  -DLEDA_BUILD_TOOLS=ON
cmake --build leda/build
```

To consume an installed Spar package, omit `SPAR_SOURCE_DIR` and set `CMAKE_PREFIX_PATH` to the Spar
installation prefix. The public repository builds without the gitignored local test/example trees.

## Scientific autocomplete

`leda_complete` loads the unchanged `SPARCKPT v1` checkpoint and `SPARTOKN v1` tokenizer:

```bash
leda/build/tools/leda_complete \
  --checkpoint checkpoint-56550.sparckpt \
  --tokenizer tokenizer-8192.spartokn \
  --prompt "Activation of CD8+ T cells requires" \
  --max-new-tokens 64 \
  --temperature 0.7 \
  --top-p 0.9 \
  --seed 42 \
  --show-top 10
```

Use `--greedy` for argmax decoding and `--device cuda` on a CUDA-enabled Spar build. Prompt plus
completion must fit within 512 tokens; the tool never truncates silently. Generated raw bytes are
streamed without splitting valid UTF-8 code points, and EOD is never passed to tokenizer decoding.

The frozen public autocomplete default is temperature `0.7`, top-k `50`, top-p `0.90`, and 32 new
tokens. It was selected on validation diagnostics; greedy decoding remains available but is not the
default because it repeats pathologically. See [the Phase-33 evaluation protocol](docs/phase33_evaluation.md)
for the exact metric and TEST-freeze procedure.

## Client-side WebGPU runtime

The inference-only runtime in [`web/`](web/) executes the frozen Demo-v0 graph directly with our
own TypeScript controller and WGSL compute shaders. It runs in a Web Worker and uses device-resident
GQA K/V caches; it does not port Spar's training stack or use a third-party model runtime. A native
export tool converts the authoritative `SPARCKPT v1` checkpoint into the deterministic,
model-only `LEDAWEB v1` browser artifact.

This is the Phase-34 runtime and diagnostic harness, not the final website. It requires a secure
context and a WebGPU-capable browser. Once the immutable runtime, tokenizer, and model assets have
loaded, generation makes no network requests and prompt text stays in the browser. See
[the Phase-34 architecture and validation report](docs/phase34_webgpu.md).
