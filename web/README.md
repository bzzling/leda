# Leda browser inference runtime

This directory contains the purpose-built, inference-only WebGPU runtime for the frozen Leda Demo
v0 model. It uses TypeScript for model/session control and our own WGSL compute shaders. It does not
port Spar's training stack and does not depend on ONNX, Transformers.js, TensorFlow.js, WebLLM, or
another model runtime.

The runtime executes in a Web Worker. Initialization downloads and verifies immutable `LEDAWEB v1`
model weights and the frozen `SPARTOKN v1` tokenizer, uploads weights to WebGPU, and exposes progress
events. Once initialization finishes, completion performs no network requests. K/V state remains on
the GPU in unexpanded `[2, 512, 64]` grouped-query form for every layer.

This is a runtime and diagnostic harness, not the Phase-35 website.

## Build and test

```sh
npm install
npm test
```

To include validation of a local frozen model export:

```sh
LEDA_WEB_ARTIFACT=/path/to/leda-demo-v0.ledaweb npm test
```

The minimal local browser harness can serve explicit, untracked artifacts:

```sh
npm run serve -- \
  --model /path/to/leda-demo-v0.ledaweb \
  --tokenizer /path/to/leda-tokenizer.spartokn \
  --reference /path/to/native-reference-v1.ledaref
```

Then open `http://127.0.0.1:4173/dev/`. Localhost is treated as a secure context by browsers. The
runtime otherwise requires static HTTPS hosting and a WebGPU-capable browser.

`http://127.0.0.1:4173/dev/verify.html` runs the complete native-vs-WebGPU reference matrix in a
Worker when the optional reference fixture is supplied.

## Frozen identities

- source checkpoint SHA-256: `a94e2a15977c811b2bc8a7bb1e87be45510e3d350f08b21ef599644ba3fd1bed`
- tokenizer SHA-256: `04345687338f13b1b8e8d219dae265094a708bf8d1b409541748cb2d89424970`
- Float32 LEDAWEB SHA-256: `2afdc9c772af3f6f6cceb71506e98e528cded43143c927304875614f04e7adac`

Large model/tokenizer artifacts are deliberately not versioned here.

## LEDAWEB v1

The native `leda_export_web` tool deterministically writes an inference-only binary:

1. a fixed 192-byte little-endian header containing format identity, exact model configuration,
   section offsets, and the 32-byte source-checkpoint digest;
2. eight-byte-aligned tensor records containing name, rank, dtype, element count, payload offset,
   and dimensions;
3. a 64-byte-aligned sequence of raw little-endian Float32 parameter tensors.

Version 1 contains 134 tensors and exactly 40,385,024 values. It has no optimizer state, gradient,
iterator, RNG, or training-progress data. `SPARCKPT v1` remains unchanged and authoritative for
native training/checkpointing.
