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

Current work is local C++ inference and evaluation. There is no website or serving API yet.
