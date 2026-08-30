# Leda v0 reference specification

Leda is a science and biomedical language-model family built with the separate Spar framework.
Version 0 is a correctness-first reference definition, not a competitive pretrained model or a
production-scale claim.

## Architecture

Leda v0 is a dense causal decoder-only Transformer:

```text
token embedding
N × pre-norm Transformer block
  RMSNorm
  grouped-query causal self-attention
    QK-Norm
    rotary position embeddings (RoPE)
  residual connection
  RMSNorm
  SwiGLU MLP
  residual connection
final RMSNorm
tied token-embedding projection
vocabulary logits
```

The reference presets use no attention or MLP biases. The output projection reuses the token
embedding Parameter and is not counted twice. Leda owns one ordinary `spar::nn::DecoderLM`; its
configuration is mapped field by field and it does not duplicate the neural-network implementation.

## Project and API boundary

Leda's public modules are `leda`, `leda.config`, `leda.model`, `leda.inference`, and
`leda.pretraining`, with public types in namespace `leda`. They consume only exported Spar modules.
Canonical model Parameter names
remain those of `DecoderLM`, such as `token_embedding.weight`, `blocks.0...`, and
`final_norm.weight`, without wrapper prefixes.

## Tokenization and document boundaries

Leda v0 uses Spar's raw-byte BPE tokenizer. Tokenizer artifacts remain `SPARTOKN v1` and contain raw
bytes plus learned merges only. The packed data layer reserves one model-level EOD token at
`tokenizer.vocab_size()`, so the decoder vocabulary has one additional entry. Every source document,
including an empty or final document, receives exactly one EOD. Reference windows may attend across
an EOD boundary; there is no boundary-reset mask in v0.

## Reference presets

These are CPU reference-development presets:

| preset | model dim | hidden dim | layers | query heads | KV heads | QK-Norm |
|---|---:|---:|---:|---:|---:|---:|
| `leda_tiny` | 128 | 384 | 4 | 4 | 2 | yes |
| `leda_small` | 256 | 768 | 8 | 8 | 2 | yes |

Both default to Float32, RMSNorm epsilon `1e-5`, QK-Norm epsilon `1e-6`, and RoPE theta `10000`.

The frozen `leda_demo_v0` inference target uses vocabulary 8193, model dimension 512, hidden
dimension 1536, 12 layers, eight query heads, two KV heads, and Float32. It has exactly 40,385,024
unique Parameters and an explicitly supported context of 512 tokens.

For a representative model vocabulary of 257, the independently audited no-bias parameter formula
is:

```text
V*D
+ L * (2*D*D + 2*D*(D*Hkv/Hq) + 2*(D/Hq) + 3*D*hidden + 2*D)
+ D
```

This gives 820,736 Parameters for `leda_tiny` and 6,099,968 for `leda_small`, including the token
embedding once and the final RMSNorm.

## Reference pretraining recipe

The deterministic reference run splits documents before tokenization, trains ByteBPE on training
documents only, and constructs independent training and validation corpora:

```text
sequence length:            16
stride:                     12
microbatch size:            1
accumulation steps:         2
AdamW beta1 / beta2:        0.9 / 0.95
AdamW epsilon:              1e-8
weight decay:               0.01
maximum gradient norm:      1.0
learning-rate schedule:     linear warmup then cosine decay
peak / minimum LR:          2e-3 / 2e-4
warmup / decay steps:       2 / 12
```

Each microbatch uses Sum-reduced language-model cross-entropy. Gradients are divided once by the
actual target count, clipped once, and passed to AdamW. `TrainingProgress.global_step` counts
optimizer updates and `tokens_seen` counts objective targets. Final partial accumulation groups are
retained.

The executable uses 12 original synthetic science-prose documents, holds out two before tokenizer
training, performs 12 updates, evaluates at the beginning, midpoint, and end, and performs a clean
checkpoint round-trip after update 6. Fixed seeds govern splitting, shuffling, and initialization.

One Phase 19 Release run with tokenizer vocabulary 272 and model vocabulary 273 reduced full
training CE from 5.814369 to 4.183725 and held-out CE from 5.809016 to 4.292330 over 12 updates and
360 objective targets. Loss values are deterministic expectations; wall-clock timing is diagnostic.

## Frozen Demo-v0 artifact

The clean base-pretraining artifact at approximately 100M context-128 objective tokens is preserved.
The frozen inference target continued for exactly 14,998,872 context-512 objective tokens, reaching
114,997,656 total objective tokens. Its final held-out validation cross-entropies were 3.78729 on
common-128 and 3.50241 on native-512. These are historical research measurements; the frozen TEST
split remained untouched.

The final checkpoint is `checkpoint-56550.sparckpt`, SHA-256
`a94e2a15977c811b2bc8a7bb1e87be45510e3d350f08b21ef599644ba3fd1bed`. Checkpoints and tokenizer
binaries are intentionally not committed.

## Checkpoint compatibility

Checkpointing remains `SPARCKPT v1`. Leda v0 checkpoints its underlying DecoderLM, and compatibility
is defined by the mapped `spar::nn::DecoderConfig` and canonical DecoderLM Parameters. Loading first
constructs generic Spar state, then `leda::Leda::from_decoder` requires exact equality between that
checkpoint configuration and `leda::decoder_config(LedaConfig)`. Iterator state remains an adjacent
caller-owned value. This validation is not weakened by the repository extraction.

## Memory and performance constraints

The reported persistent AdamW estimate includes Parameter, gradient, first-moment, and second-moment
storage. It excludes activations, autograd graph metadata, temporary operator tensors, allocator
overhead, and data batches.

The attention score tensor scales approximately as `B × Hq × T × T` elements. On the Phase 19 arm64
macOS Release baseline, one `leda_tiny` update took approximately 1.05, 2.15, and 4.61 seconds at
sequence lengths 32, 64, and 128. Backward occupied about 67% of update time, while isolated causal
attention exhibited the expected quadratic trend. These are observations, not benchmark contracts.

## Phase-32 inference semantics

Uncached complete-prefix forward remains the correctness oracle. Cached inference owns one
fixed-capacity K/V cache per decoder layer. K/V remain unexpanded in `[B,Hkv,S,Dh]` form, append only
new positions, stay on their model Device, and are never Parameters or checkpoint/state-dict
entries. Prefill begins at RoPE position zero; each single-token decode uses the previous cache
length as its position and attends to the complete cached prefix plus itself.

Spar's thread-local RAII inference mode suppresses graph construction without mutating Parameter
metadata. Leda sampling is application behavior: explicit greedy or deterministic seeded
temperature sampling, then top-k, softmax, top-p, renormalization, and sampling. EOD and the 512-token
limit are strict.

## Current limitations

- full quadratic attention during prefill and uncached oracle execution
- Float32 Demo-v0 weights; no BF16 or quantized inference
- no distributed training
- no FlashAttention or fused Transformer kernels
- no continuous batching, paged attention, or speculative decoding
- no generation server or website
- no instruction tuning; output is scientific autocomplete rather than question answering

These limitations are deliberate. The reference implementation remains the correctness oracle for
future Spar backend and kernel work.
