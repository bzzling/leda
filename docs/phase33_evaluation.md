# Scientific-autocomplete evaluation protocol

Phase 33 evaluates the frozen Leda Demo v0 checkpoint as a causal scientific-text continuation
model. It is not evaluated as a chatbot, instruction model, factual question-answering system, or
medical decision aid. No training, tokenizer, corpus, architecture, checkpoint, Spar, or production
Leda inference changes are permitted by this protocol.

## Frozen model identity

- checkpoint: `checkpoint-56550.sparckpt`
- checkpoint SHA-256: `a94e2a15977c811b2bc8a7bb1e87be45510e3d350f08b21ef599644ba3fd1bed`
- tokenizer format: `SPARTOKN v1`
- tokenizer SHA-256: `04345687338f13b1b8e8d219dae265094a708bf8d1b409541748cb2d89424970`
- corpus fingerprint: `d921f74cd244df7097529c2b9121ee398ad1c0b4d82e3d78e9bfdf4ca4abb1a3`
- unique Parameters: 40,385,024
- supported context: 512 tokens

The corpus fingerprint above is copied from the frozen corpus record. Evaluation tooling verifies
the actual checkpoint and tokenizer file hashes before use.

## Quantitative procedure

Documents reset context. Every in-document target after the first token is scored exactly once by
overlapping consecutive 512-token chunks by one token. Token-weighted cross entropy and
`exp(cross_entropy)` perplexity are reported globally and by the six frozen broad domains.
Confidence intervals use 10,000 percentile bootstrap draws over whole documents with seed 3302;
windows are never treated as independent bootstrap units.

Next-token metrics use the 20 lowest stable hash-keyed full chunks in each domain under selection
seed 3301: 61,320 positions total. The metrics are top-1/top-5/top-10 accuracy, mean probability of
the correct token, and mean distribution entropy.

Teacher-forced autocomplete uses 64 identical true continuation targets for eight deterministic
positions per domain at prefix lengths 32, 64, 128, 256, and 400. At 256 and 400, the same target is
rescored using only the final 32, 64, 128, 256, or 400 available prefix tokens. This measures the
benefit of available context without claiming causal understanding.

## Frozen decoding default

The public default is:

```text
temperature:    0.7
top-k:          50
top-p:          0.90
max_new_tokens: 32
```

Sampling order remains temperature scaling, top-k, softmax, top-p, renormalization, then sampling.
The validation grid contained greedy decoding and four compact sampled alternatives. At 32 tokens,
the selected recipe reduced mean repeated 3-gram fraction from 0.363 under greedy to 0.132, reduced
the mean longest repeated span from 12.17 tokens to 5.00, and had no premature EOD among 126 seeded
samples. Temperature 0.8 was more diverse but more prone to abrupt topic changes in the fixed
qualitative suite. Ordinary sampling sufficiently reduced repetition, so no repetition penalty or
no-repeat n-gram rule is introduced.

The committed prompt suite contains 42 original prompts balanced across immunology/biomedicine,
molecular biology, chemistry, physics, mathematics, CS/ML, and scientific explanatory prose.
Generation seeds are 3303, 3304, and 3305. Generated outputs and raw metric tables are experiment
artifacts and are not committed.

## TEST freeze

Before TEST is opened, `autocomplete_study.py freeze-spec` records the artifact hashes, corpus
fingerprint, context, selected decoding recipe, selection rules, metric definitions, precision, and
the exact committed evaluation-tool SHA. The canonical JSON specification is SHA-256 hashed. TEST
is then evaluated once with that specification. No model, recipe, or evaluation tuning is allowed
after inspecting its results.

## Available checkpoint story

Compatible preserved checkpoints exist at 99,998,784, 104,998,408, 109,998,032, and 114,997,656
objective tokens. Earlier approximately 1M, 5M, 25M, and 50M artifacts are not available and will
not be recreated. Comparison is restricted to checkpoints with identical model geometry,
tokenizer, and corpus identity.

## Phase-34 data-bundle plan

Phase 34 should generate one small deterministic JSON bundle from the frozen results containing
architecture numbers, training-token milestones, corpus composition, global/domain CE, held-out
TEST metrics, long-context ablations, fixed representative completions, next-token candidates,
performance, and limitations. Checkpoints, tokenizer binaries, corpora, raw tables, and large output
collections remain external artifacts.
