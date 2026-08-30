import assert from "node:assert/strict";
import test from "node:test";

import { completeText, type CachedLogitModel } from "../src/completion.js";
import { MODEL } from "../src/constants.js";
import { ByteBPETokenizer } from "../src/tokenizer.js";

function bytesOnlyTokenizer(): ByteBPETokenizer {
  const bytes = new ArrayBuffer(24);
  new Uint8Array(bytes).set(new TextEncoder().encode("SPARTOKN"));
  const view = new DataView(bytes);
  view.setUint32(8, 1, true);
  view.setUint32(12, 256, true);
  view.setBigUint64(16, 0n, true);
  return ByteBPETokenizer.parse(bytes);
}

class FakeModel implements CachedLogitModel {
  cacheLength = 0;
  readonly calls: string[] = [];

  async prefill(tokens: readonly number[]): Promise<Float32Array> {
    this.calls.push(`prefill:${tokens.length}`);
    this.cacheLength = tokens.length;
    return this.logits(65);
  }

  async decode(token: number): Promise<Float32Array> {
    this.calls.push(`decode:${token}`);
    ++this.cacheLength;
    return this.logits(this.cacheLength % 2 === 0 ? 66 : 65);
  }

  reset(): void {
    this.calls.push("reset");
    this.cacheLength = 0;
  }

  private logits(best: number): Float32Array {
    const values = new Float32Array(MODEL.vocabSize).fill(-10);
    values[best] = 10;
    return values;
  }
}

class EodModel extends FakeModel {
  override async prefill(tokens: readonly number[]): Promise<Float32Array> {
    await super.prefill(tokens);
    const values = new Float32Array(MODEL.vocabSize).fill(-10);
    values[MODEL.eodToken] = 10;
    return values;
  }
}

test("completion is strict, deterministic, sequentially reusable, and performs no fetch", async () => {
  const originalFetch = globalThis.fetch;
  let fetches = 0;
  globalThis.fetch = (() => {
    ++fetches;
    throw new Error("Inference must not fetch");
  }) as typeof fetch;
  try {
    const model = new FakeModel();
    const tokenizer = bytesOnlyTokenizer();
    const first = await completeText(model, tokenizer, "x", { greedy: true, maxNewTokens: 3 });
    const second = await completeText(model, tokenizer, "y", { greedy: true, maxNewTokens: 2 });
    assert.deepEqual(first.generatedTokens, [65, 66, 65]);
    assert.deepEqual(second.generatedTokens, [65, 66]);
    assert.equal(new TextDecoder().decode(first.generatedBytes), "ABA");
    assert.equal(fetches, 0);
    assert.deepEqual(model.calls, [
      "reset", "prefill:1", "decode:65", "decode:66",
      "reset", "prefill:1", "decode:65",
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("completion rejects context overflow before model execution", async () => {
  const model = new FakeModel();
  await assert.rejects(
    completeText(model, bytesOnlyTokenizer(), "x".repeat(512), { greedy: true, maxNewTokens: 1 }),
    /512-token context/,
  );
  assert.deepEqual(model.calls, []);
});

test("completion cancellation is observed between token steps", async () => {
  const model = new FakeModel();
  let emitted = 0;
  await assert.rejects(
    completeText(
      model,
      bytesOnlyTokenizer(),
      "x",
      { greedy: true, maxNewTokens: 3 },
      () => emitted === 1,
      () => { ++emitted; },
    ),
    (error: unknown) => error instanceof DOMException && error.name === "AbortError",
  );
  assert.equal(emitted, 1);
});

test("EOD stops generation without passing model-only EOD through the tokenizer", async () => {
  const result = await completeText(
    new EodModel(), bytesOnlyTokenizer(), "x", { greedy: true, maxNewTokens: 8 },
  );
  assert.equal(result.stoppedByEod, true);
  assert.deepEqual(result.generatedTokens, []);
  assert.equal(result.generatedBytes.length, 0);
});
