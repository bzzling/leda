import assert from "node:assert/strict";
import test from "node:test";

import { greedyToken, sampleToken, samplingProbabilities, SplitMix64, topCandidates } from "../src/sampling.js";

test("SplitMix64 matches the native Spar sequence", () => {
  const random = new SplitMix64(0);
  assert.equal(random.nextU64(), 0xe220a8397b1dcdafn);
  assert.equal(random.nextU64(), 0x6e789e6aa1b965f4n);
  assert.equal(random.nextU64(), 0x06c45d188009454fn);
});

test("greedy ties use the smaller token ID", () => {
  assert.equal(greedyToken([1, 3, 3, 2]), 1);
});

test("top-k then top-p retains the smallest sufficient descending prefix", () => {
  const probabilities = samplingProbabilities([4, 3, 2, 1], 1, 3, 0.8);
  assert.ok(probabilities[0]! > probabilities[1]!);
  assert.equal(probabilities[2], 0);
  assert.equal(probabilities[3], 0);
  assert.ok(Math.abs(probabilities[0]! + probabilities[1]! - 1) < 1e-15);
});

test("seeded samples and top candidates are deterministic", () => {
  const logits = [0.1, 0.2, 3, 2, -1];
  const first = sampleToken(logits, new SplitMix64(42), { temperature: 0.7, topK: 3, topP: 0.9 });
  const second = sampleToken(logits, new SplitMix64(42), { temperature: 0.7, topK: 3, topP: 0.9 });
  assert.equal(first, second);
  assert.deepEqual(topCandidates(logits, 3).map((item) => item.token), [2, 3, 1]);
});
