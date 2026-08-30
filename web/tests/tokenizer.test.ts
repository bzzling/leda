import assert from "node:assert/strict";
import test from "node:test";

import { ByteBPETokenizer, StreamingTokenDecoder } from "../src/tokenizer.js";

function tokenizerArtifact(merges: readonly (readonly [number, number])[]): ArrayBuffer {
  const bytes = new ArrayBuffer(24 + merges.length * 8);
  const output = new Uint8Array(bytes);
  output.set(new TextEncoder().encode("SPARTOKN"));
  const view = new DataView(bytes);
  view.setUint32(8, 1, true);
  view.setUint32(12, 256, true);
  view.setBigUint64(16, BigInt(merges.length), true);
  merges.forEach(([left, right], rank) => {
    view.setUint32(24 + rank * 8, left, true);
    view.setUint32(28 + rank * 8, right, true);
  });
  return bytes;
}

test("SPARTOKN parses, merges by rank/position, and round-trips arbitrary bytes", () => {
  const tokenizer = ByteBPETokenizer.parse(tokenizerArtifact([
    [97, 98],
    [256, 99],
    [257, 257],
  ]));
  assert.equal(tokenizer.vocabularySize, 259);
  assert.deepEqual(tokenizer.encodeText("abcabc"), [258]);
  assert.deepEqual(tokenizer.decodeBytes([258]), new TextEncoder().encode("abcabc"));

  const arbitrary = Uint8Array.of(0, 255, 128, 0, 97);
  const encoded = tokenizer.encodeBytes(arbitrary);
  assert.deepEqual(tokenizer.decodeBytes(encoded), arbitrary);
});

test("overlapping equal-rank pairs merge leftmost first", () => {
  const tokenizer = ByteBPETokenizer.parse(tokenizerArtifact([[97, 97], [256, 97]]));
  assert.deepEqual(tokenizer.encodeText("aaa"), [257]);
});

test("streaming output preserves UTF-8 split across token boundaries", () => {
  const decoder = new StreamingTokenDecoder();
  assert.equal(decoder.push(Uint8Array.of(0xce)), "");
  assert.equal(decoder.push(Uint8Array.of(0xb2)), "β");
  assert.equal(decoder.finish(), "");
});

test("SPARTOKN rejects malformed artifacts", () => {
  assert.throws(() => ByteBPETokenizer.parse(new ArrayBuffer(3)), /shorter/);
  const invalid = tokenizerArtifact([[256, 0]]);
  assert.throws(() => ByteBPETokenizer.parse(invalid), /not yet defined/);
});
