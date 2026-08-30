import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";

import { parseReferenceFixture } from "../src/reference.js";

const REFERENCE_SHA256 = "49812a582c8d12b7e415c1e7583e4dff6f68a74939b99697d1c0a0519af3cd0b";

test("the committed LEDAREF fixture has the exact frozen inventory and identity", async () => {
  const file = await readFile(resolve("tests/fixtures/native-reference-v1.ledaref"));
  const bytes = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
  const fixture = await parseReferenceFixture(bytes, REFERENCE_SHA256);
  assert.equal(fixture.tokenizerCases.length, 7);
  assert.deepEqual(fixture.modelCases.map((item) => item.promptTokens.length),
    [1, 17, 32, 127, 128, 256, 400, 500]);
  assert.equal(fixture.modelCases.at(-1)?.stages.length, 13);

  const corrupted = bytes.slice(0);
  const view = new Uint8Array(corrupted);
  view[view.length - 1] = (view[view.length - 1] ?? 0) ^ 1;
  await assert.rejects(parseReferenceFixture(corrupted, REFERENCE_SHA256), /SHA-256 mismatch/);
});
