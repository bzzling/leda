import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { parseLedaWebArtifact } from "../src/artifact.js";
import { FROZEN_F32_ARTIFACT_SHA256, MODEL } from "../src/constants.js";

test("LEDAWEB rejects short and corrupt artifacts", async () => {
  await assert.rejects(parseLedaWebArtifact(new ArrayBuffer(10)), /shorter/);
  await assert.rejects(parseLedaWebArtifact(new ArrayBuffer(192)), /magic/);
});

test("the locally exported frozen artifact parses exactly when available", async (context) => {
  const path = process.env.LEDA_WEB_ARTIFACT;
  if (path === undefined) {
    context.skip("set LEDA_WEB_ARTIFACT to run the 154 MiB artifact test");
    return;
  }
  const file = await readFile(path);
  const bytes = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
  const artifact = await parseLedaWebArtifact(bytes, FROZEN_F32_ARTIFACT_SHA256);
  assert.equal(artifact.tensors.size, MODEL.tensorCount);
  assert.equal(artifact.sourceCheckpointSha256.length, 64);

  const corrupt = bytes.slice(0);
  const values = new Uint8Array(corrupt);
  values[values.length - 1] = (values[values.length - 1] ?? 0) ^ 1;
  await assert.rejects(parseLedaWebArtifact(corrupt, FROZEN_F32_ARTIFACT_SHA256), /SHA-256 mismatch/);
});
