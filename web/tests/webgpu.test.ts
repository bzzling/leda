import assert from "node:assert/strict";
import test from "node:test";

import { detectWebGpu } from "../src/webgpu.js";

test("WebGPU capability detection fails cleanly without a browser GPU", async () => {
  const capability = await detectWebGpu();
  assert.equal(capability.supported, false);
  if (!capability.supported) {
    assert.match(capability.reason, /WebGPU/);
  }
});
