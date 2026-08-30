import { equalHex, sha256Hex } from "./hash.js";

export interface DownloadProgress {
  readonly receivedBytes: number;
  readonly totalBytes?: number;
  readonly fromCache: boolean;
}

export type DownloadProgressHandler = (progress: DownloadProgress) => void;

const CACHE_NAME = "leda-web-model-v1";

function cacheKey(url: string, sha256: string): Request {
  const key = new URL(url, globalThis.location?.href ?? "https://leda.invalid/");
  key.searchParams.set("leda-artifact-sha256", sha256.toLowerCase());
  return new Request(key.toString(), { method: "GET" });
}

async function verified(bytes: ArrayBuffer, expectedSha256: string): Promise<boolean> {
  return equalHex(await sha256Hex(bytes), expectedSha256);
}

async function readWithProgress(
  response: Response,
  progress: DownloadProgressHandler | undefined,
): Promise<ArrayBuffer> {
  if (!response.ok) {
    throw new Error(`Static asset download failed with HTTP ${response.status}`);
  }
  const totalHeader = response.headers.get("content-length");
  const total = totalHeader === null ? undefined : Number(totalHeader);
  if (response.body === null) {
    const bytes = await response.arrayBuffer();
    progress?.({ receivedBytes: bytes.byteLength, ...(total === undefined ? {} : { totalBytes: total }), fromCache: false });
    return bytes;
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    if (value !== undefined) {
      chunks.push(value);
      received += value.byteLength;
      progress?.({ receivedBytes: received, ...(total === undefined ? {} : { totalBytes: total }), fromCache: false });
    }
  }
  const result = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result.buffer;
}

export async function loadVerifiedStaticAsset(
  url: string,
  expectedSha256: string,
  progress?: DownloadProgressHandler,
): Promise<ArrayBuffer> {
  const canCache = typeof caches !== "undefined";
  const key = cacheKey(url, expectedSha256);
  if (canCache) {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(key);
    if (cached !== undefined) {
      const bytes = await cached.arrayBuffer();
      if (await verified(bytes, expectedSha256)) {
        progress?.({ receivedBytes: bytes.byteLength, totalBytes: bytes.byteLength, fromCache: true });
        return bytes;
      }
      await cache.delete(key);
    }
  }

  const bytes = await readWithProgress(await fetch(url, { cache: "no-store" }), progress);
  const actual = await sha256Hex(bytes);
  if (!equalHex(actual, expectedSha256)) {
    throw new Error(`Static asset SHA-256 mismatch: expected ${expectedSha256}, got ${actual}`);
  }
  if (canCache) {
    const cache = await caches.open(CACHE_NAME);
    await cache.put(key, new Response(bytes, {
      headers: {
        "content-type": "application/octet-stream",
        "x-leda-sha256": actual,
      },
    }));
  }
  return bytes;
}

export async function invalidateLedaWebCache(): Promise<boolean> {
  return typeof caches !== "undefined" && caches.delete(CACHE_NAME);
}
