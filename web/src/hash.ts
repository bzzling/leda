export function bytesToHex(bytes: Uint8Array): string {
  let result = "";
  for (const value of bytes) {
    result += value.toString(16).padStart(2, "0");
  }
  return result;
}

export async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return bytesToHex(new Uint8Array(digest));
}

export function equalHex(actual: string, expected: string): boolean {
  return actual.toLowerCase() === expected.toLowerCase();
}
