const SPARTOKN_MAGIC = "SPARTOKN";
const SPARTOKN_VERSION = 1;
const BASE_VOCABULARY = 256;
const HEADER_BYTES = 24;

export type TokenId = number;

interface Merge {
  readonly left: TokenId;
  readonly right: TokenId;
}

interface Node {
  token: TokenId;
  previous: number;
  next: number;
  alive: boolean;
}

interface Candidate {
  readonly rank: number;
  readonly left: number;
  readonly right: number;
}

const NO_NODE = -1;

function pairKey(left: TokenId, right: TokenId): bigint {
  return (BigInt(left) << 32n) | BigInt(right);
}

class CandidateHeap {
  private readonly values: Candidate[] = [];

  get empty(): boolean {
    return this.values.length === 0;
  }

  private earlier(left: Candidate, right: Candidate): boolean {
    return left.rank < right.rank ||
      (left.rank === right.rank &&
        (left.left < right.left || (left.left === right.left && left.right < right.right)));
  }

  push(value: Candidate): void {
    let index = this.values.length;
    this.values.push(value);
    while (index !== 0) {
      const parent = Math.floor((index - 1) / 2);
      const parentValue = this.values[parent];
      if (parentValue === undefined || this.earlier(parentValue, value)) {
        break;
      }
      this.values[index] = parentValue;
      index = parent;
    }
    this.values[index] = value;
  }

  pop(): Candidate {
    const result = this.values[0];
    const tail = this.values.pop();
    if (result === undefined || tail === undefined) {
      throw new Error("Internal tokenizer candidate heap underflow");
    }
    if (this.values.length !== 0) {
      let index = 0;
      while (true) {
        const left = index * 2 + 1;
        if (left >= this.values.length) {
          break;
        }
        const right = left + 1;
        let child = left;
        const leftValue = this.values[left];
        const rightValue = this.values[right];
        if (leftValue === undefined) {
          throw new Error("Internal tokenizer candidate heap invariant failed");
        }
        if (rightValue !== undefined && this.earlier(rightValue, leftValue)) {
          child = right;
        }
        const childValue = this.values[child];
        if (childValue === undefined || this.earlier(tail, childValue)) {
          break;
        }
        this.values[index] = childValue;
        index = child;
      }
      this.values[index] = tail;
    }
    return result;
  }
}

export class ByteBPETokenizer {
  private readonly mergeRanks = new Map<bigint, number>();
  private readonly pieces: readonly Uint8Array[];

  private constructor(readonly merges: readonly Merge[]) {
    const pieces: Uint8Array[] = [];
    for (let byte = 0; byte < BASE_VOCABULARY; ++byte) {
      pieces.push(Uint8Array.of(byte));
    }
    for (let rank = 0; rank < merges.length; ++rank) {
      const merge = merges[rank];
      if (merge === undefined || merge.left >= BASE_VOCABULARY + rank ||
          merge.right >= BASE_VOCABULARY + rank) {
        throw new Error(`SPARTOKN merge ${rank} references a token not yet defined`);
      }
      const key = pairKey(merge.left, merge.right);
      if (this.mergeRanks.has(key)) {
        throw new Error(`SPARTOKN merge ${rank} duplicates an earlier pair`);
      }
      this.mergeRanks.set(key, rank);
      const left = pieces[merge.left];
      const right = pieces[merge.right];
      if (left === undefined || right === undefined) {
        throw new Error("SPARTOKN piece construction invariant failed");
      }
      const piece = new Uint8Array(left.length + right.length);
      piece.set(left);
      piece.set(right, left.length);
      pieces.push(piece);
    }
    this.pieces = pieces;
  }

  static parse(bytes: ArrayBuffer): ByteBPETokenizer {
    if (bytes.byteLength < HEADER_BYTES) {
      throw new Error("SPARTOKN artifact is shorter than its header");
    }
    const view = new DataView(bytes);
    const magic = String.fromCharCode(...new Uint8Array(bytes, 0, 8));
    if (magic !== SPARTOKN_MAGIC) {
      throw new Error("Invalid SPARTOKN artifact magic");
    }
    if (view.getUint32(8, true) !== SPARTOKN_VERSION) {
      throw new Error("Unsupported SPARTOKN artifact version");
    }
    if (view.getUint32(12, true) !== BASE_VOCABULARY) {
      throw new Error("SPARTOKN base vocabulary must contain 256 bytes");
    }
    const mergeCountBig = view.getBigUint64(16, true);
    if (mergeCountBig > BigInt(0xffff_ffff - BASE_VOCABULARY + 1)) {
      throw new Error("SPARTOKN merge count exceeds the TokenId range");
    }
    const mergeCount = Number(mergeCountBig);
    if (bytes.byteLength !== HEADER_BYTES + mergeCount * 8) {
      throw new Error("SPARTOKN artifact size does not match its merge count");
    }
    const merges: Merge[] = [];
    for (let rank = 0; rank < mergeCount; ++rank) {
      const offset = HEADER_BYTES + rank * 8;
      merges.push({ left: view.getUint32(offset, true), right: view.getUint32(offset + 4, true) });
    }
    return new ByteBPETokenizer(merges);
  }

  get vocabularySize(): number {
    return this.pieces.length;
  }

  encodeText(text: string): TokenId[] {
    return this.encodeBytes(new TextEncoder().encode(text));
  }

  encodeBytes(bytes: Uint8Array): TokenId[] {
    if (bytes.length === 0) {
      return [];
    }
    const nodes: Node[] = Array.from(bytes, (byte, index) => ({
      token: byte,
      previous: index === 0 ? NO_NODE : index - 1,
      next: index + 1 === bytes.length ? NO_NODE : index + 1,
      alive: true,
    }));
    const candidates = new CandidateHeap();
    const enqueue = (leftIndex: number): void => {
      if (leftIndex === NO_NODE) {
        return;
      }
      const left = nodes[leftIndex];
      if (left === undefined || !left.alive || left.next === NO_NODE) {
        return;
      }
      const right = nodes[left.next];
      if (right === undefined || !right.alive) {
        return;
      }
      const rank = this.mergeRanks.get(pairKey(left.token, right.token));
      if (rank !== undefined) {
        candidates.push({ rank, left: leftIndex, right: left.next });
      }
    };
    for (let index = 0; index + 1 < nodes.length; ++index) {
      enqueue(index);
    }
    while (!candidates.empty) {
      const candidate = candidates.pop();
      const left = nodes[candidate.left];
      const right = nodes[candidate.right];
      if (left === undefined || right === undefined || !left.alive || !right.alive ||
          left.next !== candidate.right ||
          this.mergeRanks.get(pairKey(left.token, right.token)) !== candidate.rank) {
        continue;
      }
      left.token = BASE_VOCABULARY + candidate.rank;
      left.next = right.next;
      right.alive = false;
      if (right.next !== NO_NODE) {
        const next = nodes[right.next];
        if (next === undefined) {
          throw new Error("Internal tokenizer linked-list invariant failed");
        }
        next.previous = candidate.left;
      }
      enqueue(left.previous);
      enqueue(candidate.left);
    }
    const result: TokenId[] = [];
    for (let index = 0; index !== NO_NODE;) {
      const node = nodes[index];
      if (node === undefined || !node.alive) {
        throw new Error("Internal tokenizer output traversal invariant failed");
      }
      result.push(node.token);
      index = node.next;
    }
    return result;
  }

  tokenBytes(token: TokenId): Uint8Array {
    const piece = this.pieces[token];
    if (piece === undefined) {
      throw new RangeError("Cannot decode a TokenId outside the BPE vocabulary");
    }
    return piece.slice();
  }

  decodeBytes(tokens: readonly TokenId[]): Uint8Array {
    let size = 0;
    for (const token of tokens) {
      const piece = this.pieces[token];
      if (piece === undefined) {
        throw new RangeError("Cannot decode a TokenId outside the BPE vocabulary");
      }
      size += piece.length;
      if (!Number.isSafeInteger(size)) {
        throw new RangeError("Decoded byte sequence is too large");
      }
    }
    const result = new Uint8Array(size);
    let offset = 0;
    for (const token of tokens) {
      const piece = this.pieces[token];
      if (piece === undefined) {
        throw new Error("Internal tokenizer decode invariant failed");
      }
      result.set(piece, offset);
      offset += piece.length;
    }
    return result;
  }

  decodeText(tokens: readonly TokenId[], fatal = false): string {
    return new TextDecoder("utf-8", { fatal }).decode(this.decodeBytes(tokens));
  }
}

export class StreamingTokenDecoder {
  private readonly decoder = new TextDecoder("utf-8", { fatal: false });

  push(bytes: Uint8Array): string {
    return this.decoder.decode(bytes, { stream: true });
  }

  finish(): string {
    return this.decoder.decode();
  }
}
