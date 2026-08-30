export const EMBEDDING_SHADER = /* wgsl */ `
struct Params { rows: u32, columns: u32, vocabulary: u32, unused: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  let size = params.rows * params.columns;
  if (index >= size) { return; }
  let row = index / params.columns;
  let column = index % params.columns;
  let token = tokens[row];
  if (token < params.vocabulary) {
    output[index] = weights[token * params.columns + column];
  }
}
`;

export const RMS_NORM_SHADER = /* wgsl */ `
struct Params { rows: u32, columns: u32, epsilon: f32, unused: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let row = group.x;
  let lane = local.x;
  if (row >= params.rows) { return; }
  var sum = 0.0;
  var column = lane;
  while (column < params.columns) {
    let value = input[row * params.columns + column];
    sum += value * value;
    column += 256u;
  }
  partial[lane] = sum;
  workgroupBarrier();
  var stride = 128u;
  while (stride > 0u) {
    if (lane < stride) { partial[lane] += partial[lane + stride]; }
    workgroupBarrier();
    stride >>= 1u;
  }
  let scale = inverseSqrt(partial[0] / f32(params.columns) + params.epsilon);
  column = lane;
  while (column < params.columns) {
    let index = row * params.columns + column;
    output[index] = input[index] * scale * weight[column];
    column += 256u;
  }
}
`;

export const MATMUL_SHADER = /* wgsl */ `
struct Params { rows: u32, columns: u32, inner: u32, unused: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> left: array<f32>;
@group(0) @binding(2) var<storage, read> right: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> tile_left: array<f32, 256>;
var<workgroup> tile_right: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let row = group.y * 16u + local.y;
  let column = group.x * 16u + local.x;
  let lane = local.y * 16u + local.x;
  let tiles = (params.inner + 15u) / 16u;
  var value = 0.0;
  for (var tile = 0u; tile < tiles; tile++) {
    let left_column = tile * 16u + local.x;
    let right_row = tile * 16u + local.y;
    tile_left[lane] = select(0.0, left[row * params.inner + left_column],
      row < params.rows && left_column < params.inner);
    tile_right[lane] = select(0.0, right[right_row * params.columns + column],
      right_row < params.inner && column < params.columns);
    workgroupBarrier();
    for (var inner = 0u; inner < 16u; inner++) {
      value += tile_left[local.y * 16u + inner] * tile_right[inner * 16u + local.x];
    }
    workgroupBarrier();
  }
  if (row < params.rows && column < params.columns) {
    output[row * params.columns + column] = value;
  }
}
`;

export const TIED_OUTPUT_SHADER = /* wgsl */ `
struct Params { vocabulary: u32, columns: u32, unused0: u32, unused1: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> hidden: array<f32>;
@group(0) @binding(2) var<storage, read> embedding: array<f32>;
@group(0) @binding(3) var<storage, read_write> logits: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let token = gid.x;
  if (token >= params.vocabulary) { return; }
  var value = 0.0;
  let offset = token * params.columns;
  for (var column = 0u; column < params.columns; column++) {
    value += hidden[column] * embedding[offset + column];
  }
  logits[token] = value;
}
`;

export const ADD_SHADER = /* wgsl */ `
struct Params { elements: u32, unused0: u32, unused1: u32, unused2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> left: array<f32>;
@group(0) @binding(2) var<storage, read> right: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < params.elements) { output[gid.x] = left[gid.x] + right[gid.x]; }
}
`;

export const SWIGLU_SHADER = /* wgsl */ `
struct Params { elements: u32, unused0: u32, unused1: u32, unused2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index < params.elements) {
    let value = gate[index];
    gate[index] = (value / (1.0 + exp(-value))) * up[index];
  }
}
`;

export const ROPE_SHADER = /* wgsl */ `
struct Params { rows: u32, heads: u32, head_dim: u32, start_position: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> values: array<f32>;
@group(0) @binding(2) var<storage, read> rope_table: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pairs = params.head_dim / 2u;
  let size = params.rows * params.heads * pairs;
  let index = gid.x;
  if (index >= size) { return; }
  let pair = index % pairs;
  let row_head = index / pairs;
  let row = row_head / params.heads;
  let head = row_head % params.heads;
  let table_index = ((params.start_position + row) * pairs + pair) * 2u;
  let cosine = rope_table[table_index];
  let sine = rope_table[table_index + 1u];
  let base = (row * params.heads + head) * params.head_dim + 2u * pair;
  let first = values[base];
  let second = values[base + 1u];
  values[base] = first * cosine - second * sine;
  values[base + 1u] = first * sine + second * cosine;
}
`;

export const CACHE_APPEND_SHADER = /* wgsl */ `
struct Params { rows: u32, heads: u32, head_dim: u32, start_position: u32, capacity: u32, unused0: u32, unused1: u32, unused2: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> cache: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = params.rows * params.heads * params.head_dim;
  let index = gid.x;
  if (index >= size) { return; }
  let dimension = index % params.head_dim;
  let row_head = index / params.head_dim;
  let head = row_head % params.heads;
  let row = row_head / params.heads;
  let destination = (head * params.capacity + params.start_position + row) * params.head_dim + dimension;
  cache[destination] = input[index];
}
`;

export const ATTENTION_SHADER = /* wgsl */ `
struct Params {
  rows: u32, query_heads: u32, kv_heads: u32, head_dim: u32,
  start_position: u32, capacity: u32, unused0: u32, unused1: u32
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read> key_cache: array<f32>;
@group(0) @binding(3) var<storage, read> value_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
var<workgroup> scores: array<f32, 512>;
var<workgroup> reduction: array<f32, 256>;
var<workgroup> shared_maximum: f32;
var<workgroup> shared_sum: f32;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let row = group.x;
  let query_head = group.y;
  let lane = local.x;
  if (row >= params.rows || query_head >= params.query_heads) { return; }
  let kv_head = query_head / (params.query_heads / params.kv_heads);
  let key_count = params.start_position + row + 1u;
  let query_base = (row * params.query_heads + query_head) * params.head_dim;
  var key = lane;
  var local_max = -3.402823466e+38;
  while (key < key_count) {
    let key_base = (kv_head * params.capacity + key) * params.head_dim;
    var dot = 0.0;
    for (var dimension = 0u; dimension < params.head_dim; dimension++) {
      dot += query[query_base + dimension] * key_cache[key_base + dimension];
    }
    let score = dot * inverseSqrt(f32(params.head_dim));
    scores[key] = score;
    local_max = max(local_max, score);
    key += 256u;
  }
  reduction[lane] = local_max;
  workgroupBarrier();
  var stride = 128u;
  while (stride > 0u) {
    if (lane < stride) { reduction[lane] = max(reduction[lane], reduction[lane + stride]); }
    workgroupBarrier();
    stride >>= 1u;
  }
  if (lane == 0u) { shared_maximum = reduction[0]; }
  workgroupBarrier();
  let maximum = shared_maximum;
  key = lane;
  var local_sum = 0.0;
  while (key < key_count) {
    let probability = exp(scores[key] - maximum);
    scores[key] = probability;
    local_sum += probability;
    key += 256u;
  }
  reduction[lane] = local_sum;
  workgroupBarrier();
  stride = 128u;
  while (stride > 0u) {
    if (lane < stride) { reduction[lane] += reduction[lane + stride]; }
    workgroupBarrier();
    stride >>= 1u;
  }
  if (lane == 0u) { shared_sum = reduction[0]; }
  workgroupBarrier();
  let denominator = shared_sum;
  var dimension = lane;
  while (dimension < params.head_dim) {
    var value = 0.0;
    for (var position = 0u; position < key_count; position++) {
      let value_index = (kv_head * params.capacity + position) * params.head_dim + dimension;
      value += (scores[position] / denominator) * value_cache[value_index];
    }
    output[query_base + dimension] = value;
    dimension += 256u;
  }
}
`;
