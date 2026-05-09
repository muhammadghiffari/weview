/**
 * Browser ESM wrapper for ruvector-attention-wasm v2.0.5
 *
 * The upstream pkg/ was built with wasm-pack --target nodejs (CJS + fs.readFileSync).
 * This wrapper loads the same WASM binary via fetch() for browser use.
 *
 * Usage:
 *   import initWasm, { WasmMultiHeadAttention, ... } from './ruvector_attention_browser.js';
 *   await initWasm();
 *   const attn = new WasmMultiHeadAttention(dim, heads);
 */

let _wasm;
let _initialized = false;

// The entire CJS module runs inside this IIFE to avoid polluting global scope.
// We capture all exports in _mod.
const _mod = {};

(function(exports, wasm_getter) {

// ── wasm-bindgen heap management ──────────────────────────────────
const heap = new Array(128).fill(undefined);
heap.push(undefined, null, true, false);
let heap_next = heap.length;

function addHeapObject(obj) {
  if (heap_next === heap.length) heap.push(heap.length + 1);
  const idx = heap_next;
  heap_next = heap[idx];
  heap[idx] = obj;
  return idx;
}
function getObject(idx) { return heap[idx]; }
function dropObject(idx) {
  if (idx < 132) return;
  heap[idx] = heap_next;
  heap_next = idx;
}
function takeObject(idx) {
  const ret = getObject(idx);
  dropObject(idx);
  return ret;
}
function isLikeNone(x) { return x === undefined || x === null; }

// ── Memory views ──────────────────────────────────────────────────
let cachedDataViewMemory0 = null;
let cachedUint8ArrayMemory0 = null;
let cachedFloat32ArrayMemory0 = null;

function wasm() { return wasm_getter(); }

function getDataViewMemory0() {
  if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer !== wasm().memory.buffer)
    cachedDataViewMemory0 = new DataView(wasm().memory.buffer);
  return cachedDataViewMemory0;
}
function getUint8ArrayMemory0() {
  if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.buffer !== wasm().memory.buffer)
    cachedUint8ArrayMemory0 = new Uint8Array(wasm().memory.buffer);
  return cachedUint8ArrayMemory0;
}
function getFloat32ArrayMemory0() {
  if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.buffer !== wasm().memory.buffer)
    cachedFloat32ArrayMemory0 = new Float32Array(wasm().memory.buffer);
  return cachedFloat32ArrayMemory0;
}
function getArrayF32FromWasm0(ptr, len) {
  ptr = ptr >>> 0;
  return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}
function getArrayU8FromWasm0(ptr, len) {
  ptr = ptr >>> 0;
  return getUint8ArrayMemory0().subarray(ptr, ptr + len);
}

let WASM_VECTOR_LEN = 0;

function passArrayF32ToWasm0(arg, malloc) {
  const ptr = malloc(arg.length * 4, 4) >>> 0;
  getFloat32ArrayMemory0().set(arg, ptr / 4);
  WASM_VECTOR_LEN = arg.length;
  return ptr;
}

const cachedTextEncoder = new TextEncoder();
const cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();

function getStringFromWasm0(ptr, len) {
  ptr = ptr >>> 0;
  return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

function passStringToWasm0(arg, malloc, realloc) {
  const buf = cachedTextEncoder.encode(arg);
  const ptr = malloc(buf.length, 1) >>> 0;
  getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
  WASM_VECTOR_LEN = buf.length;
  return ptr;
}

function debugString(val) {
  const type = typeof val;
  if (type == 'number' || type == 'boolean' || val == null) return `${val}`;
  if (type == 'string') return `"${val}"`;
  if (type == 'symbol') return val.description ? `Symbol(${val.description})` : 'Symbol';
  if (type == 'function') return 'Function';
  if (Array.isArray(val)) return `[${val.map(debugString).join(', ')}]`;
  try {
    const keys = Object.keys(val);
    return `{${keys.map(k => `${k}: ${debugString(val[k])}`).join(', ')}}`;
  } catch (_) { return Object.prototype.toString.call(val); }
}

function handleError(f, args) {
  try { return f.apply(this, args); }
  catch (e) { wasm().__wbindgen_export3(addHeapObject(e)); }
}

// ── FinalizationRegistry ──────────────────────────────────────────
const FR = typeof FinalizationRegistry !== 'undefined'
  ? FinalizationRegistry
  : class { register() {} unregister() {} };

const WasmMultiHeadAttentionFinalization = new FR(ptr => wasm().__wbg_wasmmultiheadattention_free(ptr >>> 0, 1));
const WasmFlashAttentionFinalization = new FR(ptr => wasm().__wbg_wasmflashattention_free(ptr >>> 0, 1));
const WasmHyperbolicAttentionFinalization = new FR(ptr => wasm().__wbg_wasmhyperbolicattention_free(ptr >>> 0, 1));
const WasmMoEAttentionFinalization = new FR(ptr => wasm().__wbg_wasmmoeattention_free(ptr >>> 0, 1));
const WasmLinearAttentionFinalization = new FR(ptr => wasm().__wbg_wasmlinearattention_free(ptr >>> 0, 1));
const WasmLocalGlobalAttentionFinalization = new FR(ptr => wasm().__wbg_wasmlocalglobalattention_free(ptr >>> 0, 1));

// ── Classes ───────────────────────────────────────────────────────

class WasmMultiHeadAttention {
  constructor(dim, num_heads) {
    const retptr = wasm().__wbindgen_add_to_stack_pointer(-16);
    try {
      wasm().wasmmultiheadattention_new(retptr, dim, num_heads);
      var r0 = getDataViewMemory0().getInt32(retptr + 0, true);
      var r1 = getDataViewMemory0().getInt32(retptr + 4, true);
      var r2 = getDataViewMemory0().getInt32(retptr + 8, true);
      if (r2) throw takeObject(r1);
      this.__wbg_ptr = r0 >>> 0;
      WasmMultiHeadAttentionFinalization.register(this, this.__wbg_ptr, this);
    } finally {
      wasm().__wbindgen_add_to_stack_pointer(16);
    }
  }
  free() {
    const ptr = this.__wbg_ptr; this.__wbg_ptr = 0;
    WasmMultiHeadAttentionFinalization.unregister(this);
    wasm().__wbg_wasmmultiheadattention_free(ptr, 0);
  }
  get dim() { return wasm().wasmmultiheadattention_dim(this.__wbg_ptr); }
  get num_heads() { return wasm().wasmmultiheadattention_num_heads(this.__wbg_ptr); }
  compute(query, keys, values) {
    const retptr = wasm().__wbindgen_add_to_stack_pointer(-16);
    try {
      const ptr0 = passArrayF32ToWasm0(query, wasm().__wbindgen_export);
      const len0 = WASM_VECTOR_LEN;
      wasm().wasmmultiheadattention_compute(retptr, this.__wbg_ptr, ptr0, len0, addHeapObject(keys), addHeapObject(values));
      var r0 = getDataViewMemory0().getInt32(retptr + 0, true);
      var r1 = getDataViewMemory0().getInt32(retptr + 4, true);
      var r2 = getDataViewMemory0().getInt32(retptr + 8, true);
      var r3 = getDataViewMemory0().getInt32(retptr + 12, true);
      if (r3) throw takeObject(r2);
      var v1 = getArrayF32FromWasm0(r0, r1).slice();
      wasm().__wbindgen_export4(r0, r1 * 4, 4);
      return v1;
    } finally {
      wasm().__wbindgen_add_to_stack_pointer(16);
    }
  }
}

class WasmFlashAttention {
  constructor(dim, block_size) {
    const ret = wasm().wasmflashattention_new(dim, block_size);
    this.__wbg_ptr = ret >>> 0;
    WasmFlashAttentionFinalization.register(this, this.__wbg_ptr, this);
  }
  free() {
    const ptr = this.__wbg_ptr; this.__wbg_ptr = 0;
    WasmFlashAttentionFinalization.unregister(this);
    wasm().__wbg_wasmflashattention_free(ptr, 0);
  }
  compute(query, keys, values) {
    const retptr = wasm().__wbindgen_add_to_stack_pointer(-16);
    try {
      const ptr0 = passArrayF32ToWasm0(query, wasm().__wbindgen_export);
      const len0 = WASM_VECTOR_LEN;
      wasm().wasmflashattention_compute(retptr, this.__wbg_ptr, ptr0, len0, addHeapObject(keys), addHeapObject(values));
      var r0 = getDataViewMemory0().getInt32(retptr + 0, true);
      var r1 = getDataViewMemory0().getInt32(retptr + 4, true);
      var r2 = getDataViewMemory0().getInt32(retptr + 8, true);
      var r3 = getDataViewMemory0().getInt32(retptr + 12, true);
      if (r3) throw takeObject(r2);
      var v1 = getArrayF32FromWasm0(r0, r1).slice();
      wasm().__wbindgen_export4(r0, r1 * 4, 4);
      return v1;
    } finally {
      wasm().__wbindgen_add_to_stack_pointer(16);
    }
  }
}

class WasmHyperbolicAttention {
  constructor(dim, curvature) {
    const ret = wasm().wasmhyperbolicattention_new(dim, curvature);
    this.__wbg_ptr = ret >>> 0;
    WasmHyperbolicAttentionFinalization.register(this, this.__wbg_ptr, this);
  }
  free() {
    const ptr = this.__wbg_ptr; this.__wbg_ptr = 0;
    WasmHyperbolicAttentionFinalization.unregister(this);
    wasm().__wbg_wasmhyperbolicattention_free(ptr, 0);
  }
  get curvature() { return wasm().wasmhyperbolicattention_curvature(this.__wbg_ptr); }
  compute(query, keys, values) {
    const retptr = wasm().__wbindgen_add_to_stack_pointer(-16);
    try {
      const ptr0 = passArrayF32ToWasm0(query, wasm().__wbindgen_export);
      const len0 = WASM_VECTOR_LEN;
      wasm().wasmhyperbolicattention_compute(retptr, this.__wbg_ptr, ptr0, len0, addHeapObject(keys), addHeapObject(values));
      var r0 = getDataViewMemory0().getInt32(retptr + 0, true);
      var r1 = getDataViewMemory0().getInt32(retptr + 4, true);
      var r2 = getDataViewMemory0().getInt32(retptr + 8, true);
      var r3 = getDataViewMemory0().getInt32(retptr + 12, true);
      if (r3) throw takeObject(r2);
      var v1 = getArrayF32FromWasm0(r0, r1).slice();
      wasm().__wbindgen_export4(r0, r1 * 4, 4);
      return v1;
    } finally {
      wasm().__wbindgen_add_to_stack_pointer(16);
    }
  }
}

class WasmMoEAttention {
  constructor(dim, num_experts, top_k) {
    const ret = wasm().wasmmoeattention_new(dim, num_experts, top_k);
    this.__wbg_ptr = ret >>> 0;
    WasmMoEAttentionFinalization.register(this, this.__wbg_ptr, this);
  }
  free() {
    const ptr = this.__wbg_ptr; this.__wbg_ptr = 0;
    WasmMoEAttentionFinalization.unregister(this);
    wasm().__wbg_wasmmoeattention_free(ptr, 0);
  }
  compute(query, keys, values) {
    const retptr = wasm().__wbindgen_add_to_stack_pointer(-16);
    try {
      const ptr0 = passArrayF32ToWasm0(query, wasm().__wbindgen_export);
      const len0 = WASM_VECTOR_LEN;
      wasm().wasmmoeattention_compute(retptr, this.__wbg_ptr, ptr0, len0, addHeapObject(keys), addHeapObject(values));
      var r0 = getDataViewMemory0().getInt32(retptr + 0, true);
      var r1 = getDataViewMemory0().getInt32(retptr + 4, true);
      var r2 = getDataViewMemory0().getInt32(retptr + 8, true);
      var r3 = getDataViewMemory0().getInt32(retptr + 12, true);
      if (r3) throw takeObject(r2);
      var v1 = getArrayF32FromWasm0(r0, r1).slice();
      wasm().__wbindgen_export4(r0, r1 * 4, 4);
      return v1;
    } finally {
      wasm().__wbindgen_add_to_stack_pointer(16);
    }
  }
}

class WasmLinearAttention {
  constructor(dim, num_features) {
    const ret = wasm().wasmlinearattention_new(dim, num_features || dim);
    this.__wbg_ptr = ret >>> 0;
    WasmLinearAttentionFinalization.register(this, this.__wbg_ptr, this);
  }
  free() {
    const ptr = this.__wbg_ptr; this.__wbg_ptr = 0;
    WasmLinearAttentionFinalization.unregister(this);
    wasm().__wbg_wasmlinearattention_free(ptr, 0);
  }
  compute(query, keys, values) {
    const retptr = wasm().__wbindgen_add_to_stack_pointer(-16);
    try {
      const ptr0 = passArrayF32ToWasm0(query, wasm().__wbindgen_export);
      const len0 = WASM_VECTOR_LEN;
      wasm().wasmlinearattention_compute(retptr, this.__wbg_ptr, ptr0, len0, addHeapObject(keys), addHeapObject(values));
      var r0 = getDataViewMemory0().getInt32(retptr + 0, true);
      var r1 = getDataViewMemory0().getInt32(retptr + 4, true);
      var r2 = getDataViewMemory0().getInt32(retptr + 8, true);
      var r3 = getDataViewMemory0().getInt32(retptr + 12, true);
      if (r3) throw takeObject(r2);
      var v1 = getArrayF32FromWasm0(r0, r1).slice();
      wasm().__wbindgen_export4(r0, r1 * 4, 4);
      return v1;
    } finally {
      wasm().__wbindgen_add_to_stack_pointer(16);
    }
  }
}

class WasmLocalGlobalAttention {
  constructor(dim, local_window, global_tokens) {
    const ret = wasm().wasmlocalglobalattention_new(dim, local_window || 4, global_tokens || 2);
    this.__wbg_ptr = ret >>> 0;
    WasmLocalGlobalAttentionFinalization.register(this, this.__wbg_ptr, this);
  }
  free() {
    const ptr = this.__wbg_ptr; this.__wbg_ptr = 0;
    WasmLocalGlobalAttentionFinalization.unregister(this);
    wasm().__wbg_wasmlocalglobalattention_free(ptr, 0);
  }
  compute(query, keys, values) {
    const retptr = wasm().__wbindgen_add_to_stack_pointer(-16);
    try {
      const ptr0 = passArrayF32ToWasm0(query, wasm().__wbindgen_export);
      const len0 = WASM_VECTOR_LEN;
      wasm().wasmlocalglobalattention_compute(retptr, this.__wbg_ptr, ptr0, len0, addHeapObject(keys), addHeapObject(values));
      var r0 = getDataViewMemory0().getInt32(retptr + 0, true);
      var r1 = getDataViewMemory0().getInt32(retptr + 4, true);
      var r2 = getDataViewMemory0().getInt32(retptr + 8, true);
      var r3 = getDataViewMemory0().getInt32(retptr + 12, true);
      if (r3) throw takeObject(r2);
      var v1 = getArrayF32FromWasm0(r0, r1).slice();
      wasm().__wbindgen_export4(r0, r1 * 4, 4);
      return v1;
    } finally {
      wasm().__wbindgen_add_to_stack_pointer(16);
    }
  }
}

// ── Standalone functions ──────────────────────────────────────────

function cosine_similarity(a, b) {
  const retptr = wasm().__wbindgen_add_to_stack_pointer(-16);
  try {
    const ptr0 = passArrayF32ToWasm0(a, wasm().__wbindgen_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(b, wasm().__wbindgen_export);
    const len1 = WASM_VECTOR_LEN;
    wasm().cosine_similarity(retptr, ptr0, len0, ptr1, len1);
    var r0 = getDataViewMemory0().getFloat64(retptr + 0, true);
    var r1 = getDataViewMemory0().getInt32(retptr + 8, true);
    var r2 = getDataViewMemory0().getInt32(retptr + 12, true);
    if (r2) throw takeObject(r1);
    return r0;
  } finally {
    wasm().__wbindgen_add_to_stack_pointer(16);
  }
}

function normalize(vec) {
  const ptr0 = passArrayF32ToWasm0(vec, wasm().__wbindgen_export);
  const len0 = WASM_VECTOR_LEN;
  wasm().normalize(ptr0, len0, addHeapObject(vec));
}

function l2_norm(vec) {
  const retptr = wasm().__wbindgen_add_to_stack_pointer(-16);
  try {
    const ptr0 = passArrayF32ToWasm0(vec, wasm().__wbindgen_export);
    const len0 = WASM_VECTOR_LEN;
    wasm().l2_norm(retptr, ptr0, len0);
    var r0 = getDataViewMemory0().getFloat64(retptr + 0, true);
    var r1 = getDataViewMemory0().getInt32(retptr + 8, true);
    var r2 = getDataViewMemory0().getInt32(retptr + 12, true);
    if (r2) throw takeObject(r1);
    return r0;
  } finally {
    wasm().__wbindgen_add_to_stack_pointer(16);
  }
}

function softmax(vec) {
  const ptr0 = passArrayF32ToWasm0(vec, wasm().__wbindgen_export);
  const len0 = WASM_VECTOR_LEN;
  wasm().softmax(ptr0, len0, addHeapObject(vec));
}

function batch_normalize(vectors, epsilon) {
  const retptr = wasm().__wbindgen_add_to_stack_pointer(-16);
  try {
    wasm().batch_normalize(retptr, addHeapObject(vectors), isLikeNone(epsilon) ? 0x100000001 : Math.fround(epsilon));
    var r0 = getDataViewMemory0().getInt32(retptr + 0, true);
    var r1 = getDataViewMemory0().getInt32(retptr + 4, true);
    var r2 = getDataViewMemory0().getInt32(retptr + 8, true);
    var r3 = getDataViewMemory0().getInt32(retptr + 12, true);
    if (r3) throw takeObject(r2);
    var v1 = getArrayF32FromWasm0(r0, r1).slice();
    wasm().__wbindgen_export4(r0, r1 * 4, 4);
    return v1;
  } finally {
    wasm().__wbindgen_add_to_stack_pointer(16);
  }
}

function pairwise_distances(vectors) {
  const retptr = wasm().__wbindgen_add_to_stack_pointer(-16);
  try {
    wasm().pairwise_distances(retptr, addHeapObject(vectors));
    var r0 = getDataViewMemory0().getInt32(retptr + 0, true);
    var r1 = getDataViewMemory0().getInt32(retptr + 4, true);
    var r2 = getDataViewMemory0().getInt32(retptr + 8, true);
    var r3 = getDataViewMemory0().getInt32(retptr + 12, true);
    if (r3) throw takeObject(r2);
    var v1 = getArrayF32FromWasm0(r0, r1).slice();
    wasm().__wbindgen_export4(r0, r1 * 4, 4);
    return v1;
  } finally {
    wasm().__wbindgen_add_to_stack_pointer(16);
  }
}

function scaled_dot_attention(query, keys, values, scale) {
  const retptr = wasm().__wbindgen_add_to_stack_pointer(-16);
  try {
    const ptr0 = passArrayF32ToWasm0(query, wasm().__wbindgen_export);
    const len0 = WASM_VECTOR_LEN;
    wasm().scaled_dot_attention(retptr, ptr0, len0, addHeapObject(keys), addHeapObject(values), isLikeNone(scale) ? 0x100000001 : Math.fround(scale));
    var r0 = getDataViewMemory0().getInt32(retptr + 0, true);
    var r1 = getDataViewMemory0().getInt32(retptr + 4, true);
    var r2 = getDataViewMemory0().getInt32(retptr + 8, true);
    var r3 = getDataViewMemory0().getInt32(retptr + 12, true);
    if (r3) throw takeObject(r2);
    var v1 = getArrayF32FromWasm0(r0, r1).slice();
    wasm().__wbindgen_export4(r0, r1 * 4, 4);
    return v1;
  } finally {
    wasm().__wbindgen_add_to_stack_pointer(16);
  }
}

function attention_weights(scores, temperature) {
  const ptr0 = passArrayF32ToWasm0(scores, wasm().__wbindgen_export);
  const len0 = WASM_VECTOR_LEN;
  wasm().attention_weights(ptr0, len0, addHeapObject(scores), isLikeNone(temperature) ? 0x100000001 : Math.fround(temperature));
}

function available_mechanisms() {
  const ret = wasm().available_mechanisms();
  return takeObject(ret);
}

function random_orthogonal_matrix(dim) {
  const retptr = wasm().__wbindgen_add_to_stack_pointer(-16);
  try {
    wasm().random_orthogonal_matrix(retptr, dim);
    var r0 = getDataViewMemory0().getInt32(retptr + 0, true);
    var r1 = getDataViewMemory0().getInt32(retptr + 4, true);
    var v1 = getArrayF32FromWasm0(r0, r1).slice();
    wasm().__wbindgen_export4(r0, r1 * 4, 4);
    return v1;
  } finally {
    wasm().__wbindgen_add_to_stack_pointer(16);
  }
}

function rv_init() { wasm().init(); }

function rv_version() {
  let d0, d1;
  const retptr = wasm().__wbindgen_add_to_stack_pointer(-16);
  try {
    wasm().version(retptr);
    d0 = getDataViewMemory0().getInt32(retptr + 0, true);
    d1 = getDataViewMemory0().getInt32(retptr + 4, true);
    return getStringFromWasm0(d0, d1);
  } finally {
    wasm().__wbindgen_add_to_stack_pointer(16);
    if (d0 !== undefined) wasm().__wbindgen_export4(d0, d1, 1);
  }
}

// ── Collect exports ───────────────────────────────────────────────
exports.WasmMultiHeadAttention = WasmMultiHeadAttention;
exports.WasmFlashAttention = WasmFlashAttention;
exports.WasmHyperbolicAttention = WasmHyperbolicAttention;
exports.WasmMoEAttention = WasmMoEAttention;
exports.WasmLinearAttention = WasmLinearAttention;
exports.WasmLocalGlobalAttention = WasmLocalGlobalAttention;
exports.cosine_similarity = cosine_similarity;
exports.normalize = normalize;
exports.l2_norm = l2_norm;
exports.softmax = softmax;
exports.batch_normalize = batch_normalize;
exports.pairwise_distances = pairwise_distances;
exports.scaled_dot_attention = scaled_dot_attention;
exports.attention_weights = attention_weights;
exports.available_mechanisms = available_mechanisms;
exports.random_orthogonal_matrix = random_orthogonal_matrix;
exports.init = rv_init;
exports.version = rv_version;

// ── Build WASM import object ──────────────────────────────────────
exports.__wbg_get_imports = function() {
  const import0 = {
    __proto__: null,
    __wbg_Error_4577686b3a6d9b3a: (arg0, arg1) => addHeapObject(Error(getStringFromWasm0(arg0, arg1))),
    __wbg_String_8564e559799eccda: (arg0, arg1) => {
      const ret = String(getObject(arg1));
      const ptr1 = passStringToWasm0(ret, wasm().__wbindgen_export, wasm().__wbindgen_export2);
      const len1 = WASM_VECTOR_LEN;
      getDataViewMemory0().setInt32(arg0 + 4, len1, true);
      getDataViewMemory0().setInt32(arg0, ptr1, true);
    },
    __wbg___wbindgen_boolean_get_18c4ed9422296fff: (arg0) => {
      const v = getObject(arg0);
      const ret = typeof v === 'boolean' ? v : undefined;
      return isLikeNone(ret) ? 0xFFFFFF : ret ? 1 : 0;
    },
    __wbg___wbindgen_copy_to_typed_array_5294f8e46aecc086: (arg0, arg1, arg2) => {
      new Uint8Array(getObject(arg2).buffer, getObject(arg2).byteOffset, getObject(arg2).byteLength).set(getArrayU8FromWasm0(arg0, arg1));
    },
    __wbg___wbindgen_debug_string_ddde1867f49c2442: (arg0, arg1) => {
      const ret = debugString(getObject(arg1));
      const ptr1 = passStringToWasm0(ret, wasm().__wbindgen_export, wasm().__wbindgen_export2);
      const len1 = WASM_VECTOR_LEN;
      getDataViewMemory0().setInt32(arg0 + 4, len1, true);
      getDataViewMemory0().setInt32(arg0, ptr1, true);
    },
    __wbg___wbindgen_is_function_d633e708baf0d146: (arg0) => typeof getObject(arg0) === 'function',
    __wbg___wbindgen_is_object_4b3de556756ee8a8: (arg0) => {
      const val = getObject(arg0);
      return typeof val === 'object' && val !== null;
    },
    __wbg___wbindgen_jsval_loose_eq_1562ceb9af84e990: (arg0, arg1) => getObject(arg0) == getObject(arg1),
    __wbg___wbindgen_number_get_5854912275df1894: (arg0, arg1) => {
      const obj = getObject(arg1);
      const ret = typeof obj === 'number' ? obj : undefined;
      getDataViewMemory0().setFloat64(arg0 + 8, isLikeNone(ret) ? 0 : ret, true);
      getDataViewMemory0().setInt32(arg0, !isLikeNone(ret), true);
    },
    __wbg___wbindgen_string_get_3e5751597f39a112: (arg0, arg1) => {
      const obj = getObject(arg1);
      const ret = typeof obj === 'string' ? obj : undefined;
      var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm().__wbindgen_export, wasm().__wbindgen_export2);
      var len1 = WASM_VECTOR_LEN;
      getDataViewMemory0().setInt32(arg0 + 4, len1, true);
      getDataViewMemory0().setInt32(arg0, ptr1, true);
    },
    __wbg___wbindgen_throw_39bc967c0e5a9b58: (arg0, arg1) => { throw new Error(getStringFromWasm0(arg0, arg1)); },
    __wbg_call_73af281463ec8b58: function() { return handleError(function(arg0, arg1) {
      return addHeapObject(getObject(arg0).call(getObject(arg1)));
    }, arguments); },
    __wbg_done_5aad55ec6b1954b1: (arg0) => getObject(arg0).done,
    __wbg_error_a6fa202b58aa1cd3: (arg0, arg1) => {
      try { console.error(getStringFromWasm0(arg0, arg1)); }
      finally { wasm().__wbindgen_export4(arg0, arg1, 1); }
    },
    __wbg_error_ad28debb48b5c6bb: (arg0) => console.error(getObject(arg0)),
    __wbg_get_4920fefd3451364b: function() { return handleError(function(arg0, arg1) {
      return addHeapObject(Reflect.get(getObject(arg0), getObject(arg1)));
    }, arguments); },
    __wbg_get_unchecked_3d0f4b91c8eca4f0: (arg0, arg1) => addHeapObject(getObject(arg0)[arg1 >>> 0]),
    __wbg_instanceof_ArrayBuffer_15859862b80b732d: (arg0) => {
      try { return getObject(arg0) instanceof ArrayBuffer; } catch (_) { return false; }
    },
    __wbg_instanceof_Uint8Array_2240b7046ac16f05: (arg0) => {
      try { return getObject(arg0) instanceof Uint8Array; } catch (_) { return false; }
    },
    __wbg_isArray_fad08a0d12828686: (arg0) => Array.isArray(getObject(arg0)),
    __wbg_iterator_fc7ad8d33bab9e26: () => addHeapObject(Symbol.iterator),
    __wbg_length_5855c1f289dfffc1: (arg0) => getObject(arg0).length,
    __wbg_length_a31e05262e09b7f8: (arg0) => getObject(arg0).length,
    __wbg_log_3c5e4b64af29e724: (arg0) => console.log(getObject(arg0)),
    __wbg_new_09959f7b4c92c246: (arg0) => addHeapObject(new Uint8Array(getObject(arg0))),
    __wbg_new_227d7c05414eb861: () => addHeapObject(new Error()),
    __wbg_new_cbee8c0d5c479eac: () => addHeapObject(new Array()),
    __wbg_next_a5fe6f328f7affc2: (arg0) => addHeapObject(getObject(arg0).next),
    __wbg_next_e592122bb4ed4c67: function() { return handleError(function(arg0) {
      return addHeapObject(getObject(arg0).next());
    }, arguments); },
    __wbg_prototypesetcall_f034d444741426c3: (arg0, arg1, arg2) => {
      Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), getObject(arg2));
    },
    __wbg_random_2b7bed8995d680fb: () => Math.random(),
    __wbg_set_4c81cfb5dc3a333c: (arg0, arg1, arg2) => { getObject(arg0)[arg1 >>> 0] = takeObject(arg2); },
    __wbg_stack_3b0d974bbf31e44f: (arg0, arg1) => {
      const ret = getObject(arg1).stack;
      const ptr1 = passStringToWasm0(ret, wasm().__wbindgen_export, wasm().__wbindgen_export2);
      const len1 = WASM_VECTOR_LEN;
      getDataViewMemory0().setInt32(arg0 + 4, len1, true);
      getDataViewMemory0().setInt32(arg0, ptr1, true);
    },
    __wbg_value_667dcb90597486a6: (arg0) => addHeapObject(getObject(arg0).value),
    __wbindgen_cast_0000000000000001: (arg0, arg1) => addHeapObject(getStringFromWasm0(arg0, arg1)),
    __wbindgen_object_drop_ref: (arg0) => takeObject(arg0),
  };
  return { __proto__: null, "./ruvector_attention_wasm_bg.js": import0 };
};

})(_mod, () => _wasm);


// ── Async WASM init (fetch-based for browsers) ───────────────────

export default async function initWasm() {
  if (_initialized) return;
  const wasmUrl = new URL('ruvector_attention_wasm_bg.wasm', import.meta.url);
  const imports = _mod.__wbg_get_imports();
  let result;
  if (typeof WebAssembly.instantiateStreaming === 'function') {
    try {
      result = await WebAssembly.instantiateStreaming(fetch(wasmUrl), imports);
    } catch (e) {
      // Fallback if streaming fails (e.g. wrong MIME type)
      const bytes = await (await fetch(wasmUrl)).arrayBuffer();
      result = await WebAssembly.instantiate(bytes, imports);
    }
  } else {
    const bytes = await (await fetch(wasmUrl)).arrayBuffer();
    result = await WebAssembly.instantiate(bytes, imports);
  }
  _wasm = result.instance.exports;
  _wasm.__wbindgen_start();
  _initialized = true;
}

// ── ESM re-exports ────────────────────────────────────────────────
// Attention mechanism classes
export const WasmMultiHeadAttention = _mod.WasmMultiHeadAttention;
export const WasmFlashAttention = _mod.WasmFlashAttention;
export const WasmHyperbolicAttention = _mod.WasmHyperbolicAttention;
export const WasmMoEAttention = _mod.WasmMoEAttention;
export const WasmLinearAttention = _mod.WasmLinearAttention;
export const WasmLocalGlobalAttention = _mod.WasmLocalGlobalAttention;
// Utility functions
export const cosine_similarity = _mod.cosine_similarity;
export const normalize = _mod.normalize;
export const l2_norm = _mod.l2_norm;
export const softmax = _mod.softmax;
export const batch_normalize = _mod.batch_normalize;
export const pairwise_distances = _mod.pairwise_distances;
export const scaled_dot_attention = _mod.scaled_dot_attention;
export const attention_weights = _mod.attention_weights;
export const random_orthogonal_matrix = _mod.random_orthogonal_matrix;
export const available_mechanisms = _mod.available_mechanisms;
// Lifecycle
export const init = _mod.init;
export const version = _mod.version;
