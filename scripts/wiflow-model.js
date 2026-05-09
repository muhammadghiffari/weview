#!/usr/bin/env node
/**
 * WiFlow Pose Estimation Architecture (arXiv:2602.08661)
 *
 * Pure JavaScript implementation for ruvllm-based CSI-to-pose inference.
 * Adapted from the published WiFlow paper for single TX/RX ESP32 deployment:
 *   - Stage 1: Temporal Convolutional Network (dilated causal convolutions)
 *   - Stage 2: Asymmetric Convolution Encoder (subcarrier-dimension spatial)
 *   - Stage 3: Axial Self-Attention (width + height, O(H^2W + HW^2))
 *   - Decoder: Adaptive average pooling + linear projection to 17 COCO keypoints
 *
 * Input:  [batch, 128 subcarriers, 20 time steps] (CSI amplitude)
 * Output: [batch, 17 keypoints, 2 coordinates] normalized to [0,1]
 *
 * ADR: docs/adr/ADR-072-wiflow-architecture.md
 */

'use strict';

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift32)
// ---------------------------------------------------------------------------

function createRng(seed) {
  let s = seed | 0 || 42;
  return () => {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}

/** Box-Muller transform for Gaussian samples */
function gaussianRng(rng) {
  return () => {
    const u1 = rng() || 1e-10;
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
}

// ---------------------------------------------------------------------------
// Tensor utility functions (Float32Array based)
// ---------------------------------------------------------------------------

/** Initialize weight array with Kaiming He (fan_in) for ReLU layers */
function initKaiming(fanIn, fanOut, rng) {
  const std = Math.sqrt(2.0 / fanIn);
  const gauss = gaussianRng(rng);
  const arr = new Float32Array(fanIn * fanOut);
  for (let i = 0; i < arr.length; i++) arr[i] = gauss() * std;
  return arr;
}

/** Initialize weight array with Xavier/Glorot */
function initXavier(fanIn, fanOut, rng) {
  const std = Math.sqrt(2.0 / (fanIn + fanOut));
  const gauss = gaussianRng(rng);
  const arr = new Float32Array(fanIn * fanOut);
  for (let i = 0; i < arr.length; i++) arr[i] = gauss() * std;
  return arr;
}

/** ReLU activation in-place */
function relu(arr) {
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < 0) arr[i] = 0;
  }
  return arr;
}

/** Softmax over a 1D array (or over last dimension of a strided view) */
function softmax(arr, offset, length) {
  offset = offset || 0;
  length = length || arr.length;
  let maxVal = -Infinity;
  for (let i = offset; i < offset + length; i++) {
    if (arr[i] > maxVal) maxVal = arr[i];
  }
  let sum = 0;
  for (let i = offset; i < offset + length; i++) {
    arr[i] = Math.exp(arr[i] - maxVal);
    sum += arr[i];
  }
  if (sum > 0) {
    for (let i = offset; i < offset + length; i++) arr[i] /= sum;
  }
  return arr;
}

/** SmoothL1 loss (Huber loss with beta) */
function smoothL1(predicted, target, beta) {
  beta = beta || 0.1;
  let loss = 0;
  const n = Math.min(predicted.length, target.length);
  for (let i = 0; i < n; i++) {
    const diff = Math.abs(predicted[i] - target[i]);
    if (diff < beta) {
      loss += 0.5 * diff * diff / beta;
    } else {
      loss += diff - 0.5 * beta;
    }
  }
  return loss / n;
}

/** SmoothL1 gradient */
function smoothL1Grad(predicted, target, beta) {
  beta = beta || 0.1;
  const n = Math.min(predicted.length, target.length);
  const grad = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const diff = predicted[i] - target[i];
    const absDiff = Math.abs(diff);
    if (absDiff < beta) {
      grad[i] = diff / beta / n;
    } else {
      grad[i] = (diff > 0 ? 1 : -1) / n;
    }
  }
  return grad;
}

// ---------------------------------------------------------------------------
// 1D Convolution (causal and non-causal)
// ---------------------------------------------------------------------------

/**
 * Conv1D: [channels_in, time] -> [channels_out, time]
 * Weight shape: [out_ch, in_ch, kernel]
 * Supports dilation and causal (left-only) padding.
 */
class Conv1d {
  /**
   * @param {number} inCh
   * @param {number} outCh
   * @param {number} kernel
   * @param {object} opts - { dilation, stride, causal, bias }
   */
  constructor(inCh, outCh, kernel, opts = {}) {
    this.inCh = inCh;
    this.outCh = outCh;
    this.kernel = kernel;
    this.dilation = opts.dilation || 1;
    this.stride = opts.stride || 1;
    this.causal = opts.causal !== undefined ? opts.causal : false;
    this.hasBias = opts.bias !== false;

    const rng = createRng(opts.seed || (inCh * 1000 + outCh * 7 + kernel * 31));
    // Kaiming init for ReLU
    this.weight = initKaiming(inCh * kernel, outCh, rng);
    this.bias = this.hasBias ? new Float32Array(outCh) : null;

    // Gradient accumulators
    this.weightGrad = new Float32Array(this.weight.length);
    this.biasGrad = this.hasBias ? new Float32Array(outCh) : null;
  }

  /** Count parameters */
  numParams() {
    return this.weight.length + (this.hasBias ? this.bias.length : 0);
  }

  /**
   * Forward pass.
   * @param {Float32Array} input - shape [inCh, T]
   * @param {number} T - temporal length
   * @returns {{ output: Float32Array, T_out: number }}
   */
  forward(input, T) {
    const effectiveK = this.kernel + (this.kernel - 1) * (this.dilation - 1);

    let padLeft, padRight;
    if (this.causal) {
      padLeft = effectiveK - 1;
      padRight = 0;
    } else {
      padLeft = Math.floor((effectiveK - 1) / 2);
      padRight = Math.ceil((effectiveK - 1) / 2);
    }

    const T_padded = T + padLeft + padRight;
    const T_out = Math.floor((T_padded - effectiveK) / this.stride) + 1;

    // Pad input with zeros
    const padded = new Float32Array(this.inCh * T_padded);
    for (let c = 0; c < this.inCh; c++) {
      for (let t = 0; t < T; t++) {
        padded[c * T_padded + (t + padLeft)] = input[c * T + t];
      }
    }

    // Convolution
    const output = new Float32Array(this.outCh * T_out);
    for (let oc = 0; oc < this.outCh; oc++) {
      for (let t = 0; t < T_out; t++) {
        let sum = this.hasBias ? this.bias[oc] : 0;
        const tStart = t * this.stride;

        for (let ic = 0; ic < this.inCh; ic++) {
          for (let k = 0; k < this.kernel; k++) {
            const tIdx = tStart + k * this.dilation;
            if (tIdx >= 0 && tIdx < T_padded) {
              const wIdx = oc * (this.inCh * this.kernel) + ic * this.kernel + k;
              sum += this.weight[wIdx] * padded[ic * T_padded + tIdx];
            }
          }
        }
        output[oc * T_out + t] = sum;
      }
    }

    return { output, T_out };
  }
}

// ---------------------------------------------------------------------------
// Batch Normalization 1D
// ---------------------------------------------------------------------------

class BatchNorm1d {
  constructor(numFeatures, opts = {}) {
    this.numFeatures = numFeatures;
    this.eps = opts.eps || 1e-5;
    this.momentum = opts.momentum || 0.1;

    this.gamma = new Float32Array(numFeatures).fill(1.0);
    this.beta = new Float32Array(numFeatures);
    this.runMean = new Float32Array(numFeatures);
    this.runVar = new Float32Array(numFeatures).fill(1.0);
    this.initialized = false;
    this.training = true;
  }

  numParams() {
    return this.numFeatures * 2; // gamma + beta
  }

  /**
   * Forward: normalize across time dimension.
   * @param {Float32Array} input - [channels, T]
   * @param {number} T - time steps
   * @returns {Float32Array} - [channels, T]
   */
  forward(input, T) {
    const output = new Float32Array(input.length);

    if (this.training && T > 1) {
      // Compute batch stats per channel
      for (let c = 0; c < this.numFeatures; c++) {
        let mean = 0;
        for (let t = 0; t < T; t++) mean += input[c * T + t];
        mean /= T;

        let variance = 0;
        for (let t = 0; t < T; t++) variance += (input[c * T + t] - mean) ** 2;
        variance /= T;

        // Update running stats
        if (this.initialized) {
          this.runMean[c] = (1 - this.momentum) * this.runMean[c] + this.momentum * mean;
          this.runVar[c] = (1 - this.momentum) * this.runVar[c] + this.momentum * variance;
        } else {
          this.runMean[c] = mean;
          this.runVar[c] = variance;
        }

        // Normalize
        const invStd = 1.0 / Math.sqrt(variance + this.eps);
        for (let t = 0; t < T; t++) {
          output[c * T + t] = this.gamma[c] * (input[c * T + t] - mean) * invStd + this.beta[c];
        }
      }
      this.initialized = true;
    } else {
      // Use running stats (inference mode)
      for (let c = 0; c < this.numFeatures; c++) {
        const invStd = 1.0 / Math.sqrt(this.runVar[c] + this.eps);
        for (let t = 0; t < T; t++) {
          output[c * T + t] = this.gamma[c] * (input[c * T + t] - this.runMean[c]) * invStd + this.beta[c];
        }
      }
    }

    return output;
  }
}

// ---------------------------------------------------------------------------
// Stage 1: Temporal Convolutional Network (TCN)
// ---------------------------------------------------------------------------

/**
 * Single TCN block: DilatedCausalConv1d -> BN -> ReLU -> residual
 */
class TCNBlock {
  constructor(inCh, outCh, kernel, dilation, seed) {
    this.conv = new Conv1d(inCh, outCh, kernel, {
      dilation,
      causal: true,
      seed: seed || (inCh * 100 + dilation * 13),
    });
    this.bn = new BatchNorm1d(outCh);

    // 1x1 residual projection if channels differ
    this.residual = null;
    if (inCh !== outCh) {
      this.residual = new Conv1d(inCh, outCh, 1, {
        seed: seed ? seed + 999 : inCh * 200 + outCh * 7,
      });
    }
  }

  numParams() {
    let p = this.conv.numParams() + this.bn.numParams();
    if (this.residual) p += this.residual.numParams();
    return p;
  }

  forward(input, T) {
    const { output: convOut, T_out } = this.conv.forward(input, T);
    const bnOut = this.bn.forward(convOut, T_out);
    relu(bnOut);

    // Residual connection
    let res;
    if (this.residual) {
      const { output: resOut } = this.residual.forward(input, T);
      res = resOut;
    } else {
      res = input;
    }

    // Add residual (T_out should equal T for causal conv with same stride)
    const outCh = this.conv.outCh;
    for (let c = 0; c < outCh; c++) {
      for (let t = 0; t < T_out; t++) {
        bnOut[c * T_out + t] += res[c * T_out + t] || 0;
      }
    }

    return { output: bnOut, T_out };
  }
}

/**
 * Full TCN: 4 blocks with dilation (1, 2, 4, 8), kernel=7
 * Channel progression: inputCh -> 256 -> 192 -> 128 -> 128
 * Scaled to reach ~2.5M total model parameters with 128-subcarrier input.
 */
class TemporalConvNet {
  constructor(inputCh, seed) {
    seed = seed || 42;
    this.blocks = [
      new TCNBlock(inputCh, 256, 7, 1, seed),
      new TCNBlock(256, 192, 7, 2, seed + 100),
      new TCNBlock(192, 128, 7, 4, seed + 200),
      new TCNBlock(128, 128, 7, 8, seed + 300),
    ];
    this.outCh = 128;
  }

  numParams() {
    return this.blocks.reduce((s, b) => s + b.numParams(), 0);
  }

  forward(input, T) {
    let x = input;
    let t = T;
    for (const block of this.blocks) {
      const result = block.forward(x, t);
      x = result.output;
      t = result.T_out;
    }
    return { output: x, T_out: t, channels: this.outCh };
  }
}

// ---------------------------------------------------------------------------
// Stage 2: Asymmetric Convolution Encoder
// ---------------------------------------------------------------------------

/**
 * Single asymmetric conv block: 1xk conv in subcarrier dim + BN + ReLU + residual
 * Operates on [channels, H, W] where H = subcarrier features, W = time
 *
 * After TCN, data is [48, T]. We reshape to [1, 48, T] and treat dim-1 as
 * "subcarrier features" and dim-2 as "time".
 * Each block does a 1×3 conv in the subcarrier dimension with stride (1,2) downsampling.
 */
class AsymmetricConvBlock {
  constructor(inCh, outCh, kernel, strideH, seed) {
    this.inCh = inCh;
    this.outCh = outCh;
    this.kernel = kernel;
    this.strideH = strideH || 1;

    const rng = createRng(seed || (inCh * 37 + outCh * 11));

    // Weight: [outCh, inCh, kernel] applied along H dimension
    this.weight = initKaiming(inCh * kernel, outCh, rng);
    this.bias = new Float32Array(outCh);
    this.bn = new BatchNorm1d(outCh);

    // Residual 1x1 + stride
    this.residual = null;
    if (inCh !== outCh || strideH > 1) {
      this.residualWeight = initKaiming(inCh, outCh, createRng(seed ? seed + 500 : inCh * 53));
      this.residualBias = new Float32Array(outCh);
    }
  }

  numParams() {
    let p = this.weight.length + this.bias.length + this.bn.numParams();
    if (this.residualWeight) p += this.residualWeight.length + this.residualBias.length;
    return p;
  }

  /**
   * Forward pass.
   * @param {Float32Array} input - [inCh, H, W] flattened
   * @param {number} H - height (subcarrier features)
   * @param {number} W - width (time)
   * @returns {{ output: Float32Array, H_out: number, W_out: number }}
   */
  forward(input, H, W) {
    const pad = Math.floor((this.kernel - 1) / 2);
    const H_out = Math.floor((H + 2 * pad - this.kernel) / this.strideH) + 1;
    const W_out = W;

    // 1×k conv along H dimension
    const convOut = new Float32Array(this.outCh * H_out * W_out);

    for (let oc = 0; oc < this.outCh; oc++) {
      for (let h = 0; h < H_out; h++) {
        const hStart = h * this.strideH - pad;
        for (let w = 0; w < W_out; w++) {
          let sum = this.bias[oc];

          for (let ic = 0; ic < this.inCh; ic++) {
            for (let k = 0; k < this.kernel; k++) {
              const hIdx = hStart + k;
              if (hIdx >= 0 && hIdx < H) {
                const wIdx = oc * (this.inCh * this.kernel) + ic * this.kernel + k;
                sum += this.weight[wIdx] * input[ic * H * W + hIdx * W + w];
              }
            }
          }
          convOut[oc * H_out * W_out + h * W_out + w] = sum;
        }
      }
    }

    // BN across H_out * W_out as "time" dimension
    const bnOut = this.bn.forward(convOut, H_out * W_out);
    relu(bnOut);

    // Residual
    if (this.residualWeight) {
      // 1x1 conv + stride for residual
      for (let oc = 0; oc < this.outCh; oc++) {
        for (let h = 0; h < H_out; h++) {
          const hSrc = h * this.strideH;
          if (hSrc >= H) continue;
          for (let w = 0; w < W_out; w++) {
            let resVal = this.residualBias[oc];
            for (let ic = 0; ic < this.inCh; ic++) {
              resVal += this.residualWeight[oc * this.inCh + ic] * input[ic * H * W + hSrc * W + w];
            }
            bnOut[oc * H_out * W_out + h * W_out + w] += resVal;
          }
        }
      }
    } else {
      // Direct residual add
      const minH = Math.min(H_out, H);
      for (let c = 0; c < Math.min(this.outCh, this.inCh); c++) {
        for (let h = 0; h < minH; h++) {
          for (let w = 0; w < W_out; w++) {
            bnOut[c * H_out * W_out + h * W_out + w] += input[c * H * W + h * W + w];
          }
        }
      }
    }

    return { output: bnOut, H_out, W_out };
  }
}

/**
 * Full asymmetric encoder: 4 blocks
 * Channel progression: 1 -> 32 -> 64 -> 128 -> 256
 * H progression (with stride 2): 128 -> 64 -> 32 -> 16 -> 8
 */
class AsymmetricConvEncoder {
  constructor(seed) {
    seed = seed || 1000;
    this.blocks = [
      new AsymmetricConvBlock(1, 32, 3, 2, seed),
      new AsymmetricConvBlock(32, 64, 3, 2, seed + 100),
      new AsymmetricConvBlock(64, 128, 3, 2, seed + 200),
      new AsymmetricConvBlock(128, 256, 3, 2, seed + 300),
    ];
    this.outCh = 256;
  }

  numParams() {
    return this.blocks.reduce((s, b) => s + b.numParams(), 0);
  }

  /**
   * Forward: takes TCN output [48, T] and processes spatially.
   * Reshapes to [1, 48, T], then applies 4 blocks.
   * @param {Float32Array} input - [channels, T] from TCN
   * @param {number} channels - TCN output channels (48)
   * @param {number} T - time steps
   * @returns {{ output: Float32Array, channels: number, H: number, W: number }}
   */
  forward(input, channels, T) {
    // Reshape [channels, T] -> [1, channels, T]
    // block input: [inCh, H, W] where inCh=1, H=channels, W=T
    let x = new Float32Array(1 * channels * T);
    for (let h = 0; h < channels; h++) {
      for (let w = 0; w < T; w++) {
        x[0 * channels * T + h * T + w] = input[h * T + w];
      }
    }
    let H = channels;
    let W = T;
    let ch = 1;

    for (const block of this.blocks) {
      const result = block.forward(x, H, W);
      x = result.output;
      H = result.H_out;
      W = result.W_out;
      ch = block.outCh;
    }

    return { output: x, channels: ch, H, W };
  }
}

// ---------------------------------------------------------------------------
// Stage 3: Axial Self-Attention
// ---------------------------------------------------------------------------

/**
 * Single-axis attention: Q, K, V linear projections + scaled dot-product.
 * Operates along one axis (width or height) of [channels, H, W] tensor.
 */
class AxialAttention {
  constructor(channels, numHeads, axis, seed) {
    this.channels = channels;
    this.numHeads = numHeads;
    this.headDim = Math.floor(channels / numHeads);
    this.axis = axis; // 'width' (temporal) or 'height' (feature)

    const rng = createRng(seed || (channels * 17 + numHeads * 3));

    // Q, K, V projections: channels -> channels
    this.Wq = initXavier(channels, channels, rng);
    this.Wk = initXavier(channels, channels, createRng((seed || 0) + 1));
    this.Wv = initXavier(channels, channels, createRng((seed || 0) + 2));
    this.Wo = initXavier(channels, channels, createRng((seed || 0) + 3));

    // Biases
    this.bq = new Float32Array(channels);
    this.bk = new Float32Array(channels);
    this.bv = new Float32Array(channels);
    this.bo = new Float32Array(channels);

    // Learnable positional encoding (max length 128)
    this.maxLen = 128;
    const posRng = createRng((seed || 0) + 10);
    this.posEnc = new Float32Array(this.maxLen * channels);
    const posScale = 0.02;
    for (let i = 0; i < this.posEnc.length; i++) {
      this.posEnc[i] = (posRng() - 0.5) * posScale;
    }
  }

  numParams() {
    return this.Wq.length + this.Wk.length + this.Wv.length + this.Wo.length +
           this.bq.length + this.bk.length + this.bv.length + this.bo.length +
           this.posEnc.length;
  }

  /**
   * Linear projection: x [N, C] @ W [C, C] + b [C] -> [N, C]
   */
  _project(x, N, C, W, b) {
    const out = new Float32Array(N * C);
    for (let n = 0; n < N; n++) {
      for (let j = 0; j < C; j++) {
        let sum = b[j];
        for (let i = 0; i < C; i++) {
          sum += x[n * C + i] * W[i * C + j];
        }
        out[n * C + j] = sum;
      }
    }
    return out;
  }

  /**
   * Forward: applies attention along the specified axis.
   * @param {Float32Array} input - [channels, H, W] flattened
   * @param {number} H
   * @param {number} W
   * @returns {Float32Array} - same shape
   */
  forward(input, H, W) {
    const C = this.channels;
    const output = new Float32Array(input.length);

    if (this.axis === 'width') {
      // Attention along W (temporal axis) for each row h
      for (let h = 0; h < H; h++) {
        // Extract row: [W, C] where each position has C channels
        const row = new Float32Array(W * C);
        for (let w = 0; w < W; w++) {
          for (let c = 0; c < C; c++) {
            row[w * C + c] = input[c * H * W + h * W + w];
          }
          // Add positional encoding
          if (w < this.maxLen) {
            for (let c = 0; c < C; c++) {
              row[w * C + c] += this.posEnc[w * C + c];
            }
          }
        }

        // Q, K, V projections: [W, C]
        const Q = this._project(row, W, C, this.Wq, this.bq);
        const K = this._project(row, W, C, this.Wk, this.bk);
        const V = this._project(row, W, C, this.Wv, this.bv);

        // Multi-head attention
        const attnOut = this._multiheadAttention(Q, K, V, W);

        // Output projection
        const projected = this._project(attnOut, W, C, this.Wo, this.bo);

        // Write back + residual
        for (let w = 0; w < W; w++) {
          for (let c = 0; c < C; c++) {
            output[c * H * W + h * W + w] = input[c * H * W + h * W + w] + projected[w * C + c];
          }
        }
      }
    } else {
      // Attention along H (feature axis) for each column w
      for (let w = 0; w < W; w++) {
        const col = new Float32Array(H * C);
        for (let h = 0; h < H; h++) {
          for (let c = 0; c < C; c++) {
            col[h * C + c] = input[c * H * W + h * W + w];
          }
          if (h < this.maxLen) {
            for (let c = 0; c < C; c++) {
              col[h * C + c] += this.posEnc[h * C + c];
            }
          }
        }

        const Q = this._project(col, H, C, this.Wq, this.bq);
        const K = this._project(col, H, C, this.Wk, this.bk);
        const V = this._project(col, H, C, this.Wv, this.bv);

        const attnOut = this._multiheadAttention(Q, K, V, H);
        const projected = this._project(attnOut, H, C, this.Wo, this.bo);

        for (let h = 0; h < H; h++) {
          for (let c = 0; c < C; c++) {
            output[c * H * W + h * W + w] = input[c * H * W + h * W + w] + projected[h * C + c];
          }
        }
      }
    }

    return output;
  }

  /**
   * Multi-head scaled dot-product attention.
   * @param {Float32Array} Q - [N, C]
   * @param {Float32Array} K - [N, C]
   * @param {Float32Array} V - [N, C]
   * @param {number} N - sequence length
   * @returns {Float32Array} - [N, C]
   */
  _multiheadAttention(Q, K, V, N) {
    const C = this.channels;
    const H = this.numHeads;
    const D = this.headDim;
    const scale = 1.0 / Math.sqrt(D);

    const output = new Float32Array(N * C);

    for (let head = 0; head < H; head++) {
      const dOff = head * D;

      // Compute attention scores: [N, N]
      const scores = new Float32Array(N * N);
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          let dot = 0;
          for (let d = 0; d < D; d++) {
            dot += Q[i * C + dOff + d] * K[j * C + dOff + d];
          }
          scores[i * N + j] = dot * scale;
        }
        // Softmax over j for this row i
        softmax(scores, i * N, N);
      }

      // Apply attention to V: [N, D]
      for (let i = 0; i < N; i++) {
        for (let d = 0; d < D; d++) {
          let sum = 0;
          for (let j = 0; j < N; j++) {
            sum += scores[i * N + j] * V[j * C + dOff + d];
          }
          output[i * C + dOff + d] = sum;
        }
      }
    }

    return output;
  }
}

/**
 * Axial Self-Attention: width attention (temporal) then height attention (feature).
 */
class AxialSelfAttention {
  constructor(channels, numHeads, seed) {
    seed = seed || 2000;
    this.widthAttn = new AxialAttention(channels, numHeads, 'width', seed);
    this.heightAttn = new AxialAttention(channels, numHeads, 'height', seed + 500);
    this.channels = channels;
  }

  numParams() {
    return this.widthAttn.numParams() + this.heightAttn.numParams();
  }

  forward(input, H, W) {
    const afterWidth = this.widthAttn.forward(input, H, W);
    const afterHeight = this.heightAttn.forward(afterWidth, H, W);
    return afterHeight;
  }
}

// ---------------------------------------------------------------------------
// Decoder: Adaptive Average Pooling + Linear -> 17 COCO keypoints x 2
// ---------------------------------------------------------------------------

/**
 * COCO skeleton: 17 keypoints
 * 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
 * 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
 * 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
 * 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
 */
const COCO_KEYPOINTS = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
];

const BONE_CONNECTIONS = [
  [0, 1], [0, 2],         // nose -> eyes
  [1, 3], [2, 4],         // eyes -> ears
  [5, 7], [7, 9],         // left arm
  [6, 8], [8, 10],        // right arm
  [5, 11], [6, 12],       // torso
  [11, 13], [13, 15],     // left leg
  [12, 14], [14, 16],     // right leg
  [5, 6],                 // shoulder width
];

/** Bone length priors normalized to person height */
const BONE_LENGTH_PRIORS = [
  0.06, 0.06,   // nose-eye (x2)
  0.06, 0.06,   // eye-ear (x2)
  0.15, 0.13,   // left shoulder-elbow, elbow-wrist
  0.15, 0.13,   // right shoulder-elbow, elbow-wrist
  0.26, 0.26,   // shoulder-hip (x2)
  0.25, 0.25,   // left hip-knee, knee-ankle
  0.25, 0.25,   // right hip-knee, knee-ankle
  0.20,         // shoulder width
];

class PoseDecoder {
  constructor(inFeatures, numKeypoints, seed) {
    this.inFeatures = inFeatures;
    this.numKeypoints = numKeypoints || 17;
    this.outDim = this.numKeypoints * 2;

    const rng = createRng(seed || 3000);
    // Linear: inFeatures -> numKeypoints * 2
    this.weight = initXavier(inFeatures, this.outDim, rng);
    this.bias = new Float32Array(this.outDim);

    // Initialize bias to center of room (0.5, 0.5) for each keypoint
    for (let k = 0; k < this.numKeypoints; k++) {
      this.bias[k * 2] = 0.5;     // x
      this.bias[k * 2 + 1] = 0.5; // y
    }
  }

  numParams() {
    return this.weight.length + this.bias.length;
  }

  /**
   * Forward: adaptive average pooling over temporal dim, then linear.
   * @param {Float32Array} input - [channels, H, W]
   * @param {number} channels
   * @param {number} H
   * @param {number} W
   * @returns {Float32Array} - [numKeypoints * 2] keypoint coordinates
   */
  forward(input, channels, H, W) {
    // Adaptive average pooling: [channels, H, W] -> [channels * H]
    // Average over W (temporal dimension)
    const pooled = new Float32Array(channels * H);
    for (let c = 0; c < channels; c++) {
      for (let h = 0; h < H; h++) {
        let sum = 0;
        for (let w = 0; w < W; w++) {
          sum += input[c * H * W + h * W + w];
        }
        pooled[c * H + h] = sum / W;
      }
    }

    // Linear projection: [channels * H] -> [numKeypoints * 2]
    const featureDim = channels * H;
    const out = new Float32Array(this.outDim);

    // If featureDim != inFeatures, truncate or zero-pad
    const useDim = Math.min(featureDim, this.inFeatures);

    for (let j = 0; j < this.outDim; j++) {
      let sum = this.bias[j];
      for (let i = 0; i < useDim; i++) {
        sum += pooled[i] * this.weight[i * this.outDim + j];
      }
      // Sigmoid to normalize output to [0, 1]
      out[j] = 1.0 / (1.0 + Math.exp(-sum));
    }

    return out;
  }
}

// ---------------------------------------------------------------------------
// WiFlow Model: Full Pipeline
// ---------------------------------------------------------------------------

class WiFlowModel {
  /**
   * @param {object} config
   * @param {number} config.inputChannels - CSI subcarrier count (default: 128)
   * @param {number} config.timeSteps - temporal window (default: 20)
   * @param {number} config.numKeypoints - COCO keypoints (default: 17)
   * @param {number} config.numHeads - attention heads (default: 8)
   * @param {number} config.seed - random seed (default: 42)
   */
  constructor(config = {}) {
    this.inputChannels = config.inputChannels || 128;
    this.timeSteps = config.timeSteps || 20;
    this.numKeypoints = config.numKeypoints || 17;
    this.numHeads = config.numHeads || 8;
    this.seed = config.seed || 42;
    this.training = true;

    // Stage 1: TCN (inputChannels -> 128 channels, preserves time)
    this.tcn = new TemporalConvNet(this.inputChannels, this.seed);

    // Stage 2: Asymmetric Conv (128 TCN features -> 8 via stride-2 downsampling)
    // Input: [1, 128, T] -> [256, 8, T]
    this.spatialEncoder = new AsymmetricConvEncoder(this.seed + 1000);

    // Stage 3: Axial Self-Attention on [256, 8, T]
    this.axialAttention = new AxialSelfAttention(256, this.numHeads, this.seed + 2000);

    // Decoder: [256, 8, T] -> 17 * 2
    // After pooling over T: feature dim = 256 * 8 = 2048
    this.decoder = new PoseDecoder(2048, this.numKeypoints, this.seed + 3000);
  }

  /** Total parameter count */
  numParams() {
    return this.tcn.numParams() +
           this.spatialEncoder.numParams() +
           this.axialAttention.numParams() +
           this.decoder.numParams();
  }

  /** Parameter breakdown by stage */
  paramBreakdown() {
    return {
      tcn: this.tcn.numParams(),
      spatialEncoder: this.spatialEncoder.numParams(),
      axialAttention: this.axialAttention.numParams(),
      decoder: this.decoder.numParams(),
      total: this.numParams(),
    };
  }

  /** Set training/eval mode */
  setTraining(mode) {
    this.training = mode;
    // Propagate to BatchNorm layers
    const setBnMode = (obj) => {
      if (obj && obj.bn) obj.bn.training = mode;
      if (obj && obj.blocks) obj.blocks.forEach(b => setBnMode(b));
      if (obj && obj.conv && obj.conv.bn) obj.conv.bn = mode;
    };
    setBnMode(this.tcn);
    setBnMode(this.spatialEncoder);
  }

  /**
   * Forward pass: CSI amplitude -> 17 keypoint coordinates.
   *
   * @param {Float32Array} csiAmplitude - [inputChannels, timeSteps] flattened
   *   or [batch, inputChannels, timeSteps] for batched inference.
   * @param {number} [batchSize=1]
   * @returns {Float32Array|Float32Array[]} - [numKeypoints * 2] or array of them
   */
  forward(csiAmplitude, batchSize) {
    batchSize = batchSize || 1;

    if (batchSize === 1) {
      return this._forwardSingle(csiAmplitude);
    }

    // Batched inference
    const results = [];
    const singleSize = this.inputChannels * this.timeSteps;
    for (let b = 0; b < batchSize; b++) {
      const slice = csiAmplitude.slice(b * singleSize, (b + 1) * singleSize);
      results.push(this._forwardSingle(slice));
    }
    return results;
  }

  /**
   * Single-sample forward pass.
   * @param {Float32Array} input - [inputChannels, timeSteps]
   * @returns {Float32Array} - [numKeypoints * 2]
   */
  _forwardSingle(input) {
    // Stage 1: TCN
    const tcnResult = this.tcn.forward(input, this.timeSteps);

    // Stage 2: Asymmetric Conv
    const spatialResult = this.spatialEncoder.forward(
      tcnResult.output, tcnResult.channels, tcnResult.T_out
    );

    // Stage 3: Axial Attention
    const attnOutput = this.axialAttention.forward(
      spatialResult.output, spatialResult.H, spatialResult.W
    );

    // Decoder
    const keypoints = this.decoder.forward(
      attnOutput, spatialResult.channels, spatialResult.H, spatialResult.W
    );

    return keypoints;
  }

  /**
   * Compute WiFlow loss: L = L_H + 0.2 * L_B
   * L_H = SmoothL1(predicted, target, beta=0.1)
   * L_B = bone length constraint violation
   *
   * @param {Float32Array} predicted - [numKeypoints * 2]
   * @param {Float32Array} target - [numKeypoints * 2]
   * @param {boolean} boneConstraints - include bone length loss
   * @returns {{ total: number, smoothL1: number, boneLoss: number }}
   */
  computeLoss(predicted, target, boneConstraints) {
    if (boneConstraints === undefined) boneConstraints = true;

    const lH = smoothL1(predicted, target, 0.1);

    let lB = 0;
    if (boneConstraints) {
      for (let b = 0; b < BONE_CONNECTIONS.length; b++) {
        const [i, j] = BONE_CONNECTIONS[b];
        const prior = BONE_LENGTH_PRIORS[b];

        const dx = predicted[i * 2] - predicted[j * 2];
        const dy = predicted[i * 2 + 1] - predicted[j * 2 + 1];
        const boneLen = Math.sqrt(dx * dx + dy * dy);

        // Penalty for deviation from prior (squared difference)
        const deviation = boneLen - prior;
        lB += deviation * deviation;
      }
      lB /= BONE_CONNECTIONS.length;
    }

    return {
      total: lH + 0.2 * lB,
      smoothL1: lH,
      boneLoss: lB,
    };
  }

  /**
   * Compute loss gradient w.r.t. predicted keypoints.
   * @param {Float32Array} predicted - [numKeypoints * 2]
   * @param {Float32Array} target - [numKeypoints * 2]
   * @returns {Float32Array} - gradient [numKeypoints * 2]
   */
  computeLossGrad(predicted, target) {
    const n = predicted.length;
    const grad = smoothL1Grad(predicted, target, 0.1);

    // Bone constraint gradient
    for (let b = 0; b < BONE_CONNECTIONS.length; b++) {
      const [i, j] = BONE_CONNECTIONS[b];
      const prior = BONE_LENGTH_PRIORS[b];

      const dx = predicted[i * 2] - predicted[j * 2];
      const dy = predicted[i * 2 + 1] - predicted[j * 2 + 1];
      const boneLen = Math.sqrt(dx * dx + dy * dy) || 1e-8;

      const deviation = boneLen - prior;
      const scale = 0.2 * 2 * deviation / (boneLen * BONE_CONNECTIONS.length);

      grad[i * 2] += scale * dx;
      grad[i * 2 + 1] += scale * dy;
      grad[j * 2] -= scale * dx;
      grad[j * 2 + 1] -= scale * dy;
    }

    return grad;
  }

  /**
   * Compute PCK@threshold (Percentage of Correct Keypoints).
   * @param {Float32Array} predicted - [numKeypoints * 2]
   * @param {Float32Array} target - [numKeypoints * 2]
   * @param {number} threshold - distance threshold (normalized coords)
   * @returns {number} - fraction of keypoints within threshold
   */
  static pck(predicted, target, threshold) {
    threshold = threshold || 0.2;
    let correct = 0;
    const nk = Math.floor(predicted.length / 2);
    for (let k = 0; k < nk; k++) {
      const dx = predicted[k * 2] - target[k * 2];
      const dy = predicted[k * 2 + 1] - target[k * 2 + 1];
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist <= threshold) correct++;
    }
    return correct / nk;
  }

  /**
   * Compute bone length violation rate.
   * @param {Float32Array} predicted - [numKeypoints * 2]
   * @param {number} tolerance - allowed deviation as fraction of prior
   * @returns {{ violationRate: number, violations: number[] }}
   */
  static boneViolations(predicted, tolerance) {
    tolerance = tolerance || 0.5; // 50% deviation tolerance
    const violations = [];
    for (let b = 0; b < BONE_CONNECTIONS.length; b++) {
      const [i, j] = BONE_CONNECTIONS[b];
      const prior = BONE_LENGTH_PRIORS[b];

      const dx = predicted[i * 2] - predicted[j * 2];
      const dy = predicted[i * 2 + 1] - predicted[j * 2 + 1];
      const boneLen = Math.sqrt(dx * dx + dy * dy);

      if (Math.abs(boneLen - prior) > prior * tolerance) {
        violations.push(b);
      }
    }
    return {
      violationRate: violations.length / BONE_CONNECTIONS.length,
      violations,
    };
  }

  /**
   * Get all weights as a flat Float32Array (for quantization / export).
   */
  getAllWeights() {
    const arrays = [];

    // Collect all weight arrays from each stage
    const collectConv = (conv) => {
      arrays.push(conv.weight);
      if (conv.bias) arrays.push(conv.bias);
    };
    const collectBN = (bn) => {
      arrays.push(bn.gamma);
      arrays.push(bn.beta);
    };

    // TCN
    for (const block of this.tcn.blocks) {
      collectConv(block.conv);
      collectBN(block.bn);
      if (block.residual) collectConv(block.residual);
    }

    // Spatial encoder
    for (const block of this.spatialEncoder.blocks) {
      arrays.push(block.weight);
      arrays.push(block.bias);
      collectBN(block.bn);
      if (block.residualWeight) {
        arrays.push(block.residualWeight);
        arrays.push(block.residualBias);
      }
    }

    // Axial attention
    for (const attn of [this.axialAttention.widthAttn, this.axialAttention.heightAttn]) {
      arrays.push(attn.Wq, attn.Wk, attn.Wv, attn.Wo);
      arrays.push(attn.bq, attn.bk, attn.bv, attn.bo);
      arrays.push(attn.posEnc);
    }

    // Decoder
    arrays.push(this.decoder.weight);
    arrays.push(this.decoder.bias);

    // Flatten
    let totalLen = 0;
    for (const a of arrays) totalLen += a.length;
    const flat = new Float32Array(totalLen);
    let offset = 0;
    for (const a of arrays) {
      flat.set(a, offset);
      offset += a.length;
    }
    return flat;
  }

  /**
   * Export model as a named tensor map (for SafeTensors).
   * @returns {Map<string, Float32Array>}
   */
  toTensorMap() {
    const tensors = new Map();

    // TCN
    for (let i = 0; i < this.tcn.blocks.length; i++) {
      const b = this.tcn.blocks[i];
      tensors.set(`tcn.block${i}.conv.weight`, b.conv.weight);
      if (b.conv.bias) tensors.set(`tcn.block${i}.conv.bias`, b.conv.bias);
      tensors.set(`tcn.block${i}.bn.gamma`, b.bn.gamma);
      tensors.set(`tcn.block${i}.bn.beta`, b.bn.beta);
      tensors.set(`tcn.block${i}.bn.runMean`, b.bn.runMean);
      tensors.set(`tcn.block${i}.bn.runVar`, b.bn.runVar);
      if (b.residual) {
        tensors.set(`tcn.block${i}.residual.weight`, b.residual.weight);
        if (b.residual.bias) tensors.set(`tcn.block${i}.residual.bias`, b.residual.bias);
      }
    }

    // Spatial encoder
    for (let i = 0; i < this.spatialEncoder.blocks.length; i++) {
      const b = this.spatialEncoder.blocks[i];
      tensors.set(`spatial.block${i}.weight`, b.weight);
      tensors.set(`spatial.block${i}.bias`, b.bias);
      tensors.set(`spatial.block${i}.bn.gamma`, b.bn.gamma);
      tensors.set(`spatial.block${i}.bn.beta`, b.bn.beta);
      tensors.set(`spatial.block${i}.bn.runMean`, b.bn.runMean);
      tensors.set(`spatial.block${i}.bn.runVar`, b.bn.runVar);
      if (b.residualWeight) {
        tensors.set(`spatial.block${i}.residual.weight`, b.residualWeight);
        tensors.set(`spatial.block${i}.residual.bias`, b.residualBias);
      }
    }

    // Axial attention
    for (const [name, attn] of [['width', this.axialAttention.widthAttn], ['height', this.axialAttention.heightAttn]]) {
      tensors.set(`axial.${name}.Wq`, attn.Wq);
      tensors.set(`axial.${name}.Wk`, attn.Wk);
      tensors.set(`axial.${name}.Wv`, attn.Wv);
      tensors.set(`axial.${name}.Wo`, attn.Wo);
      tensors.set(`axial.${name}.bq`, attn.bq);
      tensors.set(`axial.${name}.bk`, attn.bk);
      tensors.set(`axial.${name}.bv`, attn.bv);
      tensors.set(`axial.${name}.bo`, attn.bo);
      tensors.set(`axial.${name}.posEnc`, attn.posEnc);
    }

    // Decoder
    tensors.set('decoder.weight', this.decoder.weight);
    tensors.set('decoder.bias', this.decoder.bias);

    return tensors;
  }

  /**
   * Load weights from a tensor map (from SafeTensors).
   * @param {Map<string, Float32Array>} tensors
   */
  fromTensorMap(tensors) {
    const load = (key, target) => {
      const src = tensors.get(key);
      if (src && src.length === target.length) {
        target.set(src);
      }
    };

    for (let i = 0; i < this.tcn.blocks.length; i++) {
      const b = this.tcn.blocks[i];
      load(`tcn.block${i}.conv.weight`, b.conv.weight);
      if (b.conv.bias) load(`tcn.block${i}.conv.bias`, b.conv.bias);
      load(`tcn.block${i}.bn.gamma`, b.bn.gamma);
      load(`tcn.block${i}.bn.beta`, b.bn.beta);
      load(`tcn.block${i}.bn.runMean`, b.bn.runMean);
      load(`tcn.block${i}.bn.runVar`, b.bn.runVar);
      if (b.residual) {
        load(`tcn.block${i}.residual.weight`, b.residual.weight);
        if (b.residual.bias) load(`tcn.block${i}.residual.bias`, b.residual.bias);
      }
    }

    for (let i = 0; i < this.spatialEncoder.blocks.length; i++) {
      const b = this.spatialEncoder.blocks[i];
      load(`spatial.block${i}.weight`, b.weight);
      load(`spatial.block${i}.bias`, b.bias);
      load(`spatial.block${i}.bn.gamma`, b.bn.gamma);
      load(`spatial.block${i}.bn.beta`, b.bn.beta);
      load(`spatial.block${i}.bn.runMean`, b.bn.runMean);
      load(`spatial.block${i}.bn.runVar`, b.bn.runVar);
      if (b.residualWeight) {
        load(`spatial.block${i}.residual.weight`, b.residualWeight);
        load(`spatial.block${i}.residual.bias`, b.residualBias);
      }
    }

    for (const [name, attn] of [['width', this.axialAttention.widthAttn], ['height', this.axialAttention.heightAttn]]) {
      load(`axial.${name}.Wq`, attn.Wq);
      load(`axial.${name}.Wk`, attn.Wk);
      load(`axial.${name}.Wv`, attn.Wv);
      load(`axial.${name}.Wo`, attn.Wo);
      load(`axial.${name}.bq`, attn.bq);
      load(`axial.${name}.bk`, attn.bk);
      load(`axial.${name}.bv`, attn.bv);
      load(`axial.${name}.bo`, attn.bo);
      load(`axial.${name}.posEnc`, attn.posEnc);
    }

    load('decoder.weight', this.decoder.weight);
    load('decoder.bias', this.decoder.bias);
  }
}

// ---------------------------------------------------------------------------
// FLOPs estimation
// ---------------------------------------------------------------------------

/**
 * Estimate FLOPs per forward pass for each stage.
 */
function estimateFLOPs(config) {
  config = config || {};
  const C = config.inputChannels || 128;
  const T = config.timeSteps || 20;
  const K = 7; // TCN kernel

  let flops = {};

  // Stage 1: TCN - 4 dilated causal conv blocks
  // Each conv: 2 * inCh * outCh * K * T
  const tcnLayers = [
    { inCh: C, outCh: 256 },
    { inCh: 256, outCh: 192 },
    { inCh: 192, outCh: 128 },
    { inCh: 128, outCh: 128 },
  ];
  flops.tcn = 0;
  for (const l of tcnLayers) {
    flops.tcn += 2 * l.inCh * l.outCh * K * T;
    // BN: 4 * outCh * T
    flops.tcn += 4 * l.outCh * T;
    // Residual 1x1 if channels differ
    if (l.inCh !== l.outCh) flops.tcn += 2 * l.inCh * l.outCh * T;
  }

  // Stage 2: Asymmetric conv
  const spatialLayers = [
    { inCh: 1, outCh: 32, Hin: 128, Hout: 64 },
    { inCh: 32, outCh: 64, Hin: 64, Hout: 32 },
    { inCh: 64, outCh: 128, Hin: 32, Hout: 16 },
    { inCh: 128, outCh: 256, Hin: 16, Hout: 8 },
  ];
  flops.spatialEncoder = 0;
  for (const l of spatialLayers) {
    flops.spatialEncoder += 2 * l.inCh * l.outCh * 3 * l.Hout * T;
    flops.spatialEncoder += 4 * l.outCh * l.Hout * T;
    flops.spatialEncoder += 2 * l.inCh * l.outCh * l.Hout * T; // residual
  }

  // Stage 3: Axial attention
  // Width attention: H * (3 * C * C + C * W * W) for each of H rows
  const attnC = 256, attnH = 8, attnW = T;
  flops.axialAttention = 0;
  // Width: for each of H rows, project W tokens, compute W*W attention
  flops.axialAttention += attnH * (3 * attnW * attnC * attnC + attnW * attnW * attnC + attnW * attnC * attnC);
  // Height: for each of W cols, project H tokens, compute H*H attention
  flops.axialAttention += attnW * (3 * attnH * attnC * attnC + attnH * attnH * attnC + attnH * attnC * attnC);

  // Decoder
  const featureDim = 256 * 8; // after pooling
  flops.decoder = 2 * featureDim * 34; // 17*2 outputs

  flops.total = flops.tcn + flops.spatialEncoder + flops.axialAttention + flops.decoder;

  return flops;
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

module.exports = {
  // Core model classes
  WiFlowModel,
  TemporalConvNet,
  AsymmetricConvEncoder,
  AxialSelfAttention,
  AxialAttention,
  PoseDecoder,
  Conv1d,
  BatchNorm1d,
  TCNBlock,
  AsymmetricConvBlock,

  // Constants
  COCO_KEYPOINTS,
  BONE_CONNECTIONS,
  BONE_LENGTH_PRIORS,

  // Utility functions
  smoothL1,
  smoothL1Grad,
  softmax,
  relu,
  initKaiming,
  initXavier,
  createRng,
  gaussianRng,
  estimateFLOPs,
};
