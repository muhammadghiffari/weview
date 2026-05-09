#!/usr/bin/env node
/**
 * WiFi-DensePose Camera-Free Training Pipeline
 *
 * Extends train-ruvllm.js with multi-modal supervision from Cognitum Seed sensors.
 * Trains a full pose estimation model using 10 sensor signals — NO camera required.
 *
 * Supervision signals:
 *   1. PIR sensor (Seed GPIO 6) — binary presence ground truth
 *   2. BME280 temperature (Seed I2C 0x76) — occupancy proxy
 *   3. BME280 humidity (Seed I2C 0x76) — breathing confirmation
 *   4. Cross-node RSSI differential — rough XY position
 *   5. Vitals stability — HR/BR variance → activity level
 *   6. Temporal CSI patterns — periodic=walking, stable=sitting, flat=empty
 *   7. kNN cluster labels — natural groupings in vector store
 *   8. Boundary fragility — Stoer-Wagner min-cut detects regime changes
 *   9. Reed switch (Seed GPIO 5) — door open/close events
 *  10. Vibration sensor (Seed GPIO 13) — footstep detection
 *
 * Usage:
 *   node scripts/train-camera-free.js --data data/recordings/pretrain-*.csi.jsonl
 *   node scripts/train-camera-free.js --data data/recordings/*.csi.jsonl --seed-url https://169.254.42.1:8443
 *   node scripts/train-camera-free.js --data data/recordings/*.csi.jsonl --output models/csi-camerafree-v1 --benchmark
 *
 * Falls back to CSI-only training (train-ruvllm.js pipeline) if Seed is unavailable.
 *
 * ADR: docs/adr/ADR-071-ruvllm-training-pipeline.md (Camera-Free Supervision section)
 */

'use strict';

const fs = require('fs');
const path = require('path');
const https = require('https');
const { parseArgs } = require('util');

// ---------------------------------------------------------------------------
// Resolve ruvllm from vendor tree
// ---------------------------------------------------------------------------
const RUVLLM_PATH = path.resolve(__dirname, '..', 'vendor', 'ruvector', 'npm', 'packages', 'ruvllm', 'src');

const {
  ContrastiveTrainer,
  cosineSimilarity,
  tripletLoss,
  infoNCELoss,
  computeGradient,
} = require(path.join(RUVLLM_PATH, 'contrastive.js'));

const {
  TrainingPipeline,
} = require(path.join(RUVLLM_PATH, 'training.js'));

const {
  LoraAdapter,
  LoraManager,
} = require(path.join(RUVLLM_PATH, 'lora.js'));

const {
  EwcManager,
  ReasoningBank,
  SonaCoordinator,
} = require(path.join(RUVLLM_PATH, 'sona.js'));

const {
  SafeTensorsWriter,
  ModelExporter,
  DatasetExporter,
} = require(path.join(RUVLLM_PATH, 'export.js'));

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------
const { values: args } = parseArgs({
  options: {
    data:             { type: 'string',  short: 'd' },
    output:           { type: 'string',  short: 'o', default: 'models/csi-camerafree' },
    benchmark:        { type: 'boolean', short: 'b', default: false },
    epochs:           { type: 'string',  short: 'e', default: '20' },
    'batch-size':     { type: 'string',  default: '32' },
    'lora-rank':      { type: 'string',  default: '4' },
    'quantize-bits':  { type: 'string',  default: '4' },
    'seed-url':       { type: 'string',  default: 'https://169.254.42.1:8443' },
    'seed-token':     { type: 'string',  default: '' },
    'seed-collect-sec': { type: 'string', default: '120' },
    'self-refine':    { type: 'string',  default: '3' },
    'no-seed':        { type: 'boolean', default: false },
    verbose:          { type: 'boolean', short: 'v', default: false },
  },
  strict: true,
});

if (!args.data) {
  console.error('Usage: node scripts/train-camera-free.js --data <path-to-csi-jsonl> [--seed-url URL] [--output dir]');
  process.exit(1);
}

const CONFIG = {
  dataGlob: args.data,
  outputDir: args.output,
  benchmark: args.benchmark,
  epochs: parseInt(args.epochs, 10),
  batchSize: parseInt(args['batch-size'], 10),
  loraRank: parseInt(args['lora-rank'], 10),
  quantizeBits: parseInt(args['quantize-bits'], 10),
  verbose: args.verbose,

  // Seed connection
  seedUrl: args['seed-url'],
  seedToken: args['seed-token'] || process.env.SEED_TOKEN || '',
  seedCollectSec: parseInt(args['seed-collect-sec'], 10),
  noSeed: args['no-seed'],

  // Self-refinement rounds
  selfRefineRounds: parseInt(args['self-refine'], 10),

  // Contrastive training hyperparameters
  margin: 0.3,
  temperature: 0.07,
  hardNegativeRatio: 0.7,
  learningRate: 0.001,

  // Temporal window thresholds (seconds)
  positiveWindowSec: 1.0,
  negativeWindowSec: 10.0,

  // Data augmentation
  augmentMultiplier: 10,

  // Feature dimensions
  inputDim: 8,
  hiddenDim: 64,
  embeddingDim: 128,

  // Multi-modal dimensions
  seedEmbeddingDim: 45,
  multiModalInputDim: 8 + 8 + 4 + 4 + 45 + 6 + 2,  // 77-dim combined
  poseKeypoints5: 5,     // head, L_hand, R_hand, L_foot, R_foot
  poseKeypoints17: 17,   // COCO 17-keypoint format
  positionGridSize: 5,   // 5x5 grid = 25 zones

  // Anthropometric skeleton constraints (meters)
  skeleton: {
    upperArmLen:  0.30,
    forearmLen:   0.25,
    thighLen:     0.42,
    shinLen:      0.40,
    torsoLen:     0.50,
    shoulderWidth: 0.40,
    hipWidth:     0.28,
  },
};

// ---------------------------------------------------------------------------
// Seed API client (HTTPS with self-signed cert support)
// ---------------------------------------------------------------------------

class SeedClient {
  constructor(baseUrl, token) {
    this.baseUrl = baseUrl;
    this.token = token;
    this.available = false;
  }

  /**
   * Make an HTTPS GET request to the Seed API.
   * Returns parsed JSON or null on failure.
   */
  _get(endpoint) {
    return new Promise((resolve) => {
      const url = `${this.baseUrl}${endpoint}`;
      const headers = {};
      if (this.token) headers['Authorization'] = `Bearer ${this.token}`;

      const req = https.get(url, {
        rejectUnauthorized: false,  // self-signed cert on Seed
        headers,
        timeout: 5000,
      }, (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          try { resolve(JSON.parse(data)); }
          catch (_) { resolve(null); }
        });
      });
      req.on('error', () => resolve(null));
      req.on('timeout', () => { req.destroy(); resolve(null); });
    });
  }

  /**
   * Make an HTTPS POST request.
   */
  _post(endpoint, body) {
    return new Promise((resolve) => {
      const url = new URL(`${this.baseUrl}${endpoint}`);
      const bodyStr = body ? JSON.stringify(body) : '';
      const headers = { 'Content-Type': 'application/json' };
      if (this.token) headers['Authorization'] = `Bearer ${this.token}`;

      const opts = {
        hostname: url.hostname,
        port: url.port,
        path: url.pathname,
        method: 'POST',
        rejectUnauthorized: false,
        headers,
        timeout: 10000,
      };

      const req = https.request(opts, (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          try { resolve(JSON.parse(data)); }
          catch (_) { resolve(null); }
        });
      });
      req.on('error', () => resolve(null));
      req.on('timeout', () => { req.destroy(); resolve(null); });
      req.write(bodyStr);
      req.end();
    });
  }

  /** Check if the Seed API is reachable. */
  async probe() {
    const result = await this._get('/api/v1/sensor/list');
    this.available = result !== null;
    return this.available;
  }

  /** Get latest 45-dim sensor embedding. */
  async getEmbedding() {
    return this._get('/api/v1/sensor/embedding/latest');
  }

  /** Get sensor readings list. */
  async getSensors() {
    return this._get('/api/v1/sensor/list');
  }

  /** Get boundary fragility score. */
  async getBoundary() {
    return this._get('/api/v1/boundary');
  }

  /** Get coherence profile (temporal phase boundaries). */
  async getCoherence() {
    return this._get('/api/v1/coherence/profile');
  }

  /** Get drift detection status. */
  async getDrift() {
    return this._get('/api/v1/sensor/drift/status');
  }

  /** kNN query in vector store. */
  async queryStore(embedding, k = 5) {
    return this._post('/api/v1/store/query', { embedding, k });
  }

  /** Cognitive snapshot (spectral graph analysis). */
  async getCognitiveSnapshot() {
    return this._get('/api/v1/cognitive/snapshot');
  }

  /** Trigger boundary recomputation. */
  async recomputeBoundary() {
    return this._post('/api/v1/boundary/recompute', {});
  }

  /**
   * Open an SSE stream of sensor readings for durationMs.
   * Collects all events and returns them as an array.
   */
  streamSensors(durationMs) {
    return new Promise((resolve) => {
      const events = [];
      const url = `${this.baseUrl}/api/v1/sensor/stream`;
      const headers = {};
      if (this.token) headers['Authorization'] = `Bearer ${this.token}`;

      const req = https.get(url, {
        rejectUnauthorized: false,
        headers,
        timeout: durationMs + 5000,
      }, (res) => {
        let buffer = '';
        res.on('data', (chunk) => {
          buffer += chunk;
          const lines = buffer.split('\n');
          buffer = lines.pop(); // keep incomplete line
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                events.push(JSON.parse(line.slice(6)));
              } catch (_) {}
            }
          }
        });
        setTimeout(() => { req.destroy(); resolve(events); }, durationMs);
      });
      req.on('error', () => resolve(events));
    });
  }
}

// ---------------------------------------------------------------------------
// Data loading (reused from train-ruvllm.js)
// ---------------------------------------------------------------------------

function loadCsiData(filePath) {
  const features = [];
  const vitals = [];
  const rawCsi = [];

  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n').filter(l => l.trim());

  for (const line of lines) {
    try {
      const frame = JSON.parse(line);
      switch (frame.type) {
        case 'feature':
          features.push({
            timestamp: frame.timestamp,
            nodeId: frame.node_id,
            features: frame.features,
            rssi: frame.rssi,
            seq: frame.seq,
          });
          break;
        case 'vitals':
          vitals.push({
            timestamp: frame.timestamp,
            nodeId: frame.node_id,
            breathingBpm: frame.breathing_bpm,
            heartrateBpm: frame.heartrate_bpm,
            nPersons: frame.n_persons,
            motionEnergy: frame.motion_energy,
            presenceScore: frame.presence_score,
            rssi: frame.rssi,
          });
          break;
        case 'raw_csi':
          rawCsi.push({
            timestamp: frame.timestamp,
            nodeId: frame.node_id,
            subcarriers: frame.subcarriers,
            iqHex: frame.iq_hex,
            rssi: frame.rssi,
          });
          break;
      }
    } catch (_) {}
  }

  return { features, vitals, rawCsi };
}

function resolveGlob(pattern) {
  if (!pattern.includes('*')) {
    return fs.existsSync(pattern) ? [pattern] : [];
  }
  const dir = path.dirname(pattern);
  const base = path.basename(pattern);
  const regex = new RegExp('^' + base.replace(/\*/g, '.*') + '$');
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir)
    .filter(f => regex.test(f))
    .map(f => path.join(dir, f));
}

// ---------------------------------------------------------------------------
// CsiEncoder (8 -> 64 -> 128, same as train-ruvllm.js)
// ---------------------------------------------------------------------------

class CsiEncoder {
  constructor(inputDim, hiddenDim, outputDim, seed = 42) {
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    this.outputDim = outputDim;

    const rng = this._createRng(seed);
    this.w1 = this._initXavier(inputDim, hiddenDim, rng);
    this.b1 = new Float64Array(hiddenDim);
    this.w2 = this._initXavier(hiddenDim, outputDim, rng);
    this.b2 = new Float64Array(outputDim);

    this.bn1_gamma = new Float64Array(hiddenDim).fill(1.0);
    this.bn1_beta = new Float64Array(hiddenDim);
    this.bn2_gamma = new Float64Array(outputDim).fill(1.0);
    this.bn2_beta = new Float64Array(outputDim);

    this.bn1_runMean = new Float64Array(hiddenDim);
    this.bn1_runVar = new Float64Array(hiddenDim).fill(1.0);
    this.bn2_runMean = new Float64Array(outputDim);
    this.bn2_runVar = new Float64Array(outputDim).fill(1.0);
    this._bnMomentum = 0.1;
    this._bnEps = 1e-5;
    this._bnInitialized = false;
  }

  encode(input) {
    const hidden = new Float64Array(this.hiddenDim);
    for (let j = 0; j < this.hiddenDim; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < this.inputDim; i++) {
        sum += (input[i] || 0) * this.w1[i * this.hiddenDim + j];
      }
      hidden[j] = sum;
    }
    for (let j = 0; j < this.hiddenDim; j++) {
      const normed = (hidden[j] - this.bn1_runMean[j]) / Math.sqrt(this.bn1_runVar[j] + this._bnEps);
      hidden[j] = Math.max(0, this.bn1_gamma[j] * normed + this.bn1_beta[j]);
    }
    const output = new Float64Array(this.outputDim);
    for (let j = 0; j < this.outputDim; j++) {
      let sum = this.b2[j];
      for (let i = 0; i < this.hiddenDim; i++) {
        sum += hidden[i] * this.w2[i * this.outputDim + j];
      }
      output[j] = sum;
    }
    for (let j = 0; j < this.outputDim; j++) {
      const normed = (output[j] - this.bn2_runMean[j]) / Math.sqrt(this.bn2_runVar[j] + this._bnEps);
      output[j] = this.bn2_gamma[j] * normed + this.bn2_beta[j];
    }
    let norm = 0;
    for (let i = 0; i < output.length; i++) norm += output[i] * output[i];
    norm = Math.sqrt(norm) || 1;
    const result = new Array(this.outputDim);
    for (let i = 0; i < this.outputDim; i++) result[i] = output[i] / norm;
    return result;
  }

  encodeRaw(input) {
    const hidden = new Float64Array(this.hiddenDim);
    for (let j = 0; j < this.hiddenDim; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < this.inputDim; i++) {
        sum += (input[i] || 0) * this.w1[i * this.hiddenDim + j];
      }
      hidden[j] = sum;
    }
    for (let j = 0; j < this.hiddenDim; j++) {
      const normed = (hidden[j] - this.bn1_runMean[j]) / Math.sqrt(this.bn1_runVar[j] + this._bnEps);
      hidden[j] = Math.max(0, this.bn1_gamma[j] * normed + this.bn1_beta[j]);
    }
    const output = new Float64Array(this.outputDim);
    for (let j = 0; j < this.outputDim; j++) {
      let sum = this.b2[j];
      for (let i = 0; i < this.hiddenDim; i++) {
        sum += hidden[i] * this.w2[i * this.outputDim + j];
      }
      output[j] = sum;
    }
    return { hidden, output };
  }

  encodeBatch(inputs) {
    if (inputs.length === 0) return [];
    const n = inputs.length;
    const batchHidden = [];
    for (const input of inputs) {
      const h = new Float64Array(this.hiddenDim);
      for (let j = 0; j < this.hiddenDim; j++) {
        let sum = this.b1[j];
        for (let i = 0; i < this.inputDim; i++) sum += (input[i] || 0) * this.w1[i * this.hiddenDim + j];
        h[j] = sum;
      }
      batchHidden.push(h);
    }

    for (let j = 0; j < this.hiddenDim; j++) {
      let bMean = 0, bVar = 0;
      for (let b = 0; b < n; b++) bMean += batchHidden[b][j];
      bMean /= n;
      for (let b = 0; b < n; b++) bVar += (batchHidden[b][j] - bMean) ** 2;
      bVar /= n;
      if (this._bnInitialized) {
        this.bn1_runMean[j] = (1 - this._bnMomentum) * this.bn1_runMean[j] + this._bnMomentum * bMean;
        this.bn1_runVar[j] = (1 - this._bnMomentum) * this.bn1_runVar[j] + this._bnMomentum * bVar;
      } else {
        this.bn1_runMean[j] = bMean;
        this.bn1_runVar[j] = bVar;
      }
    }

    const batchOutput = [];
    for (let b = 0; b < n; b++) {
      for (let j = 0; j < this.hiddenDim; j++) {
        const normed = (batchHidden[b][j] - this.bn1_runMean[j]) / Math.sqrt(this.bn1_runVar[j] + this._bnEps);
        batchHidden[b][j] = Math.max(0, this.bn1_gamma[j] * normed + this.bn1_beta[j]);
      }
      const out = new Float64Array(this.outputDim);
      for (let j = 0; j < this.outputDim; j++) {
        let sum = this.b2[j];
        for (let i = 0; i < this.hiddenDim; i++) sum += batchHidden[b][i] * this.w2[i * this.outputDim + j];
        out[j] = sum;
      }
      batchOutput.push(out);
    }

    for (let j = 0; j < this.outputDim; j++) {
      let bMean = 0, bVar = 0;
      for (let b = 0; b < n; b++) bMean += batchOutput[b][j];
      bMean /= n;
      for (let b = 0; b < n; b++) bVar += (batchOutput[b][j] - bMean) ** 2;
      bVar /= n;
      if (this._bnInitialized) {
        this.bn2_runMean[j] = (1 - this._bnMomentum) * this.bn2_runMean[j] + this._bnMomentum * bMean;
        this.bn2_runVar[j] = (1 - this._bnMomentum) * this.bn2_runVar[j] + this._bnMomentum * bVar;
      } else {
        this.bn2_runMean[j] = bMean;
        this.bn2_runVar[j] = bVar;
      }
    }
    this._bnInitialized = true;

    const results = [];
    for (let b = 0; b < n; b++) {
      for (let j = 0; j < this.outputDim; j++) {
        const normed = (batchOutput[b][j] - this.bn2_runMean[j]) / Math.sqrt(this.bn2_runVar[j] + this._bnEps);
        batchOutput[b][j] = this.bn2_gamma[j] * normed + this.bn2_beta[j];
      }
      let norm = 0;
      for (let i = 0; i < this.outputDim; i++) norm += batchOutput[b][i] ** 2;
      norm = Math.sqrt(norm) || 1;
      const result = new Array(this.outputDim);
      for (let i = 0; i < this.outputDim; i++) result[i] = batchOutput[b][i] / norm;
      results.push(result);
    }
    return results;
  }

  _createRng(seed) {
    let s = seed;
    return () => {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      return ((s >>> 0) / 4294967296) - 0.5;
    };
  }

  _initXavier(rows, cols, rng) {
    const scale = Math.sqrt(2.0 / (rows + cols));
    const arr = new Float64Array(rows * cols);
    for (let i = 0; i < arr.length; i++) arr[i] = rng() * 2 * scale;
    return arr;
  }
}

// ---------------------------------------------------------------------------
// PresenceHead (128 -> 1, sigmoid)
// ---------------------------------------------------------------------------

class PresenceHead {
  constructor(inputDim, seed = 123) {
    this.inputDim = inputDim;
    let s = seed;
    const nextRng = () => { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return ((s >>> 0) / 4294967296) - 0.5; };
    const scale = Math.sqrt(2.0 / (inputDim + 1));
    this.weights = new Float64Array(inputDim);
    for (let i = 0; i < inputDim; i++) this.weights[i] = nextRng() * 2 * scale;
    this.bias = 0;
  }

  forward(embedding) {
    let z = this.bias;
    for (let i = 0; i < this.inputDim; i++) z += this.weights[i] * (embedding[i] || 0);
    return 1.0 / (1.0 + Math.exp(-z));
  }

  trainStep(embedding, target, lr) {
    const pred = this.forward(embedding);
    const dz = pred - target;
    for (let i = 0; i < this.inputDim; i++) {
      this.weights[i] -= lr * dz * (embedding[i] || 0);
    }
    this.bias -= lr * dz;
    const eps = 1e-7;
    return -(target * Math.log(pred + eps) + (1 - target) * Math.log(1 - pred + eps));
  }

  getWeights() {
    return { weights: Array.from(this.weights), bias: this.bias };
  }

  loadWeights(saved) {
    if (saved.weights) this.weights = new Float64Array(saved.weights);
    if (typeof saved.bias === 'number') this.bias = saved.bias;
  }
}

// ---------------------------------------------------------------------------
// PoseDecoder: 128-dim embedding -> 5 keypoints (x, y) -> 17 keypoints (x, y)
// Two-layer FC: 128 -> 64 (ReLU) -> 10 (5 keypoints x 2 coords)
// ---------------------------------------------------------------------------

class PoseDecoder5 {
  constructor(inputDim = 128, seed = 314) {
    this.inputDim = inputDim;
    this.hiddenDim = 64;
    this.outputDim = 10; // 5 keypoints * 2 (x, y)

    let s = seed;
    const rng = () => { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return ((s >>> 0) / 4294967296) - 0.5; };

    const scale1 = Math.sqrt(2.0 / (inputDim + this.hiddenDim));
    this.w1 = new Float64Array(inputDim * this.hiddenDim);
    for (let i = 0; i < this.w1.length; i++) this.w1[i] = rng() * 2 * scale1;
    this.b1 = new Float64Array(this.hiddenDim);

    const scale2 = Math.sqrt(2.0 / (this.hiddenDim + this.outputDim));
    this.w2 = new Float64Array(this.hiddenDim * this.outputDim);
    for (let i = 0; i < this.w2.length; i++) this.w2[i] = rng() * 2 * scale2;
    this.b2 = new Float64Array(this.outputDim);
  }

  /**
   * Forward pass: embedding -> 5 keypoints [{x, y}, ...]
   * Output coords are in [0, 1] range (normalized to room grid).
   */
  forward(embedding) {
    // Layer 1: ReLU
    const hidden = new Float64Array(this.hiddenDim);
    for (let j = 0; j < this.hiddenDim; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < this.inputDim; i++) {
        sum += (embedding[i] || 0) * this.w1[i * this.hiddenDim + j];
      }
      hidden[j] = Math.max(0, sum);
    }

    // Layer 2: sigmoid output (constrains coords to [0,1])
    const output = new Float64Array(this.outputDim);
    for (let j = 0; j < this.outputDim; j++) {
      let sum = this.b2[j];
      for (let i = 0; i < this.hiddenDim; i++) {
        sum += hidden[i] * this.w2[i * this.outputDim + j];
      }
      output[j] = 1.0 / (1.0 + Math.exp(-sum)); // sigmoid
    }

    // Parse into 5 keypoints: head, L_hand, R_hand, L_foot, R_foot
    return [
      { name: 'head',    x: output[0], y: output[1] },
      { name: 'L_hand',  x: output[2], y: output[3] },
      { name: 'R_hand',  x: output[4], y: output[5] },
      { name: 'L_foot',  x: output[6], y: output[7] },
      { name: 'R_foot',  x: output[8], y: output[9] },
    ];
  }

  /**
   * Train one step with MSE loss and skeleton physics constraints.
   * target: [x0, y0, x1, y1, ..., x4, y4] (10 floats)
   * Returns loss.
   */
  trainStep(embedding, target, lr, skeletonConstraints) {
    // Forward
    const hidden = new Float64Array(this.hiddenDim);
    for (let j = 0; j < this.hiddenDim; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < this.inputDim; i++) sum += (embedding[i] || 0) * this.w1[i * this.hiddenDim + j];
      hidden[j] = Math.max(0, sum);
    }
    const rawOutput = new Float64Array(this.outputDim);
    const output = new Float64Array(this.outputDim);
    for (let j = 0; j < this.outputDim; j++) {
      let sum = this.b2[j];
      for (let i = 0; i < this.hiddenDim; i++) sum += hidden[i] * this.w2[i * this.outputDim + j];
      rawOutput[j] = sum;
      output[j] = 1.0 / (1.0 + Math.exp(-sum));
    }

    // MSE loss
    let loss = 0;
    const dOutput = new Float64Array(this.outputDim);
    for (let j = 0; j < this.outputDim; j++) {
      const diff = output[j] - (target[j] || 0);
      loss += diff * diff;
      // dL/d(raw) = 2 * diff * sigmoid'(raw) = 2 * diff * output * (1 - output)
      dOutput[j] = 2 * diff * output[j] * (1 - output[j]);
    }
    loss /= this.outputDim;

    // Skeleton physics penalty: penalize impossible bone lengths
    if (skeletonConstraints) {
      const kp = [];
      for (let k = 0; k < 5; k++) kp.push({ x: output[k * 2], y: output[k * 2 + 1] });
      const penaltyGrads = this._skeletonPenaltyGrad(kp, skeletonConstraints);
      const penaltyWeight = 0.5;
      for (let j = 0; j < this.outputDim; j++) {
        dOutput[j] += penaltyWeight * penaltyGrads[j];
      }
    }

    // Backprop to w2, b2
    for (let j = 0; j < this.outputDim; j++) {
      this.b2[j] -= lr * dOutput[j];
      for (let i = 0; i < this.hiddenDim; i++) {
        this.w2[i * this.outputDim + j] -= lr * dOutput[j] * hidden[i];
      }
    }

    // Backprop to w1, b1
    for (let i = 0; i < this.hiddenDim; i++) {
      let dHidden = 0;
      for (let j = 0; j < this.outputDim; j++) {
        dHidden += dOutput[j] * this.w2[i * this.outputDim + j];
      }
      if (hidden[i] <= 0) continue; // ReLU gate
      this.b1[i] -= lr * dHidden;
      for (let k = 0; k < this.inputDim; k++) {
        this.w1[k * this.hiddenDim + i] -= lr * dHidden * (embedding[k] || 0);
      }
    }

    return loss;
  }

  /**
   * Compute gradient of skeleton physics penalty.
   * Penalizes bone lengths that exceed anthropometric limits.
   */
  _skeletonPenaltyGrad(kp, limits) {
    const grads = new Float64Array(10);
    // Bone pairs: (head=0, L_hand=1), (head=0, R_hand=2), (head=0, L_foot=3), (head=0, R_foot=4)
    // Approximate max bone lengths as fraction of room size
    // Head to hand ~ 0.55m / 5m room = 0.11
    // Head to foot ~ 0.92m / 5m room = 0.184
    const maxArmLen = (limits.upperArmLen + limits.forearmLen) / 5.0;
    const maxLegLen = (limits.torsoLen + limits.thighLen + limits.shinLen) / 5.0;

    const pairs = [
      { from: 0, to: 1, maxLen: maxArmLen },
      { from: 0, to: 2, maxLen: maxArmLen },
      { from: 0, to: 3, maxLen: maxLegLen },
      { from: 0, to: 4, maxLen: maxLegLen },
    ];

    for (const pair of pairs) {
      const dx = kp[pair.to].x - kp[pair.from].x;
      const dy = kp[pair.to].y - kp[pair.from].y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1e-8;
      if (dist > pair.maxLen) {
        const excess = dist - pair.maxLen;
        // Gradient pushes endpoints closer
        const gx = excess * dx / dist;
        const gy = excess * dy / dist;
        grads[pair.to * 2] += gx;
        grads[pair.to * 2 + 1] += gy;
        grads[pair.from * 2] -= gx;
        grads[pair.from * 2 + 1] -= gy;
      }
    }
    return grads;
  }

  getWeights() {
    return {
      w1: Array.from(this.w1), b1: Array.from(this.b1),
      w2: Array.from(this.w2), b2: Array.from(this.b2),
    };
  }
}

// ---------------------------------------------------------------------------
// Phase 0: Multi-modal data collection
// ---------------------------------------------------------------------------

/**
 * Collect multi-modal data: CSI frames from UDP + Seed sensor data via HTTPS.
 * Builds synchronized MultiModalFrame timeline.
 */
async function collectMultiModalData(seedClient, durationSec, existingFeatures, existingVitals) {
  const timeline = [];

  // If Seed is not available, build timeline from CSI-only data
  if (!seedClient.available) {
    console.log('  Seed unavailable — building CSI-only timeline.');
    return buildCsiOnlyTimeline(existingFeatures, existingVitals);
  }

  console.log(`  Collecting Seed sensor data for ${durationSec}s...`);

  // Collect sensor stream from Seed
  const sensorEvents = await seedClient.streamSensors(durationSec * 1000);
  console.log(`  Received ${sensorEvents.length} Seed sensor events.`);

  // Collect periodic boundary/coherence snapshots
  const boundarySnapshots = [];
  const coherenceSnapshots = [];
  const snapshotInterval = 10000; // every 10s
  const numSnapshots = Math.ceil(durationSec / 10);

  for (let i = 0; i < numSnapshots; i++) {
    const [boundary, coherence] = await Promise.all([
      seedClient.getBoundary(),
      seedClient.getCoherence(),
    ]);
    if (boundary) boundarySnapshots.push({ timestamp: Date.now() / 1000, ...boundary });
    if (coherence) coherenceSnapshots.push({ timestamp: Date.now() / 1000, ...coherence });
    if (i < numSnapshots - 1) {
      await new Promise(r => setTimeout(r, snapshotInterval));
    }
  }

  // Build synchronized timeline: for each CSI feature frame, find nearest Seed data
  for (const feat of existingFeatures) {
    const frame = {
      timestamp: feat.timestamp,
      csi_features: feat.features,
      nodeId: feat.nodeId,
      rssi: feat.rssi,
      seed_embedding: null,
      seed_sensors: null,
      boundary_fragility: null,
      coherence_phase: null,
    };

    // Find nearest sensor event
    let bestSensor = null;
    let bestDist = Infinity;
    for (const evt of sensorEvents) {
      const dist = Math.abs((evt.timestamp || 0) - feat.timestamp);
      if (dist < bestDist) { bestDist = dist; bestSensor = evt; }
    }
    if (bestSensor && bestDist < 2.0) {
      frame.seed_sensors = {
        temp: bestSensor.temperature || bestSensor.temp || null,
        humidity: bestSensor.humidity || null,
        pressure: bestSensor.pressure || null,
        pir: bestSensor.pir != null ? bestSensor.pir : null,
        reed: bestSensor.reed != null ? bestSensor.reed : null,
        vibration: bestSensor.vibration != null ? bestSensor.vibration : null,
      };
      frame.seed_embedding = bestSensor.embedding || null;
    }

    // Find nearest boundary snapshot
    let bestBoundary = null;
    let bestBDist = Infinity;
    for (const snap of boundarySnapshots) {
      const dist = Math.abs(snap.timestamp - feat.timestamp);
      if (dist < bestBDist) { bestBDist = dist; bestBoundary = snap; }
    }
    if (bestBoundary) {
      frame.boundary_fragility = bestBoundary.fragility || bestBoundary.score || 0;
    }

    // Find nearest coherence snapshot
    let bestCoherence = null;
    let bestCDist = Infinity;
    for (const snap of coherenceSnapshots) {
      const dist = Math.abs(snap.timestamp - feat.timestamp);
      if (dist < bestCDist) { bestCDist = dist; bestCoherence = snap; }
    }
    if (bestCoherence) {
      frame.coherence_phase = bestCoherence.phase || bestCoherence.boundary || null;
    }

    timeline.push(frame);
  }

  console.log(`  Built ${timeline.length} multi-modal frames.`);
  const seedFrames = timeline.filter(f => f.seed_sensors !== null).length;
  console.log(`  Frames with Seed data: ${seedFrames} (${(seedFrames / timeline.length * 100).toFixed(1)}%)`);

  return timeline;
}

/**
 * Build a CSI-only timeline when Seed is unavailable.
 */
function buildCsiOnlyTimeline(features, vitals) {
  const timeline = [];
  for (const feat of features) {
    // Find nearest vitals
    let nearVitals = null;
    let bestDist = Infinity;
    for (const v of vitals) {
      if (v.nodeId !== feat.nodeId) continue;
      const dist = Math.abs(v.timestamp - feat.timestamp);
      if (dist < bestDist) { bestDist = dist; nearVitals = v; }
    }

    timeline.push({
      timestamp: feat.timestamp,
      csi_features: feat.features,
      nodeId: feat.nodeId,
      rssi: feat.rssi,
      vitals: nearVitals && bestDist < 2.0 ? nearVitals : null,
      seed_embedding: null,
      seed_sensors: null,
      boundary_fragility: null,
      coherence_phase: null,
    });
  }
  return timeline;
}

// ---------------------------------------------------------------------------
// Phase 1: Weak label generation (no camera)
// ---------------------------------------------------------------------------

/**
 * Generate weak labels from sensor fusion for a multi-modal frame.
 * Returns labels for: presence, position, activity, occupancy, body_region,
 * entry_exit, breathing_zone, pose_proxy_5kp.
 */
function generateWeakLabels(frame, allFrames, vitals, nodeIds) {
  const labels = {};

  // -- 1. Presence label: PIR || CSI presence > 0.3 || temp rising > 0.1C/min --
  const pirPresent = frame.seed_sensors?.pir === 1;

  // Get CSI presence from nearest vitals
  let csiPresence = 0;
  if (frame.vitals) {
    csiPresence = frame.vitals.presenceScore || 0;
  } else {
    // Search vitals array
    let nearVitals = null;
    let bestDist = Infinity;
    for (const v of vitals) {
      if (v.nodeId !== frame.nodeId) continue;
      const d = Math.abs(v.timestamp - frame.timestamp);
      if (d < bestDist) { bestDist = d; nearVitals = v; }
    }
    if (nearVitals && bestDist < 2.0) csiPresence = nearVitals.presenceScore || 0;
  }

  // Temperature rising: check if temp increased > 0.1C over last 60s
  let tempRising = false;
  if (frame.seed_sensors?.temp != null) {
    const past = allFrames.filter(f =>
      f.seed_sensors?.temp != null &&
      f.timestamp >= frame.timestamp - 60 &&
      f.timestamp < frame.timestamp - 10
    );
    if (past.length > 0) {
      const pastAvgTemp = past.reduce((s, f) => s + f.seed_sensors.temp, 0) / past.length;
      tempRising = (frame.seed_sensors.temp - pastAvgTemp) > 0.1;
    }
  }

  labels.presence = (pirPresent || csiPresence > 0.3 || tempRising) ? 1.0 : 0.0;

  // -- 2. Position label: RSSI differential -> 5x5 grid cell --
  // Get RSSI from both nodes at this timestamp
  const sameTimeFrames = allFrames.filter(f =>
    Math.abs(f.timestamp - frame.timestamp) < 1.0 && f.nodeId !== frame.nodeId
  );
  const otherNodeFrame = sameTimeFrames[0] || null;

  if (otherNodeFrame && frame.rssi != null && otherNodeFrame.rssi != null) {
    const rssiDiff = frame.rssi - otherNodeFrame.rssi;
    // Map RSSI diff to X position: rssiDiff in [-30, +30] -> [0, 4]
    const xPos = Math.max(0, Math.min(4, Math.round((rssiDiff + 30) / 60 * 4)));
    // Y position estimated from signal strength average (closer = higher RSSI)
    const avgRssi = (frame.rssi + otherNodeFrame.rssi) / 2;
    // avgRssi typically in [-80, -20], map to [0, 4]
    const yPos = Math.max(0, Math.min(4, Math.round((avgRssi + 80) / 60 * 4)));
    labels.position = { gridX: xPos, gridY: yPos, gridIdx: yPos * 5 + xPos };
    // Normalized position for pose: [0, 1]
    labels.posNormX = xPos / 4;
    labels.posNormY = yPos / 4;
  } else {
    labels.position = { gridX: 2, gridY: 2, gridIdx: 12 }; // center default
    labels.posNormX = 0.5;
    labels.posNormY = 0.5;
  }

  // -- 3. Activity label: from temporal CSI patterns --
  // Compute CSI variance over last 2 seconds
  const recentFrames = allFrames.filter(f =>
    f.nodeId === frame.nodeId &&
    f.timestamp >= frame.timestamp - 2.0 &&
    f.timestamp <= frame.timestamp
  );

  let csiVariance = 0;
  if (recentFrames.length >= 3) {
    const means = new Float64Array(8);
    for (const rf of recentFrames) {
      for (let i = 0; i < 8; i++) means[i] += (rf.csi_features[i] || 0);
    }
    for (let i = 0; i < 8; i++) means[i] /= recentFrames.length;
    for (const rf of recentFrames) {
      for (let i = 0; i < 8; i++) csiVariance += ((rf.csi_features[i] || 0) - means[i]) ** 2;
    }
    csiVariance /= recentFrames.length * 8;
  }

  // FFT peak detection for periodicity (simplified: autocorrelation at 0.5-2Hz)
  let periodic = false;
  if (recentFrames.length >= 6) {
    // Simple autocorrelation check at lag ~0.5s (walking cadence)
    const halfLen = Math.floor(recentFrames.length / 2);
    let corr = 0, norm1 = 0, norm2 = 0;
    for (let i = 0; i < halfLen && i + halfLen < recentFrames.length; i++) {
      const v1 = recentFrames[i].csi_features[0] || 0;
      const v2 = recentFrames[i + halfLen].csi_features[0] || 0;
      corr += v1 * v2;
      norm1 += v1 * v1;
      norm2 += v2 * v2;
    }
    const normCorr = corr / (Math.sqrt(norm1 * norm2) || 1);
    periodic = normCorr > 0.5;
  }

  if (labels.presence < 0.5) {
    labels.activity = 'empty';
    labels.activityVec = [0, 0, 0, 1]; // [stationary, walking, gesture, empty]
  } else if (csiVariance < 0.1 && labels.presence > 0.5) {
    labels.activity = 'stationary';
    labels.activityVec = [1, 0, 0, 0];
  } else if (periodic) {
    labels.activity = 'walking';
    labels.activityVec = [0, 1, 0, 0];
  } else {
    labels.activity = 'gesture';
    labels.activityVec = [0, 0, 1, 0];
  }

  // -- 4. Occupancy count: max(node1_persons, node2_persons), validated by temp --
  let occupancy = 0;
  for (const v of vitals) {
    if (Math.abs(v.timestamp - frame.timestamp) < 2.0) {
      occupancy = Math.max(occupancy, v.nPersons || 0);
    }
  }
  labels.occupancy = occupancy;

  // -- 5. Body region activity: which subcarrier groups are active --
  // Top 4 subcarriers = upper body, bottom 4 = lower body
  if (frame.csi_features && frame.csi_features.length >= 8) {
    const upper = frame.csi_features.slice(0, 4);
    const lower = frame.csi_features.slice(4, 8);
    const upperEnergy = upper.reduce((s, v) => s + Math.abs(v), 0) / 4;
    const lowerEnergy = lower.reduce((s, v) => s + Math.abs(v), 0) / 4;
    labels.bodyRegion = {
      upperActive: upperEnergy > 0.2,
      lowerActive: lowerEnergy > 0.2,
      upperEnergy,
      lowerEnergy,
    };
  } else {
    labels.bodyRegion = { upperActive: false, lowerActive: false, upperEnergy: 0, lowerEnergy: 0 };
  }

  // -- 6. Entry/exit events: reed switch + PIR change + boundary fragility spike --
  labels.entryExit = 'none';
  if (frame.seed_sensors?.reed === 1) {
    // Door is open — check PIR transition
    const prevFrames = allFrames.filter(f =>
      f.timestamp >= frame.timestamp - 5 &&
      f.timestamp < frame.timestamp &&
      f.seed_sensors?.pir != null
    );
    const prevPir = prevFrames.length > 0 ? prevFrames[prevFrames.length - 1].seed_sensors.pir : 0;
    if (prevPir === 0 && pirPresent) labels.entryExit = 'entry';
    else if (prevPir === 1 && !pirPresent) labels.entryExit = 'exit';
  }
  if (frame.boundary_fragility != null && frame.boundary_fragility > 0.7) {
    if (labels.entryExit === 'none') labels.entryExit = 'regime_change';
  }

  // -- 7. Breathing zone: humidity change rate --
  labels.breathingZone = null;
  if (frame.seed_sensors?.humidity != null) {
    const pastHumidity = allFrames.filter(f =>
      f.seed_sensors?.humidity != null &&
      f.timestamp >= frame.timestamp - 30 &&
      f.timestamp < frame.timestamp - 5
    );
    if (pastHumidity.length > 0) {
      const pastAvg = pastHumidity.reduce((s, f) => s + f.seed_sensors.humidity, 0) / pastHumidity.length;
      const humidityDelta = frame.seed_sensors.humidity - pastAvg;
      // Positive delta near person location suggests breathing
      if (humidityDelta > 0.05) {
        labels.breathingZone = labels.position;
      }
    }
  }

  // -- 8. Pose proxy: 5-keypoint coarse pose from sensor fusion --
  if (labels.presence > 0.5) {
    const headX = labels.posNormX;
    const headY = labels.posNormY;

    // Hands: subcarrier variance asymmetry between 2 nodes
    let lHandOffset = 0;
    let rHandOffset = 0;
    if (otherNodeFrame && frame.csi_features && otherNodeFrame.csi_features) {
      // Compare per-subcarrier energy between nodes for left/right asymmetry
      const node1Upper = frame.csi_features.slice(0, 4);
      const node2Upper = otherNodeFrame.csi_features ? otherNodeFrame.csi_features.slice(0, 4) : [0, 0, 0, 0];
      const leftEnergy = node1Upper.reduce((s, v) => s + Math.abs(v), 0);
      const rightEnergy = node2Upper.reduce((s, v) => s + Math.abs(v), 0);
      const totalEnergy = leftEnergy + rightEnergy || 1;
      lHandOffset = (leftEnergy / totalEnergy - 0.5) * 0.2;
      rHandOffset = (rightEnergy / totalEnergy - 0.5) * 0.2;
    }

    // Feet: vibration sensor + RSSI ground reflection
    let footSpread = 0.05;
    if (frame.seed_sensors?.vibration === 1) {
      footSpread = 0.1; // wider stance when stepping
    }

    const poseProxy = [
      headX,                                      // head.x
      headY - 0.05,                               // head.y (slightly above center)
      Math.max(0, Math.min(1, headX - 0.1 + lHandOffset)),  // L_hand.x
      headY + 0.15,                               // L_hand.y
      Math.max(0, Math.min(1, headX + 0.1 + rHandOffset)),  // R_hand.x
      headY + 0.15,                               // R_hand.y
      Math.max(0, Math.min(1, headX - footSpread)),  // L_foot.x
      Math.min(1, headY + 0.35),                  // L_foot.y
      Math.max(0, Math.min(1, headX + footSpread)),  // R_foot.x
      Math.min(1, headY + 0.35),                  // R_foot.y
    ];
    labels.poseProxy5 = poseProxy;
  } else {
    labels.poseProxy5 = null;
  }

  // -- Confidence score: how many sensor signals agree --
  let signalsActive = 0;
  if (frame.seed_sensors?.pir != null) signalsActive++;
  if (frame.seed_sensors?.temp != null) signalsActive++;
  if (frame.seed_sensors?.humidity != null) signalsActive++;
  if (frame.seed_sensors?.reed != null) signalsActive++;
  if (frame.seed_sensors?.vibration != null) signalsActive++;
  if (otherNodeFrame) signalsActive++; // RSSI differential
  if (csiPresence > 0) signalsActive++; // CSI presence
  if (frame.boundary_fragility != null) signalsActive++;
  if (frame.seed_embedding != null) signalsActive++;
  labels.confidence = signalsActive / 10; // 10 possible signals

  return labels;
}

// ---------------------------------------------------------------------------
// Triplet generation (extended from train-ruvllm.js)
// ---------------------------------------------------------------------------

function generateTriplets(features, vitals, config) {
  const triplets = [];
  const byNode = {};
  for (const f of features) {
    if (!byNode[f.nodeId]) byNode[f.nodeId] = [];
    byNode[f.nodeId].push(f);
  }
  const nodeIds = Object.keys(byNode).map(Number);
  for (const nid of nodeIds) byNode[nid].sort((a, b) => a.timestamp - b.timestamp);

  // Strategy 1+2: Temporal positive/negative
  for (const nid of nodeIds) {
    const frames = byNode[nid];
    for (let i = 0; i < frames.length; i++) {
      const anchor = frames[i];
      for (let j = i + 1; j < frames.length && j < i + 20; j++) {
        const candidate = frames[j];
        const timeDiff = Math.abs(candidate.timestamp - anchor.timestamp);
        if (timeDiff <= config.positiveWindowSec) {
          for (let k = 0; k < frames.length; k++) {
            const neg = frames[k];
            if (Math.abs(neg.timestamp - anchor.timestamp) >= config.negativeWindowSec) {
              triplets.push({
                anchor: anchor.features, positive: candidate.features, negative: neg.features,
                isHard: Math.abs(neg.timestamp - anchor.timestamp) < config.negativeWindowSec * 2,
                type: 'temporal',
                anchorLabel: `node${nid}-t${anchor.timestamp.toFixed(2)}`,
                posLabel: `node${nid}-t${candidate.timestamp.toFixed(2)}`,
                negLabel: `node${nid}-t${neg.timestamp.toFixed(2)}`,
              });
              break;
            }
          }
        }
      }
    }
  }

  // Strategy 3: Cross-node positive
  if (nodeIds.length >= 2) {
    const n1Frames = byNode[nodeIds[0]] || [];
    const n2Frames = byNode[nodeIds[1]] || [];
    for (const f1 of n1Frames) {
      let bestMatch = null, bestDist = Infinity;
      for (const f2 of n2Frames) {
        const d = Math.abs(f2.timestamp - f1.timestamp);
        if (d < bestDist) { bestDist = d; bestMatch = f2; }
      }
      if (bestMatch && bestDist < config.positiveWindowSec) {
        for (const f2neg of n2Frames) {
          if (Math.abs(f2neg.timestamp - f1.timestamp) >= config.negativeWindowSec) {
            triplets.push({
              anchor: f1.features, positive: bestMatch.features, negative: f2neg.features,
              isHard: false, type: 'cross-node',
              anchorLabel: `node${f1.nodeId}-t${f1.timestamp.toFixed(2)}`,
              posLabel: `node${bestMatch.nodeId}-t${bestMatch.timestamp.toFixed(2)}`,
              negLabel: `node${f2neg.nodeId}-t${f2neg.timestamp.toFixed(2)}`,
            });
            break;
          }
        }
      }
    }
  }

  // Strategy 5: Hard negatives near transitions
  const sortedVitals = [...vitals].sort((a, b) => a.timestamp - b.timestamp);
  const transitionTimes = [];
  for (let i = 1; i < sortedVitals.length; i++) {
    if (Math.abs(sortedVitals[i].motionEnergy - sortedVitals[i - 1].motionEnergy) > 2.0) {
      transitionTimes.push(sortedVitals[i].timestamp);
    }
  }
  for (const transTime of transitionTimes.slice(0, 50)) {
    for (const nid of nodeIds) {
      const frames = byNode[nid];
      let before = null, after = null;
      for (const f of frames) {
        if (f.timestamp < transTime) before = f;
        if (f.timestamp > transTime && !after) after = f;
      }
      if (before && after) {
        const anchorIdx = Math.max(0, frames.indexOf(before) - 5);
        const anchor = frames[anchorIdx];
        if (anchor) {
          triplets.push({
            anchor: anchor.features, positive: before.features, negative: after.features,
            isHard: true, type: 'transition-hard',
            anchorLabel: `node${nid}-pre-transition`,
            posLabel: `node${nid}-before`,
            negLabel: `node${nid}-after`,
          });
        }
      }
    }
  }

  // Strategy 6: Scenario boundary
  for (const nid of nodeIds) {
    const frames = byNode[nid];
    if (frames.length < 10) continue;
    const tMid = (frames[0].timestamp + frames[frames.length - 1].timestamp) / 2;
    const firstHalf = frames.filter(f => f.timestamp < tMid);
    const secondHalf = frames.filter(f => f.timestamp >= tMid);
    if (firstHalf.length < 3 || secondHalf.length < 3) continue;
    const nBoundary = Math.min(50, firstHalf.length, secondHalf.length);
    for (let i = 0; i < nBoundary; i++) {
      const posIdx = Math.min(i + 1, firstHalf.length - 1);
      const negIdx = Math.min(i, secondHalf.length - 1);
      triplets.push({
        anchor: firstHalf[i].features, positive: firstHalf[posIdx].features,
        negative: secondHalf[negIdx].features, isHard: true, type: 'scenario-boundary',
        anchorLabel: `node${nid}-first-${i}`,
        posLabel: `node${nid}-first-${posIdx}`,
        negLabel: `node${nid}-second-${negIdx}`,
      });
    }
  }

  return triplets;
}

// ---------------------------------------------------------------------------
// Extended contrastive triplets: multi-modal (Phase 2 enhanced)
// ---------------------------------------------------------------------------

/**
 * Generate additional contrastive triplets from multi-modal data.
 * - Multi-modal positive: CSI + Seed at same time agree
 * - Sensor-verified negative: PIR=0 vs PIR=1
 * - Activity boundary: before/after fragility spike
 * - Cross-modal: CSI embedding close to Seed embedding for same state
 */
function generateMultiModalTriplets(timeline, encoder) {
  const triplets = [];

  // Sensor-verified negative: PIR=0 vs PIR=1
  const pirOff = timeline.filter(f => f.seed_sensors?.pir === 0);
  const pirOn = timeline.filter(f => f.seed_sensors?.pir === 1);

  if (pirOff.length >= 2 && pirOn.length >= 1) {
    const nPairs = Math.min(100, pirOff.length, pirOn.length);
    for (let i = 0; i < nPairs; i++) {
      const anchor = pirOn[i % pirOn.length];
      // Positive: another PIR=1 frame
      const positive = pirOn[(i + 1) % pirOn.length];
      // Negative: PIR=0 frame
      const negative = pirOff[i % pirOff.length];
      triplets.push({
        anchor: anchor.csi_features, positive: positive.csi_features,
        negative: negative.csi_features, isHard: true, type: 'sensor-verified',
        anchorLabel: `pir-on-${i}`, posLabel: `pir-on-${i + 1}`, negLabel: `pir-off-${i}`,
      });
    }
  }

  // Activity boundary: before/after boundary fragility spike
  const fragilityFrames = timeline.filter(f =>
    f.boundary_fragility != null && f.boundary_fragility > 0.5
  );
  for (const spike of fragilityFrames.slice(0, 50)) {
    const before = timeline.filter(f =>
      f.timestamp < spike.timestamp && f.timestamp >= spike.timestamp - 5
    );
    const after = timeline.filter(f =>
      f.timestamp > spike.timestamp && f.timestamp <= spike.timestamp + 5
    );
    if (before.length >= 2 && after.length >= 1) {
      triplets.push({
        anchor: before[before.length - 1].csi_features,
        positive: before[before.length - 2].csi_features,
        negative: after[0].csi_features,
        isHard: true, type: 'activity-boundary',
        anchorLabel: `pre-spike-${spike.timestamp.toFixed(1)}`,
        posLabel: `pre-spike-prev`,
        negLabel: `post-spike`,
      });
    }
  }

  // Cross-modal: CSI embedding close to Seed embedding for same state
  // Use CSI features as anchor, Seed embedding as projection target
  const seedFrames = timeline.filter(f => f.seed_embedding != null && f.seed_embedding.length > 0);
  if (seedFrames.length >= 3) {
    for (let i = 0; i < Math.min(100, seedFrames.length); i++) {
      const anchor = seedFrames[i];
      // Positive: temporally adjacent frame (same state)
      const posIdx = (i + 1) % seedFrames.length;
      const positive = seedFrames[posIdx];
      // Negative: temporally distant frame
      const negIdx = (i + Math.floor(seedFrames.length / 2)) % seedFrames.length;
      const negative = seedFrames[negIdx];

      if (Math.abs(anchor.timestamp - positive.timestamp) < 2.0 &&
          Math.abs(anchor.timestamp - negative.timestamp) > 5.0) {
        triplets.push({
          anchor: anchor.csi_features, positive: positive.csi_features,
          negative: negative.csi_features, isHard: false, type: 'cross-modal',
          anchorLabel: `seed-csi-${i}`, posLabel: `seed-csi-${posIdx}`, negLabel: `seed-csi-${negIdx}`,
        });
      }
    }
  }

  return triplets;
}

// ---------------------------------------------------------------------------
// Quantization (same as train-ruvllm.js)
// ---------------------------------------------------------------------------

function quantizeWeights(weights, bits) {
  const maxVal = 2 ** bits - 1;
  let wMin = Infinity, wMax = -Infinity;
  for (let i = 0; i < weights.length; i++) {
    if (weights[i] < wMin) wMin = weights[i];
    if (weights[i] > wMax) wMax = weights[i];
  }
  const range = wMax - wMin || 1e-10;
  const scale = range / maxVal;
  const zeroPoint = Math.round(-wMin / scale);
  const qValues = new Uint8Array(weights.length);
  for (let i = 0; i < weights.length; i++) {
    let q = Math.round((weights[i] - wMin) / scale);
    qValues[i] = Math.max(0, Math.min(maxVal, q));
  }

  let packed;
  if (bits === 8) {
    packed = new Uint8Array(weights.length);
    for (let i = 0; i < weights.length; i++) packed[i] = qValues[i];
  } else if (bits === 4) {
    packed = new Uint8Array(Math.ceil(weights.length / 2));
    for (let i = 0; i < weights.length; i += 2) {
      const hi = qValues[i] & 0x0F;
      const lo = (i + 1 < weights.length) ? (qValues[i + 1] & 0x0F) : 0;
      packed[i >> 1] = (hi << 4) | lo;
    }
  } else if (bits === 2) {
    packed = new Uint8Array(Math.ceil(weights.length / 4));
    for (let i = 0; i < weights.length; i += 4) {
      let byte = 0;
      for (let k = 0; k < 4; k++) {
        const val = (i + k < weights.length) ? (qValues[i + k] & 0x03) : 0;
        byte |= val << (6 - k * 2);
      }
      packed[Math.floor(i / 4)] = byte;
    }
  } else {
    packed = new Uint8Array(weights.length);
    for (let i = 0; i < weights.length; i++) packed[i] = qValues[i];
  }

  const originalSize = weights.length * 4;
  return { quantized: packed, scale, zeroPoint, bits, numWeights: weights.length,
    originalSize, quantizedSize: packed.length, compressionRatio: originalSize / packed.length };
}

function dequantizeWeights(packed, scale, zeroPoint, bits, numWeights) {
  const result = new Float32Array(numWeights);
  if (bits === 8) {
    for (let i = 0; i < numWeights; i++) result[i] = (packed[i] - zeroPoint) * scale;
  } else if (bits === 4) {
    for (let i = 0; i < numWeights; i++) {
      const byteIdx = i >> 1;
      const nibble = (i % 2 === 0) ? (packed[byteIdx] >> 4) & 0x0F : packed[byteIdx] & 0x0F;
      result[i] = (nibble - zeroPoint) * scale;
    }
  } else if (bits === 2) {
    for (let i = 0; i < numWeights; i++) {
      const byteIdx = Math.floor(i / 4);
      const shift = 6 - (i % 4) * 2;
      result[i] = ((packed[byteIdx] >> shift) & 0x03 - zeroPoint) * scale;
    }
  } else {
    for (let i = 0; i < numWeights; i++) result[i] = (packed[i] - zeroPoint) * scale;
  }
  return result;
}

function quantizationQuality(original, dequantized) {
  let sumSqErr = 0;
  const n = Math.min(original.length, dequantized.length);
  for (let i = 0; i < n; i++) {
    const diff = original[i] - dequantized[i];
    sumSqErr += diff * diff;
  }
  return Math.sqrt(sumSqErr / n);
}

// ---------------------------------------------------------------------------
// Data augmentation (same as train-ruvllm.js)
// ---------------------------------------------------------------------------

function augmentData(features, multiplier = 10) {
  if (features.length < 2 || multiplier <= 1) return features;
  const augmented = [...features];
  const targetSize = features.length * multiplier;
  const rng = { s: 7919 };
  const nextRand = () => { rng.s ^= rng.s << 13; rng.s ^= rng.s >> 17; rng.s ^= rng.s << 5; return (rng.s >>> 0) / 4294967296; };
  const nextGaussian = () => {
    const u1 = nextRand() || 1e-10;
    const u2 = nextRand();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };

  const byNode = {};
  for (const f of features) { if (!byNode[f.nodeId]) byNode[f.nodeId] = []; byNode[f.nodeId].push(f); }
  for (const nid of Object.keys(byNode)) byNode[nid].sort((a, b) => a.timestamp - b.timestamp);
  const nodeIds = Object.keys(byNode).map(Number);

  while (augmented.length < targetSize) {
    const strategy = nextRand();
    if (strategy < 0.5) {
      const nid = nodeIds[Math.floor(nextRand() * nodeIds.length)];
      const frames = byNode[nid];
      if (frames.length < 2) continue;
      const idx = Math.floor(nextRand() * (frames.length - 1));
      const alpha = 0.2 + nextRand() * 0.6;
      const blended = frames[idx].features.map((v, i) => v * alpha + (frames[idx + 1].features[i] || 0) * (1 - alpha));
      augmented.push({ timestamp: frames[idx].timestamp * alpha + frames[idx + 1].timestamp * (1 - alpha),
        nodeId: nid, features: blended, rssi: frames[idx].rssi, seq: -1 });
    } else if (strategy < 0.8) {
      const idx = Math.floor(nextRand() * features.length);
      const f = features[idx];
      const noisy = f.features.map(v => v + nextGaussian() * 0.02);
      augmented.push({ timestamp: f.timestamp + (nextRand() - 0.5) * 0.1, nodeId: f.nodeId, features: noisy, rssi: f.rssi, seq: -1 });
    } else {
      if (nodeIds.length < 2) {
        const idx = Math.floor(nextRand() * features.length);
        const f = features[idx];
        augmented.push({ ...f, features: f.features.map(v => v + nextGaussian() * 0.01), seq: -1 });
        continue;
      }
      const frames1 = byNode[nodeIds[0]], frames2 = byNode[nodeIds[1]];
      const idx1 = Math.floor(nextRand() * frames1.length);
      const f1 = frames1[idx1];
      let bestIdx = 0, bestDist = Infinity;
      for (let j = 0; j < frames2.length; j++) {
        const d = Math.abs(frames2[j].timestamp - f1.timestamp);
        if (d < bestDist) { bestDist = d; bestIdx = j; }
      }
      if (bestDist < 2.0) {
        const f2 = frames2[bestIdx];
        const alpha = 0.3 + nextRand() * 0.4;
        augmented.push({
          timestamp: f1.timestamp, nodeId: nodeIds[0],
          features: f1.features.map((v, i) => v * alpha + (f2.features[i] || 0) * (1 - alpha)),
          rssi: Math.round(f1.rssi * alpha + f2.rssi * (1 - alpha)), seq: -1,
        });
      }
    }
  }
  return augmented;
}

// ---------------------------------------------------------------------------
// Live UDP data collection (same as train-ruvllm.js)
// ---------------------------------------------------------------------------

async function collectLiveData(port = 5006, durationSec = 60) {
  let dgram;
  try { dgram = require('dgram'); } catch (_) { return { features: [], vitals: [] }; }

  return new Promise((resolve) => {
    const features = [], vitals = [];
    const sock = dgram.createSocket('udp4');
    let resolved = false;
    const finish = () => {
      if (resolved) return; resolved = true;
      try { sock.close(); } catch (_) {}
      resolve({ features, vitals });
    };
    sock.on('message', (msg) => {
      try {
        const frame = JSON.parse(msg.toString());
        if (frame.type === 'feature') features.push({
          timestamp: frame.timestamp, nodeId: frame.node_id, features: frame.features, rssi: frame.rssi, seq: frame.seq,
        });
        else if (frame.type === 'vitals') vitals.push({
          timestamp: frame.timestamp, nodeId: frame.node_id, breathingBpm: frame.breathing_bpm,
          heartrateBpm: frame.heartrate_bpm, nPersons: frame.n_persons, motionEnergy: frame.motion_energy,
          presenceScore: frame.presence_score, rssi: frame.rssi,
        });
      } catch (_) {}
    });
    sock.on('error', () => finish());
    sock.bind(port, () => {
      console.log(`  Listening on UDP :${port} for ${durationSec}s...`);
      setTimeout(finish, durationSec * 1000);
    });
    setTimeout(() => finish(), (durationSec + 2) * 1000);
  });
}

// ---------------------------------------------------------------------------
// Phase 4: Interpolate 5 keypoints -> COCO 17 keypoints
// ---------------------------------------------------------------------------

/**
 * Interpolate from 5 coarse keypoints (head, L_hand, R_hand, L_foot, R_foot)
 * to COCO 17 keypoints using skeleton priors.
 *
 * COCO 17 order: nose, L_eye, R_eye, L_ear, R_ear,
 *   L_shoulder, R_shoulder, L_elbow, R_elbow, L_wrist, R_wrist,
 *   L_hip, R_hip, L_knee, R_knee, L_ankle, R_ankle
 */
function interpolateTo17Keypoints(kp5, skeleton) {
  const head    = { x: kp5[0], y: kp5[1] };
  const lHand   = { x: kp5[2], y: kp5[3] };
  const rHand   = { x: kp5[4], y: kp5[5] };
  const lFoot   = { x: kp5[6], y: kp5[7] };
  const rFoot   = { x: kp5[8], y: kp5[9] };

  // Shoulders: 0.3*head + 0.7*hands
  const lShoulder = { x: 0.3 * head.x + 0.7 * lHand.x, y: 0.3 * head.y + 0.7 * lHand.y };
  const rShoulder = { x: 0.3 * head.x + 0.7 * rHand.x, y: 0.3 * head.y + 0.7 * rHand.y };

  // Elbows: midpoint(shoulder, hand)
  const lElbow = { x: (lShoulder.x + lHand.x) / 2, y: (lShoulder.y + lHand.y) / 2 };
  const rElbow = { x: (rShoulder.x + rHand.x) / 2, y: (rShoulder.y + rHand.y) / 2 };

  // Hips: midpoint(head, feet)
  const lHip = { x: (head.x + lFoot.x) / 2, y: (head.y + lFoot.y) / 2 };
  const rHip = { x: (head.x + rFoot.x) / 2, y: (head.y + rFoot.y) / 2 };

  // Knees: midpoint(hip, foot)
  const lKnee = { x: (lHip.x + lFoot.x) / 2, y: (lHip.y + lFoot.y) / 2 };
  const rKnee = { x: (rHip.x + rFoot.x) / 2, y: (rHip.y + rFoot.y) / 2 };

  // Face keypoints derived from head
  const nose  = { x: head.x, y: head.y };
  const lEye  = { x: head.x - 0.01, y: head.y - 0.005 };
  const rEye  = { x: head.x + 0.01, y: head.y - 0.005 };
  const lEar  = { x: head.x - 0.02, y: head.y };
  const rEar  = { x: head.x + 0.02, y: head.y };

  // Clamp all to [0, 1]
  const clamp = (v) => Math.max(0, Math.min(1, v));

  // COCO 17 order
  const kp17 = [
    clamp(nose.x),      clamp(nose.y),       // 0: nose
    clamp(lEye.x),      clamp(lEye.y),       // 1: L_eye
    clamp(rEye.x),      clamp(rEye.y),       // 2: R_eye
    clamp(lEar.x),      clamp(lEar.y),       // 3: L_ear
    clamp(rEar.x),      clamp(rEar.y),       // 4: R_ear
    clamp(lShoulder.x), clamp(lShoulder.y),  // 5: L_shoulder
    clamp(rShoulder.x), clamp(rShoulder.y),  // 6: R_shoulder
    clamp(lElbow.x),    clamp(lElbow.y),      // 7: L_elbow
    clamp(rElbow.x),    clamp(rElbow.y),      // 8: R_elbow
    clamp(lHand.x),     clamp(lHand.y),       // 9: L_wrist
    clamp(rHand.x),     clamp(rHand.y),       // 10: R_wrist
    clamp(lHip.x),      clamp(lHip.y),        // 11: L_hip
    clamp(rHip.x),      clamp(rHip.y),        // 12: R_hip
    clamp(lKnee.x),     clamp(lKnee.y),       // 13: L_knee
    clamp(rKnee.x),     clamp(rKnee.y),       // 14: R_knee
    clamp(lFoot.x),     clamp(lFoot.y),       // 15: L_ankle
    clamp(rFoot.x),     clamp(rFoot.y),       // 16: R_ankle
  ];

  // Apply bone length constraints
  return applyBoneLengthConstraints(kp17, skeleton);
}

/**
 * Apply anthropometric bone length constraints to 17 keypoints.
 * Iteratively pull joints to satisfy max bone length limits.
 */
function applyBoneLengthConstraints(kp17, skeleton) {
  // Room-normalized max bone lengths (5m room assumption)
  const roomScale = 5.0;
  const maxUpperArm   = skeleton.upperArmLen / roomScale;
  const maxForearm    = skeleton.forearmLen / roomScale;
  const maxThigh      = skeleton.thighLen / roomScale;
  const maxShin       = skeleton.shinLen / roomScale;
  const maxShoulder   = skeleton.shoulderWidth / roomScale;

  // Bone connections: [parentIdx, childIdx, maxLength]
  const bones = [
    [5,  7,  maxUpperArm],   // L_shoulder -> L_elbow
    [7,  9,  maxForearm],    // L_elbow -> L_wrist
    [6,  8,  maxUpperArm],   // R_shoulder -> R_elbow
    [8,  10, maxForearm],    // R_elbow -> R_wrist
    [11, 13, maxThigh],      // L_hip -> L_knee
    [13, 15, maxShin],       // L_knee -> L_ankle
    [12, 14, maxThigh],      // R_hip -> R_knee
    [14, 16, maxShin],       // R_knee -> R_ankle
    [5,  6,  maxShoulder],   // L_shoulder -> R_shoulder
  ];

  const result = [...kp17]; // copy

  // 3 iterations of constraint projection
  for (let iter = 0; iter < 3; iter++) {
    for (const [pIdx, cIdx, maxLen] of bones) {
      const px = result[pIdx * 2], py = result[pIdx * 2 + 1];
      const cx = result[cIdx * 2], cy = result[cIdx * 2 + 1];
      const dx = cx - px, dy = cy - py;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > maxLen && dist > 1e-8) {
        const excess = (dist - maxLen) / 2;
        const nx = dx / dist, ny = dy / dist;
        // Move both joints toward each other
        result[pIdx * 2]     += excess * nx;
        result[pIdx * 2 + 1] += excess * ny;
        result[cIdx * 2]     -= excess * nx;
        result[cIdx * 2 + 1] -= excess * ny;
        // Re-clamp
        result[pIdx * 2]     = Math.max(0, Math.min(1, result[pIdx * 2]));
        result[pIdx * 2 + 1] = Math.max(0, Math.min(1, result[pIdx * 2 + 1]));
        result[cIdx * 2]     = Math.max(0, Math.min(1, result[cIdx * 2]));
        result[cIdx * 2 + 1] = Math.max(0, Math.min(1, result[cIdx * 2 + 1]));
      }
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// createLabels (from vitals only, for CSI-only fallback)
// ---------------------------------------------------------------------------

function createLabels(featureFrame, vitals) {
  let nearest = null, bestDist = Infinity;
  for (const v of vitals) {
    if (v.nodeId !== featureFrame.nodeId) continue;
    const dist = Math.abs(v.timestamp - featureFrame.timestamp);
    if (dist < bestDist) { bestDist = dist; nearest = v; }
  }
  if (!nearest || bestDist > 2.0) return null;

  const presence = nearest.presenceScore > 0.3 ? 1.0 : 0.0;
  let activity;
  if (nearest.presenceScore <= 0.1) activity = [0, 0, 1];
  else if (nearest.motionEnergy > 2.0) activity = [0, 1, 0];
  else activity = [1, 0, 0];

  return {
    presence,
    activity,
    vitalsTarget: [nearest.breathingBpm / 30.0, nearest.heartrateBpm / 120.0],
  };
}

// ============================================================================
// MAIN PIPELINE
// ============================================================================

async function main() {
  const startTime = Date.now();
  console.log('=== WiFi-DensePose Camera-Free Training Pipeline ===');
  console.log(`Config: epochs=${CONFIG.epochs} batch=${CONFIG.batchSize} lora_rank=${CONFIG.loraRank} quant=${CONFIG.quantizeBits}bit`);
  console.log(`Seed URL: ${CONFIG.seedUrl} (${CONFIG.noSeed ? 'disabled' : 'enabled'})`);
  console.log('');

  // =========================================================================
  // Step 1: Load CSI data
  // =========================================================================
  console.log('[1/12] Loading CSI data...');
  const files = resolveGlob(CONFIG.dataGlob);
  if (files.length === 0) { console.error(`No files found: ${CONFIG.dataGlob}`); process.exit(1); }

  let allFeatures = [];
  let allVitals = [];
  let allRawCsi = [];

  for (const file of files) {
    console.log(`  Loading: ${path.basename(file)}`);
    const { features, vitals, rawCsi } = loadCsiData(file);
    allFeatures = allFeatures.concat(features);
    allVitals = allVitals.concat(vitals);
    allRawCsi = allRawCsi.concat(rawCsi);
  }

  console.log(`  Loaded: ${allFeatures.length} features, ${allVitals.length} vitals, ${allRawCsi.length} raw CSI`);
  const nodeIds = [...new Set(allFeatures.map(f => f.nodeId))];
  console.log(`  Nodes: ${nodeIds.join(', ')}`);

  if (allFeatures.length === 0) {
    console.error('No feature frames found.'); process.exit(1);
  }

  // Live data supplement if dataset is small
  if (allFeatures.length < 500) {
    console.log(`\n[1b/12] Dataset small (${allFeatures.length} < 500), attempting live UDP collection...`);
    try {
      const live = await collectLiveData(5006, 60);
      if (live.features.length > 0) {
        allFeatures = allFeatures.concat(live.features);
        allVitals = allVitals.concat(live.vitals);
        console.log(`  Collected ${live.features.length} features, ${live.vitals.length} vitals from UDP.`);
      } else {
        console.log('  No live data received. Proceeding with existing data.');
      }
    } catch (e) {
      console.log(`  Live collection failed: ${e.message}`);
    }
  }

  // Augment
  const originalCount = allFeatures.length;
  allFeatures = augmentData(allFeatures, CONFIG.augmentMultiplier);
  console.log(`\n[1c/12] Augmentation: ${originalCount} -> ${allFeatures.length} features (${CONFIG.augmentMultiplier}x)`);

  // =========================================================================
  // Step 2: Probe Seed and collect multi-modal data (Phase 0)
  // =========================================================================
  console.log('\n[2/12] Phase 0: Multi-modal data collection...');
  const seedClient = new SeedClient(CONFIG.seedUrl, CONFIG.seedToken);
  let seedAvailable = false;

  if (!CONFIG.noSeed) {
    try {
      seedAvailable = await seedClient.probe();
      if (seedAvailable) {
        console.log('  Cognitum Seed connected. Collecting multi-modal data...');
      } else {
        console.log('  Cognitum Seed not reachable. Falling back to CSI-only pipeline.');
      }
    } catch (e) {
      console.log(`  Seed probe failed: ${e.message}. Falling back to CSI-only pipeline.`);
    }
  } else {
    console.log('  --no-seed flag set. Running CSI-only pipeline.');
  }

  const timeline = await collectMultiModalData(seedClient, CONFIG.seedCollectSec, allFeatures, allVitals);
  console.log(`  Timeline: ${timeline.length} frames`);

  // =========================================================================
  // Step 3: Generate weak labels (Phase 1)
  // =========================================================================
  console.log('\n[3/12] Phase 1: Weak label generation (no camera)...');
  const labeledTimeline = [];
  let poseLabelCount = 0;
  let sensorLabelCount = 0;

  for (const frame of timeline) {
    const labels = generateWeakLabels(frame, timeline, allVitals, nodeIds);
    labeledTimeline.push({ ...frame, labels });
    if (labels.poseProxy5) poseLabelCount++;
    if (labels.confidence > 0.3) sensorLabelCount++;
  }

  console.log(`  Total frames labeled: ${labeledTimeline.length}`);
  console.log(`  Frames with pose proxy: ${poseLabelCount}`);
  console.log(`  Frames with sensor labels (conf > 0.3): ${sensorLabelCount}`);
  console.log(`  Activity distribution:`);
  const actDist = { stationary: 0, walking: 0, gesture: 0, empty: 0 };
  for (const f of labeledTimeline) actDist[f.labels.activity]++;
  for (const [k, v] of Object.entries(actDist)) {
    console.log(`    ${k}: ${v} (${(v / labeledTimeline.length * 100).toFixed(1)}%)`);
  }

  // =========================================================================
  // Step 4: Generate contrastive triplets
  // =========================================================================
  console.log('\n[4/12] Generating contrastive triplets...');
  const baseTriplets = generateTriplets(allFeatures, allVitals, CONFIG);

  // Build the encoder first so we can generate multi-modal triplets
  const encoder = new CsiEncoder(CONFIG.inputDim, CONFIG.hiddenDim, CONFIG.embeddingDim);

  // Multi-modal triplets (if Seed available)
  let multiModalTriplets = [];
  if (seedAvailable) {
    multiModalTriplets = generateMultiModalTriplets(timeline, encoder);
    console.log(`  Multi-modal triplets: ${multiModalTriplets.length}`);
  }

  const allTriplets = [...baseTriplets, ...multiModalTriplets];
  console.log(`  Total triplets: ${allTriplets.length}`);
  console.log(`  Temporal: ${allTriplets.filter(t => t.type === 'temporal').length}`);
  console.log(`  Cross-node: ${allTriplets.filter(t => t.type === 'cross-node').length}`);
  console.log(`  Sensor-verified: ${allTriplets.filter(t => t.type === 'sensor-verified').length}`);
  console.log(`  Activity-boundary: ${allTriplets.filter(t => t.type === 'activity-boundary').length}`);
  console.log(`  Cross-modal: ${allTriplets.filter(t => t.type === 'cross-modal').length}`);
  console.log(`  Hard negatives: ${allTriplets.filter(t => t.isHard).length}`);

  if (allTriplets.length === 0) {
    console.error('No triplets generated.'); process.exit(1);
  }

  // =========================================================================
  // Step 5: Encode features (batch mode for BN stats)
  // =========================================================================
  console.log('\n[5/12] Building encoder and encoding features...');
  const encodingStart = Date.now();
  const allInputs = allFeatures.map(f => f.features);
  const batchSizeEnc = 64;
  let allEmbeddings = [];
  for (let i = 0; i < allInputs.length; i += batchSizeEnc) {
    allEmbeddings = allEmbeddings.concat(encoder.encodeBatch(allInputs.slice(i, i + batchSizeEnc)));
  }
  const encodedFeatures = allFeatures.map((f, i) => ({ ...f, embedding: allEmbeddings[i] }));
  console.log(`  Encoded ${encodedFeatures.length} frames in ${Date.now() - encodingStart}ms`);

  // =========================================================================
  // Step 6: Phase 2 — Enhanced contrastive pretraining
  // =========================================================================
  console.log('\n[6/12] Phase 2: Enhanced contrastive pretraining...');

  const contrastiveTrainer = new ContrastiveTrainer({
    epochs: CONFIG.epochs, batchSize: CONFIG.batchSize, margin: CONFIG.margin,
    temperature: CONFIG.temperature, hardNegativeRatio: CONFIG.hardNegativeRatio,
    learningRate: CONFIG.learningRate, outputPath: path.join(CONFIG.outputDir, 'contrastive'),
  });

  for (const triplet of allTriplets) {
    const aEmb = encoder.encode(triplet.anchor);
    const pEmb = encoder.encode(triplet.positive);
    const nEmb = encoder.encode(triplet.negative);
    contrastiveTrainer.addTriplet(triplet.anchorLabel, aEmb, triplet.posLabel, pEmb, triplet.negLabel, nEmb, triplet.isHard);
  }

  console.log(`  Triplets loaded: ${contrastiveTrainer.getTripletCount()}`);
  const contrastiveResult = contrastiveTrainer.train();

  // Gradient update of encoder weights
  console.log('  Applying gradient updates to encoder...');
  let initialContrastiveLoss = 0;
  for (const t of allTriplets) {
    initialContrastiveLoss += tripletLoss(encoder.encode(t.anchor), encoder.encode(t.positive), encoder.encode(t.negative), CONFIG.margin);
  }
  initialContrastiveLoss /= allTriplets.length || 1;

  let finalContrastiveLoss = 0;
  for (let epoch = 0; epoch < CONFIG.epochs; epoch++) {
    let epochLoss = 0;
    const shuffled = [...allTriplets];
    let shuffleSeed = epoch * 31 + 17;
    for (let i = shuffled.length - 1; i > 0; i--) {
      shuffleSeed ^= shuffleSeed << 13; shuffleSeed ^= shuffleSeed >> 17; shuffleSeed ^= shuffleSeed << 5;
      const j = (shuffleSeed >>> 0) % (i + 1);
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }

    for (const t of shuffled) {
      const aEmb = encoder.encode(t.anchor);
      const pEmb = encoder.encode(t.positive);
      const nEmb = encoder.encode(t.negative);
      const loss = tripletLoss(aEmb, pEmb, nEmb, CONFIG.margin);
      epochLoss += loss;
      if (loss > 0) {
        const grad = computeGradient(aEmb, pEmb, nEmb, CONFIG.learningRate);
        const { hidden } = encoder.encodeRaw(t.anchor);
        for (let j = 0; j < encoder.outputDim; j++) {
          for (let i = 0; i < encoder.hiddenDim; i++) {
            if (hidden[i] > 0) encoder.w2[i * encoder.outputDim + j] += grad[j] * hidden[i] * 0.01;
          }
          encoder.b2[j] += grad[j] * 0.01;
        }
      }
    }
    epochLoss /= shuffled.length || 1;
    if (epoch === CONFIG.epochs - 1 || epoch % 5 === 0) {
      if (CONFIG.verbose) console.log(`    Epoch ${epoch + 1}: loss=${epochLoss.toFixed(6)}`);
    }
    finalContrastiveLoss = epochLoss;
  }

  // Re-encode with updated encoder
  let reEncodedEmbs = [];
  for (let i = 0; i < allInputs.length; i += batchSizeEnc) {
    reEncodedEmbs = reEncodedEmbs.concat(encoder.encodeBatch(allInputs.slice(i, i + batchSizeEnc)));
  }
  for (let i = 0; i < encodedFeatures.length; i++) encodedFeatures[i].embedding = reEncodedEmbs[i];

  const improvement = initialContrastiveLoss > 0
    ? ((initialContrastiveLoss - finalContrastiveLoss) / initialContrastiveLoss * 100) : 0;
  console.log(`  Initial loss: ${initialContrastiveLoss.toFixed(6)}, Final: ${finalContrastiveLoss.toFixed(6)}, Improvement: ${improvement.toFixed(1)}%`);

  contrastiveResult.initialLoss = initialContrastiveLoss;
  contrastiveResult.finalLoss = finalContrastiveLoss;
  contrastiveResult.improvement = improvement;

  const contrastiveOutDir = contrastiveTrainer.exportTrainingData();
  console.log(`  Exported to: ${contrastiveOutDir}`);

  // =========================================================================
  // Step 7: Task head training (presence + activity + vitals)
  // =========================================================================
  console.log('\n[7/12] Task head training...');

  const taskAdapter = new LoraAdapter(
    { rank: CONFIG.loraRank * 2, alpha: CONFIG.loraRank * 4, dropout: 0.05, targetModules: ['encoder', 'task_heads'] },
    CONFIG.embeddingDim, CONFIG.embeddingDim
  );

  const taskPipeline = new TrainingPipeline({
    learningRate: CONFIG.learningRate, batchSize: CONFIG.batchSize,
    epochs: Math.max(5, Math.floor(CONFIG.epochs / 2)),
    scheduler: 'cosine', warmupSteps: 50, earlyStoppingPatience: 5,
    checkpointInterval: 2, ewcLambda: 2000, validationSplit: 0.1,
  }, taskAdapter);

  let labeledCount = 0;
  const taskTrainingData = [];

  for (const ef of encodedFeatures) {
    const labels = createLabels(ef, allVitals);
    if (!labels) continue;
    const target = new Array(CONFIG.embeddingDim).fill(0);
    target[0] = labels.presence;
    target[1] = labels.activity[0]; target[2] = labels.activity[1]; target[3] = labels.activity[2];
    target[4] = labels.vitalsTarget[0]; target[5] = labels.vitalsTarget[1];
    taskTrainingData.push({ input: ef.embedding, target, quality: 1.0 });
    labeledCount++;
  }

  console.log(`  Labeled samples: ${labeledCount} / ${encodedFeatures.length}`);
  if (taskTrainingData.length > 0) {
    taskPipeline.addData(taskTrainingData);
    const taskResult = taskPipeline.train();
    console.log(`  Epochs: ${taskResult.epochs}, Final loss: ${taskResult.finalLoss.toFixed(6)}`);
  }

  // Presence head
  console.log('\n[7b/12] Presence head training...');
  const presenceHead = new PresenceHead(CONFIG.embeddingDim);
  const presenceTrainData = [];
  for (const ef of encodedFeatures) {
    const labels = createLabels(ef, allVitals);
    if (!labels) continue;
    presenceTrainData.push({ embedding: ef.embedding, target: labels.presence });
  }
  if (presenceTrainData.length > 0) {
    let presenceLoss = 0;
    for (let epoch = 0; epoch < 30; epoch++) {
      presenceLoss = 0;
      let pSeed = epoch * 41 + 7;
      const pShuffled = [...presenceTrainData];
      for (let i = pShuffled.length - 1; i > 0; i--) {
        pSeed ^= pSeed << 13; pSeed ^= pSeed >> 17; pSeed ^= pSeed << 5;
        const j = (pSeed >>> 0) % (i + 1);
        [pShuffled[i], pShuffled[j]] = [pShuffled[j], pShuffled[i]];
      }
      for (const sample of pShuffled) presenceLoss += presenceHead.trainStep(sample.embedding, sample.target, 0.01);
      presenceLoss /= pShuffled.length;
    }
    let presCorrect = 0;
    for (const s of presenceTrainData) if ((presenceHead.forward(s.embedding) > 0.5 ? 1 : 0) === s.target) presCorrect++;
    console.log(`  Presence accuracy: ${(presCorrect / presenceTrainData.length * 100).toFixed(1)}% (loss: ${presenceLoss.toFixed(6)})`);
  }

  // =========================================================================
  // Step 8: Phase 3 — Pose proxy training (5 keypoints, no camera)
  // =========================================================================
  console.log('\n[8/12] Phase 3: Pose proxy training (5-keypoint, no camera)...');
  const poseDecoder = new PoseDecoder5(CONFIG.embeddingDim);

  // Collect pose training data from weak labels
  const poseTrainData = [];
  for (const ef of encodedFeatures) {
    // Find corresponding timeline frame
    const tlFrame = labeledTimeline.find(f =>
      f.nodeId === ef.nodeId && Math.abs(f.timestamp - ef.timestamp) < 0.1
    );
    if (tlFrame && tlFrame.labels && tlFrame.labels.poseProxy5) {
      poseTrainData.push({
        embedding: ef.embedding,
        target: tlFrame.labels.poseProxy5,
        confidence: tlFrame.labels.confidence,
      });
    }
  }

  console.log(`  Pose training samples: ${poseTrainData.length}`);

  if (poseTrainData.length > 10) {
    const poseEpochs = 30;
    const poseLr = 0.005;
    let poseLoss = 0;

    for (let epoch = 0; epoch < poseEpochs; epoch++) {
      poseLoss = 0;
      let pSeed = epoch * 53 + 11;
      const shuffled = [...poseTrainData];
      for (let i = shuffled.length - 1; i > 0; i--) {
        pSeed ^= pSeed << 13; pSeed ^= pSeed >> 17; pSeed ^= pSeed << 5;
        const j = (pSeed >>> 0) % (i + 1);
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }

      for (const sample of shuffled) {
        // Weight by confidence: higher confidence = higher learning rate
        const sampleLr = poseLr * Math.max(0.2, sample.confidence);
        poseLoss += poseDecoder.trainStep(sample.embedding, sample.target, sampleLr, CONFIG.skeleton);
      }
      poseLoss /= shuffled.length;

      if (CONFIG.verbose && (epoch % 10 === 0 || epoch === poseEpochs - 1)) {
        console.log(`    Pose epoch ${epoch + 1}: loss=${poseLoss.toFixed(6)}`);
      }
    }
    console.log(`  Final pose loss: ${poseLoss.toFixed(6)}`);
  } else {
    console.log('  WARN: Too few pose samples. Skipping pose proxy training.');
  }

  // =========================================================================
  // Step 9: Phase 4 — Upgrade to 17 keypoints (interpolation)
  // =========================================================================
  console.log('\n[9/12] Phase 4: 5-keypoint -> 17-keypoint interpolation...');

  // Verify interpolation on sample frames
  let kp17Count = 0;
  const kp17Samples = [];
  for (const sample of poseTrainData.slice(0, 100)) {
    const kp5 = poseDecoder.forward(sample.embedding);
    const kp5flat = [kp5[0].x, kp5[0].y, kp5[1].x, kp5[1].y, kp5[2].x, kp5[2].y, kp5[3].x, kp5[3].y, kp5[4].x, kp5[4].y];
    const kp17 = interpolateTo17Keypoints(kp5flat, CONFIG.skeleton);
    kp17Samples.push(kp17);
    kp17Count++;
  }
  console.log(`  Interpolated ${kp17Count} frames from 5 to 17 keypoints.`);

  if (kp17Samples.length > 0) {
    // Verify bone length constraints
    let constraintViolations = 0;
    for (const kp17 of kp17Samples) {
      // Check shoulder width
      const sw = Math.sqrt((kp17[10] - kp17[12]) ** 2 + (kp17[11] - kp17[13]) ** 2);
      if (sw > CONFIG.skeleton.shoulderWidth / 5.0 * 1.1) constraintViolations++;
    }
    console.log(`  Skeleton constraint violations: ${constraintViolations}/${kp17Samples.length}`);
  }

  // =========================================================================
  // Step 10: Phase 5 — Self-refinement loop
  // =========================================================================
  console.log(`\n[10/12] Phase 5: Self-refinement (${CONFIG.selfRefineRounds} rounds)...`);

  if (poseTrainData.length > 10) {
    for (let round = 0; round < CONFIG.selfRefineRounds; round++) {
      // Run inference on all data
      const confidentPredictions = [];
      for (const ef of encodedFeatures) {
        const presence = presenceHead.forward(ef.embedding);
        if (presence < 0.3) continue; // skip empty frames

        const kp5 = poseDecoder.forward(ef.embedding);
        // Compute prediction confidence: consistency between forward passes
        // Proxy: variance of keypoint positions across nearby frames
        const nearbyFrames = encodedFeatures.filter(f =>
          f.nodeId === ef.nodeId && Math.abs(f.timestamp - ef.timestamp) < 0.5 && f !== ef
        );
        let variance = 0;
        if (nearbyFrames.length > 0) {
          for (const nf of nearbyFrames) {
            const nkp = poseDecoder.forward(nf.embedding);
            for (let k = 0; k < 5; k++) {
              variance += (kp5[k].x - nkp[k].x) ** 2 + (kp5[k].y - nkp[k].y) ** 2;
            }
          }
          variance /= nearbyFrames.length * 10;
        }
        const confidence = 1.0 / (1.0 + variance * 100);

        if (confidence > 0.8) {
          confidentPredictions.push({
            embedding: ef.embedding,
            target: [kp5[0].x, kp5[0].y, kp5[1].x, kp5[1].y, kp5[2].x, kp5[2].y, kp5[3].x, kp5[3].y, kp5[4].x, kp5[4].y],
            confidence,
          });
        }
      }

      console.log(`  Round ${round + 1}: ${confidentPredictions.length} confident predictions (>0.8)`);

      if (confidentPredictions.length < 5) {
        console.log('  Too few confident predictions, stopping refinement.');
        break;
      }

      // Retrain pose decoder with pseudo-labels
      const refineLr = CONFIG.learningRate * 0.1 * (1.0 / (round + 1)); // decay LR each round
      let refineLoss = 0;
      for (let epoch = 0; epoch < 10; epoch++) {
        refineLoss = 0;
        for (const sample of confidentPredictions) {
          refineLoss += poseDecoder.trainStep(sample.embedding, sample.target, refineLr * sample.confidence, CONFIG.skeleton);
        }
        refineLoss /= confidentPredictions.length;
      }
      console.log(`  Round ${round + 1} refinement loss: ${refineLoss.toFixed(6)}`);
    }
  } else {
    console.log('  Skipping self-refinement (no pose training data).');
  }

  // =========================================================================
  // Step 11: LoRA refinement + Quantization + EWC (same as train-ruvllm.js)
  // =========================================================================
  console.log('\n[11/12] LoRA refinement + quantization + EWC...');

  // LoRA per-node
  const loraManager = new LoraManager({
    rank: CONFIG.loraRank, alpha: CONFIG.loraRank * 2, dropout: 0.1, targetModules: ['room_adapt'],
  });

  for (const nodeId of nodeIds) {
    const nodeAdapter = loraManager.create(`node-${nodeId}`,
      { rank: CONFIG.loraRank, alpha: CONFIG.loraRank * 2, dropout: 0.1 },
      CONFIG.embeddingDim, CONFIG.embeddingDim
    );
    const nodeFeatures = encodedFeatures.filter(f => f.nodeId === nodeId);
    const nodePipeline = new TrainingPipeline({
      learningRate: CONFIG.learningRate * 0.5,
      batchSize: Math.min(CONFIG.batchSize, nodeFeatures.length),
      epochs: 5, scheduler: 'cosine', ewcLambda: 3000,
    }, nodeAdapter);

    const nodeData = [];
    for (const nf of nodeFeatures) {
      const labels = createLabels(nf, allVitals);
      if (!labels) continue;
      const target = new Array(CONFIG.embeddingDim).fill(0);
      target[0] = labels.presence;
      target[1] = labels.activity[0]; target[2] = labels.activity[1]; target[3] = labels.activity[2];
      target[4] = labels.vitalsTarget[0]; target[5] = labels.vitalsTarget[1];
      nodeData.push({ input: nf.embedding, target, quality: 1.0 });
    }
    if (nodeData.length > 0) {
      nodePipeline.addData(nodeData);
      const nr = nodePipeline.train();
      console.log(`  Node ${nodeId}: ${nodeData.length} samples, loss=${nr.finalLoss.toFixed(6)}`);
    }
  }
  console.log(`  LoRA adapters: ${loraManager.list().join(', ')}`);

  // Quantization
  console.log('  Quantization...');
  const mergedWeights = taskAdapter.merge();
  const flatWeights = new Float32Array(mergedWeights.flat());
  const quantResults = {};
  for (const bits of [2, 4, 8]) {
    const qr = quantizeWeights(flatWeights, bits);
    const deq = dequantizeWeights(qr.quantized, qr.scale, qr.zeroPoint, bits, qr.numWeights);
    const rmse = quantizationQuality(flatWeights, deq);
    quantResults[bits] = { ...qr, rmse };
    console.log(`  ${bits}-bit: ${qr.compressionRatio.toFixed(1)}x compression, RMSE=${rmse.toFixed(6)}`);
  }

  // EWC
  console.log('  EWC consolidation...');
  const ewcManager = taskPipeline.getEwcManager();
  ewcManager.registerTask('csi-camerafree-v1', taskAdapter.merge().flat());
  for (const nodeId of nodeIds) {
    const na = loraManager.get(`node-${nodeId}`);
    if (na) ewcManager.registerTask(`node-${nodeId}-adapt`, na.merge().flat());
  }
  const ewcStats = ewcManager.stats();
  console.log(`  EWC tasks: ${ewcStats.tasksLearned}, forgetting rate: ${ewcStats.forgettingRate.toFixed(4)}`);

  // =========================================================================
  // Step 12: Export
  // =========================================================================
  console.log('\n[12/12] Exporting models...');
  fs.mkdirSync(CONFIG.outputDir, { recursive: true });

  const exporter = new ModelExporter();
  const exportModel = {
    metadata: {
      name: 'wifi-densepose-camerafree',
      version: '1.0.0',
      architecture: 'csi-encoder-8-64-128-pose17',
      pipelineType: 'camera-free',
      seedAvailable: seedAvailable,
      supervisionSignals: seedAvailable ? 10 : 3,
      training: {
        steps: contrastiveResult.history.length * contrastiveTrainer.getTripletCount(),
        loss: contrastiveResult.finalLoss,
        learningRate: CONFIG.learningRate,
        selfRefineRounds: CONFIG.selfRefineRounds,
      },
      custom: {
        inputDim: CONFIG.inputDim,
        hiddenDim: CONFIG.hiddenDim,
        embeddingDim: CONFIG.embeddingDim,
        poseKeypoints5: CONFIG.poseKeypoints5,
        poseKeypoints17: CONFIG.poseKeypoints17,
        positionGridSize: CONFIG.positionGridSize,
        totalFrames: allFeatures.length,
        totalTriplets: allTriplets.length,
        multiModalTriplets: multiModalTriplets.length,
        nodes: nodeIds,
        quantizationBits: CONFIG.quantizeBits,
      },
    },
    loraWeights: taskAdapter.getWeights(),
    loraConfig: taskAdapter.getConfig(),
    ewcStats,
    tensors: new Map(),
  };

  // Encoder tensors
  exportModel.tensors.set('encoder.w1', new Float32Array(encoder.w1));
  exportModel.tensors.set('encoder.b1', new Float32Array(encoder.b1));
  exportModel.tensors.set('encoder.w2', new Float32Array(encoder.w2));
  exportModel.tensors.set('encoder.b2', new Float32Array(encoder.b2));
  exportModel.tensors.set('encoder.bn1_gamma', new Float32Array(encoder.bn1_gamma));
  exportModel.tensors.set('encoder.bn1_beta', new Float32Array(encoder.bn1_beta));
  exportModel.tensors.set('encoder.bn1_runMean', new Float32Array(encoder.bn1_runMean));
  exportModel.tensors.set('encoder.bn1_runVar', new Float32Array(encoder.bn1_runVar));
  exportModel.tensors.set('encoder.bn2_gamma', new Float32Array(encoder.bn2_gamma));
  exportModel.tensors.set('encoder.bn2_beta', new Float32Array(encoder.bn2_beta));
  exportModel.tensors.set('encoder.bn2_runMean', new Float32Array(encoder.bn2_runMean));
  exportModel.tensors.set('encoder.bn2_runVar', new Float32Array(encoder.bn2_runVar));

  // Presence head
  exportModel.tensors.set('presence_head.weights', new Float32Array(presenceHead.weights));
  exportModel.tensors.set('presence_head.bias', new Float32Array([presenceHead.bias]));

  // Pose decoder
  exportModel.tensors.set('pose_decoder.w1', new Float32Array(poseDecoder.w1));
  exportModel.tensors.set('pose_decoder.b1', new Float32Array(poseDecoder.b1));
  exportModel.tensors.set('pose_decoder.w2', new Float32Array(poseDecoder.w2));
  exportModel.tensors.set('pose_decoder.b2', new Float32Array(poseDecoder.b2));

  // SafeTensors
  const safetensorsBuffer = exporter.toSafeTensors(exportModel);
  fs.writeFileSync(path.join(CONFIG.outputDir, 'model.safetensors'), safetensorsBuffer);
  console.log(`  SafeTensors: model.safetensors (${(safetensorsBuffer.length / 1024).toFixed(1)} KB)`);

  // HuggingFace config
  const hfExport = exporter.toHuggingFace(exportModel);
  fs.writeFileSync(path.join(CONFIG.outputDir, 'config.json'), hfExport.config);
  console.log(`  HF config: config.json`);

  // JSON model
  const jsonExport = exporter.toJSON(exportModel);
  fs.writeFileSync(path.join(CONFIG.outputDir, 'model.json'), jsonExport);

  // Presence head JSON
  fs.writeFileSync(path.join(CONFIG.outputDir, 'presence-head.json'), JSON.stringify(presenceHead.getWeights()));

  // Pose decoder JSON
  fs.writeFileSync(path.join(CONFIG.outputDir, 'pose-decoder.json'), JSON.stringify(poseDecoder.getWeights()));
  console.log(`  Pose decoder: pose-decoder.json`);

  // Quantized models
  const quantDir = path.join(CONFIG.outputDir, 'quantized');
  fs.mkdirSync(quantDir, { recursive: true });
  for (const [bits, qr] of Object.entries(quantResults)) {
    const qPath = path.join(quantDir, `model-q${bits}.bin`);
    fs.writeFileSync(qPath, Buffer.from(qr.quantized));
    console.log(`  Quantized ${bits}-bit: model-q${bits}.bin (${(qr.quantizedSize / 1024).toFixed(1)} KB)`);
  }

  // LoRA adapters
  const loraDir = path.join(CONFIG.outputDir, 'lora');
  fs.mkdirSync(loraDir, { recursive: true });
  for (const adapterId of loraManager.list()) {
    const adapter = loraManager.get(adapterId);
    fs.writeFileSync(path.join(loraDir, `${adapterId}.json`), adapter.toJSON());
    console.log(`  LoRA: ${adapterId}.json`);
  }

  // RVF manifest
  const rvfPath = path.join(CONFIG.outputDir, 'model.rvf.jsonl');
  const rvfLines = [
    JSON.stringify({ type: 'metadata', ...exportModel.metadata }),
    JSON.stringify({ type: 'encoder', w1_shape: [CONFIG.inputDim, CONFIG.hiddenDim], w2_shape: [CONFIG.hiddenDim, CONFIG.embeddingDim] }),
    JSON.stringify({ type: 'pose_decoder', architecture: '128-64-10', keypoints5: true, keypoints17: 'interpolated' }),
    JSON.stringify({ type: 'lora', config: taskAdapter.getConfig(), parameters: taskAdapter.numParameters() }),
    JSON.stringify({ type: 'ewc', stats: ewcStats }),
    JSON.stringify({ type: 'quantization', default_bits: CONFIG.quantizeBits, variants: [2, 4, 8] }),
    JSON.stringify({ type: 'camera_free_supervision', signals: seedAvailable ? 10 : 3,
      sources: seedAvailable
        ? ['PIR', 'BME280_temp', 'BME280_humidity', 'RSSI_diff', 'vitals_stability',
           'temporal_CSI', 'kNN_clusters', 'boundary_fragility', 'reed_switch', 'vibration']
        : ['RSSI_diff', 'vitals_stability', 'temporal_CSI'] }),
  ];
  fs.writeFileSync(rvfPath, rvfLines.join('\n'));
  console.log(`  RVF manifest: model.rvf.jsonl`);

  // Training metrics
  const metricsPath = path.join(CONFIG.outputDir, 'training-metrics.json');
  const metrics = {
    timestamp: new Date().toISOString(),
    pipelineType: 'camera-free',
    totalDurationMs: Date.now() - startTime,
    seedAvailable,
    supervisionSignals: seedAvailable ? 10 : 3,
    data: {
      files: files.map(f => path.basename(f)),
      totalFeatures: allFeatures.length,
      totalVitals: allVitals.length,
      totalRawCsi: allRawCsi.length,
      nodes: nodeIds,
      multiModalFrames: timeline.length,
      seedFrames: timeline.filter(f => f.seed_sensors !== null).length,
    },
    weakLabels: {
      totalLabeled: labeledTimeline.length,
      poseProxyFrames: poseLabelCount,
      sensorLabeled: sensorLabelCount,
      activityDistribution: actDist,
    },
    contrastive: {
      triplets: allTriplets.length,
      temporal: allTriplets.filter(t => t.type === 'temporal').length,
      crossNode: allTriplets.filter(t => t.type === 'cross-node').length,
      sensorVerified: allTriplets.filter(t => t.type === 'sensor-verified').length,
      activityBoundary: allTriplets.filter(t => t.type === 'activity-boundary').length,
      crossModal: allTriplets.filter(t => t.type === 'cross-modal').length,
      hardNegatives: allTriplets.filter(t => t.isHard).length,
      initialLoss: contrastiveResult.initialLoss,
      finalLoss: contrastiveResult.finalLoss,
      improvement: contrastiveResult.improvement,
    },
    poseTraining: {
      samples: poseTrainData.length,
      keypoints5: CONFIG.poseKeypoints5,
      keypoints17: CONFIG.poseKeypoints17,
      selfRefineRounds: CONFIG.selfRefineRounds,
    },
    lora: { adapters: loraManager.list(), totalParameters: loraManager.stats().totalParameters },
    quantization: Object.fromEntries(
      Object.entries(quantResults).map(([bits, qr]) => [`q${bits}`, { compressionRatio: qr.compressionRatio, rmse: qr.rmse, sizeKB: qr.quantizedSize / 1024 }])
    ),
    ewc: ewcStats,
    config: CONFIG,
  };
  fs.writeFileSync(metricsPath, JSON.stringify(metrics, null, 2));
  console.log(`  Metrics: training-metrics.json`);

  // =========================================================================
  // Summary
  // =========================================================================
  const totalDuration = Date.now() - startTime;
  console.log('\n=== Training Complete ===');
  console.log(`  Pipeline: Camera-Free (${seedAvailable ? '10 signals' : 'CSI-only fallback, 3 signals'})`);
  console.log(`  Duration: ${(totalDuration / 1000).toFixed(1)}s`);
  console.log(`  Output: ${path.resolve(CONFIG.outputDir)}`);
  console.log(`  Model (fp32): ${(safetensorsBuffer.length / 1024).toFixed(1)} KB`);
  console.log(`  Model (q${CONFIG.quantizeBits}): ${(quantResults[CONFIG.quantizeBits]?.quantizedSize / 1024 || 0).toFixed(1)} KB`);
  console.log(`  Pose: 5-keypoint (trained) -> 17-keypoint (interpolated, COCO format)`);
  console.log(`  LoRA adapters: ${loraManager.count()}`);
  console.log(`  EWC tasks: ${ewcStats.tasksLearned}`);

  // =========================================================================
  // Optional benchmark
  // =========================================================================
  if (CONFIG.benchmark) {
    console.log('\n=== Benchmark Mode ===');
    runBenchmark(encoder, taskAdapter, presenceHead, poseDecoder, allFeatures, allVitals, quantResults);
  }
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------
function runBenchmark(encoder, adapter, presenceHead, poseDecoder, features, vitals, quantResults) {
  const N = Math.min(1000, features.length);
  const testFeatures = features.slice(0, N);

  // Inference latency
  console.log(`\nInference latency (${N} samples, encoder + adapter + presence + pose):`);
  const latencies = [];
  for (const f of testFeatures) {
    const start = process.hrtime.bigint();
    const emb = encoder.encode(f.features);
    adapter.forward(emb);
    presenceHead.forward(emb);
    const kp5 = poseDecoder.forward(emb);
    const kp5flat = [kp5[0].x, kp5[0].y, kp5[1].x, kp5[1].y, kp5[2].x, kp5[2].y, kp5[3].x, kp5[3].y, kp5[4].x, kp5[4].y];
    interpolateTo17Keypoints(kp5flat, CONFIG.skeleton);
    const elapsed = Number(process.hrtime.bigint() - start) / 1e6;
    latencies.push(elapsed);
  }

  latencies.sort((a, b) => a - b);
  const mean = latencies.reduce((a, b) => a + b, 0) / latencies.length;
  const p95 = latencies[Math.floor(latencies.length * 0.95)];
  const p99 = latencies[Math.floor(latencies.length * 0.99)];
  console.log(`  Mean:  ${mean.toFixed(3)}ms`);
  console.log(`  P95:   ${p95.toFixed(3)}ms`);
  console.log(`  P99:   ${p99.toFixed(3)}ms`);
  console.log(`  Throughput: ${(1000 / mean).toFixed(0)} poses/sec`);

  // Embedding quality
  console.log('\nEmbedding quality (temporal pairs):');
  let posSim = [], negSim = [];
  for (let i = 0; i < Math.min(features.length - 1, 200); i++) {
    const emb1 = encoder.encode(features[i].features);
    const emb2 = encoder.encode(features[i + 1].features);
    const sim = cosineSimilarity(emb1, emb2);
    const td = Math.abs(features[i + 1].timestamp - features[i].timestamp);
    if (td <= 1.0) posSim.push(sim);
    else if (td >= CONFIG.negativeWindowSec) negSim.push(sim);
  }
  if (posSim.length > 0) console.log(`  Positive pair avg: ${(posSim.reduce((a, b) => a + b, 0) / posSim.length).toFixed(4)} (n=${posSim.length})`);
  if (negSim.length > 0) console.log(`  Negative pair avg: ${(negSim.reduce((a, b) => a + b, 0) / negSim.length).toFixed(4)} (n=${negSim.length})`);

  // Presence detection accuracy
  console.log('\nPresence detection accuracy:');
  let correct = 0, total = 0;
  for (const f of testFeatures) {
    const labels = createLabels(f, vitals);
    if (!labels) continue;
    const emb = encoder.encode(f.features);
    if ((presenceHead.forward(emb) > 0.5 ? 1 : 0) === labels.presence) correct++;
    total++;
  }
  if (total > 0) console.log(`  Accuracy: ${(correct / total * 100).toFixed(1)}% (${correct}/${total})`);

  // Pose prediction sample
  console.log('\nPose prediction (first 3 frames):');
  for (let i = 0; i < Math.min(3, testFeatures.length); i++) {
    const emb = encoder.encode(testFeatures[i].features);
    const pres = presenceHead.forward(emb);
    if (pres < 0.3) { console.log(`  Frame ${i}: empty (presence=${pres.toFixed(2)})`); continue; }
    const kp5 = poseDecoder.forward(emb);
    console.log(`  Frame ${i}: presence=${pres.toFixed(2)} head=(${kp5[0].x.toFixed(2)},${kp5[0].y.toFixed(2)}) ` +
      `Lhand=(${kp5[1].x.toFixed(2)},${kp5[1].y.toFixed(2)}) Rhand=(${kp5[2].x.toFixed(2)},${kp5[2].y.toFixed(2)}) ` +
      `Lfoot=(${kp5[3].x.toFixed(2)},${kp5[3].y.toFixed(2)}) Rfoot=(${kp5[4].x.toFixed(2)},${kp5[4].y.toFixed(2)})`);
  }

  // Memory usage
  console.log('\nQuantization:');
  console.log('  Bits | Size (KB) | Compression | RMSE');
  console.log('  -----|-----------|-------------|------');
  for (const [bits, qr] of Object.entries(quantResults)) {
    console.log(`  ${bits.padStart(4)} | ${(qr.quantizedSize / 1024).toFixed(1).padStart(9)} | ${qr.compressionRatio.toFixed(1).padStart(11)}x | ${qr.rmse.toFixed(6)}`);
  }
  console.log(`  fp32 | ${(quantResults[Object.keys(quantResults)[0]].originalSize / 1024).toFixed(1).padStart(9)} | ${' '.padStart(10)}1x | 0.000000`);
}

// Run
main().catch(err => {
  console.error('Camera-free training pipeline failed:', err);
  process.exit(1);
});
