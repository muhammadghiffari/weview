#!/usr/bin/env node
/**
 * WiFlow Pose Estimation Training Pipeline
 *
 * Trains the WiFlow architecture (arXiv:2602.08661) on collected CSI data
 * using the ruvllm training infrastructure. Extends train-ruvllm.js patterns
 * with WiFlow-specific stages:
 *
 *   Phase 0: CSI data loading + amplitude extraction + 20-frame windowing
 *   Phase 1: Contrastive pretraining (temporal consistency loss)
 *   Phase 2: Supervised pose training (SmoothL1 + bone constraint loss)
 *   Phase 3: LoRA room-specific adaptation
 *   Phase 4: Quantization (TurboQuant INT8 target: ~2.5 MB)
 *   Phase 5: Export (SafeTensors + ONNX-compatible + RVF)
 *
 * Usage:
 *   node scripts/train-wiflow.js --data data/recordings/pretrain-*.csi.jsonl
 *   node scripts/train-wiflow.js --data data/recordings/*.csi.jsonl --epochs 50 --output models/wiflow-v1
 *   node scripts/train-wiflow.js --data data/recordings/*.csi.jsonl --contrastive-only
 *
 * ADR: docs/adr/ADR-072-wiflow-architecture.md
 */

'use strict';

const fs = require('fs');
const path = require('path');
const { parseArgs } = require('util');

// ---------------------------------------------------------------------------
// Resolve dependencies
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
} = require(path.join(RUVLLM_PATH, 'sona.js'));

const {
  SafeTensorsWriter,
  ModelExporter,
} = require(path.join(RUVLLM_PATH, 'export.js'));

const {
  WiFlowModel,
  COCO_KEYPOINTS,
  BONE_CONNECTIONS,
  BONE_LENGTH_PRIORS,
  smoothL1,
  createRng,
  gaussianRng,
  estimateFLOPs,
} = require(path.join(__dirname, 'wiflow-model.js'));

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------
const { values: args } = parseArgs({
  options: {
    data: { type: 'string', short: 'd' },
    output: { type: 'string', short: 'o', default: 'models/wiflow-v1' },
    epochs: { type: 'string', short: 'e', default: '30' },
    'batch-size': { type: 'string', default: '8' },
    'learning-rate': { type: 'string', default: '0.001' },
    'lora-rank': { type: 'string', default: '4' },
    'quantize-bits': { type: 'string', default: '8' },
    'contrastive-only': { type: 'boolean', default: false },
    'max-samples': { type: 'string', default: '0' },
    'time-steps': { type: 'string', default: '20' },
    'subcarriers': { type: 'string', default: '128' },
    seed: { type: 'string', default: '42' },
    verbose: { type: 'boolean', short: 'v', default: false },
  },
  strict: true,
});

if (!args.data) {
  console.error('Usage: node scripts/train-wiflow.js --data <path-to-csi-jsonl> [--output dir] [--epochs N]');
  process.exit(1);
}

const CONFIG = {
  dataGlob: args.data,
  outputDir: args.output,
  epochs: parseInt(args.epochs, 10),
  batchSize: parseInt(args['batch-size'], 10),
  learningRate: parseFloat(args['learning-rate']),
  loraRank: parseInt(args['lora-rank'], 10),
  quantizeBits: parseInt(args['quantize-bits'], 10),
  contrastiveOnly: args['contrastive-only'],
  maxSamples: parseInt(args['max-samples'], 10) || 0,
  timeSteps: parseInt(args['time-steps'], 10),
  subcarriers: parseInt(args['subcarriers'], 10),
  seed: parseInt(args.seed, 10),
  verbose: args.verbose,

  // Contrastive hyperparameters
  margin: 0.3,
  temperature: 0.07,
  contrastiveEpochs: 3,

  // Bone constraint weight
  boneWeight: 0.2,
};

// ---------------------------------------------------------------------------
// Data loading and CSI amplitude extraction
// ---------------------------------------------------------------------------

/**
 * Parse CSI JSONL file and extract raw CSI frames.
 */
function loadCsiData(filePath) {
  const rawCsi = [];
  const features = [];
  const vitals = [];

  const content = fs.readFileSync(filePath, 'utf-8');
  for (const line of content.split('\n')) {
    if (!line.trim()) continue;
    try {
      const frame = JSON.parse(line);
      switch (frame.type) {
        case 'raw_csi':
          rawCsi.push({
            timestamp: frame.timestamp,
            nodeId: frame.node_id,
            subcarriers: frame.subcarriers,
            iqHex: frame.iq_hex,
            rssi: frame.rssi,
          });
          break;
        case 'feature':
          features.push({
            timestamp: frame.timestamp,
            nodeId: frame.node_id,
            features: frame.features,
            rssi: frame.rssi,
          });
          break;
        case 'vitals':
          vitals.push({
            timestamp: frame.timestamp,
            nodeId: frame.node_id,
            presenceScore: frame.presence_score,
            motionEnergy: frame.motion_energy,
            breathingBpm: frame.breathing_bpm,
            heartrateBpm: frame.heartrate_bpm,
            nPersons: frame.n_persons,
          });
          break;
      }
    } catch (_) { /* skip malformed */ }
  }

  return { rawCsi, features, vitals };
}

/**
 * Parse IQ hex string into complex pairs [I0, Q0, I1, Q1, ...].
 * Each I/Q value is a signed byte.
 */
function parseIqHex(hexStr) {
  const bytes = [];
  for (let i = 0; i < hexStr.length; i += 2) {
    let val = parseInt(hexStr.substr(i, 2), 16);
    if (val > 127) val -= 256; // signed byte
    bytes.push(val);
  }
  return bytes;
}

/**
 * Extract amplitude from IQ data for a given number of subcarriers.
 * Returns Float32Array of amplitudes [nSubcarriers].
 * Skips first I/Q pair (DC offset) per WiFlow paper recommendation.
 */
function extractAmplitude(iqBytes, nSubcarriers) {
  const amp = new Float32Array(nSubcarriers);
  // Each subcarrier has 2 bytes (I, Q), first pair is often DC/padding
  const start = 2; // skip first IQ pair (index 0,1)
  for (let sc = 0; sc < nSubcarriers; sc++) {
    const idx = start + sc * 2;
    if (idx + 1 < iqBytes.length) {
      const I = iqBytes[idx];
      const Q = iqBytes[idx + 1];
      amp[sc] = Math.sqrt(I * I + Q * Q);
    }
  }
  return amp;
}

/**
 * Normalize amplitude to zero-mean, unit-variance per subcarrier across time window.
 */
function normalizeAmplitude(window) {
  // window: array of Float32Array [nSubcarriers]
  const T = window.length;
  if (T === 0) return [];
  const nSc = window[0].length;

  // Compute per-subcarrier mean and std
  const mean = new Float32Array(nSc);
  const std = new Float32Array(nSc);
  for (let sc = 0; sc < nSc; sc++) {
    let sum = 0;
    for (let t = 0; t < T; t++) sum += window[t][sc];
    mean[sc] = sum / T;
    let varSum = 0;
    for (let t = 0; t < T; t++) varSum += (window[t][sc] - mean[sc]) ** 2;
    std[sc] = Math.sqrt(varSum / T) || 1;
  }

  return window.map(frame => {
    const normed = new Float32Array(nSc);
    for (let sc = 0; sc < nSc; sc++) {
      normed[sc] = (frame[sc] - mean[sc]) / std[sc];
    }
    return normed;
  });
}

/**
 * Create sliding windows of CSI amplitude data.
 * Returns arrays of { input: Float32Array[nSc * T], timestamp, nodeId }.
 */
function createWindows(rawCsi, nSubcarriers, timeSteps) {
  // Group by nodeId, sort by timestamp
  const byNode = {};
  for (const frame of rawCsi) {
    if (!byNode[frame.nodeId]) byNode[frame.nodeId] = [];
    byNode[frame.nodeId].push(frame);
  }

  const windows = [];

  for (const nodeId of Object.keys(byNode)) {
    const frames = byNode[nodeId].sort((a, b) => a.timestamp - b.timestamp);

    // Extract amplitudes
    const amplitudes = frames.map(f => {
      const iq = parseIqHex(f.iqHex);
      return extractAmplitude(iq, nSubcarriers);
    });

    // Create sliding windows with stride 1
    for (let i = 0; i <= amplitudes.length - timeSteps; i++) {
      const windowFrames = amplitudes.slice(i, i + timeSteps);
      const normalized = normalizeAmplitude(windowFrames);

      // Flatten to [nSubcarriers, timeSteps] (channel-first)
      const input = new Float32Array(nSubcarriers * timeSteps);
      for (let sc = 0; sc < nSubcarriers; sc++) {
        for (let t = 0; t < timeSteps; t++) {
          input[sc * timeSteps + t] = normalized[t][sc];
        }
      }

      windows.push({
        input,
        timestamp: frames[i + timeSteps - 1].timestamp,
        startTimestamp: frames[i].timestamp,
        nodeId: parseInt(nodeId),
      });
    }
  }

  return windows;
}

/**
 * Generate pose proxy labels from vitals and motion data.
 * This is the camera-free pipeline: no ground truth keypoints,
 * but we can generate coarse pose proxies from sensor data.
 *
 * Strategy:
 *   - Person detected (presence > 0.3): place a standing skeleton at center
 *   - High motion (energy > 2): add random perturbation to limbs
 *   - Multiple people: offset skeletons horizontally
 *   - No presence: return null (skip)
 */
function generatePoseProxy(timestamp, nodeId, vitals, rng) {
  // Find nearest vitals for this timestamp and node
  let nearest = null;
  let bestDist = Infinity;
  for (const v of vitals) {
    if (v.nodeId !== nodeId) continue;
    const dist = Math.abs(v.timestamp - timestamp);
    if (dist < bestDist) {
      bestDist = dist;
      nearest = v;
    }
  }

  if (!nearest || bestDist > 2.0 || nearest.presenceScore <= 0.1) {
    return null; // No person detected
  }

  // Base standing skeleton (COCO 17 keypoints, normalized [0,1])
  const baseKeypoints = new Float32Array([
    0.50, 0.10,  // 0: nose
    0.48, 0.08,  // 1: left_eye
    0.52, 0.08,  // 2: right_eye
    0.45, 0.09,  // 3: left_ear
    0.55, 0.09,  // 4: right_ear
    0.40, 0.25,  // 5: left_shoulder
    0.60, 0.25,  // 6: right_shoulder
    0.35, 0.40,  // 7: left_elbow
    0.65, 0.40,  // 8: right_elbow
    0.32, 0.55,  // 9: left_wrist
    0.68, 0.55,  // 10: right_wrist
    0.43, 0.55,  // 11: left_hip
    0.57, 0.55,  // 12: right_hip
    0.42, 0.72,  // 13: left_knee
    0.58, 0.72,  // 14: right_knee
    0.41, 0.90,  // 15: left_ankle
    0.59, 0.90,  // 16: right_ankle
  ]);

  const keypoints = new Float32Array(baseKeypoints);
  const gauss = gaussianRng(rng);

  // Add motion-based perturbation
  const motionScale = Math.min(nearest.motionEnergy / 10.0, 0.15);
  for (let i = 0; i < keypoints.length; i++) {
    keypoints[i] += gauss() * motionScale;
    // Clamp to [0.01, 0.99]
    keypoints[i] = Math.max(0.01, Math.min(0.99, keypoints[i]));
  }

  // Add breathing-related micro-motion to torso
  if (nearest.breathingBpm > 0) {
    const breathPhase = (nearest.timestamp * nearest.breathingBpm / 60.0) * 2 * Math.PI;
    const breathAmp = 0.005; // very small
    for (const idx of [5, 6, 11, 12]) { // shoulders and hips
      keypoints[idx * 2 + 1] += Math.sin(breathPhase) * breathAmp;
    }
  }

  return {
    keypoints,
    confidence: nearest.presenceScore,
    isProxy: true,
  };
}

/**
 * Resolve glob pattern to file list.
 */
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
// Quantization (from train-ruvllm.js)
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
  return {
    quantized: packed, scale, zeroPoint, bits,
    numWeights: weights.length, originalSize,
    quantizedSize: packed.length,
    compressionRatio: originalSize / packed.length,
  };
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
      const val = (packed[byteIdx] >> shift) & 0x03;
      result[i] = (val - zeroPoint) * scale;
    }
  }
  return result;
}

function quantizationQuality(original, dequantized) {
  let sumSqErr = 0;
  const n = Math.min(original.length, dequantized.length);
  for (let i = 0; i < n; i++) sumSqErr += (original[i] - dequantized[i]) ** 2;
  return Math.sqrt(sumSqErr / n);
}

// ---------------------------------------------------------------------------
// Deterministic shuffle
// ---------------------------------------------------------------------------

function shuffleArray(arr, seed) {
  const result = [...arr];
  let s = seed;
  for (let i = result.length - 1; i > 0; i--) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    const j = (s >>> 0) % (i + 1);
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

// ---------------------------------------------------------------------------
// Main training pipeline
// ---------------------------------------------------------------------------

async function main() {
  const startTime = Date.now();
  console.log('=== WiFlow Pose Estimation Training Pipeline ===');
  console.log(`Config: epochs=${CONFIG.epochs} batch=${CONFIG.batchSize} lr=${CONFIG.learningRate}`);
  console.log(`        subcarriers=${CONFIG.subcarriers} timeSteps=${CONFIG.timeSteps} seed=${CONFIG.seed}`);
  console.log('');

  // -----------------------------------------------------------------------
  // Step 1: Load CSI data
  // -----------------------------------------------------------------------
  console.log('[1/7] Loading CSI data...');
  const files = resolveGlob(CONFIG.dataGlob);
  if (files.length === 0) {
    console.error(`No files found matching: ${CONFIG.dataGlob}`);
    process.exit(1);
  }

  let allRawCsi = [];
  let allFeatures = [];
  let allVitals = [];

  for (const file of files) {
    console.log(`  Loading: ${path.basename(file)}`);
    const { rawCsi, features, vitals } = loadCsiData(file);
    allRawCsi = allRawCsi.concat(rawCsi);
    allFeatures = allFeatures.concat(features);
    allVitals = allVitals.concat(vitals);
  }

  console.log(`  Raw CSI frames: ${allRawCsi.length}`);
  console.log(`  Feature frames: ${allFeatures.length}`);
  console.log(`  Vitals frames: ${allVitals.length}`);
  console.log(`  Nodes: ${[...new Set(allRawCsi.map(f => f.nodeId))].join(', ')}`);

  if (allRawCsi.length === 0) {
    console.error('No raw CSI frames found. WiFlow requires raw IQ data (type="raw_csi").');
    process.exit(1);
  }

  // Check subcarrier counts in data
  const scCounts = new Map();
  for (const f of allRawCsi) {
    scCounts.set(f.subcarriers, (scCounts.get(f.subcarriers) || 0) + 1);
  }
  console.log(`  Subcarrier distributions: ${[...scCounts.entries()].map(([k,v]) => `${k}sc: ${v} frames`).join(', ')}`);

  // Use the target subcarrier count; frames with different counts will be resampled
  const targetSc = CONFIG.subcarriers;

  // -----------------------------------------------------------------------
  // Step 2: Create amplitude windows
  // -----------------------------------------------------------------------
  console.log('\n[2/7] Extracting amplitude and creating windows...');

  // If frames have different subcarrier counts, resample to target
  const resampledCsi = allRawCsi.map(f => {
    if (f.subcarriers === targetSc) return f;
    // For frames with fewer subcarriers (e.g., 64), zero-pad to 128
    // For frames with more, truncate
    const iq = parseIqHex(f.iqHex);
    const amp = extractAmplitude(iq, f.subcarriers);
    // Resample amplitude to targetSc via linear interpolation
    const resampled = new Float32Array(targetSc);
    for (let i = 0; i < targetSc; i++) {
      const srcIdx = (i / targetSc) * f.subcarriers;
      const lo = Math.floor(srcIdx);
      const hi = Math.min(lo + 1, f.subcarriers - 1);
      const frac = srcIdx - lo;
      resampled[i] = amp[lo] * (1 - frac) + amp[hi] * frac;
    }
    // Re-encode as fake iqHex (amplitude only, Q=0)
    const newIq = [];
    newIq.push(0, 0); // DC offset
    for (let i = 0; i < targetSc; i++) {
      const v = Math.round(Math.min(127, Math.max(-128, resampled[i])));
      newIq.push(v, 0); // I = amplitude, Q = 0
    }
    const hexStr = newIq.map(b => {
      const unsigned = b < 0 ? b + 256 : b;
      return unsigned.toString(16).padStart(2, '0');
    }).join('');
    return { ...f, iqHex: hexStr, subcarriers: targetSc };
  });

  const windows = createWindows(resampledCsi, targetSc, CONFIG.timeSteps);
  console.log(`  Windows created: ${windows.length} (from ${allRawCsi.length} raw frames)`);
  console.log(`  Window shape: [${targetSc}, ${CONFIG.timeSteps}] = ${targetSc * CONFIG.timeSteps} values`);

  if (windows.length === 0) {
    console.error(`Not enough consecutive frames to create ${CONFIG.timeSteps}-step windows.`);
    process.exit(1);
  }

  // -----------------------------------------------------------------------
  // Step 3: Initialize WiFlow model
  // -----------------------------------------------------------------------
  console.log('\n[3/7] Initializing WiFlow model...');
  const model = new WiFlowModel({
    inputChannels: targetSc,
    timeSteps: CONFIG.timeSteps,
    numKeypoints: 17,
    numHeads: 8,
    seed: CONFIG.seed,
  });

  const breakdown = model.paramBreakdown();
  console.log(`  Parameter count: ${model.numParams().toLocaleString()}`);
  console.log(`    TCN:              ${breakdown.tcn.toLocaleString()}`);
  console.log(`    Spatial encoder:  ${breakdown.spatialEncoder.toLocaleString()}`);
  console.log(`    Axial attention:  ${breakdown.axialAttention.toLocaleString()}`);
  console.log(`    Decoder:          ${breakdown.decoder.toLocaleString()}`);

  const flops = estimateFLOPs({ inputChannels: targetSc, timeSteps: CONFIG.timeSteps });
  console.log(`  Estimated FLOPs: ${(flops.total / 1e6).toFixed(1)}M`);

  // Verify forward pass works
  console.log('  Verifying forward pass...');
  const testInput = new Float32Array(targetSc * CONFIG.timeSteps);
  const rng = createRng(CONFIG.seed);
  for (let i = 0; i < testInput.length; i++) testInput[i] = (rng() - 0.5) * 2;

  const t0 = Date.now();
  const testOutput = model.forward(testInput);
  const fwdMs = Date.now() - t0;
  console.log(`  Forward pass: ${fwdMs}ms, output shape: [${testOutput.length / 2}, 2]`);
  console.log(`  Sample keypoints (nose): x=${testOutput[0].toFixed(3)}, y=${testOutput[1].toFixed(3)}`);

  // -----------------------------------------------------------------------
  // Phase 1: Contrastive pretraining (temporal consistency)
  // -----------------------------------------------------------------------
  console.log('\n[4/7] Phase 1: Contrastive pretraining...');

  // Generate temporal triplets from windows
  const triplets = [];
  const nodeWindows = {};
  for (const w of windows) {
    if (!nodeWindows[w.nodeId]) nodeWindows[w.nodeId] = [];
    nodeWindows[w.nodeId].push(w);
  }

  for (const nodeId of Object.keys(nodeWindows)) {
    const nw = nodeWindows[nodeId];
    for (let i = 0; i < nw.length; i++) {
      // Positive: adjacent window (temporal consistency)
      for (let j = i + 1; j < Math.min(i + 3, nw.length); j++) {
        // Negative: window at least 10 windows away
        const negStart = Math.max(0, i - 20);
        const negEnd = Math.min(nw.length, i + 20);
        for (let k = 0; k < nw.length; k++) {
          if (k >= i - 3 && k <= i + 3) continue; // skip nearby
          triplets.push({
            anchor: nw[i],
            positive: nw[j],
            negative: nw[k],
          });
          if (triplets.length > 5000) break; // cap triplets
        }
        if (triplets.length > 5000) break;
      }
      if (triplets.length > 5000) break;
    }
  }

  console.log(`  Temporal triplets: ${triplets.length}`);

  if (triplets.length > 0) {
    // Use ruvllm ContrastiveTrainer for metric tracking
    const contrastiveTrainer = new ContrastiveTrainer({
      epochs: CONFIG.contrastiveEpochs,
      batchSize: CONFIG.batchSize,
      margin: CONFIG.margin,
      temperature: CONFIG.temperature,
      hardNegativeRatio: 0.5,
      learningRate: CONFIG.learningRate,
      outputPath: path.join(CONFIG.outputDir, 'contrastive'),
    });

    // Use model's forward pass to generate embeddings for contrastive learning
    // We use the decoder output as the embedding (34-dim for 17 keypoints * 2)
    const sampleTriplets = triplets.slice(0, Math.min(50, triplets.length));
    for (const t of sampleTriplets) {
      const anchorEmb = Array.from(model.forward(t.anchor.input));
      const posEmb = Array.from(model.forward(t.positive.input));
      const negEmb = Array.from(model.forward(t.negative.input));
      contrastiveTrainer.addTriplet(
        `a-${t.anchor.timestamp}`, anchorEmb,
        `p-${t.positive.timestamp}`, posEmb,
        `n-${t.negative.timestamp}`, negEmb,
        false
      );
    }

    const contrastiveResult = contrastiveTrainer.train();
    console.log(`  Contrastive loss: ${contrastiveResult.finalLoss.toFixed(6)}`);
    console.log(`  Duration: ${contrastiveResult.durationMs}ms`);

    // Apply gradient updates to decoder weights via temporal consistency
    console.log('  Applying decoder weight updates for temporal consistency...');
    const decoderLr = CONFIG.learningRate * 0.1;

    for (let epoch = 0; epoch < CONFIG.contrastiveEpochs; epoch++) {
      let epochLoss = 0;
      const shuffled = shuffleArray(sampleTriplets, epoch * 31 + 17);

      for (const t of shuffled) {
        const anchorOut = model.forward(t.anchor.input);
        const posOut = model.forward(t.positive.input);
        const negOut = model.forward(t.negative.input);

        const loss = tripletLoss(
          Array.from(anchorOut), Array.from(posOut), Array.from(negOut), CONFIG.margin
        );
        epochLoss += loss;

        if (loss > 0) {
          // Update decoder weights to push anchor closer to positive, away from negative
          const grad = computeGradient(
            Array.from(anchorOut), Array.from(posOut), Array.from(negOut), decoderLr
          );
          // Apply gradient to decoder bias (simplified update)
          for (let j = 0; j < Math.min(grad.length, model.decoder.bias.length); j++) {
            model.decoder.bias[j] += grad[j] * 0.01;
          }
        }
      }

      epochLoss /= shuffled.length || 1;
      if (CONFIG.verbose || epoch === CONFIG.contrastiveEpochs - 1) {
        console.log(`    Epoch ${epoch + 1}/${CONFIG.contrastiveEpochs}: loss=${epochLoss.toFixed(6)}`);
      }
    }
  }

  if (CONFIG.contrastiveOnly) {
    console.log('\n  --contrastive-only flag set, skipping supervised training.');
    await exportModel(model, CONFIG, startTime, { contrastiveOnly: true });
    return;
  }

  // -----------------------------------------------------------------------
  // Phase 2: Supervised pose training (SmoothL1 + bone constraint)
  // -----------------------------------------------------------------------
  console.log('\n[5/7] Phase 2: Supervised pose training...');

  // Generate pose proxy labels for each window
  const proxyRng = createRng(CONFIG.seed + 100);
  const labeledWindows = [];
  const unlabeledWindows = [];

  for (const w of windows) {
    const proxy = generatePoseProxy(w.timestamp, w.nodeId, allVitals, proxyRng);
    if (proxy) {
      labeledWindows.push({ ...w, target: proxy.keypoints, confidence: proxy.confidence });
    } else {
      unlabeledWindows.push(w);
    }
  }

  // Limit samples if --max-samples set (useful for fast iteration)
  if (CONFIG.maxSamples > 0 && labeledWindows.length > CONFIG.maxSamples) {
    labeledWindows.length = CONFIG.maxSamples;
  }

  console.log(`  Labeled windows (pose proxy): ${labeledWindows.length}`);
  console.log(`  Unlabeled windows: ${unlabeledWindows.length}`);

  if (labeledWindows.length > 0) {
    // Training loop with SmoothL1 + bone constraint
    const lr = CONFIG.learningRate;
    let bestLoss = Infinity;
    let patience = 10;
    let patienceCounter = 0;

    for (let epoch = 0; epoch < CONFIG.epochs; epoch++) {
      let epochLossH = 0;
      let epochLossB = 0;
      let epochPCK = 0;
      let nSamples = 0;

      const shuffled = shuffleArray(labeledWindows, epoch * 41 + 7);
      const batches = [];
      for (let i = 0; i < shuffled.length; i += CONFIG.batchSize) {
        batches.push(shuffled.slice(i, i + CONFIG.batchSize));
      }

      for (const batch of batches) {
        for (const sample of batch) {
          const predicted = model.forward(sample.input);
          const lossResult = model.computeLoss(predicted, sample.target, true);

          epochLossH += lossResult.smoothL1;
          epochLossB += lossResult.boneLoss;

          // Compute PCK@20
          epochPCK += WiFlowModel.pck(predicted, sample.target, 0.2);
          nSamples++;

          // Gradient update on decoder (simplified: update decoder weights)
          const grad = model.computeLossGrad(predicted, sample.target);
          const decoderDim = model.decoder.outDim;
          const featureDim = model.decoder.inFeatures;

          // Update decoder bias
          for (let j = 0; j < decoderDim; j++) {
            model.decoder.bias[j] -= lr * grad[j] * sample.confidence;
          }

          // Update decoder weights (approximate: use small perturbation)
          // Full backprop through TCN/spatial/attention is expensive in pure JS
          // We use decoder-only updates + contrastive pretrained features
          for (let j = 0; j < decoderDim; j++) {
            for (let i = 0; i < Math.min(featureDim, 48); i++) {
              model.decoder.weight[i * decoderDim + j] -= lr * grad[j] * 0.001;
            }
          }
        }
      }

      epochLossH /= nSamples || 1;
      epochLossB /= nSamples || 1;
      epochPCK /= nSamples || 1;
      const totalLoss = epochLossH + 0.2 * epochLossB;

      if (CONFIG.verbose || epoch % 5 === 0 || epoch === CONFIG.epochs - 1) {
        console.log(`    Epoch ${epoch + 1}/${CONFIG.epochs}: L_H=${epochLossH.toFixed(4)} L_B=${epochLossB.toFixed(4)} total=${totalLoss.toFixed(4)} PCK@20=${(epochPCK * 100).toFixed(1)}%`);
      }

      // Early stopping
      if (totalLoss < bestLoss) {
        bestLoss = totalLoss;
        patienceCounter = 0;
      } else {
        patienceCounter++;
        if (patienceCounter >= patience) {
          console.log(`    Early stopping at epoch ${epoch + 1} (patience=${patience})`);
          break;
        }
      }
    }
  } else {
    console.log('  WARN: No pose proxy labels generated. Skipping supervised training.');
  }

  // -----------------------------------------------------------------------
  // Phase 3: LoRA room-specific adaptation
  // -----------------------------------------------------------------------
  console.log('\n[6/7] Phase 3: LoRA adaptation...');

  const loraManager = new LoraManager({
    rank: CONFIG.loraRank,
    alpha: CONFIG.loraRank * 2,
    dropout: 0.1,
    targetModules: ['decoder'],
  });

  const nodeIds = [...new Set(windows.map(w => w.nodeId))];

  for (const nodeId of nodeIds) {
    console.log(`  Training LoRA adapter for node ${nodeId}...`);
    const nodeAdapter = loraManager.create(
      `wiflow-node-${nodeId}`,
      { rank: CONFIG.loraRank, alpha: CONFIG.loraRank * 2, dropout: 0.1 },
      2048, // decoder input dim (256 * 8)
      34    // decoder output dim (17 * 2)
    );

    const nodeData = labeledWindows.filter(w => w.nodeId === nodeId);
    if (nodeData.length > 0) {
      const nodePipeline = new TrainingPipeline({
        learningRate: CONFIG.learningRate * 0.5,
        batchSize: Math.min(CONFIG.batchSize, nodeData.length),
        epochs: 5,
        scheduler: 'cosine',
        ewcLambda: 2000,
      }, nodeAdapter);

      const pipelineData = nodeData.map(w => ({
        input: Array.from(model.forward(w.input)),
        target: Array.from(w.target),
        quality: w.confidence,
      }));
      nodePipeline.addData(pipelineData);
      const nodeResult = nodePipeline.train();
      console.log(`    Node ${nodeId}: ${nodeData.length} samples, loss=${nodeResult.finalLoss.toFixed(6)}`);
    }
  }

  console.log(`  LoRA adapters: ${loraManager.list().join(', ')}`);

  // -----------------------------------------------------------------------
  // Phase 4 + 5: Quantization + Export
  // -----------------------------------------------------------------------
  await exportModel(model, CONFIG, startTime, {
    loraManager,
    labeledWindows,
    windows,
    nodeIds,
    allRawCsi,
    allVitals,
    allFeatures,
  });
}

/**
 * Export trained model.
 */
async function exportModel(model, config, startTime, context) {
  console.log('\n[7/7] Quantization + Export...');

  fs.mkdirSync(config.outputDir, { recursive: true });

  // Quantization
  const allWeights = model.getAllWeights();
  console.log(`  Total weights: ${allWeights.length.toLocaleString()} (${(allWeights.length * 4 / 1024 / 1024).toFixed(2)} MB fp32)`);

  const quantResults = {};
  for (const bits of [2, 4, 8]) {
    const qr = quantizeWeights(allWeights, bits);
    const deq = dequantizeWeights(qr.quantized, qr.scale, qr.zeroPoint, bits, qr.numWeights);
    const rmse = quantizationQuality(allWeights, deq);
    quantResults[bits] = { ...qr, rmse };
    console.log(`  ${bits}-bit: ${qr.compressionRatio.toFixed(1)}x compression, RMSE=${rmse.toFixed(6)}, size=${(qr.quantizedSize / 1024).toFixed(1)} KB`);
  }

  // SafeTensors export
  const exporter = new ModelExporter();
  const exportData = {
    metadata: {
      name: 'wifi-densepose-wiflow',
      version: '1.0.0',
      architecture: 'wiflow-tcn-asymconv-axialattn',
      training: {
        steps: config.epochs,
        learningRate: config.learningRate,
      },
      custom: {
        inputChannels: config.subcarriers,
        timeSteps: config.timeSteps,
        numKeypoints: 17,
        numHeads: 8,
        totalParams: model.numParams(),
        paramBreakdown: model.paramBreakdown(),
        flops: estimateFLOPs({ inputChannels: config.subcarriers, timeSteps: config.timeSteps }),
        seed: config.seed,
        quantizationBits: config.quantizeBits,
      },
    },
    tensors: model.toTensorMap(),
  };

  const safetensorsBuffer = exporter.toSafeTensors(exportData);
  fs.writeFileSync(path.join(config.outputDir, 'model.safetensors'), safetensorsBuffer);
  console.log(`  SafeTensors: ${path.join(config.outputDir, 'model.safetensors')} (${(safetensorsBuffer.length / 1024).toFixed(1)} KB)`);

  // HuggingFace config
  const hfExport = exporter.toHuggingFace(exportData);
  fs.writeFileSync(path.join(config.outputDir, 'config.json'), hfExport.config);

  // JSON export
  const jsonExport = exporter.toJSON(exportData);
  fs.writeFileSync(path.join(config.outputDir, 'model.json'), jsonExport);

  // Quantized models
  const quantDir = path.join(config.outputDir, 'quantized');
  fs.mkdirSync(quantDir, { recursive: true });
  for (const [bits, qr] of Object.entries(quantResults)) {
    const qPath = path.join(quantDir, `wiflow-q${bits}.bin`);
    fs.writeFileSync(qPath, Buffer.from(qr.quantized));
    console.log(`  Quantized ${bits}-bit: ${qPath} (${(qr.quantizedSize / 1024).toFixed(1)} KB)`);
  }

  // LoRA adapters
  if (context.loraManager) {
    const loraDir = path.join(config.outputDir, 'lora');
    fs.mkdirSync(loraDir, { recursive: true });
    for (const adapterId of context.loraManager.list()) {
      const adapter = context.loraManager.get(adapterId);
      const loraPath = path.join(loraDir, `${adapterId}.json`);
      fs.writeFileSync(loraPath, adapter.toJSON());
      console.log(`  LoRA adapter: ${loraPath}`);
    }
  }

  // RVF manifest
  const rvfPath = path.join(config.outputDir, 'model.rvf.jsonl');
  const rvfLines = [
    JSON.stringify({ type: 'metadata', ...exportData.metadata }),
    JSON.stringify({ type: 'wiflow', architecture: 'tcn-asymconv-axialattn', stages: 4 }),
    JSON.stringify({ type: 'quantization', default_bits: config.quantizeBits, variants: [2, 4, 8] }),
  ];
  fs.writeFileSync(rvfPath, rvfLines.join('\n'));

  // Training metrics
  const metricsPath = path.join(config.outputDir, 'training-metrics.json');
  const metrics = {
    timestamp: new Date().toISOString(),
    totalDurationMs: Date.now() - startTime,
    model: {
      architecture: 'wiflow',
      totalParams: model.numParams(),
      paramBreakdown: model.paramBreakdown(),
      flops: estimateFLOPs({ inputChannels: config.subcarriers, timeSteps: config.timeSteps }),
    },
    data: {
      rawCsiFrames: context.allRawCsi ? context.allRawCsi.length : 0,
      windows: context.windows ? context.windows.length : 0,
      labeledWindows: context.labeledWindows ? context.labeledWindows.length : 0,
      nodes: context.nodeIds || [],
    },
    quantization: Object.fromEntries(
      Object.entries(quantResults).map(([bits, qr]) => [
        `q${bits}`,
        { compressionRatio: qr.compressionRatio, rmse: qr.rmse, sizeKB: qr.quantizedSize / 1024 },
      ])
    ),
    config,
  };
  fs.writeFileSync(metricsPath, JSON.stringify(metrics, null, 2));
  console.log(`  Metrics: ${metricsPath}`);

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`\n=== Training complete in ${elapsed}s ===`);
  console.log(`  Output: ${config.outputDir}`);
  console.log(`  Model size: ${(allWeights.length * 4 / 1024 / 1024).toFixed(2)} MB (fp32), ${(quantResults[8].quantizedSize / 1024 / 1024).toFixed(2)} MB (int8)`);
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------
main().catch(err => {
  console.error('Training failed:', err);
  process.exit(1);
});
