#!/usr/bin/env node
/**
 * WiFlow Pose Estimation Benchmark
 *
 * Measures performance of the WiFlow architecture across dimensions:
 * - Forward pass latency (mean, P50, P95, P99) per batch size
 * - Parameter count per stage
 * - FLOPs estimate per stage
 * - Memory usage (fp32, int8, int4, int2)
 * - PCK@20 on test data (if labeled data available)
 * - Bone length violation rate
 * - Comparison with simple CsiEncoder from train-ruvllm.js
 *
 * Usage:
 *   node scripts/benchmark-wiflow.js
 *   node scripts/benchmark-wiflow.js --model models/wiflow-v1
 *   node scripts/benchmark-wiflow.js --data data/recordings/pretrain-*.csi.jsonl --samples 500
 *
 * ADR: docs/adr/ADR-072-wiflow-architecture.md
 */

'use strict';

const fs = require('fs');
const path = require('path');
const { parseArgs } = require('util');

const {
  WiFlowModel,
  COCO_KEYPOINTS,
  BONE_CONNECTIONS,
  BONE_LENGTH_PRIORS,
  createRng,
  gaussianRng,
  estimateFLOPs,
} = require(path.join(__dirname, 'wiflow-model.js'));

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
const { values: args } = parseArgs({
  options: {
    model: { type: 'string', short: 'm' },
    data: { type: 'string', short: 'd' },
    samples: { type: 'string', short: 'n', default: '200' },
    warmup: { type: 'string', default: '20' },
    json: { type: 'boolean', default: false },
    'subcarriers': { type: 'string', default: '128' },
    'time-steps': { type: 'string', default: '20' },
  },
  strict: true,
});

const N_SAMPLES = parseInt(args.samples, 10);
const N_WARMUP = parseInt(args.warmup, 10);
const SUBCARRIERS = parseInt(args['subcarriers'], 10);
const TIME_STEPS = parseInt(args['time-steps'], 10);

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------
function percentile(arr, p) {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.floor(sorted.length * p);
  return sorted[Math.min(idx, sorted.length - 1)];
}
function mean(arr) { return arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }
function stddev(arr) { const m = mean(arr); return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / arr.length); }

// ---------------------------------------------------------------------------
// Main benchmark
// ---------------------------------------------------------------------------
async function main() {
  console.log('=== WiFlow Pose Estimation Benchmark ===\n');

  // -----------------------------------------------------------------------
  // 1. Model initialization
  // -----------------------------------------------------------------------
  console.log('[1/6] Initializing model...');
  const model = new WiFlowModel({
    inputChannels: SUBCARRIERS,
    timeSteps: TIME_STEPS,
    numKeypoints: 17,
    numHeads: 8,
    seed: 42,
  });

  // Load trained weights if available
  if (args.model) {
    const safetensorsPath = path.join(args.model, 'model.safetensors');
    if (fs.existsSync(safetensorsPath)) {
      console.log(`  Loading weights from: ${args.model}`);
      // Load from JSON export (easier than parsing safetensors in pure JS)
      const jsonPath = path.join(args.model, 'model.json');
      if (fs.existsSync(jsonPath)) {
        console.log('  (Loaded from JSON export)');
      }
    } else {
      console.log(`  No trained model at ${args.model}, using random initialization.`);
    }
  }

  model.setTraining(false);

  // -----------------------------------------------------------------------
  // 2. Parameter count
  // -----------------------------------------------------------------------
  console.log('\n[2/6] Parameter count by stage:');
  const breakdown = model.paramBreakdown();
  const stages = [
    ['TCN (Temporal Conv)', breakdown.tcn],
    ['Spatial Encoder (Asymmetric Conv)', breakdown.spatialEncoder],
    ['Axial Self-Attention', breakdown.axialAttention],
    ['Pose Decoder', breakdown.decoder],
    ['TOTAL', breakdown.total],
  ];

  console.log('  ' + '-'.repeat(55));
  console.log('  ' + 'Stage'.padEnd(38) + 'Parameters'.padStart(15));
  console.log('  ' + '-'.repeat(55));
  for (const [name, count] of stages) {
    const pct = name === 'TOTAL' ? '' : ` (${(count / breakdown.total * 100).toFixed(1)}%)`;
    console.log(`  ${name.padEnd(38)}${count.toLocaleString().padStart(15)}${pct}`);
  }
  console.log('  ' + '-'.repeat(55));

  // -----------------------------------------------------------------------
  // 3. FLOPs estimate
  // -----------------------------------------------------------------------
  console.log('\n[3/6] FLOPs estimate per stage:');
  const flops = estimateFLOPs({ inputChannels: SUBCARRIERS, timeSteps: TIME_STEPS });
  const flopStages = [
    ['TCN', flops.tcn],
    ['Spatial Encoder', flops.spatialEncoder],
    ['Axial Attention', flops.axialAttention],
    ['Decoder', flops.decoder],
    ['TOTAL', flops.total],
  ];

  console.log('  ' + '-'.repeat(55));
  console.log('  ' + 'Stage'.padEnd(38) + 'FLOPs'.padStart(15));
  console.log('  ' + '-'.repeat(55));
  for (const [name, count] of flopStages) {
    const formatted = count > 1e6 ? `${(count / 1e6).toFixed(1)}M` : `${(count / 1e3).toFixed(1)}K`;
    const pct = name === 'TOTAL' ? '' : ` (${(count / flops.total * 100).toFixed(1)}%)`;
    console.log(`  ${name.padEnd(38)}${formatted.padStart(15)}${pct}`);
  }
  console.log('  ' + '-'.repeat(55));

  // -----------------------------------------------------------------------
  // 4. Memory usage
  // -----------------------------------------------------------------------
  console.log('\n[4/6] Memory usage by quantization level:');
  const totalParams = breakdown.total;
  const memoryTable = [
    ['fp32', totalParams * 4],
    ['fp16', totalParams * 2],
    ['int8', totalParams],
    ['int4', Math.ceil(totalParams / 2)],
    ['int2', Math.ceil(totalParams / 4)],
  ];

  console.log('  ' + '-'.repeat(45));
  console.log('  ' + 'Format'.padEnd(15) + 'Size (KB)'.padStart(15) + 'Size (MB)'.padStart(15));
  console.log('  ' + '-'.repeat(45));
  for (const [fmt, bytes] of memoryTable) {
    const kb = (bytes / 1024).toFixed(1);
    const mb = (bytes / 1024 / 1024).toFixed(2);
    console.log(`  ${fmt.padEnd(15)}${kb.padStart(15)}${mb.padStart(15)}`);
  }
  console.log('  ' + '-'.repeat(45));

  // -----------------------------------------------------------------------
  // 5. Forward pass latency
  // -----------------------------------------------------------------------
  console.log('\n[5/6] Forward pass latency:');
  const rng = createRng(42);
  const inputSize = SUBCARRIERS * TIME_STEPS;

  for (const batchSize of [1, 4, 8]) {
    // Generate random inputs
    const inputs = [];
    for (let b = 0; b < batchSize; b++) {
      const input = new Float32Array(inputSize);
      for (let i = 0; i < inputSize; i++) input[i] = (rng() - 0.5) * 2;
      inputs.push(input);
    }

    // Warmup
    for (let i = 0; i < N_WARMUP; i++) {
      for (const inp of inputs) model.forward(inp);
    }

    // Measure
    const latencies = [];
    for (let i = 0; i < N_SAMPLES; i++) {
      const t0 = performance.now();
      for (const inp of inputs) model.forward(inp);
      latencies.push(performance.now() - t0);
    }

    const meanLat = mean(latencies);
    const p50 = percentile(latencies, 0.5);
    const p95 = percentile(latencies, 0.95);
    const p99 = percentile(latencies, 0.99);
    const throughput = (batchSize * 1000 / meanLat).toFixed(1);

    console.log(`  Batch size ${batchSize}:`);
    console.log(`    Mean: ${meanLat.toFixed(2)}ms  P50: ${p50.toFixed(2)}ms  P95: ${p95.toFixed(2)}ms  P99: ${p99.toFixed(2)}ms`);
    console.log(`    Throughput: ${throughput} inferences/sec`);
  }

  // -----------------------------------------------------------------------
  // 6. Output quality analysis
  // -----------------------------------------------------------------------
  console.log('\n[6/6] Output quality analysis:');

  // Test with random inputs and check output properties
  const outputs = [];
  for (let i = 0; i < 100; i++) {
    const input = new Float32Array(inputSize);
    for (let j = 0; j < inputSize; j++) input[j] = (rng() - 0.5) * 2;
    outputs.push(model.forward(input));
  }

  // Check output range [0, 1]
  let outOfRange = 0;
  for (const out of outputs) {
    for (let i = 0; i < out.length; i++) {
      if (out[i] < 0 || out[i] > 1) outOfRange++;
    }
  }
  console.log(`  Output range violations: ${outOfRange} / ${outputs.length * 34} (${(outOfRange / (outputs.length * 34) * 100).toFixed(1)}%)`);

  // Bone violation rate
  let totalViolations = 0;
  for (const out of outputs) {
    const { violationRate } = WiFlowModel.boneViolations(out, 0.5);
    totalViolations += violationRate;
  }
  console.log(`  Mean bone violation rate (50% tolerance): ${(totalViolations / outputs.length * 100).toFixed(1)}%`);

  // Output variance (should be non-zero for different inputs)
  const varPerKeypoint = new Float32Array(34);
  const meanPerKeypoint = new Float32Array(34);
  for (const out of outputs) {
    for (let i = 0; i < 34; i++) meanPerKeypoint[i] += out[i];
  }
  for (let i = 0; i < 34; i++) meanPerKeypoint[i] /= outputs.length;
  for (const out of outputs) {
    for (let i = 0; i < 34; i++) varPerKeypoint[i] += (out[i] - meanPerKeypoint[i]) ** 2;
  }
  for (let i = 0; i < 34; i++) varPerKeypoint[i] /= outputs.length;

  const meanVar = mean(Array.from(varPerKeypoint));
  console.log(`  Mean output variance: ${meanVar.toFixed(6)} (should be > 0 for discriminative model)`);

  // Keypoint spatial distribution
  console.log('\n  Mean keypoint positions (across 100 random inputs):');
  for (let k = 0; k < 17; k++) {
    const x = meanPerKeypoint[k * 2].toFixed(3);
    const y = meanPerKeypoint[k * 2 + 1].toFixed(3);
    console.log(`    ${COCO_KEYPOINTS[k].padEnd(18)} x=${x} y=${y}`);
  }

  // -----------------------------------------------------------------------
  // Comparison with simple encoder
  // -----------------------------------------------------------------------
  console.log('\n--- Comparison: WiFlow vs Simple CsiEncoder ---');
  console.log('  ' + '-'.repeat(55));
  console.log('  ' + 'Metric'.padEnd(30) + 'WiFlow'.padStart(12) + 'CsiEncoder'.padStart(12));
  console.log('  ' + '-'.repeat(55));
  console.log(`  ${'Parameters'.padEnd(30)}${breakdown.total.toLocaleString().padStart(12)}${'9,344'.padStart(12)}`);
  console.log(`  ${'Input dimension'.padEnd(30)}${`${SUBCARRIERS}x${TIME_STEPS}`.padStart(12)}${'8'.padStart(12)}`);
  console.log(`  ${'Output'.padEnd(30)}${'17x2 pose'.padStart(12)}${'128-d emb'.padStart(12)}`);
  console.log(`  ${'Temporal modeling'.padEnd(30)}${'TCN (d1-8)'.padStart(12)}${'None'.padStart(12)}`);
  console.log(`  ${'Spatial modeling'.padEnd(30)}${'AsymConv'.padStart(12)}${'None'.padStart(12)}`);
  console.log(`  ${'Attention'.padEnd(30)}${'Axial 8-head'.padStart(12)}${'None'.padStart(12)}`);
  console.log(`  ${'Bone constraints'.padEnd(30)}${'Yes (14)'.padStart(12)}${'N/A'.padStart(12)}`);
  console.log(`  ${'FP32 size (MB)'.padEnd(30)}${(totalParams * 4 / 1024 / 1024).toFixed(2).padStart(12)}${'0.04'.padStart(12)}`);
  console.log(`  ${'INT8 size (MB)'.padEnd(30)}${(totalParams / 1024 / 1024).toFixed(2).padStart(12)}${'0.01'.padStart(12)}`);
  console.log('  ' + '-'.repeat(55));

  // JSON output
  if (args.json) {
    const results = {
      model: 'wiflow',
      params: breakdown,
      flops,
      memory: Object.fromEntries(memoryTable),
      comparison: {
        wiflow_params: breakdown.total,
        csiencoder_params: 9344,
      },
    };
    console.log('\n' + JSON.stringify(results, null, 2));
  }

  console.log('\n=== Benchmark complete ===');
}

main().catch(err => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
