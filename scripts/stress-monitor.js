#!/usr/bin/env node
/**
 * ADR-077: Stress Monitor — HRV-based emotional state detection
 *
 * Computes RMSSD and LF/HF ratio from heart rate time series to produce
 * a stress score (0-100). Uses 5-minute sliding windows with FFT analysis.
 *
 * DISCLAIMER: This is an informational wellness tool, NOT a medical device.
 * Do not use for clinical diagnosis.
 *
 * Usage:
 *   node scripts/stress-monitor.js --replay data/recordings/overnight-1775217646.csi.jsonl
 *   node scripts/stress-monitor.js --port 5006
 *   node scripts/stress-monitor.js --replay FILE --json
 *
 * ADR: docs/adr/ADR-077-novel-rf-sensing-applications.md
 */

'use strict';

const dgram = require('dgram');
const fs = require('fs');
const readline = require('readline');
const { parseArgs } = require('util');

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
const { values: args } = parseArgs({
  options: {
    port:     { type: 'string', short: 'p', default: '5006' },
    replay:   { type: 'string', short: 'r' },
    json:     { type: 'boolean', default: false },
    interval: { type: 'string', short: 'i', default: '5000' },
    window:   { type: 'string', short: 'w', default: '300' },
  },
  strict: true,
});

const PORT        = parseInt(args.port, 10);
const JSON_OUTPUT = args.json;
const INTERVAL_MS = parseInt(args.interval, 10);
const WINDOW_SEC  = parseInt(args.window, 10);

// ---------------------------------------------------------------------------
// ADR-018 packet constants
// ---------------------------------------------------------------------------
const VITALS_MAGIC = 0xC5110002;
const FUSED_MAGIC  = 0xC5110004;

// ---------------------------------------------------------------------------
// Simple FFT (radix-2 DIT, power-of-2 only)
// ---------------------------------------------------------------------------
function fft(re, im) {
  const n = re.length;
  if (n <= 1) return;

  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  // Cooley-Tukey
  for (let len = 2; len <= n; len *= 2) {
    const half = len / 2;
    const angle = -2 * Math.PI / len;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let j = 0; j < half; j++) {
        const tRe = curRe * re[i + j + half] - curIm * im[i + j + half];
        const tIm = curRe * im[i + j + half] + curIm * re[i + j + half];

        re[i + j + half] = re[i + j] - tRe;
        im[i + j + half] = im[i + j] - tIm;
        re[i + j] += tRe;
        im[i + j] += tIm;

        const newCurRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newCurRe;
      }
    }
  }
}

function nextPow2(n) {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

// ---------------------------------------------------------------------------
// HRV analysis engine
// ---------------------------------------------------------------------------
class HRVAnalyzer {
  constructor(windowSec) {
    this.windowSec = windowSec;
    this.hrSamples = []; // { timestamp, hr }
    this.history = [];   // { timestamp, rmssd, lfhf, stress, motionMean }
    this.maxHistory = 500;
  }

  push(timestamp, hr, motion) {
    this.hrSamples.push({ timestamp, hr, motion: motion || 0 });
    // Prune old samples
    const cutoff = timestamp - this.windowSec;
    while (this.hrSamples.length > 0 && this.hrSamples[0].timestamp < cutoff) {
      this.hrSamples.shift();
    }
  }

  analyze(timestamp) {
    const samples = this.hrSamples;
    const n = samples.length;
    if (n < 10) return null;

    // Compute RR intervals (from HR in BPM -> interval in ms)
    // HR = 60000 / RR_ms, so RR_ms = 60000 / HR
    const rr = [];
    for (const s of samples) {
      if (s.hr > 20 && s.hr < 200) {
        rr.push(60000 / s.hr);
      }
    }
    if (rr.length < 5) return null;

    // RMSSD: root mean square of successive differences
    let sumSqDiff = 0;
    let diffCount = 0;
    for (let i = 1; i < rr.length; i++) {
      const diff = rr[i] - rr[i - 1];
      sumSqDiff += diff * diff;
      diffCount++;
    }
    const rmssd = diffCount > 0 ? Math.sqrt(sumSqDiff / diffCount) : 0;

    // FFT-based LF/HF ratio
    // Resample RR series to uniform ~1 Hz for FFT
    const fs = 1.0; // 1 Hz sampling (approximate, given ~1 Hz vitals)
    const nfft = nextPow2(Math.max(rr.length, 64));
    const re = new Float64Array(nfft);
    const im = new Float64Array(nfft);

    // De-mean and window (Hann)
    const mean = rr.reduce((a, b) => a + b, 0) / rr.length;
    for (let i = 0; i < rr.length; i++) {
      const hann = 0.5 * (1 - Math.cos(2 * Math.PI * i / (rr.length - 1)));
      re[i] = (rr[i] - mean) * hann;
    }

    fft(re, im);

    // Compute power spectral density
    const freqRes = fs / nfft;
    let lfPower = 0, hfPower = 0;
    for (let k = 0; k < nfft / 2; k++) {
      const freq = k * freqRes;
      const power = re[k] * re[k] + im[k] * im[k];

      if (freq >= 0.04 && freq <= 0.15) lfPower += power;
      if (freq >= 0.15 && freq <= 0.40) hfPower += power;
    }

    const lfhf = hfPower > 0.001 ? lfPower / hfPower : 0;

    // Stress score (0-100)
    // High RMSSD = relaxed (low stress), high LF/HF = stressed
    const maxRmssd = 100; // typical max RMSSD for WiFi-derived HR
    const rmssdNorm = Math.min(rmssd / maxRmssd, 1.0);
    const lfhfNorm = Math.min(lfhf / 4.0, 1.0);
    const stress = Math.round(50 * (1 - rmssdNorm) + 50 * lfhfNorm);

    // Average motion in window
    let motionSum = 0;
    for (const s of samples) motionSum += s.motion;
    const motionMean = motionSum / n;

    // HR stats
    const hrValues = samples.map(s => s.hr).filter(h => h > 20 && h < 200);
    const hrMean = hrValues.reduce((a, b) => a + b, 0) / hrValues.length;

    const result = {
      timestamp,
      rmssd: +rmssd.toFixed(2),
      lfPower: +lfPower.toFixed(2),
      hfPower: +hfPower.toFixed(2),
      lfhf: +lfhf.toFixed(3),
      stress,
      hrMean: +hrMean.toFixed(1),
      motionMean: +motionMean.toFixed(3),
      samples: n,
    };

    this.history.push(result);
    if (this.history.length > this.maxHistory) this.history.shift();

    return result;
  }

  stressLabel(score) {
    if (score < 20) return 'Very relaxed';
    if (score < 40) return 'Relaxed';
    if (score < 60) return 'Moderate';
    if (score < 80) return 'Stressed';
    return 'Very stressed';
  }

  renderTrend(width) {
    const w = width || 50;
    if (this.history.length === 0) return 'No data yet.';

    const step = Math.max(1, Math.floor(this.history.length / w));
    const bars = ['\u2581', '\u2582', '\u2583', '\u2584', '\u2585', '\u2586', '\u2587', '\u2588'];

    let line = '';
    for (let i = 0; i < this.history.length; i += step) {
      const s = this.history[i].stress;
      const idx = Math.min(7, Math.floor(s / 12.5));
      line += bars[idx];
    }
    return `Stress trend: ${line}  (low)\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588(high)`;
  }
}

// ---------------------------------------------------------------------------
// Packet parsing
// ---------------------------------------------------------------------------
function parseVitalsJsonl(record) {
  if (record.type !== 'vitals') return null;
  return {
    timestamp: record.timestamp,
    nodeId: record.node_id,
    hr: record.heartrate_bpm || 0,
    motion: record.motion_energy || 0,
  };
}

function parseVitalsUdp(buf) {
  if (buf.length < 32) return null;
  const magic = buf.readUInt32LE(0);
  if (magic !== VITALS_MAGIC && magic !== FUSED_MAGIC) return null;
  return {
    timestamp: Date.now() / 1000,
    nodeId: buf.readUInt8(4),
    hr: buf.readUInt32LE(8) / 10000,
    motion: buf.readFloatLE(16),
  };
}

// ---------------------------------------------------------------------------
// Replay mode
// ---------------------------------------------------------------------------
async function startReplay(filePath) {
  if (!fs.existsSync(filePath)) {
    console.error(`File not found: ${filePath}`);
    process.exit(1);
  }

  const analyzer = new HRVAnalyzer(WINDOW_SEC);
  const rl = readline.createInterface({
    input: fs.createReadStream(filePath),
    crlfDelay: Infinity,
  });

  let vitalsCount = 0;
  let lastAnalysisTs = 0;

  for await (const line of rl) {
    if (!line.trim()) continue;
    let record;
    try { record = JSON.parse(line); } catch { continue; }

    const v = parseVitalsJsonl(record);
    if (!v) continue;

    analyzer.push(v.timestamp, v.hr, v.motion);
    vitalsCount++;

    const tsMs = v.timestamp * 1000;
    if (lastAnalysisTs === 0) lastAnalysisTs = tsMs;

    if (tsMs - lastAnalysisTs >= INTERVAL_MS) {
      const result = analyzer.analyze(v.timestamp);

      if (result) {
        if (JSON_OUTPUT) {
          console.log(JSON.stringify(result));
        } else {
          const ts = new Date(v.timestamp * 1000).toISOString().slice(11, 19);
          const label = analyzer.stressLabel(result.stress);
          const bar = '\u2588'.repeat(Math.round(result.stress / 5));
          console.log(`[${ts}] Stress: ${String(result.stress).padStart(3)}/100 ${bar.padEnd(20)} ${label} | RMSSD ${result.rmssd} | LF/HF ${result.lfhf} | HR ${result.hrMean} | Motion ${result.motionMean}`);
        }
      }

      lastAnalysisTs = tsMs;
    }
  }

  // Final summary
  if (!JSON_OUTPUT) {
    console.log('\n' + '='.repeat(70));
    console.log('STRESS ANALYSIS SUMMARY');
    console.log('DISCLAIMER: Informational only. Not a medical device.');
    console.log('='.repeat(70));

    if (analyzer.history.length > 0) {
      const scores = analyzer.history.map(h => h.stress);
      const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
      const min = Math.min(...scores);
      const max = Math.max(...scores);

      console.log(`Average stress: ${avg.toFixed(0)}/100 (${analyzer.stressLabel(avg)})`);
      console.log(`Range:          ${min} - ${max}`);
      console.log(`Windows:        ${analyzer.history.length}`);
      console.log('');
      console.log(analyzer.renderTrend(60));

      // Activity correlation
      const highMotion = analyzer.history.filter(h => h.motionMean > 3.0);
      const lowMotion = analyzer.history.filter(h => h.motionMean < 1.0);
      if (highMotion.length > 0 && lowMotion.length > 0) {
        const avgHigh = highMotion.reduce((s, h) => s + h.stress, 0) / highMotion.length;
        const avgLow = lowMotion.reduce((s, h) => s + h.stress, 0) / lowMotion.length;
        console.log('');
        console.log(`Activity correlation:`);
        console.log(`  Active periods (motion > 3):  avg stress ${avgHigh.toFixed(0)} (${highMotion.length} windows)`);
        console.log(`  Rest periods (motion < 1):    avg stress ${avgLow.toFixed(0)} (${lowMotion.length} windows)`);
      }
    }

    console.log(`\nProcessed ${vitalsCount} vitals packets`);
  } else {
    if (analyzer.history.length > 0) {
      const scores = analyzer.history.map(h => h.stress);
      console.log(JSON.stringify({
        type: 'summary',
        avg_stress: +(scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1),
        min_stress: Math.min(...scores),
        max_stress: Math.max(...scores),
        windows: analyzer.history.length,
      }));
    }
  }
}

// ---------------------------------------------------------------------------
// Live UDP mode
// ---------------------------------------------------------------------------
function startLive() {
  const analyzer = new HRVAnalyzer(WINDOW_SEC);
  const server = dgram.createSocket('udp4');

  server.on('message', (buf) => {
    const v = parseVitalsUdp(buf);
    if (v) {
      analyzer.push(v.timestamp, v.hr, v.motion);
    }
  });

  setInterval(() => {
    const result = analyzer.analyze(Date.now() / 1000);

    if (JSON_OUTPUT) {
      if (result) console.log(JSON.stringify(result));
    } else {
      process.stdout.write('\x1B[2J\x1B[H');
      console.log('=== STRESS MONITOR (ADR-077) ===');
      console.log('DISCLAIMER: Informational only. Not a medical device.');
      console.log('');

      if (result) {
        const label = analyzer.stressLabel(result.stress);
        const bar = '\u2588'.repeat(Math.round(result.stress / 5));
        console.log(`Stress: ${result.stress}/100 ${bar} ${label}`);
        console.log(`RMSSD: ${result.rmssd} ms | LF/HF: ${result.lfhf}`);
        console.log(`HR: ${result.hrMean} BPM | Motion: ${result.motionMean}`);
        console.log(`Window: ${result.samples} samples`);
        console.log('');
        console.log(analyzer.renderTrend(50));
      } else {
        console.log('Collecting data...');
      }
    }
  }, INTERVAL_MS);

  server.bind(PORT, () => {
    if (!JSON_OUTPUT) {
      console.log(`Stress Monitor listening on UDP :${PORT} (window ${WINDOW_SEC}s)`);
    }
  });

  process.on('SIGINT', () => { server.close(); process.exit(0); });
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------
if (args.replay) {
  startReplay(args.replay);
} else {
  startLive();
}
