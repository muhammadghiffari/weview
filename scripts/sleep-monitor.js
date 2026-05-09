#!/usr/bin/env node
/**
 * ADR-077: Sleep Quality Monitor — CSI-based sleep staging
 *
 * Classifies sleep stages from breathing rate + heart rate + motion energy
 * using 5-minute sliding windows. Produces a hypnogram and summary stats.
 *
 * DISCLAIMER: This is a consumer-grade informational tool, NOT a medical device.
 * Do not use for clinical diagnosis. Consult a physician for sleep concerns.
 *
 * Usage:
 *   node scripts/sleep-monitor.js --replay data/recordings/overnight-1775217646.csi.jsonl
 *   node scripts/sleep-monitor.js --port 5006
 *   node scripts/sleep-monitor.js --replay FILE --json
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
const WINDOW_SEC  = parseInt(args.window, 10); // default 5 min = 300s

// ---------------------------------------------------------------------------
// ADR-018 packet constants
// ---------------------------------------------------------------------------
const VITALS_MAGIC = 0xC5110002;
const FUSED_MAGIC  = 0xC5110004;

// ---------------------------------------------------------------------------
// Sleep stage thresholds
// ---------------------------------------------------------------------------
const STAGES = { AWAKE: 'Awake', LIGHT: 'Light', REM: 'REM', DEEP: 'Deep' };
const STAGE_CHARS = { Awake: 'W', Light: 'L', REM: 'R', Deep: 'D' };
const STAGE_BARS  = { Awake: '\u2581', Light: '\u2583', REM: '\u2585', Deep: '\u2588' };

// ---------------------------------------------------------------------------
// Vitals buffer
// ---------------------------------------------------------------------------
class VitalsBuffer {
  constructor(maxAgeSec) {
    this.maxAgeSec = maxAgeSec;
    this.samples = []; // { timestamp, br, hr, motion }
  }

  push(timestamp, br, hr, motion) {
    this.samples.push({ timestamp, br, hr, motion });
    this._prune(timestamp);
  }

  _prune(now) {
    const cutoff = now - this.maxAgeSec;
    while (this.samples.length > 0 && this.samples[0].timestamp < cutoff) {
      this.samples.shift();
    }
  }

  get length() { return this.samples.length; }

  stats() {
    const n = this.samples.length;
    if (n < 3) return null;

    let brSum = 0, hrSum = 0, motionSum = 0;
    for (const s of this.samples) {
      brSum += s.br;
      hrSum += s.hr;
      motionSum += s.motion;
    }
    const brMean = brSum / n;
    const hrMean = hrSum / n;
    const motionMean = motionSum / n;

    // BR variance
    let brVar = 0;
    for (const s of this.samples) {
      brVar += (s.br - brMean) ** 2;
    }
    brVar /= (n - 1);

    // HR coefficient of variation
    let hrVar = 0;
    for (const s of this.samples) {
      hrVar += (s.hr - hrMean) ** 2;
    }
    hrVar /= (n - 1);
    const hrCV = hrMean > 0 ? Math.sqrt(hrVar) / hrMean : 0;

    return { brMean, brVar, hrMean, hrCV, motionMean, n };
  }

  classify() {
    const s = this.stats();
    if (!s) return null;

    // High motion => Awake
    if (s.motionMean > 5.0 || s.brMean > 25 || s.brMean < 3) {
      return { stage: STAGES.AWAKE, ...s };
    }

    // REM: irregular breathing (high variance), HR elevated
    if (s.brVar > 8.0 && s.brMean >= 15 && s.brMean <= 25) {
      return { stage: STAGES.REM, ...s };
    }

    // Deep: low BR, very regular
    if (s.brMean >= 6 && s.brMean <= 14 && s.brVar < 2.0 && s.motionMean < 2.0) {
      return { stage: STAGES.DEEP, ...s };
    }

    // Light: moderate BR and variance
    if (s.brMean >= 10 && s.brMean <= 20 && s.motionMean < 4.0) {
      return { stage: STAGES.LIGHT, ...s };
    }

    // Default to Awake
    return { stage: STAGES.AWAKE, ...s };
  }
}

// ---------------------------------------------------------------------------
// Sleep session tracker
// ---------------------------------------------------------------------------
class SleepSession {
  constructor(windowSec) {
    this.windowSec = windowSec;
    this.buffers = new Map(); // nodeId -> VitalsBuffer
    this.hypnogram = [];      // { timestamp, stage, stats }
    this.startTime = null;
    this.lastTime = null;
  }

  ingest(timestamp, nodeId, br, hr, motion) {
    if (!this.startTime) this.startTime = timestamp;
    this.lastTime = timestamp;

    if (!this.buffers.has(nodeId)) {
      this.buffers.set(nodeId, new VitalsBuffer(this.windowSec));
    }
    this.buffers.get(nodeId).push(timestamp, br, hr, motion);
  }

  analyze(timestamp) {
    // Merge stats from all nodes (take the one with most samples)
    let bestResult = null;
    let bestCount = 0;
    for (const [, buf] of this.buffers) {
      const result = buf.classify();
      if (result && result.n > bestCount) {
        bestResult = result;
        bestCount = result.n;
      }
    }

    if (bestResult) {
      this.hypnogram.push({ timestamp, ...bestResult });
    }
    return bestResult;
  }

  summary() {
    if (this.hypnogram.length === 0) return null;

    const counts = { Awake: 0, Light: 0, REM: 0, Deep: 0 };
    for (const entry of this.hypnogram) {
      counts[entry.stage]++;
    }
    const total = this.hypnogram.length;
    const sleepEntries = total - counts.Awake;
    const durationSec = this.lastTime - this.startTime;
    const durationMin = durationSec / 60;

    return {
      totalRecordedMin: durationMin,
      totalSleepMin: (sleepEntries / total) * durationMin,
      sleepEfficiency: total > 0 ? ((sleepEntries / total) * 100) : 0,
      stageMinutes: {
        Awake: (counts.Awake / total) * durationMin,
        Light: (counts.Light / total) * durationMin,
        REM: (counts.REM / total) * durationMin,
        Deep: (counts.Deep / total) * durationMin,
      },
      stagePercent: {
        Awake: total > 0 ? ((counts.Awake / total) * 100) : 0,
        Light: total > 0 ? ((counts.Light / total) * 100) : 0,
        REM: total > 0 ? ((counts.REM / total) * 100) : 0,
        Deep: total > 0 ? ((counts.Deep / total) * 100) : 0,
      },
      entries: total,
    };
  }

  renderHypnogram(width) {
    if (this.hypnogram.length === 0) return 'No data yet.';

    const w = width || 60;
    const step = Math.max(1, Math.floor(this.hypnogram.length / w));
    let bars = '';
    let labels = '';
    for (let i = 0; i < this.hypnogram.length; i += step) {
      const entry = this.hypnogram[i];
      bars += STAGE_BARS[entry.stage] || ' ';
      labels += STAGE_CHARS[entry.stage] || '?';
    }

    const lines = [];
    lines.push('Hypnogram:');
    lines.push(`  ${bars}`);
    lines.push(`  ${labels}`);
    lines.push('  W=Awake L=Light R=REM D=Deep');
    return lines.join('\n');
  }
}

// ---------------------------------------------------------------------------
// Packet parsing (from JSONL or UDP)
// ---------------------------------------------------------------------------
function parseVitalsJsonl(record) {
  if (record.type !== 'vitals') return null;
  return {
    timestamp: record.timestamp,
    nodeId: record.node_id,
    br: record.breathing_bpm || 0,
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
    br: buf.readUInt16LE(6) / 100,
    hr: buf.readUInt32LE(8) / 10000,
    motion: buf.readFloatLE(16),
  };
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------
function renderLive(session, latest) {
  const lines = [];
  lines.push('=== SLEEP QUALITY MONITOR (ADR-077) ===');
  lines.push('DISCLAIMER: Informational only. Not a medical device.');
  lines.push('');

  if (latest) {
    lines.push(`Current stage: ${latest.stage}`);
    lines.push(`  BR: ${latest.brMean.toFixed(1)} BPM (var ${latest.brVar.toFixed(2)})`);
    lines.push(`  HR: ${latest.hrMean.toFixed(1)} BPM (CV ${(latest.hrCV * 100).toFixed(1)}%)`);
    lines.push(`  Motion: ${latest.motionMean.toFixed(2)}`);
    lines.push(`  Window: ${latest.n} samples`);
  } else {
    lines.push('Collecting data...');
  }

  lines.push('');
  lines.push(session.renderHypnogram(60));

  const sum = session.summary();
  if (sum) {
    lines.push('');
    lines.push(`Duration: ${sum.totalRecordedMin.toFixed(1)} min | Sleep: ${sum.totalSleepMin.toFixed(1)} min | Efficiency: ${sum.sleepEfficiency.toFixed(1)}%`);
    lines.push(`  Deep: ${sum.stagePercent.Deep.toFixed(1)}% | Light: ${sum.stagePercent.Light.toFixed(1)}% | REM: ${sum.stagePercent.REM.toFixed(1)}% | Awake: ${sum.stagePercent.Awake.toFixed(1)}%`);
  }

  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Replay mode
// ---------------------------------------------------------------------------
async function startReplay(filePath) {
  if (!fs.existsSync(filePath)) {
    console.error(`File not found: ${filePath}`);
    process.exit(1);
  }

  const session = new SleepSession(WINDOW_SEC);
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

    session.ingest(v.timestamp, v.nodeId, v.br, v.hr, v.motion);
    vitalsCount++;

    // Analyze every INTERVAL_MS worth of time
    const tsMs = v.timestamp * 1000;
    if (lastAnalysisTs === 0) lastAnalysisTs = tsMs;

    if (tsMs - lastAnalysisTs >= INTERVAL_MS) {
      const result = session.analyze(v.timestamp);

      if (JSON_OUTPUT) {
        if (result) {
          console.log(JSON.stringify({
            timestamp: v.timestamp,
            stage: result.stage,
            br_mean: +result.brMean.toFixed(2),
            br_var: +result.brVar.toFixed(3),
            hr_mean: +result.hrMean.toFixed(2),
            hr_cv: +result.hrCV.toFixed(4),
            motion_mean: +result.motionMean.toFixed(3),
          }));
        }
      } else if (result) {
        const ts = new Date(v.timestamp * 1000).toISOString().slice(11, 19);
        console.log(`[${ts}] ${result.stage.padEnd(5)} | BR ${result.brMean.toFixed(1)} (var ${result.brVar.toFixed(2)}) | HR ${result.hrMean.toFixed(1)} | Motion ${result.motionMean.toFixed(2)}`);
      }

      lastAnalysisTs = tsMs;
    }
  }

  // Final summary
  if (!JSON_OUTPUT) {
    console.log('\n' + '='.repeat(60));
    console.log('SLEEP SESSION SUMMARY');
    console.log('='.repeat(60));
    console.log(session.renderHypnogram(60));

    const sum = session.summary();
    if (sum) {
      console.log('');
      console.log(`Total recorded: ${sum.totalRecordedMin.toFixed(1)} min`);
      console.log(`Total sleep:    ${sum.totalSleepMin.toFixed(1)} min`);
      console.log(`Efficiency:     ${sum.sleepEfficiency.toFixed(1)}%`);
      console.log(`Entries:        ${sum.entries} analysis windows`);
      console.log('');
      console.log('Stage breakdown:');
      for (const stage of ['Deep', 'Light', 'REM', 'Awake']) {
        const pct = sum.stagePercent[stage].toFixed(1);
        const min = sum.stageMinutes[stage].toFixed(1);
        const bar = '\u2588'.repeat(Math.round(sum.stagePercent[stage] / 2));
        console.log(`  ${stage.padEnd(6)} ${bar.padEnd(50)} ${pct}% (${min} min)`);
      }
    }

    console.log(`\nProcessed ${vitalsCount} vitals packets`);
  } else {
    const sum = session.summary();
    if (sum) {
      console.log(JSON.stringify({ type: 'summary', ...sum }));
    }
  }
}

// ---------------------------------------------------------------------------
// Live UDP mode
// ---------------------------------------------------------------------------
function startLive() {
  const session = new SleepSession(WINDOW_SEC);
  const server = dgram.createSocket('udp4');

  server.on('message', (buf) => {
    const v = parseVitalsUdp(buf);
    if (v) {
      session.ingest(v.timestamp, v.nodeId, v.br, v.hr, v.motion);
    }
  });

  setInterval(() => {
    const result = session.analyze(Date.now() / 1000);

    if (JSON_OUTPUT) {
      if (result) {
        console.log(JSON.stringify({
          timestamp: Date.now() / 1000,
          stage: result.stage,
          br_mean: +result.brMean.toFixed(2),
          br_var: +result.brVar.toFixed(3),
          hr_mean: +result.hrMean.toFixed(2),
          motion_mean: +result.motionMean.toFixed(3),
        }));
      }
    } else {
      process.stdout.write('\x1B[2J\x1B[H');
      process.stdout.write(renderLive(session, result) + '\n');
    }
  }, INTERVAL_MS);

  server.bind(PORT, () => {
    if (!JSON_OUTPUT) {
      console.log(`Sleep Monitor listening on UDP :${PORT} (window ${WINDOW_SEC}s)`);
      console.log('DISCLAIMER: Informational only. Not a medical device.\n');
    }
  });

  process.on('SIGINT', () => {
    if (!JSON_OUTPUT) {
      console.log('\n' + '='.repeat(60));
      const sum = session.summary();
      if (sum) {
        console.log(`Session: ${sum.totalRecordedMin.toFixed(1)} min | Sleep: ${sum.totalSleepMin.toFixed(1)} min | Efficiency: ${sum.sleepEfficiency.toFixed(1)}%`);
      }
    }
    server.close();
    process.exit(0);
  });
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------
if (args.replay) {
  startReplay(args.replay);
} else {
  startLive();
}
