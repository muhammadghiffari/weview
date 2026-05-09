#!/usr/bin/env node
/**
 * ADR-077: Breathing Disorder Screening — Apnea/Hypopnea Detection
 *
 * Monitors breathing rate time series for respiratory events (pauses > 10s)
 * and computes AHI (Apnea-Hypopnea Index) for pre-screening.
 *
 * DISCLAIMER: This is a pre-screening tool, NOT a clinical diagnostic device.
 * Consult a physician and pursue polysomnography for diagnosis.
 *
 * Usage:
 *   node scripts/apnea-detector.js --replay data/recordings/overnight-1775217646.csi.jsonl
 *   node scripts/apnea-detector.js --port 5006
 *   node scripts/apnea-detector.js --replay FILE --json
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
    'apnea-threshold':   { type: 'string', default: '3.0' },
    'hypopnea-drop':     { type: 'string', default: '0.5' },
    'min-duration':      { type: 'string', default: '10' },
  },
  strict: true,
});

const PORT             = parseInt(args.port, 10);
const JSON_OUTPUT      = args.json;
const INTERVAL_MS      = parseInt(args.interval, 10);
const APNEA_THRESH     = parseFloat(args['apnea-threshold']);   // BR below this = apnea
const HYPOPNEA_DROP    = parseFloat(args['hypopnea-drop']);     // 50% drop from baseline
const MIN_DURATION_SEC = parseInt(args['min-duration'], 10);    // min event duration

// ---------------------------------------------------------------------------
// ADR-018 packet constants
// ---------------------------------------------------------------------------
const VITALS_MAGIC = 0xC5110002;
const FUSED_MAGIC  = 0xC5110004;

// ---------------------------------------------------------------------------
// Apnea detector engine
// ---------------------------------------------------------------------------
class ApneaDetector {
  constructor(opts) {
    this.apneaThresh = opts.apneaThresh;
    this.hypopneaDrop = opts.hypopneaDrop;
    this.minDurationSec = opts.minDurationSec;

    // Rolling baseline (exponential moving average, 5-min window)
    this.baselineBR = null;
    this.baselineAlpha = 0.005; // slow adaptation

    // Event tracking
    this.events = [];           // { type, startTs, endTs, durationSec, avgBR }
    this.currentEvent = null;   // in-progress event
    this.eventSamples = [];     // BR samples during current event

    // Time tracking
    this.startTime = null;
    this.lastTime = null;
    this.totalSamples = 0;

    // Per-hour tracking
    this.hourlyEvents = new Map(); // hour_index -> count
  }

  ingest(timestamp, br) {
    if (!this.startTime) this.startTime = timestamp;
    this.lastTime = timestamp;
    this.totalSamples++;

    // Update baseline (only with "normal" breathing)
    if (br > this.apneaThresh * 2 && (!this.baselineBR || br < this.baselineBR * 2)) {
      if (this.baselineBR === null) {
        this.baselineBR = br;
      } else {
        this.baselineBR = this.baselineBR * (1 - this.baselineAlpha) + br * this.baselineAlpha;
      }
    }

    // Detect events
    const isApnea = br < this.apneaThresh;
    const isHypopnea = this.baselineBR && br < this.baselineBR * (1 - this.hypopneaDrop) && !isApnea;
    const inEvent = isApnea || isHypopnea;

    if (inEvent) {
      if (!this.currentEvent) {
        // Start new event
        this.currentEvent = {
          type: isApnea ? 'apnea' : 'hypopnea',
          startTs: timestamp,
        };
        this.eventSamples = [br];
      } else {
        this.eventSamples.push(br);
        // Upgrade hypopnea to apnea if BR drops further
        if (isApnea && this.currentEvent.type === 'hypopnea') {
          this.currentEvent.type = 'apnea';
        }
      }
    } else {
      // Event ended
      if (this.currentEvent) {
        const duration = timestamp - this.currentEvent.startTs;
        if (duration >= this.minDurationSec) {
          const avgBR = this.eventSamples.reduce((a, b) => a + b, 0) / this.eventSamples.length;
          const event = {
            type: this.currentEvent.type,
            startTs: this.currentEvent.startTs,
            endTs: timestamp,
            durationSec: duration,
            avgBR,
          };
          this.events.push(event);

          // Track hourly
          const hourIdx = Math.floor((this.currentEvent.startTs - this.startTime) / 3600);
          this.hourlyEvents.set(hourIdx, (this.hourlyEvents.get(hourIdx) || 0) + 1);
        }
        this.currentEvent = null;
        this.eventSamples = [];
      }
    }

    return { isApnea, isHypopnea, baseline: this.baselineBR, br };
  }

  getAHI() {
    const hours = this.lastTime && this.startTime
      ? (this.lastTime - this.startTime) / 3600
      : 0;
    if (hours < 0.01) return { ahi: 0, hours, events: 0, severity: 'Insufficient data' };

    const totalEvents = this.events.length;
    const ahi = totalEvents / hours;

    let severity;
    if (ahi < 5) severity = 'Normal';
    else if (ahi < 15) severity = 'Mild';
    else if (ahi < 30) severity = 'Moderate';
    else severity = 'Severe';

    return { ahi, hours, events: totalEvents, severity };
  }

  getHourlyAHI() {
    const result = [];
    for (const [hour, count] of [...this.hourlyEvents.entries()].sort((a, b) => a[0] - b[0])) {
      result.push({ hour, events: count, ahi: count }); // events per 1 hour
    }
    return result;
  }

  getEventSummary() {
    const apneas = this.events.filter(e => e.type === 'apnea');
    const hypopneas = this.events.filter(e => e.type === 'hypopnea');

    return {
      totalEvents: this.events.length,
      apneas: apneas.length,
      hypopneas: hypopneas.length,
      avgApneaDuration: apneas.length > 0
        ? apneas.reduce((s, e) => s + e.durationSec, 0) / apneas.length : 0,
      avgHypopneaDuration: hypopneas.length > 0
        ? hypopneas.reduce((s, e) => s + e.durationSec, 0) / hypopneas.length : 0,
      maxDuration: this.events.length > 0
        ? Math.max(...this.events.map(e => e.durationSec)) : 0,
      baselineBR: this.baselineBR || 0,
    };
  }
}

// ---------------------------------------------------------------------------
// Packet parsing
// ---------------------------------------------------------------------------
function parseVitalsJsonl(record) {
  if (record.type !== 'vitals') return null;
  return { timestamp: record.timestamp, nodeId: record.node_id, br: record.breathing_bpm || 0 };
}

function parseVitalsUdp(buf) {
  if (buf.length < 32) return null;
  const magic = buf.readUInt32LE(0);
  if (magic !== VITALS_MAGIC && magic !== FUSED_MAGIC) return null;
  return {
    timestamp: Date.now() / 1000,
    nodeId: buf.readUInt8(4),
    br: buf.readUInt16LE(6) / 100,
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

  const detector = new ApneaDetector({
    apneaThresh: APNEA_THRESH,
    hypopneaDrop: HYPOPNEA_DROP,
    minDurationSec: MIN_DURATION_SEC,
  });

  const rl = readline.createInterface({
    input: fs.createReadStream(filePath),
    crlfDelay: Infinity,
  });

  let vitalsCount = 0;
  let lastPrintTs = 0;

  for await (const line of rl) {
    if (!line.trim()) continue;
    let record;
    try { record = JSON.parse(line); } catch { continue; }

    const v = parseVitalsJsonl(record);
    if (!v) continue;

    const state = detector.ingest(v.timestamp, v.br);
    vitalsCount++;

    // Print new events immediately
    const lastEvent = detector.events.length > 0 ? detector.events[detector.events.length - 1] : null;
    if (lastEvent && lastEvent.endTs === v.timestamp) {
      if (JSON_OUTPUT) {
        console.log(JSON.stringify({
          type: 'event',
          event_type: lastEvent.type,
          start: lastEvent.startTs,
          end: lastEvent.endTs,
          duration_sec: +lastEvent.durationSec.toFixed(1),
          avg_br: +lastEvent.avgBR.toFixed(2),
        }));
      } else {
        const ts = new Date(lastEvent.startTs * 1000).toISOString().slice(11, 19);
        const tag = lastEvent.type === 'apnea' ? '!! APNEA  ' : '~  HYPOPNEA';
        console.log(`[${ts}] ${tag} | ${lastEvent.durationSec.toFixed(1)}s | avg BR ${lastEvent.avgBR.toFixed(1)} BPM`);
      }
    }

    // Periodic status
    const tsMs = v.timestamp * 1000;
    if (tsMs - lastPrintTs >= INTERVAL_MS * 2) {
      if (!JSON_OUTPUT) {
        const ahi = detector.getAHI();
        const ts = new Date(v.timestamp * 1000).toISOString().slice(11, 19);
        console.log(`[${ts}] BR ${v.br.toFixed(1)} | baseline ${(state.baseline || 0).toFixed(1)} | AHI ${ahi.ahi.toFixed(1)} (${ahi.severity}) | ${ahi.events} events / ${ahi.hours.toFixed(2)} hrs`);
      }
      lastPrintTs = tsMs;
    }
  }

  // Final summary
  const ahi = detector.getAHI();
  const summary = detector.getEventSummary();

  if (JSON_OUTPUT) {
    console.log(JSON.stringify({
      type: 'summary',
      ahi: +ahi.ahi.toFixed(2),
      severity: ahi.severity,
      hours: +ahi.hours.toFixed(3),
      ...summary,
      hourly: detector.getHourlyAHI(),
    }));
  } else {
    console.log('\n' + '='.repeat(60));
    console.log('APNEA SCREENING SUMMARY');
    console.log('DISCLAIMER: Pre-screening only. Consult a physician.');
    console.log('='.repeat(60));
    console.log(`Monitored:        ${ahi.hours.toFixed(2)} hours (${vitalsCount} samples)`);
    console.log(`AHI:              ${ahi.ahi.toFixed(1)} events/hour`);
    console.log(`Severity:         ${ahi.severity}`);
    console.log(`Total events:     ${summary.totalEvents}`);
    console.log(`  Apneas:         ${summary.apneas} (avg ${summary.avgApneaDuration.toFixed(1)}s)`);
    console.log(`  Hypopneas:      ${summary.hypopneas} (avg ${summary.avgHypopneaDuration.toFixed(1)}s)`);
    console.log(`  Longest event:  ${summary.maxDuration.toFixed(1)}s`);
    console.log(`Baseline BR:      ${summary.baselineBR.toFixed(1)} BPM`);

    const hourly = detector.getHourlyAHI();
    if (hourly.length > 0) {
      console.log('\nHourly breakdown:');
      for (const h of hourly) {
        const bar = '\u2588'.repeat(Math.min(h.events, 40));
        console.log(`  Hour ${h.hour}: ${bar} ${h.events} events (AHI ${h.ahi})`);
      }
    }

    // Event timeline
    if (detector.events.length > 0 && detector.events.length <= 50) {
      console.log('\nEvent timeline:');
      for (const e of detector.events) {
        const ts = new Date(e.startTs * 1000).toISOString().slice(11, 19);
        const tag = e.type === 'apnea' ? 'APNEA   ' : 'HYPOPNEA';
        console.log(`  [${ts}] ${tag} ${e.durationSec.toFixed(1)}s (BR ${e.avgBR.toFixed(1)})`);
      }
    } else if (detector.events.length > 50) {
      console.log(`\n(${detector.events.length} events total, showing first/last 5)`);
      for (const e of detector.events.slice(0, 5)) {
        const ts = new Date(e.startTs * 1000).toISOString().slice(11, 19);
        console.log(`  [${ts}] ${e.type.padEnd(8)} ${e.durationSec.toFixed(1)}s`);
      }
      console.log('  ...');
      for (const e of detector.events.slice(-5)) {
        const ts = new Date(e.startTs * 1000).toISOString().slice(11, 19);
        console.log(`  [${ts}] ${e.type.padEnd(8)} ${e.durationSec.toFixed(1)}s`);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Live UDP mode
// ---------------------------------------------------------------------------
function startLive() {
  const detector = new ApneaDetector({
    apneaThresh: APNEA_THRESH,
    hypopneaDrop: HYPOPNEA_DROP,
    minDurationSec: MIN_DURATION_SEC,
  });

  const server = dgram.createSocket('udp4');

  server.on('message', (buf) => {
    const v = parseVitalsUdp(buf);
    if (!v) return;

    const state = detector.ingest(v.timestamp, v.br);

    // Alert on new events
    const lastEvent = detector.events.length > 0 ? detector.events[detector.events.length - 1] : null;
    if (lastEvent && Math.abs(lastEvent.endTs - v.timestamp) < 2) {
      if (JSON_OUTPUT) {
        console.log(JSON.stringify({
          type: 'event', event_type: lastEvent.type,
          duration_sec: +lastEvent.durationSec.toFixed(1),
          avg_br: +lastEvent.avgBR.toFixed(2),
        }));
      } else {
        const tag = lastEvent.type === 'apnea' ? '!! APNEA' : '~  HYPOPNEA';
        console.log(`${tag} | ${lastEvent.durationSec.toFixed(1)}s | avg BR ${lastEvent.avgBR.toFixed(1)}`);
      }
    }
  });

  setInterval(() => {
    if (!JSON_OUTPUT) {
      const ahi = detector.getAHI();
      process.stdout.write('\x1B[2J\x1B[H');
      console.log('=== APNEA SCREENING (ADR-077) ===');
      console.log('DISCLAIMER: Pre-screening only. Not a diagnostic device.');
      console.log('');
      console.log(`AHI: ${ahi.ahi.toFixed(1)} events/hour | Severity: ${ahi.severity}`);
      console.log(`Events: ${ahi.events} in ${ahi.hours.toFixed(2)} hours`);
      console.log(`Baseline BR: ${(detector.baselineBR || 0).toFixed(1)} BPM`);

      if (detector.events.length > 0) {
        console.log('\nRecent events:');
        for (const e of detector.events.slice(-5)) {
          const ts = new Date(e.startTs * 1000).toISOString().slice(11, 19);
          console.log(`  [${ts}] ${e.type.padEnd(8)} ${e.durationSec.toFixed(1)}s (BR ${e.avgBR.toFixed(1)})`);
        }
      }
    }
  }, INTERVAL_MS);

  server.bind(PORT, () => {
    if (!JSON_OUTPUT) {
      console.log(`Apnea Detector listening on UDP :${PORT}`);
      console.log('DISCLAIMER: Pre-screening only. Consult a physician.\n');
    }
  });

  process.on('SIGINT', () => {
    const ahi = detector.getAHI();
    if (!JSON_OUTPUT) {
      console.log(`\nSession AHI: ${ahi.ahi.toFixed(1)} (${ahi.severity}) | ${ahi.events} events / ${ahi.hours.toFixed(2)} hrs`);
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
