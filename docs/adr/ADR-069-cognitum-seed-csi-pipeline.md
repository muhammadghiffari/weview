# ADR-069: ESP32 CSI → Cognitum Seed RVF Ingest Pipeline

| Field      | Value                                                    |
|------------|----------------------------------------------------------|
| Status     | Accepted                                                 |
| Date       | 2026-04-02                                               |
| Authors    | rUv, claude-flow                                         |
| Drivers    | #348 (multinode mesh accuracy), Research: Arena Physica   |
| Supersedes | —                                                        |
| Related    | ADR-066 (ESP32 swarm + Seed coordinator), ADR-068 (per-node state), ADR-018 (CSI binary protocol), ADR-039 (edge intelligence), ADR-065 (happiness scoring + Seed bridge) |

## Context

The wifi-densepose project has two hardware components that need to work as an integrated sensing pipeline:

1. **ESP32-S3** (COM9 / 192.168.1.105) — Captures WiFi CSI at 100 Hz, runs dual-core DSP pipeline (phase extraction, subcarrier selection, breathing/heart rate estimation, presence/fall detection), and sends ADR-018 binary frames via UDP.

2. **Cognitum Seed** (USB / 169.254.42.1 / 192.168.1.109) — A Pi Zero 2 W edge intelligence appliance running firmware v0.8.1. It provides:
   - **RVF vector store** — Append-only binary format with content-addressed IDs, kNN queries (cosine/L2/dot), and kNN graph with boundary analysis
   - **Witness chain** — SHA-256 tamper-evident audit trail for every write operation
   - **Ed25519 custody** — Device-bound keypair for cryptographic attestation
   - **Sensor pipeline** — 5 sensors (reed switch, PIR, vibration, ADS1115 4-ch ADC, BME280), 13 drift detectors, anti-spoofing
   - **Cognitive container** — Spectral graph analysis with Stoer-Wagner min-cut fragility scoring
   - **MCP proxy** — 114 tools via JSON-RPC 2.0 for AI assistant integration
   - **Thermal governor** — DVFS management with zone-based frequency scaling
   - **Temporal coherence** — Phase boundary detection across vector store evolution
   - **Swarm sync** — Epoch-based delta replication between peers
   - **Reflex rules** — 3 rules (fragility alarm, drift cutoff, HD anomaly indicator)
   - **98 HTTPS API endpoints** with per-client bearer token authentication

### Current State

| Component | Status | Details |
|-----------|--------|---------|
| ESP32 CSI capture | Working | 100 Hz, ADR-018 binary frames via UDP |
| ESP32 edge DSP | Working | 10-stage pipeline on Core 1 (phase, variance, vitals, fall) |
| ESP32 → sensing-server | Working | UDP port 5005, binary protocol |
| Cognitum Seed | Online | v0.8.1, paired, 19 vectors, epoch 25, WiFi connected |
| Seed vector store | Working | 8-dim RVF, kNN queries in 85ms for 20k vectors |
| Seed MCP proxy | Working | 114 tools, default-deny policy |
| ESP32 → Seed pipeline | **Validated** | Bridge on host laptop, UDP 5006 → HTTPS ingest (see Validation Results) |

### Gap Analysis (from Arena Physica research)

Arena Physica's approach (Heaviside-0 forward model, Marconi-0 inverse diffusion) demonstrates that neural surrogates for Maxwell's equations are production-viable. Our research identified that:

1. **Physics-informed intermediate supervision** — Evaluating pipeline stages independently catches failures that end-to-end metrics miss
2. **Vector embeddings for EM fields** — Storing CSI features as vectors enables similarity search for environment fingerprinting and anomaly detection
3. **Witness chain for sensing integrity** — Tamper-evident audit trails are critical for healthcare/safety applications (fall detection, vital signs)
4. **Edge compute for inference** — Pi Zero 2 W can run ~2.5M parameter models at 10+ Hz with INT8 quantization

### Problem

There is no pipeline connecting ESP32 CSI sensing to the Cognitum Seed's vector store. The ESP32 sends raw CSI frames to the Rust sensing-server (typically running on a laptop/desktop), but cannot leverage the Seed's:
- Persistent vector storage with kNN search
- Cryptographic witness chain for data integrity
- Cognitive container for structural analysis
- Sensor fusion with environmental sensors (BME280 temperature/humidity, PIR motion)
- Swarm sync for multi-Seed deployments

## Decision

Build a three-stage pipeline connecting ESP32 CSI capture to Cognitum Seed RVF storage:

### Architecture

```
┌──────────────────────────┐
│     ESP32-S3 (COM9)      │
│     node_id=1            │
│     192.168.1.105        │
│     Firmware v0.5.2      │
│ ┌──────────────────────┐ │
│ │ Core 0: WiFi + CSI   │ │
│ │   100 Hz capture     │ │
│ │   ADR-018 framing    │ │
│ ├──────────────────────┤ │
│ │ Core 1: Edge DSP     │ │
│ │   Phase extraction   │ │
│ │   Subcarrier select  │ │
│ │   Vital signs (HR/BR)│ │
│ │   Presence/fall det. │ │
│ │   Feature vector     │ │◄── 8-dim feature extraction
│ └──────────┬───────────┘ │
│            │ UDP          │
└────────────┼─────────────┘
             │ Port 5005 (raw CSI, magic 0xC5110001)
             │ + Port 5006 (vitals 0xC5110002 + features 0xC5110003)
             ▼
┌────────────────────────────────────────────┐
│   Host Laptop (192.168.1.20)               │
│   Bridge script (Python)                   │
│ ┌────────────────────────────────────────┐ │
│ │  Stage 1: CSI Receiver                 │ │
│ │    UDP listener on port 5006           │ │
│ │    Parses 0xC5110003 feature packets   │ │
│ │    (also accepts 0xC5110001/0002)      │ │
│ │    Batches 10 vectors per ingest       │ │
│ └──────────┬─────────────────────────────┘ │
└────────────┼───────────────────────────────┘
             │ HTTPS POST (bearer token)
             ▼
┌────────────────────────────────────────────┐
│         Cognitum Seed (Pi Zero 2 W)        │
│         169.254.42.1 / 192.168.1.109       │
│         Firmware v0.8.1                    │
│ ┌────────────────────────────────────────┐ │
│ │  Stage 2: RVF Ingest                   │ │
│ │    POST /api/v1/store/ingest           │ │
│ │    Content-addressed vector ID         │ │
│ │    Metadata: node_id, timestamp, type  │ │
│ │    Witness chain entry per batch       │ │
│ ├────────────────────────────────────────┤ │
│ │  Stage 3: Cognitive Analysis           │ │
│ │    kNN graph rebuild (every 10s)       │ │
│ │    Boundary analysis (fragility)       │ │
│ │    Temporal coherence (phase detect)   │ │
│ │    Reflex rules (alarm triggers)       │ │
│ ├────────────────────────────────────────┤ │
│ │  Existing Sensors                      │ │
│ │    BME280 → temp/humidity/pressure     │ │
│ │    PIR → motion ground truth           │ │
│ │    Reed switch → door/window state     │ │
│ │    ADS1115 → analog inputs             │ │
│ └────────────────────────────────────────┘ │
│                                            │
│  Outputs:                                  │
│    • /api/v1/store/query — kNN search      │
│    • /api/v1/boundary — fragility score    │
│    • /api/v1/coherence/profile — phases    │
│    • /api/v1/cognitive/snapshot — graph     │
│    • /api/v1/custody/attestation — signed  │
│    • MCP proxy — 114 tools for AI agents   │
└────────────────────────────────────────────┘
```

### Stage 1: ESP32 Feature Vector Extraction

The ESP32 edge processing pipeline (Core 1) already computes all signals needed. We add a compact 8-dimensional feature vector extracted from the existing DSP outputs:

| Dimension | Feature | Source | Range |
|-----------|---------|--------|-------|
| 0 | Presence score | `s_presence_score / 10.0` (clamped) | 0.0–1.0 |
| 1 | Motion energy | `s_motion_energy / 10.0` (clamped) | 0.0–1.0 |
| 2 | Breathing rate | `s_breathing_bpm / 30.0` (clamped) | 0.0–1.0 |
| 3 | Heart rate | `s_heartrate_bpm / 120.0` (clamped) | 0.0–1.0 |
| 4 | Phase variance (mean) | Top-K subcarrier Welford variance mean | 0.0–1.0 |
| 5 | Person count | `n_active_persons / 4.0` (clamped) | 0.0–1.0 |
| 6 | Fall detected | Binary: 1.0 if `s_fall_detected`, else 0.0 | 0.0 or 1.0 |
| 7 | RSSI (normalized) | `(s_latest_rssi + 100) / 100` (clamped) | 0.0–1.0 |

This maps directly to the Seed's store dimension of 8, enabling kNN queries like "find the 10 most similar sensing states to the current one."

**Packet format** (magic `0xC5110003`, defined as `edge_feature_pkt_t` in `edge_processing.h`):

```c
typedef struct __attribute__((packed)) {
    uint32_t magic;          // EDGE_FEATURE_MAGIC = 0xC5110003
    uint8_t  node_id;        // ESP32 node identifier
    uint8_t  reserved;       // alignment padding
    uint16_t seq;            // sequence number
    int64_t  timestamp_us;   // microseconds since boot
    float    features[8];    // 8-dim normalized feature vector (32 bytes)
} edge_feature_pkt_t;        // Total: 48 bytes (static_assert enforced)
```

**Transmission rate:** 1 Hz (one feature vector per second, aggregated from 100 Hz CSI). This keeps UDP bandwidth under 50 bytes/s per node and avoids overwhelming the Seed's vector store.

### Stage 2: Seed-Side RVF Ingest

A lightweight Rust service on the Seed (or a Python bridge script) listens for feature packets on UDP port 5006 and ingests them via the Seed's REST API:

```bash
# Ingest a feature vector with metadata
curl -sk -X POST https://169.254.42.1:8443/api/v1/store/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [[0, [0.85, 0.3, 0.52, 0.65, 0.4, 0.78, 0.1, -0.45]]],
    "metadata": {
      "node_id": 1,
      "type": "csi_feature",
      "timestamp": 1775166970
    }
  }'
```

**Batching:** Accumulate 10 vectors (10 seconds) per ingest call to reduce HTTP overhead (`--batch-size 10` default in `seed_csi_bridge.py`; also supports time-based flushing via `--flush-interval`). At 1 vector/second per node, a 4-node mesh generates 14,400 vectors/hour (345,600/day). Daily compaction is required to stay within the Seed's 100K vector working set (see Storage Budget).

**Witness chain:** Each ingest automatically appends a witness entry, providing a tamper-evident record of all sensing data. The epoch increments monotonically, and the SHA-256 chain can be verified at any time via `POST /api/v1/witness/verify`.

### Stage 3: Cognitive Analysis & Sensor Fusion

Once CSI feature vectors are in the RVF store, the Seed's existing subsystems activate:

1. **kNN Graph** — Rebuilt every 10 seconds. Similar sensing states cluster together. Anomalous states (intruder, fall, unusual breathing) appear as outliers.

2. **Boundary Analysis** — Stoer-Wagner min-cut computes a fragility score (0.0–1.0). High fragility indicates the vector space is splitting — a regime change in the environment (door opened, person entered/left, HVAC state change).

3. **Temporal Coherence** — Phase boundary detection across the vector store timeline identifies when the environment transitions between states (occupied → empty, day → night, normal → abnormal).

4. **Reflex Rules** — Three pre-configured rules fire automatically:
   - `fragility_alarm` (threshold 0.3) → relay actuator for presence alert
   - `drift_cutoff` (threshold 1.0) → cutoff when sensor drift detected
   - `hd_anomaly_indicator` (threshold 200) → PWM brightness for anomaly severity

5. **Sensor Fusion** — The Seed's BME280 (temperature/humidity/pressure) and PIR sensor provide environmental ground truth that correlates with CSI features:
   - PIR motion validates CSI presence detection
   - Temperature changes correlate with occupancy
   - Humidity changes correlate with breathing detection fidelity

6. **MCP Integration** — AI assistants can query the full pipeline via the 114-tool MCP proxy:
   ```json
   {"method": "tools/call", "params": {"name": "seed.memory.query", "arguments": {"vector": [0.8, 0.5, 0.4, 0.6, 0.3, 0.7, 0.1, -0.3], "k": 5}}}
   ```

### ESP32 Provisioning

The ESP32's existing NVS provisioning system supports configuring the Seed as the target:

```bash
python firmware/esp32-csi-node/provision.py \
  --port COM9 \
  --target-ip 192.168.1.20 \
  --target-port 5006 \
  --node-id 1
```

Note: `--target-ip` is the host laptop (192.168.1.20), not the Seed IP, because the bridge runs on the host and forwards to the Seed via HTTPS (see Known Issue 4).

No firmware recompilation needed — the `stream_sender` module reads target IP/port from NVS at boot.

### Data Flow Rates

| Path | Rate | Size | Bandwidth |
|------|------|------|-----------|
| CSI capture → ring buffer | 100 Hz | ~400 B | 40 KB/s (internal) |
| Edge DSP → sensing-server | 100 Hz | ~200 B | 20 KB/s (existing) |
| Edge DSP → Seed features | 1 Hz | 48 B | 48 B/s (new) |
| Seed ingest (batched) | 0.1 Hz | ~500 B | 50 B/s (HTTP) |
| Seed kNN graph rebuild | 0.1 Hz | internal | — |
| Seed witness chain | per batch | 32 B hash | — |

### Storage Budget

| Timeframe | Vectors/node | 4 nodes | RVF size | RAM |
|-----------|-------------|---------|----------|-----|
| 1 hour | 3,600 | 14,400 | ~580 KB | ~6 MB |
| 24 hours | 86,400 | 345,600 | ~14 MB | ~140 MB |
| 7 days | 604,800 | 2,419,200 | ~97 MB | exceeds |

**Compaction policy:** Run `POST /api/v1/store/compact` daily at 03:00, retaining only the last 24 hours of vectors. Archive older vectors to USB drive via `POST /api/v1/store/export` before compaction.

**Dimension reduction:** For deployments exceeding 100K vectors, reduce feature extraction rate to 0.1 Hz (one vector per 10 seconds) or increase compaction frequency.

## Validation Results

**Live hardware test performed 2026-04-02.**

### Hardware Under Test

| Component | Port | IP | Firmware | WiFi | RSSI |
|-----------|------|----|----------|------|------|
| ESP32-S3 (8MB) | COM9 | 192.168.1.105 | v0.5.2 | ruv.net (ch 5) | -34 dBm |
| Cognitum Seed | USB | 169.254.42.1 / 192.168.1.109 | v0.8.1 | ruv.net | — |
| Host laptop | — | 192.168.1.20 | — | ruv.net | — |

Seed device_id: `ecaf97dd-fc90-4b0e-b0e7-e9f896b9fbb6`. Pairing token issued to `wifi-densepose-claude`.

### Pipeline Validated

1. **UDP streaming** -- 211 packets captured in 15 seconds:
   - 196 raw CSI frames (magic `0xC5110001`)
   - 15 vitals frames (magic `0xC5110002`)

2. **Bridge pipeline** -- 20 vitals packets (`0xC5110002`) parsed, converted to 8-dim feature vectors via the bridge's `parse_vitals_packet()` fallback path, ingested in 4 batches of 5 vectors each (`--batch-size 5`). The native `0xC5110003` feature packet path is implemented in firmware but was not exercised in this validation run (firmware was v0.5.2; the `send_feature_vector()` addition requires a reflash).

3. **RVF ingest** -- All 20 vectors accepted by Seed. Epochs advanced 88 to 91. Witness chain verified valid (193 entries, SHA-256 chain intact).

4. **Seed sensors** -- BME280, PIR, reed switch, ADS1115, vibration sensor all present and healthy.

### Live Vital Signs Captured

| Metric | Observed Range | Expected | Notes |
|--------|---------------|----------|-------|
| Presence score | 1.41 -- 14.92 | 0.0 -- 1.0 | **Needs normalization** (see Known Issues) |
| Motion energy | 1.41 -- 14.92 | 0.0 -- 1.0 | Same raw value as presence score |
| Breathing rate | 19.8 -- 33.5 BPM | 12 -- 25 BPM | Plausible but slightly high |
| Heart rate | 75.3 -- 99.1 BPM | 60 -- 100 BPM | Plausible range |
| RSSI | -43 to -72 dBm | -30 to -80 dBm | Normal |
| Fall detected | No | — | Correct (no falls occurred) |
| n_persons | 4 | 1 | **Miscalibrated** (see Known Issues) |

### Known Issues Found

1. **`presence_score` exceeds 1.0 in vitals packets** -- Raw values range 1.41 to 14.92 in the vitals packet (`0xC5110002`). The bridge's vitals-to-feature conversion clamps to 1.0 for dim 0 and divides by 10.0 for dim 1 (`motion_energy / 10.0`), but dim 0 clamps without scaling. **Note:** The firmware's native feature vector (`0xC5110003`) already normalizes correctly by dividing `s_presence_score` by 10.0 (see `edge_processing.c` line 657). This issue only affects the vitals-packet fallback path in the bridge.

2. **`n_persons = 4` with 1 person present** -- The multi-person counting algorithm is miscalibrated for single-occupancy scenarios. The per-node state pipeline (ADR-068) may mitigate this when the baseline is properly trained, but the raw edge count is unreliable.

3. **Content-addressed vector IDs cause deduplication** -- Similar feature vectors hash to the same ID, causing the Seed to silently drop duplicates. **Fixed in bridge:** `seed_csi_bridge.py` now uses `_make_vector_id()` which generates a SHA-256 hash of `node_id:timestamp_us:seq_counter`, producing unique 32-bit IDs. This was observed during validation and fixed before the final test run.

4. **Bridge runs on host, not Seed** -- The ESP32 target IP must be the host laptop (192.168.1.20), not the Seed IP. The bridge script on the host forwards to the Seed via HTTPS. This adds a hop but avoids running a UDP listener on the Pi Zero 2 W.

5. **PIR GPIO read returned 404** -- `GET /api/v1/sensor/gpio/read?pin=6` returned 404. The PIR endpoint may require a different pin number or endpoint format. Ground-truth validation against PIR is deferred to Phase 3.

## Implementation Plan

### Phase 1: ESP32 Feature Extraction (firmware change) -- DONE

Implemented as `send_feature_vector()` in `edge_processing.c` (lines 644-699) and `edge_feature_pkt_t` in `edge_processing.h` (lines 112-124). The function reads from static globals (`s_presence_score`, `s_motion_energy`, `s_breathing_bpm`, `s_heartrate_bpm`, subcarrier Welford variance, person tracker, fall flag, RSSI) and normalizes each dimension to 0.0-1.0 with clamping.

Called at the same 1 Hz cadence as `send_vitals_packet()` in Step 13 of the edge processing pipeline (line 855). The compressed frame magic was reassigned from `0xC5110003` to `0xC5110005` to free up `0xC5110003` for feature vectors (`EDGE_COMPRESSED_MAGIC` in `edge_processing.h` line 29).

### Phase 2: Seed Ingest Bridge (Python script on host) -- DONE

Implemented as `scripts/seed_csi_bridge.py`. The bridge:
1. Listens on UDP port 5006 (configurable via `--udp-port`)
2. Accepts all three packet formats: `0xC5110003` (ADR-069 features), `0xC5110002` (vitals, converted to 8-dim), and `0xC5110001` (raw CSI, minimal features)
3. Generates unique vector IDs via SHA-256 hash of `node_id:timestamp:seq` (avoids content-addressed deduplication -- see Known Issue 3)
4. Batches vectors (default 10, configurable via `--batch-size`) with time-based flush fallback (`--flush-interval`)
5. POSTs to Seed's `/api/v1/store/ingest` with bearer token
6. Supports `--validate` mode (kNN query + PIR comparison after each batch)
7. Supports `--stats` mode (print Seed status, boundary, coherence, graph)
8. Supports `--compact` mode (trigger store compaction)

### Phase 3: Validation & Ground Truth -- BLOCKED

Use the Seed's PIR sensor as ground truth for presence detection:
1. Query PIR state: `GET /api/v1/sensor/gpio/read?pin=6`
2. Compare with CSI presence score (feature dim 0)
3. Log agreement/disagreement rate
4. Use kNN to find historical vectors matching current PIR state → validate CSI accuracy

**Status:** The bridge implements `--validate` mode with PIR comparison (see `_run_validation()` in `seed_csi_bridge.py`). However, the PIR endpoint returned 404 during validation (Known Issue 5). This phase is blocked until the correct PIR API endpoint is identified.

### Phase 4: Multi-Node Mesh (addresses #348)

Deploy 3 ESP32 nodes, each sending feature vectors to the bridge host (which forwards to the Seed):
- Node 1 (lobby): `--node-id 1 --target-ip 192.168.1.20 --target-port 5006`
- Node 2 (hallway): `--node-id 2 --target-ip 192.168.1.20 --target-port 5006`
- Node 3 (room): `--node-id 3 --target-ip 192.168.1.20 --target-port 5006`

All nodes target the host laptop (192.168.1.20) where the bridge script runs. The bridge batches and forwards all nodes' vectors to the Seed via HTTPS. The Seed's kNN graph naturally clusters vectors by node and by sensing state. Cross-node analysis via boundary fragility detects when a person moves between zones.

## Security Considerations

1. **Bearer token** — All write operations require the pairing token. Token stored as SHA-256 hash on device.
2. **TLS** — All API calls over HTTPS (port 8443) with device-provisioned CA certificate.
3. **Witness chain** — Every ingest is cryptographically chained. Tampering detection via `POST /api/v1/witness/verify`.
4. **Ed25519 attestation** — Device identity bound to hardware keypair. Attestation includes epoch, vector count, and witness head.
5. **Anti-spoofing** — Sensor pipeline has entropy-based spoofing detection (min 0.5 bits entropy, streak threshold 3).
6. **USB-only pairing** — Pairing window can only be opened from USB interface (169.254.42.1), not from WiFi.

## Hardware Bill of Materials

| Component | Port | IP | Cost |
|-----------|------|----|------|
| ESP32-S3 (8MB) | COM9 | 192.168.1.105 (DHCP) | ~$9 |
| Cognitum Seed (Pi Zero 2W) | USB | 169.254.42.1 / 192.168.1.109 | ~$15 |
| USB-C cable (data) | — | — | ~$3 |
| **Total** | | | **~$27** |

### Seed Sensors (included)

| Sensor | Interface | Channels | Purpose |
|--------|-----------|----------|---------|
| Reed switch | GPIO 5 | 1 | Door/window state |
| PIR motion | GPIO 6 | 1 | Motion ground truth |
| Vibration | GPIO 13 | 1 | Structural vibration |
| ADS1115 | I2C 0x48 | 4 | Analog inputs (extensible) |
| BME280 | I2C 0x76 | 3 | Temperature, humidity, pressure |

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Pi Zero thermal throttling at sustained ingest | Medium | Performance degrades | Thermal governor already manages DVFS; 1 Hz ingest is minimal load |
| WiFi congestion with ESP32 CSI + UDP | Low | Lost packets | Feature vectors are 48 bytes at 1 Hz; negligible vs CSI traffic |
| RVF store exceeds RAM at high vector count | Medium | OOM | Compaction policy + dimension reduction + daily export |
| Bearer token exposure | Low | Unauthorized writes | TLS encryption + USB-only pairing + token hashing |
| ESP32 NVS corruption | Low | Config lost | NVS is wear-leveled flash with CRC; re-provision via USB |

## Consequences

### Positive
- ESP32 CSI features become persistent, searchable, and cryptographically attested
- kNN similarity search enables environment fingerprinting and anomaly detection
- PIR + BME280 provide ground truth for CSI validation
- MCP proxy enables AI assistants to query sensing state directly
- Witness chain provides audit trail for healthcare/safety applications
- Architecture aligns with Arena Physica's insight: store embeddings, not raw signals

### Negative
- Additional firmware packet type (48 bytes, trivial)
- Bridge script needed on Seed or host machine
- Daily compaction required for long-running deployments
- Bearer token must be managed (stored securely, rotated if compromised)

### Neutral
- Existing sensing-server pipeline unchanged (ESP32 still sends to port 5005)
- Seed's existing sensors continue operating independently
- Target IP/port configurable via NVS provisioning (no recompilation for deployment changes)
- Firmware recompilation needed once to add `send_feature_vector()` (Phase 1), but subsequent node deployments only need provisioning
