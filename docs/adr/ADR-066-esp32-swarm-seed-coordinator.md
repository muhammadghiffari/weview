# ADR-066: ESP32 CSI Swarm with Cognitum Seed Coordinator

**Status:** Proposed
**Date:** 2026-03-20
**Deciders:** @ruvnet
**Related:** ADR-065 (happiness scoring + Seed bridge), ADR-039 (edge intelligence), ADR-060 (provisioning), ADR-018 (CSI binary protocol), ADR-040 (WASM runtime)

## Context

ADR-065 established a single ESP32-S3 node pushing happiness vectors to a Cognitum Seed at `169.254.42.1` (Pi Zero 2 W, firmware 0.7.0). The Seed is now on the same WiFi network (`RedCloverWifi`, `10.1.10.236`) as the ESP32 node (`10.1.10.168`).

The Seed already exposes REST APIs for:
- Peer discovery (`/api/v1/peers`) — 0 peers currently registered
- Delta sync (`/api/v1/delta/pull`, `/api/v1/delta/push`) — epoch-based replication
- Reflex rules (`/api/v1/sensor/reflex/rules`) — 3 rules (fragility alarm, drift cutoff, HD anomaly indicator)
- Actuators (`/api/v1/sensor/actuators`) — relay + PWM outputs
- Cognitive engine (`/api/v1/cognitive/tick`) — periodic inference loop
- Witness chain (`/api/v1/custody/epoch`) — epoch 316, cryptographically signed
- kNN search (`/api/v1/store/search`) — similarity queries across the full vector store

A hotel deployment requires multiple ESP32 nodes (lobby, hallway, restaurant, rooms) coordinated as a swarm with centralized analytics on the Seed.

## Decision

Implement a Seed-coordinated ESP32 swarm where each node operates autonomously for CSI sensing and edge processing, while the Seed serves as the swarm coordinator for registration, aggregation, drift detection, cross-zone inference, and actuator control.

### Architecture

```
    ESP32 Node A              ESP32 Node B              ESP32 Node C
    (Lobby)                   (Hallway)                 (Restaurant)
    node_id=1                 node_id=2                 node_id=3
    10.1.10.168               10.1.10.xxx               10.1.10.xxx
    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │ WiFi CSI     │          │ WiFi CSI     │          │ WiFi CSI     │
    │ Tier 2 DSP   │          │ Tier 2 DSP   │          │ Tier 2 DSP   │
    │ WASM Tier 3  │          │ WASM Tier 3  │          │ WASM Tier 3  │
    │ Swarm Bridge │          │ Swarm Bridge │          │ Swarm Bridge │
    └──────┬───────┘          └──────┬───────┘          └──────┬───────┘
           │ HTTP POST                │ HTTP POST                │ HTTP POST
           │ (happiness vectors,      │                          │
           │  heartbeat, events)      │                          │
           └──────────┬───────────────┴──────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ Cognitum Seed │
              │ (Coordinator) │
              │ 10.1.10.236   │
              ├───────────────┤
              │ Vector Store  │ ← 8-dim vectors tagged with node_id + zone
              │ kNN Search    │ ← Cross-zone similarity ("which room matches?")
              │ Drift Detect  │ ← Global mood trend across all zones
              │ Witness Chain │ ← Tamper-proof audit trail per node
              │ Reflex Rules  │ ← Trigger actuators on swarm-wide patterns
              │ Cognitive Eng │ ← Periodic cross-zone inference
              │ Peer Registry │ ← Node health, last-seen, capabilities
              └───────────────┘
```

### Swarm Protocol

#### 1. Node Registration (on boot)

Each ESP32 registers with the Seed via HTTP POST on startup. The Seed's peer discovery API tracks active nodes.

```
POST /api/v1/store/ingest
{
  "vectors": [{
    "id": "node-1-reg",
    "values": [0,0,0,0,0,0,0,0],
    "metadata": {
      "type": "registration",
      "node_id": 1,
      "zone": "lobby",
      "mac": "1C:DB:D4:83:D2:40",
      "ip": "10.1.10.168",
      "firmware": "0.5.0",
      "capabilities": ["csi", "tier2", "presence", "vitals", "happiness"],
      "flash_mb": 4,
      "psram_mb": 2
    }
  }]
}
```

#### 2. Heartbeat (every 30 seconds)

```
POST /api/v1/store/ingest
{
  "vectors": [{
    "id": "node-1-hb-{epoch}",
    "values": [happiness, gait, stride, fluidity, calm, posture, dwell, social],
    "metadata": {
      "type": "heartbeat",
      "node_id": 1,
      "zone": "lobby",
      "uptime_s": 3600,
      "csi_frames": 72000,
      "free_heap": 317140,
      "presence_now": true,
      "persons": 2,
      "rssi": -60
    }
  }]
}
```

#### 3. Happiness Vector Ingestion (every 5 seconds when presence detected)

```
POST /api/v1/store/ingest
{
  "vectors": [{
    "id": "node-1-h-{epoch}-{ts}",
    "values": [0.72, 0.65, 0.80, 0.71, 0.55, 0.60, 0.85, 0.45],
    "metadata": {
      "type": "happiness",
      "node_id": 1,
      "zone": "lobby",
      "timestamp_ms": 1742486400000,
      "persons": 2,
      "direction": "entering"
    }
  }]
}
```

#### 4. Cross-Zone Queries (Seed-side)

The Seed can answer questions across the entire swarm:

```
POST /api/v1/store/search
{"vector": [0.8, 0.7, 0.9, 0.8, 0.6, 0.7, 0.9, 0.5], "k": 5}

Response: nearest neighbors across all zones, showing which
rooms had the most similar mood to a "happy" reference vector.
```

#### 5. Reflex Rules for Swarm Patterns

Configure the Seed's reflex engine to act on swarm-wide patterns:

| Rule | Trigger | Action | Use Case |
|------|---------|--------|----------|
| `low_happiness_alert` | Mean happiness < 0.3 across 3+ nodes for 5 min | Activate `alarm` relay | Staff alert: guest dissatisfaction |
| `crowd_surge` | Presence count > 10 across lobby + hallway | PWM indicator brightness 100% | Lobby congestion warning |
| `zone_drift` | Drift score > 0.5 on any node | Log to witness chain | Trend change documentation |
| `ghost_anomaly` | Event 650 (anomaly) from any node | Notify + log | Security: unexpected RF disturbance |

### ESP32 Firmware: Swarm Bridge Module

New module `swarm_bridge.c` added to the CSI firmware, activated via NVS config:

```c
typedef struct {
    char     seed_url[64];       // e.g. "http://10.1.10.236"
    char     zone_name[16];      // e.g. "lobby"
    uint16_t heartbeat_sec;      // Default: 30
    uint16_t ingest_sec;         // Default: 5
    uint8_t  enabled;            // 0 = disabled, 1 = enabled
} swarm_config_t;
```

NVS keys (provisioned via `provision.py --seed-url http://10.1.10.236 --zone lobby`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `seed_url` | string | (empty) | Seed base URL; empty = swarm disabled |
| `zone_name` | string | `"default"` | Zone identifier for this node |
| `swarm_hb` | u16 | 30 | Heartbeat interval (seconds) |
| `swarm_ingest` | u16 | 5 | Vector ingest interval (seconds) |

The swarm bridge runs as a FreeRTOS task on Core 0 (separate from DSP on Core 1):

```
swarm_bridge_task (Core 0, priority 3, stack 4096)
  ├── On boot: POST registration to Seed
  ├── Every 30s: POST heartbeat with latest happiness vector
  ├── Every 5s (if presence): POST happiness vector
  └── On event 650+ (anomaly): POST immediately
```

HTTP client uses `esp_http_client` (already in ESP-IDF, no extra dependencies). JSON is formatted with `snprintf` (no cJSON dependency needed for the small payloads).

### Node Discovery and Addressing

Nodes find the Seed via:

1. **NVS provisioned URL** (primary) — `provision.py --seed-url http://10.1.10.236`
2. **mDNS fallback** — Seed advertises `_cognitum._tcp.local`; ESP32 resolves `cognitum.local`
3. **Link-local fallback** — `http://169.254.42.1` when connected via USB

### Vector ID Scheme

```
{node_id}-{type}-{epoch}-{timestamp_ms}
```

Examples:
- `1-reg` — Node 1 registration
- `1-hb-316` — Node 1 heartbeat at epoch 316
- `1-h-316-1742486400000` — Node 1 happiness vector at epoch 316, timestamp T
- `2-h-316-1742486401000` — Node 2 happiness vector at same epoch

### Witness Chain Integration

Every vector ingested into the Seed increments the epoch and extends the witness chain. The chain provides:

- **Per-node audit trail** — filter by node_id metadata to get one node's history
- **Tamper detection** — Ed25519 signed, hash-chained; break = detectable
- **Regulatory compliance** — prove "sensor X reported Y at time Z" for disputes
- **Cross-node ordering** — Seed epoch gives total order across all nodes

### Scaling Considerations

| Nodes | Vectors/hour | Seed storage/day | kNN latency |
|-------|---|---|---|
| 1 | 720 | ~1.5 MB | < 1 ms |
| 5 | 3,600 | ~7.5 MB | < 2 ms |
| 10 | 7,200 | ~15 MB | < 5 ms |
| 20 | 14,400 | ~30 MB | < 10 ms |

The Seed's Pi Zero 2 W has 512 MB RAM and typically an 8-32 GB SD card. At 30 MB/day for 20 nodes, storage lasts 250+ days before compaction is needed. The Seed's optimizer runs automatic compaction in the background.

### Provisioning for Swarm

```bash
# Node 1: Lobby (COM5, existing)
python provision.py --port COM5 \
    --ssid "RedCloverWifi" --password "redclover2.4" \
    --node-id 1 --seed-url "http://10.1.10.236" --zone "lobby"

# Node 2: Hallway (future device)
python provision.py --port COM6 \
    --ssid "RedCloverWifi" --password "redclover2.4" \
    --node-id 2 --seed-url "http://10.1.10.236" --zone "hallway"

# Node 3: Restaurant (future device)
python provision.py --port COM8 \
    --ssid "RedCloverWifi" --password "redclover2.4" \
    --node-id 3 --seed-url "http://10.1.10.236" --zone "restaurant"
```

## Consequences

### Positive

- **Zero infrastructure** — no cloud, no server, no database. Seed + ESP32s + WiFi router is the entire stack
- **Autonomous nodes** — each ESP32 runs full Tier 2 DSP independently; Seed loss degrades gracefully to local-only operation
- **Cryptographic audit** — witness chain gives tamper-proof history for every observation across all nodes
- **Real-time cross-zone analytics** — Seed kNN search answers "which zones are happy/stressed right now" in < 5 ms
- **Physical actuators** — Seed's relay/PWM outputs can trigger real-world actions (lights, alarms, displays) based on swarm-wide patterns
- **Horizontal scaling** — add ESP32 nodes by flashing firmware + running provision.py; no Seed reconfiguration needed
- **Privacy-preserving** — no cameras, no audio, no PII; only 8-dimensional feature vectors stored

### Negative

- **Single point of aggregation** — Seed failure loses cross-zone analytics (nodes continue autonomously)
- **WiFi dependency** — nodes must be on the same network as the Seed; no mesh/LoRa fallback yet
- **HTTP overhead** — REST/JSON adds ~200 bytes overhead per vector vs raw binary UDP; acceptable at 5-second intervals
- **Pi Zero 2 W limits** — 512 MB RAM, single-core ARM; adequate for 20 nodes but not 100+
- **No WASM OTA via Seed** — currently WASM modules are uploaded per-node; future work could use Seed as WASM distribution hub

### Implementation Progress

**ADR-069** implements the first stage of this swarm vision with live hardware validation (2026-04-02). A single ESP32-S3 node (COM9, firmware v0.5.2) was validated sending CSI-derived feature vectors through a host-side bridge into the Cognitum Seed's RVF store (firmware v0.8.1). The pipeline confirmed: UDP streaming (211 packets/15s), 8-dim feature extraction, batched HTTPS ingest (4 batches of 5 vectors), and witness chain integrity (193 entries, SHA-256 verified). Multi-node deployment (Phase 4 of ADR-069) is the next step toward the full swarm architecture described here.

### Future Work

- **Seed-initiated WASM push** — Seed distributes WASM modules to all nodes via their OTA endpoints
- **mDNS auto-discovery** — nodes find Seed without provisioned URL
- **Mesh fallback** — ESP-NOW peer-to-peer when WiFi is down
- **Multi-Seed federation** — multiple Seeds for multi-floor/multi-building deployments
- **Seed dashboard** — web UI on the Seed showing live swarm map with per-zone happiness
