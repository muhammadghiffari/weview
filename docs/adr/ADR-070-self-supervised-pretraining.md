# ADR-070: Self-Supervised Pretraining from Live ESP32 CSI + Cognitum Seed

| Field      | Value                                                    |
|------------|----------------------------------------------------------|
| Status     | Accepted                                                 |
| Date       | 2026-04-02                                               |
| Authors    | rUv, claude-flow                                         |
| Drivers    | README limitation "No pre-trained model weights provided"|
| Related    | ADR-069 (Cognitum Seed pipeline), ADR-027 (MERIDIAN), ADR-024 (AETHER contrastive), ADR-015 (MM-Fi dataset) |

## Context

The README lists "No pre-trained model weights are provided; training from scratch is required" as a known limitation. Users must collect their own CSI dataset and train from scratch, which is a significant barrier to adoption.

We now have the infrastructure to generate pre-trained weights directly from live hardware:

- **2 ESP32-S3 nodes** (COM8 node_id=2 at 192.168.1.104, COM9 node_id=1 at 192.168.1.105) streaming CSI + vitals + 8-dim feature vectors at 1 Hz each
- **Cognitum Seed** (Pi Zero 2 W) with RVF vector store, kNN search, witness chain, and environmental sensors (BME280, PIR, vibration)
- **Recording API** in sensing-server (`POST /api/v1/recording/start`) that saves CSI frames to `.csi.jsonl`
- **Self-supervised training** via `rapid_adapt.rs` (contrastive TTT + entropy minimization)
- **AETHER contrastive embeddings** (ADR-024) for environment-independent representations

### Why Self-Supervised?

No cameras or labels are needed. The system learns from:

1. **Temporal coherence** — Frames close in time should have similar embeddings (positive pairs), frames far apart should differ (negative pairs)
2. **Multi-node consistency** — The same person seen from 2 nodes should produce correlated features, different people should produce decorrelated features
3. **Cognitum Seed ground truth** — PIR sensor, BME280 environment changes, and kNN cluster transitions provide weak supervision without human labeling
4. **Physical constraints** — Breathing 6-30 BPM, heart rate 40-150 BPM, person count 0-4, RSSI physics

## Decision

Implement a 4-phase pretraining pipeline that collects CSI from 2 ESP32 nodes, stores feature vectors in the Cognitum Seed, and produces distributable pre-trained weights.

### Phase 1: Data Collection (30 min)

Capture labeled scenarios using the sensing-server recording API and Cognitum Seed:

| Scenario | Duration | Label | Activity |
|----------|----------|-------|----------|
| Empty room | 5 min | `empty` | No one present, establish baseline |
| 1 person stationary | 5 min | `1p-still` | Sit at desk, normal breathing |
| 1 person walking | 5 min | `1p-walk` | Walk around room, varied paths |
| 1 person varied | 5 min | `1p-varied` | Stand, sit, wave arms, turn |
| 2 people | 5 min | `2p` | Both moving in room |
| Transitions | 5 min | `transitions` | Enter/exit room, appear/disappear |

**Data rate per scenario:**
- 2 nodes × 100 Hz CSI = 200 frames/sec = 60,000 frames per 5 min
- 2 nodes × 1 Hz features = 2 vectors/sec = 600 vectors per 5 min
- Total: 360,000 CSI frames + 3,600 feature vectors per collection run

**Cognitum Seed role:**
- Stores all feature vectors with witness chain attestation
- PIR sensor provides binary presence ground truth
- BME280 tracks environmental conditions during collection
- kNN graph clusters naturally emerge from the vector distribution

### Phase 2: Contrastive Pretraining

Train a contrastive encoder on the collected CSI data:

```
Input: Raw CSI frame (128 subcarriers × 2 I/Q = 256 features)
       ↓
    TCN temporal encoder (3 layers, kernel=7)
       ↓
    Projection head → 128-dim embedding
       ↓
    Contrastive loss (InfoNCE):
      positive: frames within 0.5s window from same node
      negative: frames >5s apart or from different scenario
      cross-node positive: same timestamp, different node
```

**Self-supervised signals:**
- Temporal adjacency (frames within 500ms = positive pair)
- Cross-node agreement (same person seen from 2 viewpoints)
- PIR consistency (embedding should cluster by PIR state)
- Scenario boundary (embeddings should shift at label transitions)

### Phase 3: Downstream Head Training

Attach lightweight heads for each task:

| Head | Architecture | Output | Supervision |
|------|-------------|--------|-------------|
| Presence | Linear(128→1) + sigmoid | 0.0-1.0 | PIR sensor (free) |
| Person count | Linear(128→4) + softmax | 0-3 people | Scenario labels |
| Activity | Linear(128→4) + softmax | still/walk/varied/empty | Scenario labels |
| Vital signs | Linear(128→2) | BR, HR (BPM) | ESP32 edge vitals |

### Phase 4: Package & Distribute

Produce distributable artifacts:

| Artifact | Format | Size | Description |
|----------|--------|------|-------------|
| `pretrained-encoder.onnx` | ONNX | ~2 MB | Contrastive encoder (TCN backbone) |
| `pretrained-heads.onnx` | ONNX | ~100 KB | Task-specific heads |
| `pretrained.rvf` | RVF | ~500 KB | RuVector format with metadata |
| `room-profiles.json` | JSON | ~10 KB | Environment calibration profiles |
| `collection-witness.json` | JSON | ~5 KB | Seed witness chain attestation proving data provenance |

Include in GitHub release alongside firmware binaries. Users download and run:

```bash
# Use pre-trained model (no training needed)
cargo run -p wifi-densepose-sensing-server -- --model pretrained.rvf --http-port 3000
```

## Hardware Setup

```
                    192.168.1.20 (Host laptop)
                    ┌──────────────────────────┐
                    │  sensing-server           │
                    │    Recording API          │
                    │    Training pipeline      │
                    │                           │
                    │  seed_csi_bridge.py       │
                    │    Feature → Seed ingest  │
                    └────┬──────────┬───────────┘
                         │          │
          UDP:5006       │          │  HTTPS:8443
     ┌───────────────────┤          ├───────────────┐
     │                   │          │               │
     ▼                   ▼          ▼               │
┌──────────┐    ┌──────────┐    ┌──────────────┐    │
│ ESP32 #1 │    │ ESP32 #2 │    │Cognitum Seed │◄───┘
│ COM9     │    │ COM8     │    │ Pi Zero 2W   │
│ node=1   │    │ node=2   │    │ USB          │
│ .1.105   │    │ .1.104   │    │ .42.1/8443   │
│ v0.5.4   │    │ v0.5.4   │    │ v0.8.1       │
└──────────┘    └──────────┘    │ PIR, BME280  │
                                │ RVF store    │
                                │ Witness chain│
                                └──────────────┘
```

## Data Collection Protocol

### Step 1: Start Seed ingest (background)

```bash
export SEED_TOKEN="your-token"
python scripts/seed_csi_bridge.py \
  --seed-url https://169.254.42.1:8443 --token "$SEED_TOKEN" \
  --udp-port 5006 --batch-size 10 --validate &
```

### Step 2: Start sensing-server with recording

```bash
cargo run -p wifi-densepose-sensing-server -- \
  --source esp32 --udp-port 5006 --http-port 3000
```

### Step 3: Record each scenario

```bash
# Empty room (leave room for 5 min)
curl -X POST http://localhost:3000/api/v1/recording/start \
  -H 'Content-Type: application/json' \
  -d '{"session_name":"pretrain-empty","label":"empty","duration_secs":300}'

# 1 person stationary (sit at desk for 5 min)
curl -X POST http://localhost:3000/api/v1/recording/start \
  -d '{"session_name":"pretrain-1p-still","label":"1p-still","duration_secs":300}'

# ... repeat for each scenario
```

### Step 4: Verify with Seed

```bash
python scripts/seed_csi_bridge.py --token "$SEED_TOKEN" --stats
# Should show 3,600+ vectors from the collection run
```

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 2 nodes insufficient for spatial diversity | Medium | Lower pretraining quality | Place nodes 3-5m apart at different heights |
| PIR sensor has limited range | Low | Weak presence labels | BME280 temp changes + kNN clusters as backup |
| Contrastive pretraining collapses | Low | Useless embeddings | Temperature scheduling, hard negative mining |
| Model too large for ESP32 inference | N/A | N/A | Inference on host/Seed, not on ESP32 |
| Room-specific overfitting | Medium | Poor generalization | MERIDIAN domain randomization (ADR-027), LoRA adaptation |

## Consequences

### Positive
- Users get working model out of the box — no training needed
- Witness chain proves data provenance (when/where/which hardware)
- Pre-trained encoder transfers to new environments via LoRA fine-tuning
- Removes the #1 adoption barrier from the README

### Negative
- 30 min of manual data collection per pretraining run
- Pre-trained weights are room-specific without adaptation
- ONNX runtime dependency for inference
