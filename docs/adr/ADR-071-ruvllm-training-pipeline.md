# ADR-071: ruvllm Training Pipeline for CSI Sensing Models

- **Status**: Proposed
- **Date**: 2026-04-02
- **Deciders**: ruv
- **Relates to**: ADR-069 (Cognitum Seed CSI Pipeline), ADR-070 (Self-Supervised Pretraining), ADR-024 (Contrastive CSI Embedding / AETHER), ADR-016 (RuVector Training Pipeline)

## Context

The WiFi-DensePose project needs a training pipeline to convert collected CSI data
(`.csi.jsonl` frames from ESP32 nodes) into deployable models for presence detection,
activity classification, and vital sign estimation.

Previous ADRs established the data collection protocol (ADR-070) and Cognitum Seed
inference target (ADR-069). What was missing was the actual training, refinement,
quantization, and export pipeline connecting raw CSI recordings to deployable models.

### Why ruvllm instead of PyTorch

| Criterion | ruvllm | PyTorch | ONNX Runtime |
|-----------|--------|---------|--------------|
| Runtime dependency | Node.js only | Python + CUDA + pip | C++ runtime |
| Install size | ~5 MB (npm) | ~2 GB (torch+cuda) | ~50 MB |
| SONA adaptation | <1ms native | N/A | N/A |
| Quantization | 2/4/8-bit TurboQuant | INT8/FP16 (separate tool) | INT8 only |
| LoRA fine-tuning | Built-in LoraAdapter | Requires PEFT library | N/A |
| EWC protection | Built-in EwcManager | Manual implementation | N/A |
| SafeTensors export | Native SafeTensorsWriter | Via safetensors library | N/A |
| Contrastive training | Built-in ContrastiveTrainer | Manual triplet loss | N/A |
| Edge deployment | ESP32, Pi Zero, browser | GPU servers only | ARM (limited) |
| M4 Pro performance | 88-135 tok/s native | ~30 tok/s (MPS) | ~50 tok/s |
| Ecosystem integration | RuVector, Cognitum Seed | Standalone | Standalone |

The ruvllm package (`@ruvector/ruvllm` v2.5.4) provides the complete training
lifecycle in a single dependency: contrastive pretraining, task head training,
LoRA refinement, EWC consolidation, quantization, and SafeTensors/RVF export.
No Python dependency means the entire pipeline runs on the same Node.js runtime
as the Cognitum Seed inference engine.

## Decision

Use ruvllm's `ContrastiveTrainer`, `TrainingPipeline`, `LoraAdapter`, `EwcManager`,
`SafeTensorsWriter`, and `ModelExporter` for the complete CSI model training lifecycle.

### Training Phases

The pipeline executes five sequential phases:

#### Phase 1: Contrastive Pretraining

Learns an embedding space where temporally and spatially similar CSI states are close
and dissimilar states are far apart.

- **Encoder architecture**: 8-dim CSI feature vector -> 64-dim hidden (ReLU) -> 128-dim embedding (L2-normalized)
- **Loss functions**: Triplet loss (margin=0.3) + InfoNCE (temperature=0.07)
- **Triplet strategies**:
  - Temporal positive: frames within 1 second (same environment state)
  - Temporal negative: frames >30 seconds apart (different state)
  - Cross-node positive: same timestamp from different ESP32 nodes (same person, different viewpoint)
  - Cross-node negative: different timestamp + different node
  - Hard negatives: frames near motion energy transition boundaries
- **Hyperparameters**: 20 epochs, batch size 32, hard negative ratio 0.7
- **Implementation**: `ContrastiveTrainer.addTriplet()` + `.train()`

#### Phase 2: Task Head Training

Trains supervised heads on top of the frozen embedding for specific sensing tasks.

- **Presence head**: 128 -> 1 (sigmoid), threshold at presence_score > 0.3
- **Activity head**: 128 -> 3 (softmax: still/moving/empty), derived from motion_energy thresholds
- **Vitals head**: 128 -> 2 (linear: breathing BPM, heart rate BPM), normalized targets
- **Implementation**: `TrainingPipeline.addData()` + `.train()` with cosine LR scheduler,
  early stopping (patience=5), and quality-weighted MSE loss

#### Phase 3: LoRA Refinement

Per-node LoRA adapters for room-specific adaptation without forgetting the base model.

- **Configuration**: rank=4, alpha=8, dropout=0.1
- **Per-node training**: Each ESP32 node gets its own LoRA adapter trained on
  node-specific data with reduced learning rate (0.5x base)
- **Implementation**: `LoraManager.create()` for each node, `TrainingPipeline` with
  `LoraAdapter` passed to constructor

#### Phase 4: Quantization (TurboQuant)

Reduces model size for edge deployment with minimal quality loss.

| Bit Width | Compression | Typical RMSE | Target Device |
|-----------|-------------|-------------|---------------|
| 8-bit | 4x | <0.001 | Cognitum Seed (Pi Zero) |
| 4-bit | 8x | <0.01 | Standard edge inference |
| 2-bit | 16x | <0.05 | ESP32-S3 feature extraction |

- **Method**: Uniform affine quantization with scale/zero-point per tensor
- **Quality validation**: RMSE between original fp32 and dequantized weights

#### Phase 5: EWC Consolidation

Elastic Weight Consolidation prevents catastrophic forgetting when the model
is later fine-tuned on new room data or updated CSI conditions.

- **Fisher information**: Computed from training data gradients
- **Lambda**: 2000 (base), 3000 (per-node)
- **Tasks registered**: Base pretraining + one per ESP32 node
- **Implementation**: `EwcManager.registerTask()` for each training phase

### Data Pipeline

```
.csi.jsonl files
    |
    v
Parse frames: feature (8-dim), vitals, raw CSI
    |
    v
Generate contrastive triplets (temporal, cross-node, hard negatives)
    |
    v
Encode through CsiEncoder (8 -> 64 -> 128)
    |
    v
Phase 1: ContrastiveTrainer (triplet + InfoNCE loss)
    |
    v
Phase 2: TrainingPipeline (presence + activity + vitals heads)
    |
    v
Phase 3: LoRA per-node refinement
    |
    v
Phase 4: TurboQuant (2/4/8-bit quantization)
    |
    v
Phase 5: EWC consolidation
    |
    v
Export: SafeTensors, JSON config, RVF manifest, per-node LoRA adapters
```

### Export Formats

| Format | File | Consumer |
|--------|------|----------|
| SafeTensors | `model.safetensors` | HuggingFace ecosystem, general inference |
| JSON config | `config.json` | Model loading metadata |
| JSON model | `model.json` | Full model state for Node.js loading |
| Quantized binaries | `quantized/model-q{2,4,8}.bin` | Edge deployment |
| Per-node LoRA | `lora/node-{id}.json` | Room-specific adaptation |
| RVF manifest | `model.rvf.jsonl` | Cognitum Seed ingest (ADR-069) |
| Training metrics | `training-metrics.json` | Dashboards, CI validation |

### Hardware Targets

| Device | Role | Quantization | Expected Latency |
|--------|------|-------------|-----------------|
| Mac Mini M4 Pro | Training (primary) | fp32 | <5 min total |
| Cognitum Seed Pi Zero | Inference | 4-bit / 8-bit | <10 ms per frame |
| ESP32-S3 | Feature extraction only | 2-bit (encoder weights) | <5 ms per frame |
| Browser (WASM) | Visualization | 4-bit | <20 ms per frame |

### Performance Targets

| Metric | Target | Measured |
|--------|--------|----------|
| Training time (5,783 frames, M4 Pro) | <5 min | TBD |
| Inference latency (M4 Pro) | <1 ms | TBD |
| Inference latency (Pi Zero) | <10 ms | TBD |
| SONA adaptation | <1 ms | <0.05 ms (ruvllm spec) |
| Presence detection accuracy | >85% | TBD |
| 4-bit quality loss (RMSE) | <0.01 | TBD |
| 2-bit quality loss (RMSE) | <0.05 | TBD |

## Consequences

### Positive

- **Zero Python dependency**: The entire training and inference pipeline runs on
  Node.js, eliminating Python/CUDA/pip dependency management on training and
  deployment targets.
- **Integrated lifecycle**: Contrastive pretraining, task heads, LoRA refinement,
  EWC consolidation, and quantization in a single script using one library.
- **Edge-first**: 2-bit quantization enables running the encoder on ESP32-S3.
  4-bit quantization fits comfortably on Cognitum Seed Pi Zero.
- **Continual learning**: EWC protection means the model can be updated with new
  room data without losing previously learned patterns.
- **Per-node adaptation**: LoRA adapters allow room-specific fine-tuning with
  minimal storage overhead (rank-4 adapter ~2KB per node).
- **HuggingFace compatibility**: SafeTensors export enables sharing models on the
  HuggingFace Hub and loading in other frameworks.
- **Reproducibility**: Seeded encoder initialization and deterministic data pipeline
  ensure reproducible training runs.

### Negative

- **No GPU acceleration**: ruvllm's JS training loop does not use GPU compute.
  For the small model sizes in CSI sensing (8->64->128), this is acceptable
  (~seconds on M4 Pro), but would not scale to large vision models.
- **Simplified backpropagation**: The LoRA backward pass and contrastive training
  use approximate gradient updates rather than full automatic differentiation.
  Sufficient for the target model sizes but not equivalent to PyTorch autograd.
- **Quantization is post-training only**: No quantization-aware training (QAT).
  For 4-bit and 8-bit this produces acceptable quality loss; 2-bit may need
  QAT in future if quality degrades.

### Risks

- **Quality ceiling**: The simplified training may produce lower accuracy than a
  PyTorch-trained equivalent. Mitigated by: (a) the model is small enough that
  the training loop converges quickly, (b) SONA adaptation can compensate at
  inference time, (c) we can switch to PyTorch for training only if needed
  while keeping ruvllm for inference.
- **ruvllm API stability**: The library is at v2.5.4 with active development.
  Mitigated by vendoring the package in `vendor/ruvector/npm/packages/ruvllm/`.

## Implementation

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train-ruvllm.js` | Full 5-phase training pipeline |
| `scripts/benchmark-ruvllm.js` | Model benchmarking (latency, quality, accuracy) |

### Usage

```bash
# Train on collected CSI data
node scripts/train-ruvllm.js \
  --data data/recordings/pretrain-1775182186.csi.jsonl \
  --output models/csi-v1 \
  --epochs 20

# Train with benchmark
node scripts/train-ruvllm.js \
  --data data/recordings/pretrain-*.csi.jsonl \
  --output models/csi-v1 \
  --benchmark

# Standalone benchmark
node scripts/benchmark-ruvllm.js \
  --model models/csi-v1 \
  --data data/recordings/pretrain-*.csi.jsonl \
  --samples 5000 \
  --json
```

### Output Structure

```
models/csi-v1/
  model.safetensors          # SafeTensors (HuggingFace compatible)
  config.json                # Model configuration
  model.json                 # Full JSON model state
  model.rvf.jsonl            # RVF manifest for Cognitum Seed
  training-metrics.json      # Training loss curves, timing, config
  contrastive/
    triplets.jsonl           # Contrastive training pairs
    triplets.csv             # CSV format for analysis
    embeddings.json          # Embedding matrices
  quantized/
    model-q2.bin             # 2-bit quantized (ESP32 edge)
    model-q4.bin             # 4-bit quantized (Pi Zero default)
    model-q8.bin             # 8-bit quantized (high quality)
  lora/
    node-1.json              # LoRA adapter for ESP32 node 1
    node-2.json              # LoRA adapter for ESP32 node 2
```

## Camera-Free Supervision

### Motivation

Traditional WiFi-based pose estimation (WiFlow, Person-in-WiFi) requires camera-supervised
training: a camera captures ground-truth poses during CSI collection, and the model learns
to map CSI to those poses. This creates a deployment paradox — the camera is needed for
training but the whole point of WiFi sensing is to avoid cameras.

The camera-free pipeline (`scripts/train-camera-free.js`) replaces camera supervision with
10 sensor signals from the Cognitum Seed and 2 ESP32 nodes, generating weak labels through
sensor fusion.

### 10 Supervision Signals (No Camera)

| # | Signal | Source | Provides |
|---|--------|--------|----------|
| 1 | PIR sensor | Seed GPIO 6 | Binary presence ground truth |
| 2 | BME280 temperature | Seed I2C 0x76 | Occupancy proxy (temp rises with people) |
| 3 | BME280 humidity | Seed I2C 0x76 | Breathing confirmation / zone |
| 4 | Cross-node RSSI | 2 ESP32 nodes | Rough XY position (differential triangulation) |
| 5 | Vitals stability | ESP32 CSI | HR/BR variance indicates activity level |
| 6 | Temporal CSI patterns | ESP32 CSI | Periodic=walking, stable=sitting, flat=empty |
| 7 | kNN cluster labels | Seed vector store | Natural groupings in embedding space |
| 8 | Boundary fragility | Seed Stoer-Wagner | Regime change detection (entry/exit/activity) |
| 9 | Reed switch | Seed GPIO 5 | Door open/close events |
| 10 | Vibration sensor | Seed GPIO 13 | Footstep detection |

### Camera-Free Training Phases

The pipeline extends the base 5 phases with camera-free-specific phases:

```
Phase 0: Multi-Modal Data Collection
  ├── UDP port 5006 → ESP32 CSI features + vitals
  ├── HTTPS → Seed sensor embeddings (45-dim, every 100ms)
  ├── HTTPS → Seed boundary/coherence (every 10s)
  └── Build synchronized MultiModalFrame timeline

Phase 1: Weak Label Generation
  ├── Presence: PIR || CSI_presence > 0.3 || temp_rising > 0.1°C/min
  ├── Position: RSSI differential → 5×5 grid (25 zones)
  ├── Activity: CSI variance + FFT periodicity → stationary/walking/gesture/empty
  ├── Occupancy: max(node1_persons, node2_persons) validated by temp
  ├── Body region: upper/lower subcarrier groups → which body part moves
  ├── Entry/exit: reed_switch + PIR transition + boundary fragility spike
  ├── Breathing zone: humidity change rate → person location
  └── Pose proxy: 5-keypoint coarse pose from RSSI + subcarrier asymmetry + vibration

Phase 2: Enhanced Contrastive Pretraining
  ├── Base triplets (temporal, cross-node, transition, scenario boundary)
  ├── Sensor-verified negatives: PIR=0 vs PIR=1 must differ
  ├── Activity boundary: before/after fragility spike must differ
  └── Cross-modal: CSI embedding ≈ Seed embedding for same state

Phase 3: Pose Proxy Training (5-keypoint)
  ├── Head: RSSI centroid between 2 nodes
  ├── Hands: per-subcarrier variance asymmetry (left/right from 2 nodes)
  ├── Feet: vibration sensor + RSSI ground reflection
  └── Skeleton physics constraints (anthropometric bone length limits)

Phase 4: 17-Keypoint Interpolation
  ├── Shoulders = 0.3 × head + 0.7 × hands
  ├── Elbows = midpoint(shoulder, hand)
  ├── Hips = midpoint(head, feet)
  ├── Knees = midpoint(hip, foot)
  ├── Face = derived from head position
  └── Iterative bone length constraint projection (3 iterations)

Phase 5: Self-Refinement Loop (3 rounds)
  ├── Run inference on all collected data
  ├── Keep predictions where temporal consistency confidence > 0.8
  ├── Use as pseudo-labels for next training round
  └── Decaying learning rate per round (diminishing returns)
```

### Seed API Endpoints Used

| Endpoint | Data | Collection Rate |
|----------|------|----------------|
| `GET /api/v1/sensor/stream` | SSE sensor readings | Continuous (100ms) |
| `GET /api/v1/sensor/embedding/latest` | 45-dim sensor embedding | Per-frame |
| `GET /api/v1/boundary` | Fragility score | Every 10s |
| `GET /api/v1/coherence/profile` | Temporal phase boundaries | Every 10s |
| `GET /api/v1/store/query` | kNN similarity search | On demand |
| `POST /api/v1/boundary/recompute` | Trigger analysis | On regime change |

### Graceful Degradation

The pipeline works with or without the Cognitum Seed:

| Mode | Signals | Pose Quality |
|------|---------|-------------|
| Full (Seed + 2 ESP32) | 10 signals | 5-keypoint trained, 17-keypoint interpolated |
| CSI-only (2 ESP32) | 3 signals (RSSI, vitals, temporal) | Coarser position/activity only |
| Single node | 2 signals (vitals, temporal) | Presence + activity only |

When the Seed API is unreachable, the pipeline automatically falls back to
CSI-only training, producing the same output format (SafeTensors, HuggingFace,
quantized) with reduced label quality.

### Output Format

Same as the base pipeline (SafeTensors + HuggingFace compatible), plus:

| File | Description |
|------|-------------|
| `pose-decoder.json` | 5-keypoint pose decoder weights |
| `model.rvf.jsonl` | Extended with `camera_free_supervision` record |
| `training-metrics.json` | Includes weak label stats and multi-modal triplet counts |

### Usage

```bash
# Full pipeline with Seed
node scripts/train-camera-free.js \
  --data data/recordings/pretrain-*.csi.jsonl \
  --seed-url https://169.254.42.1:8443 \
  --output models/csi-camerafree-v1

# CSI-only (no Seed)
node scripts/train-camera-free.js \
  --data data/recordings/pretrain-*.csi.jsonl \
  --no-seed \
  --output models/csi-camerafree-v1

# With benchmark
node scripts/train-camera-free.js \
  --data data/recordings/*.csi.jsonl \
  --benchmark
```

## References

- [ruvllm source](vendor/ruvector/npm/packages/ruvllm/) — v2.5.4
- [ADR-069](ADR-069-cognitum-seed-csi-pipeline.md) — Cognitum Seed CSI Pipeline
- [ADR-070](ADR-070-self-supervised-pretraining.md) — Self-Supervised Pretraining Protocol
- [ADR-024](ADR-024-contrastive-csi-embedding.md) — Contrastive CSI Embedding / AETHER
- [ADR-016](ADR-016-ruvector-training-pipeline.md) — RuVector Training Pipeline Integration
