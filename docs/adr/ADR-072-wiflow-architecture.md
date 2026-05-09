# ADR-072: WiFlow Pose Estimation Architecture

- **Status**: Proposed
- **Date**: 2026-04-02
- **Deciders**: ruv
- **Relates to**: ADR-071 (ruvllm Training Pipeline), ADR-070 (Self-Supervised Pretraining), ADR-024 (Contrastive CSI Embedding / AETHER), ADR-069 (Cognitum Seed CSI Pipeline)

## Context

The WiFi-DensePose project needs a neural architecture that can convert raw CSI amplitude
data into 17-keypoint COCO pose estimates. The existing `train-ruvllm.js` pipeline uses a
simple 2-layer FC encoder (8 -> 64 -> 128) that produces contrastive embeddings for
presence detection but cannot output spatial keypoint coordinates.

We evaluated published WiFi-based pose estimation architectures:

| Architecture | Params | Input | Key Innovation | Publication |
|-------------|--------|-------|---------------|-------------|
| **WiFlow** | 4.82M | 540x20 | TCN + AsymConv + Axial Attention | arXiv:2602.08661 |
| WiPose | 11.2M | 3x3x30x20 | 3D CNN + heatmap regression | CVPR 2021 |
| MetaFi++ | 8.6M | 114x30x20 | Transformer + meta-learning | NeurIPS 2023 |
| Person-in-WiFi 3D | 15.3M | Multi-antenna | Deformable attention + 3D | CVPR 2024 |

WiFlow is the lightest published SOTA architecture, designed specifically for commercial
WiFi hardware. Its key advantage is operating on CSI amplitude only (no phase), which
is critical for ESP32-S3 where phase calibration is unreliable.

### Why WiFlow

1. **Lightest SOTA**: 4.82M parameters at original scale; our adaptation targets ~2.5M
2. **Amplitude-only**: Discards phase, which is noisy on consumer hardware
3. **Published architecture**: Fully specified in arXiv:2602.08661, reproducible
4. **Temporal modeling**: TCN with dilated causal convolutions captures motion dynamics
5. **Efficient attention**: Axial attention reduces O(H^2W^2) to O(H^2W + HW^2)
6. **Proven on commercial WiFi**: Validated on commodity Intel 5300 and Atheros hardware

## Decision

Implement the WiFlow architecture in pure JavaScript (ruvllm native) with the following
adaptations for our ESP32 single TX/RX deployment.

### Architecture Overview

```
CSI Amplitude [128, 20]
        |
   Stage 1: TCN (Dilated Causal Conv)
   dilation = (1, 2, 4, 8), kernel = 7
   128 -> 256 -> 192 -> 128 channels
        |
   Stage 2: Asymmetric Conv Encoder
   1xk conv (k=3), stride (1,2)
   [1, 128, 20] -> [256, 8, 20]
        |
   Stage 3: Axial Self-Attention
   Width (temporal): 8 heads
   Height (feature): 8 heads
        |
   Decoder: Adaptive Avg Pool + Linear
   [256, 8, 20] -> pool -> [2048] -> [17, 2]
        |
   17 COCO Keypoints [x, y] in [0, 1]
```

### Our Adaptation vs Original WiFlow

| Aspect | WiFlow Original | Our Adaptation | Reason |
|--------|----------------|----------------|--------|
| Input channels | 540 (18 links x 30 SC) | 128 (1 TX x 1 RX x 128 SC) | Single ESP32 link |
| Time steps | 20 | 20 | Same |
| TCN channels | 540 -> 256 -> 128 -> 64 | 128 -> 256 -> 192 -> 128 | Proportional reduction |
| Spatial blocks | 4 (stride 2) | 4 (stride 2) | Same |
| Attention heads | 8 | 8 | Same |
| Parameters | 4.82M | ~1.8M | Fewer input channels |
| Input type | Amplitude only | Amplitude only | Same |
| Output | 17 x 2 | 17 x 2 | Same |

### Parameter Budget Breakdown

| Stage | Parameters | % of Total |
|-------|-----------|------------|
| TCN (4 blocks, k=7, d=1,2,4,8) | ~969K | 54% |
| Asymmetric Conv (4 blocks, 1x3, stride 2) | ~174K | 10% |
| Axial Attention (width + height, 8 heads) | ~592K | 33% |
| Pose Decoder (pool + linear -> 17x2) | ~70K | 4% |
| **Total** | **~1.8M** | **100%** |

### Loss Function

```
L = L_H + 0.2 * L_B

L_H = SmoothL1(predicted, target, beta=0.1)
L_B = (1/14) * sum_b (bone_length_b - prior_b)^2
```

14 bone connections enforce anatomical constraints:
- Nose-eye (x2): 0.06
- Eye-ear (x2): 0.06
- Shoulder-elbow (x2): 0.15
- Elbow-wrist (x2): 0.13
- Shoulder-hip (x2): 0.26
- Hip-knee (x2): 0.25
- Knee-ankle (x2): 0.25
- Shoulder width: 0.20

All lengths normalized to person height.

### Training Strategy (Camera-Free Pipeline)

Since we have no ground-truth pose labels from cameras, training proceeds in three phases:

#### Phase 1: Contrastive Pretraining
- Temporal triplets: adjacent windows are positive pairs, distant windows are negative
- Cross-node triplets: same-time windows from different ESP32 nodes are positive
- Uses ruvllm `ContrastiveTrainer` with triplet + InfoNCE loss
- Learns a representation where similar CSI states cluster together

#### Phase 2: Pose Proxy Training
- Generate coarse pose proxies from vitals data:
  - Person detected (presence > 0.3): place standing skeleton at center
  - High motion: perturb limb positions proportional to motion energy
  - Breathing: add micro-oscillation to torso keypoints
- Train with SmoothL1 + bone constraint loss
- Confidence-weighted updates (higher presence = stronger gradient)

#### Phase 3: Self-Refinement (Future)
- Multi-node consistency: same person seen from different nodes should produce
  consistent pose after geometric transform
- Temporal smoothness: adjacent frames should produce similar poses
- Bone constraint tightening: gradually reduce tolerance

### Integration with Existing Pipeline

```
train-ruvllm.js (ADR-071)        train-wiflow.js (ADR-072)
  |                                  |
  | 8-dim features                   | 128-dim raw CSI amplitude
  | -> 128-dim embedding             | -> 17x2 keypoint coordinates
  | -> presence/activity/vitals      | -> bone-constrained pose
  |                                  |
  +-- ContrastiveTrainer -----+------+
  +-- TrainingPipeline -------+------+
  +-- LoRA per-node ----------+------+
  +-- TurboQuant quantize ----+------+
  +-- SafeTensors export -----+------+
```

Both pipelines share the ruvllm infrastructure; WiFlow adds the deeper architecture
for direct pose regression while the simple encoder handles embedding tasks.

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| PCK@20 | > 80% | On lab data with 2+ nodes |
| Forward latency | < 50ms | Pi Zero 2W at INT8 |
| Model size (INT8) | < 2 MB | TurboQuant |
| Bone violation rate | < 10% | 50% tolerance |
| Temporal jitter | < 3cm | Exponential smoothing |

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Single TX/RX has less spatial info than 18 links | High | 2-node multi-static compensates; cross-node fusion from ADR-029 |
| Camera-free labels are coarse | Medium | Bone constraints enforce anatomy; contrastive pretrain provides structure |
| Pure JS too slow for real-time | Medium | INT8 quantization; axial attention is O(H^2W+HW^2) not O(H^2W^2) |
| Overfitting with ~5K frames | Medium | Temporal augmentation + noise + cross-node interpolation |
| Phase not available (amplitude-only) | Low | WiFlow was designed amplitude-only; not a limitation |

## Consequences

### Positive
- Proven SOTA architecture adapted to our hardware constraints
- Pure JavaScript implementation runs everywhere ruvllm runs (Node.js, browser WASM)
- Bone constraints enforce physically plausible outputs even with noisy inputs
- Shares training infrastructure with existing ruvllm pipeline
- Modular: each stage (TCN, AsymConv, Axial, Decoder) is independently testable

### Negative
- ~1.8M parameters is 193x larger than simple CsiEncoder (9,344 params)
- Forward pass is slower (~50ms vs <1ms for simple encoder)
- Camera-free training will produce lower accuracy than supervised WiFlow
- No ground-truth PCK evaluation possible without camera labels
- Axial attention is O(N^2) within each axis, limiting scalability

### Neutral
- FLOPs dominated by TCN (~48%) due to dilated convolutions
- INT8 quantization brings model to ~1.7MB, viable for edge deployment
- Architecture is fixed (no NAS); future work could explore lighter variants

## Implementation

### Files Created

| File | Purpose |
|------|---------|
| `scripts/wiflow-model.js` | WiFlow architecture (all stages, loss, metrics) |
| `scripts/train-wiflow.js` | Training pipeline (contrastive + pose proxy + LoRA + quant) |
| `scripts/benchmark-wiflow.js` | Benchmarking (latency, params, FLOPs, memory, quality) |
| `docs/adr/ADR-072-wiflow-architecture.md` | This document |

### Usage

```bash
# Train on collected data
node scripts/train-wiflow.js --data data/recordings/pretrain-*.csi.jsonl

# Train with more epochs and custom output
node scripts/train-wiflow.js --data data/recordings/*.csi.jsonl --epochs 50 --output models/wiflow-v2

# Contrastive pretraining only (no labels needed)
node scripts/train-wiflow.js --data data/recordings/*.csi.jsonl --contrastive-only

# Benchmark
node scripts/benchmark-wiflow.js

# Benchmark with trained model
node scripts/benchmark-wiflow.js --model models/wiflow-v1
```

### Dependencies

- ruvllm (vendored at `vendor/ruvector/npm/packages/ruvllm/src/`)
  - `ContrastiveTrainer`, `tripletLoss`, `infoNCELoss`, `computeGradient`
  - `TrainingPipeline`
  - `LoraAdapter`, `LoraManager`
  - `EwcManager`
  - `ModelExporter`, `SafeTensorsWriter`
- No external ML frameworks (no PyTorch, no TensorFlow, no ONNX Runtime)

## References

- WiFlow: arXiv:2602.08661
- COCO Keypoints: https://cocodataset.org/#keypoints-2020
- Axial Attention: Wang et al., "Axial-DeepLab", ECCV 2020
- TCN: Bai et al., "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling", 2018
