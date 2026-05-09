# SOTA WiFi Sensing for Edge Pose Estimation (2024-2026 Update)

**Date:** 2026-04-02
**Focus:** New architectures, lightweight models, edge deployment, ESP32+Pi Zero inference
**Complements:** `wifi-sensing-ruvector-sota-2026.md` (February 2026 survey)

---

## 1. New Architectures Since Last Survey

### 1.1 WiFlow: Lightweight Continuous Pose Estimation (February 2026)

**Paper:** WiFlow: A Lightweight WiFi-based Continuous Human Pose Estimation Network with Spatio-Temporal Feature Decoupling ([arXiv:2602.08661](https://arxiv.org/html/2602.08661))

WiFlow is the most directly relevant architecture for our ESP32 + Pi Zero deployment target.

#### Architecture

Three-stage encoder-decoder with spatio-temporal decoupling:

**Stage 1: Temporal Encoder (TCN)**
- Dilated causal convolution with exponentially growing dilation factors (1, 2, 4, 8)
- Input: 540x20 tensor (18 antenna links x 30 subcarriers = 540 features, 20 time steps)
- Progressive channel compression: 540 -> 440 -> 340 -> 240
- Preserves temporal causality while achieving full receptive field coverage

**Stage 2: Spatial Encoder (Asymmetric Convolution)**
- 1xk kernels operating only in the subcarrier dimension
- 4 residual blocks: 8 -> 16 -> 32 -> 64 channels
- Subcarrier compression: 240 -> 120 -> 60 -> 30 -> 15
- Stride (1,2) downsampling -- no pooling layers

**Stage 3: Axial Self-Attention**
- Two-stage axial attention reduces complexity from O(H^2 W^2) to O(H^2 W + HW^2)
- Stage one: width direction (temporal axis), 8 groups
- Stage two: height direction (keypoint axis)
- Input reshaped to (B x K) x C x T for first stage

**Decoder:**
- Adaptive average pooling instead of fully connected layers
- Direct coordinate regression to 2D keypoint positions

#### Key Metrics

| Metric | WiFlow | WPformer | WiSPPN |
|--------|--------|----------|--------|
| Parameters | **4.82M** | 10.04M | 121.5M |
| FLOPs | **0.47B** | 35.00B | 338.45B |
| PCK@20 (random split) | **97.00%** | 70.02% | 85.87% |
| MPJPE (random split) | **0.008m** | 0.028m | 0.016m |
| PCK@20 (cross-subject) | **86.89%** | -- | -- |
| Training time (5-fold) | **18.17h** | 137.5h | -- |

**Critical observations for our project:**
- 4.82M parameters at INT8 quantization = ~4.8 MB model size -- fits in Pi Zero 2 W RAM (512 MB)
- 0.47B FLOPs suggests ~50ms inference on Cortex-A53 with NEON SIMD (estimated)
- Only uses amplitude, discards phase (phase is "heavily corrupted by CFO and SFO in commercial WiFi devices")
- ESP32-S3 CSI has similar CFO/SFO issues, so amplitude-only approach is pragmatic

**Loss function:**
```
L = L_H + lambda * L_B
L_H = SmoothL1(predicted_keypoints, ground_truth, beta=0.1)
L_B = sum of bone length constraint violations across 14 bone connections
lambda = 0.2
```

The bone constraint loss is particularly important for edge deployment where noisy predictions need physical plausibility enforcement.

#### Adaptation for ESP32 + Pi Zero

WiFlow's architecture maps well to our hardware:
- TCN runs on ESP32 (temporal feature extraction from raw CSI stream)
- Asymmetric conv + axial attention runs on Pi Zero (spatial encoding + pose regression)
- The 540-dimensional input assumes Intel 5300 NIC (18 links x 30 subcarriers); for ESP32-S3 with 1 TX x 1 RX and 52 subcarriers, input dimension is 52x20 = 1040 -- even smaller

### 1.2 MultiFormer: Multi-Person WiFi Pose (May 2025)

**Paper:** MultiFormer: A Multi-Person Pose Estimation System Based on CSI and Attention Mechanism ([arXiv:2505.22555](https://arxiv.org/html/2505.22555v1))

#### Architecture

Teacher-student framework with OpenPose teacher providing ground truth labels.

**Time-Frequency Dual-Dimensional Tokenization (TFDDT):**
- Input: CSI matrix from 1 TX, 3 RX, 30 subcarriers
- Upsampled via zero-insertion + low-pass filtering to 64x3x64
- Two parallel token streams:
  - Frequency tokens F_j: N_S tokens of length M x N_R (subcarrier-centric view)
  - Temporal tokens T_i: M tokens of length N_S x N_R (time-centric view)

**Dual Transformer Encoder:**
- 8 layers per branch (frequency and temporal)
- Multi-head self-attention: MSA(X) = (1/H) * sum(Softmax(QK^T / sqrt(d_k)) V)
- Each branch followed by FFN with ReLU, dropout, residual connections

**Multi-Stage Pose Estimation:**
- Part Confidence Maps (PCM): 19x36x36 heatmaps (18 keypoints + average)
- Part Affinity Fields (PAF): 38x36x36 directional fields for 19 limb connections
- Pose-Attentive Perception Module (PAPM): channel + spatial attention on PCM/PAF
- Multi-person assignment via Hungarian algorithm on PAF integrals

#### Model Variants

| Variant | Encoder Layers | Input | Parameters |
|---------|---------------|-------|------------|
| MultiFormer | 8 | 64x1296 | 11.93M |
| MultiFormer-24 | 8 | 64x576 | 4.05M |
| MultiFormer-18 | 6 | 64x324 | **2.80M** |

**Key result on MM-Fi dataset:** MultiFormer achieves PCK@20 of 0.7225, outperforming CSI2Pose (0.6841). The compact MultiFormer-18 at 2.80M parameters is edge-deployable.

#### Relevance to Our Project

MultiFormer's dual-token approach is valuable because:
1. It explicitly separates temporal and frequency information (like WiFlow's decoupling)
2. The PAF-based multi-person assignment using Hungarian algorithm can run on Pi Zero
3. The 2.80M parameter variant (MultiFormer-18) at INT8 = ~2.8 MB, well within Pi Zero constraints

### 1.3 Person-in-WiFi 3D (CVPR 2024)

**Paper:** Person-in-WiFi 3D: End-to-End Multi-Person 3D Pose Estimation with Wi-Fi (CVPR 2024)

First multi-person 3D WiFi pose estimation.

**Key results:**
- Single person MPJPE: 91.7mm
- Two persons: 108.1mm
- Three persons: 125.3mm
- Dataset: 97K frames, 4m x 3.5m area, 7 volunteers
- Transformer-based end-to-end architecture

**Relevance:** Establishes the accuracy ceiling for WiFi 3D pose. Our ESP32+Pi system should target comparable single-person performance (sub-100mm MPJPE) as a milestone.

### 1.4 Spatio-Temporal 3D Point Clouds from WiFi-CSI (October 2024)

**Paper:** [arXiv:2410.16303](https://arxiv.org/html/2410.16303v1)

Novel approach: generates 3D point clouds from WiFi CSI data using transformer networks.

**Key innovation:** Positional encoding with learned embeddings for antennas and subcarriers, followed by multi-head attention over antenna-subcarrier pairs. This captures both spatial (antenna geometry) and spectral (subcarrier frequency response) dependencies.

**Relevance:** Point cloud output is a richer representation than keypoints alone, enabling:
- Silhouette estimation for activity recognition
- Body volume estimation for person identification
- Occlusion reasoning when fused with multiple viewpoints

### 1.5 Graph-Based 3D Human Pose from WiFi (November 2025)

**Paper:** Graph-based 3D Human Pose Estimation using WiFi Signals ([arXiv:2511.19105](https://arxiv.org/html/2511.19105))

Uses graph neural networks where nodes represent keypoints and edges represent skeletal connections. CSI features are injected as node/edge attributes.

**Relevance:** Graph structure naturally maps to our RuvSense pose_tracker which already maintains a 17-keypoint skeleton with Kalman filtering. Adding graph-based message passing between keypoints could improve joint prediction coherence.

## 2. Edge Deployment Landscape

### 2.1 CSI-Sense-Zero: ESP32 + Pi Zero Reference Implementation

**Repository:** [github.com/winwinashwin/CSI-Sense-Zero](https://github.com/winwinashwin/CSI-Sense-Zero)

The most directly relevant prior art for our hardware target.

**Architecture:**
- Two ESP32-WROOM-32: one TX, one RX (captures CSI)
- Pi Zero: inference node
- Communication: USB serial at 921,600 baud
- Buffer: 235KB FIFO at `/tmp/csififo` (~256 CSI records)
- Inference rate: 2 Hz (configurable)
- WebSocket output for real-time visualization

**Data flow:**
```
ESP32 TX -> WiFi signal -> ESP32 RX -> Serial (921.6 kbaud) -> Pi Zero FIFO -> Model -> WebSocket
```

**Limitations:**
- Original Pi Zero (single-core ARM11) -- very slow inference
- Activity recognition only (not pose estimation)
- Python inference (not optimized for ARM)

**What we improve:**
- Pi Zero 2 W has quad-core Cortex-A53 -- roughly 5-10x faster than Pi Zero
- Rust inference (ONNX/Candle) vs Python -- 3-10x faster
- ESP32-S3 vs ESP32-WROOM-32 -- better CSI quality, more subcarriers
- Pose estimation instead of just activity classification
- UDP transport instead of USB serial -- supports multi-node mesh

### 2.2 OnnxStream: Lightweight ONNX on Pi Zero 2 W

**Repository:** [github.com/vitoplantamura/OnnxStream](https://github.com/vitoplantamura/OnnxStream)

Runs Stable Diffusion XL on Pi Zero 2 W in 298 MB RAM. Key features:
- C++ implementation, XNNPACK acceleration
- ARM NEON SIMD optimization
- Memory-efficient streaming execution (processes one operator at a time)
- Supports INT8 quantization

**Benchmark estimates for our model sizes:**

| Model | Parameters | INT8 Size | Est. Pi Zero 2 Latency |
|-------|-----------|-----------|----------------------|
| MultiFormer-18 | 2.80M | ~2.8 MB | ~30-50ms |
| WiFlow | 4.82M | ~4.8 MB | ~50-80ms |
| MultiFormer | 11.93M | ~11.9 MB | ~120-200ms |
| DensePose-WiFi | ~25M (est.) | ~25 MB | ~300-500ms |

These estimates assume XNNPACK-accelerated INT8 inference on Cortex-A53 @ 1 GHz. The WiFlow and MultiFormer-18 models can achieve 12-20 Hz inference, matching our 20 Hz TDMA cycle target.

### 2.3 ONNX Runtime on ARM

ONNX Runtime officially supports Raspberry Pi deployment with:
- ARM NEON execution provider
- INT8 quantization support
- Python and C++ APIs
- Model optimization tools (graph optimization, operator fusion)

For Rust integration, the `ort` crate (ONNX Runtime Rust bindings) supports cross-compilation to aarch64-linux-gnu.

### 2.4 EfficientFi: CSI Compression for Edge

**Paper:** EfficientFi: Towards Large-Scale Lightweight WiFi Sensing via CSI Compression ([arXiv:2204.04138](https://arxiv.org/pdf/2204.04138))

Proposes compressing CSI data on the sensing device before transmission to the inference node. Key idea: train a CSI autoencoder where the encoder runs on the constrained device and the decoder runs on the more powerful inference node.

**Relevance:** For our ESP32 -> Pi Zero pipeline, CSI compression on ESP32 reduces:
- UDP packet size (lower bandwidth, less packet loss)
- Pi Zero preprocessing time (compressed features are more compact)
- Effective latency (less data to transmit per frame)

## 3. Comparative Analysis: Architecture Selection for ESP32 + Pi Zero

### 3.1 Decision Matrix

| Criterion | WiFlow | MultiFormer-18 | DensePose-WiFi | Graph-3D |
|-----------|--------|----------------|----------------|----------|
| Parameters | 4.82M | 2.80M | ~25M | ~8M (est.) |
| FLOPs | 0.47B | ~0.3B (est.) | ~5B (est.) | ~1B (est.) |
| Multi-person | No | Yes (PAF+Hungarian) | Yes (RCNN-based) | No |
| 3D output | No (2D) | No (2D) | No (UV map) | Yes (3D) |
| Amplitude-only | Yes | Yes | No (amp+phase) | Unknown |
| Edge-viable | Yes | Yes | No | Marginal |
| Open source | Not yet | Not yet | Limited | Not yet |

### 3.2 Recommended Architecture: Hybrid WiFlow + MultiFormer

For the ESP32 + Pi Zero deployment, we recommend a hybrid architecture:

1. **WiFlow's TCN temporal encoder** on ESP32 -- extract temporal features from raw CSI
2. **MultiFormer's dual-token approach** on Pi Zero -- process both frequency and temporal views
3. **WiFlow's bone constraint loss** during training -- enforce physical skeleton plausibility
4. **RuvSense coherence gating** before inference -- reject low-quality CSI frames

This hybrid achieves:
- ~3-5M parameters (between WiFlow and MultiFormer-18)
- Amplitude-only input (robust to ESP32 CFO/SFO)
- Sub-100ms inference on Pi Zero 2 W
- Optional multi-person support via PAF module

### 3.3 Training Data Strategy

Based on the surveyed papers:

| Dataset | Subjects | Frames | Hardware | Availability |
|---------|----------|--------|----------|--------------|
| CMU DensePose-WiFi | 8 | ~250K | Intel 5300 | Limited |
| Person-in-WiFi 3D | 7 | 97K | Custom WiFi | GitHub |
| MM-Fi | Multiple | Large | WiFi + mmWave | Public |
| Wi-Pose | Multiple | Large | Intel 5300 | Public |

**Our approach:**
1. Pre-train on MM-Fi/Wi-Pose public datasets (Intel 5300 CSI format)
2. Apply domain adaptation for ESP32-S3 CSI format (different subcarrier count, CFO characteristics)
3. Fine-tune on self-collected ESP32-S3 data in target environments
4. Augment with synthetic CSI from ray-tracing forward model (Arena Physica insight)

## 4. Gap Analysis: Current wifi-densepose vs SOTA

### 4.1 What We Have

| Capability | Status | Module |
|-----------|--------|--------|
| ESP32 CSI capture | Production | `wifi-densepose-hardware` |
| Multi-node fusion | Production | `ruvsense/multistatic.rs` |
| Phase alignment | Production | `ruvsense/phase_align.rs` |
| Coherence gating | Production | `ruvsense/coherence_gate.rs` |
| 17-keypoint tracking | Production | `ruvsense/pose_tracker.rs` |
| ONNX inference engine | Production | `wifi-densepose-nn` |
| Modality translator | Production | `wifi-densepose-nn/translator.rs` |
| Training pipeline | Production | `wifi-densepose-train` |
| Subcarrier interpolation | Production | `wifi-densepose-train/subcarrier.rs` |

### 4.2 What We Are Missing

| Gap | Required For | Priority |
|-----|-------------|----------|
| **Pi Zero deployment target** | Edge inference node | Critical |
| **Lightweight model architecture** | Sub-100ms inference on Cortex-A53 | Critical |
| **Temporal causal convolution** | Real-time streaming inference | High |
| **Axial attention module** | Efficient spatial encoding | High |
| **Bone constraint loss** | Physical plausibility | High |
| **CSI compression on ESP32** | Bandwidth reduction | Medium |
| **INT8 quantization pipeline** | Model size reduction | Medium |
| **Cross-environment adaptation** | Deployment generalization | Medium |
| **Multi-person PAF decoding** | Multiple subject support | Low (Phase 2) |
| **3D pose lifting** | Z-axis estimation | Low (Phase 3) |
| **Diffusion-based pose refinement** | Uncertainty quantification | Research |

### 4.3 Architecture Gaps in Detail

**1. No lightweight inference path.** The current `wifi-densepose-nn` crate assumes GPU or high-end CPU inference. We need an `EdgeInferenceEngine` optimized for:
- INT8 ONNX models
- ARM NEON SIMD via XNNPACK
- Streaming inference (process CSI frames as they arrive, not in batches)
- Memory-mapped model loading (avoid loading entire model into RAM)

**2. No ESP32 -> Pi Zero communication protocol.** The `wifi-densepose-hardware` crate handles ESP32 CSI capture and UDP aggregation to a server, but has no lightweight protocol for ESP32 -> Pi Zero direct communication. We need:
- Compact binary frame format (not the full ADR-018 format)
- Optional CSI compression (autoencoder on ESP32 or simple PCA)
- Heartbeat and synchronization for multi-ESP32 setups

**3. No temporal convolution module.** The existing signal processing pipeline uses frame-by-frame processing. WiFlow and MultiFormer both show that temporal context (20 frames for WiFlow, 64 frames for MultiFormer) significantly improves accuracy. We need a ring buffer + TCN module in the inference path.

**4. No bone/skeleton constraint enforcement at inference time.** The `pose_tracker.rs` has Kalman filtering and skeleton constraints, but these are post-hoc corrections. WiFlow shows that baking bone constraints into the loss function during training produces better models that need less post-processing.

## 5. References

1. DensePose From WiFi, Geng et al., arXiv:2301.00250, 2023
2. Person-in-WiFi 3D, Yan et al., CVPR 2024
3. WiFlow, arXiv:2602.08661, 2026
4. MultiFormer, arXiv:2505.22555, 2025
5. CSI-Channel Spatial Decomposition, MDPI Electronics 14(4), 2025
6. CSI-Former, MDPI Entropy 25(1), 2023
7. Spatio-Temporal 3D Point Clouds from WiFi-CSI, arXiv:2410.16303, 2024
8. Graph-based 3D Human Pose from WiFi, arXiv:2511.19105, 2025
9. EfficientFi, arXiv:2204.04138, 2022
10. CSI-Sense-Zero, github.com/winwinashwin/CSI-Sense-Zero
11. OnnxStream, github.com/vitoplantamura/OnnxStream
12. Arena Physica, arenaphysica.com (Atlas RF Studio, Heaviside-0/Marconi-0)
13. Tools and Methods for WiFi Sensing in Embedded Devices, MDPI Sensors 25(19), 2025
14. Real-Time HAR using WiFi CSI and LSTM on Edge Devices, SASI-ITE 2025
