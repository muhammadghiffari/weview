# GOAP Implementation Plan: ESP32-S3 + Pi Zero 2 W WiFi Pose Estimation

**Date:** 2026-04-02
**Version:** 1.0
**Status:** Proposed
**Depends on:** ADR-029, ADR-068, SOTA survey (sota-wifi-sensing-2025.md)

---

## 1. Goal State Definition

### 1.1 Terminal Goal

A production-ready WiFi-based human pose estimation system where:
- **ESP32-S3** nodes capture WiFi CSI at 100 Hz, perform temporal feature extraction, and transmit compressed features via UDP
- **Raspberry Pi Zero 2 W** receives features from 1-4 ESP32 nodes, runs neural inference, and outputs 17-keypoint COCO poses at >= 10 Hz
- **Single-person MPJPE** < 100mm in trained environments
- **End-to-end latency** < 150ms (CSI capture to pose output)
- **Total BOM cost** < $30 per sensing zone (1x Pi Zero + 2x ESP32)

### 1.2 World State Variables

```
current_state:
  esp32_csi_capture:           true    # Already implemented
  multi_node_aggregation:      true    # ADR-018 UDP aggregator
  phase_alignment:             true    # ruvsense/phase_align.rs
  coherence_gating:            true    # ruvsense/coherence_gate.rs
  multistatic_fusion:          true    # ruvsense/multistatic.rs
  kalman_pose_tracking:        true    # ruvsense/pose_tracker.rs
  onnx_inference_engine:       true    # wifi-densepose-nn
  modality_translator:         true    # wifi-densepose-nn/translator.rs
  training_pipeline:           true    # wifi-densepose-train
  pi_zero_deployment:          false   # No Pi Zero target
  lightweight_model:           false   # No edge-optimized model
  temporal_conv_module:        false   # No TCN in inference path
  csi_compression:             false   # No ESP32-side compression
  int8_quantization:           false   # No quantization pipeline
  bone_constraint_loss:        false   # No skeleton physics in loss
  esp32_pi_protocol:           false   # No lightweight protocol
  edge_inference_engine:       false   # No ARM-optimized inference
  cross_env_adaptation:        false   # No domain adaptation
  multi_person_paf:            false   # No PAF-based multi-person
  3d_pose_lifting:             false   # No Z-axis estimation

goal_state:
  esp32_csi_capture:           true
  multi_node_aggregation:      true
  phase_alignment:             true
  coherence_gating:            true
  multistatic_fusion:          true
  kalman_pose_tracking:        true
  onnx_inference_engine:       true
  modality_translator:         true
  training_pipeline:           true
  pi_zero_deployment:          true    # TARGET
  lightweight_model:           true    # TARGET
  temporal_conv_module:        true    # TARGET
  csi_compression:             true    # TARGET
  int8_quantization:           true    # TARGET
  bone_constraint_loss:        true    # TARGET
  esp32_pi_protocol:           true    # TARGET
  edge_inference_engine:       true    # TARGET
  cross_env_adaptation:        true    # TARGET (Phase 2)
  multi_person_paf:            true    # TARGET (Phase 2)
  3d_pose_lifting:             true    # TARGET (Phase 3)
```

## 2. Action Definitions

Each action has preconditions, effects, estimated cost (developer-days), and priority.

### Action 1: Define ESP32-Pi Communication Protocol (ADR-069)

```
name:           define_esp32_pi_protocol
cost:           3 days
priority:       CRITICAL (blocks all Pi Zero work)
preconditions:  [esp32_csi_capture]
effects:        [esp32_pi_protocol := true]
```

**Description:** Design a lightweight binary protocol for ESP32 -> Pi Zero communication over UDP (WiFi) or UART (wired fallback).

**Protocol specification:**

```
Frame Header (8 bytes):
  [0:1]   magic:         0xCF01 (CSI Frame v1)
  [2]     node_id:       u8 (0-255, identifies ESP32 node)
  [3]     frame_type:    u8 (0=raw_csi, 1=compressed_features, 2=heartbeat)
  [4:5]   sequence:      u16 (monotonic frame counter, wraps at 65535)
  [6:7]   payload_len:   u16 (bytes following header)

Raw CSI Payload (frame_type=0):
  [0:3]   timestamp_us:  u32 (microseconds since boot, wraps at ~71 minutes)
  [4]     channel:       u8 (WiFi channel 1-13)
  [5]     bandwidth:     u8 (0=20MHz, 1=40MHz)
  [6]     rssi:          i8 (dBm)
  [7]     noise_floor:   i8 (dBm)
  [8:9]   num_sc:        u16 (number of subcarriers, typically 52 or 114)
  [10..]  csi_data:      [i16; num_sc * 2] (interleaved I/Q, little-endian)

Compressed Feature Payload (frame_type=1):
  [0:3]   timestamp_us:  u32
  [4]     compression:   u8 (0=none, 1=pca_16, 2=pca_32, 3=autoencoder)
  [5]     num_features:  u8 (number of feature dimensions)
  [6..]   features:      [f16; num_features] (half-precision floats)

Heartbeat Payload (frame_type=2):
  [0:3]   uptime_s:      u32
  [4:7]   frames_sent:   u32
  [8:9]   free_heap:     u16 (KB)
  [10]    wifi_rssi:     i8 (connection to AP)
  [11]    battery_pct:   u8 (0-100, 0xFF if wired)
```

**Implementation locations:**
- ESP32 firmware: `firmware/esp32-csi-node/main/protocol_v2.h`
- Rust parser: `wifi-densepose-hardware/src/protocol_v2.rs`

**Design rationale:**
- Fixed 8-byte header with magic number for frame synchronization
- Half-precision (f16) for compressed features saves 50% bandwidth vs f32
- Heartbeat enables Pi Zero to detect node failures and rebalance
- Raw CSI mode for debugging; compressed mode for production

### Action 2: Implement Lightweight Model Architecture

```
name:           implement_lightweight_model
cost:           10 days
priority:       CRITICAL (core inference capability)
preconditions:  [training_pipeline, onnx_inference_engine]
effects:        [lightweight_model := true, temporal_conv_module := true]
```

**Architecture: WiFlowPose (hybrid WiFlow + MultiFormer)**

Based on SOTA analysis, we define a custom architecture combining the best elements:

```
Input: CSI amplitude tensor [B, T, S]
  B = batch size
  T = temporal window (20 frames at 20 Hz = 1 second context)
  S = subcarriers (52 for ESP32-S3 20MHz, 114 for 40MHz)

Stage 1: Temporal Encoder (runs on ESP32 optionally, or Pi Zero)
  TCN with 4 layers, dilation [1, 2, 4, 8]
  Input:  [B, T, S] = [B, 20, 52]
  Output: [B, T', C_t] = [B, 20, 64] (temporal features)

Stage 2: Spatial Encoder (runs on Pi Zero)
  Asymmetric convolution blocks (1xk kernels on subcarrier dimension)
  4 residual blocks: 64 -> 128 -> 128 -> 64 channels
  Subcarrier compression: 52 -> 26 -> 13 -> 7
  Output: [B, 64, 7]

Stage 3: Keypoint Decoder (runs on Pi Zero)
  Axial self-attention (2-stage, 4 heads)
  Reshape to [B, 17, 64] (17 keypoints x 64 features)
  Linear projection: 64 -> 2 (x, y coordinates)
  Output: [B, 17, 2] (17 COCO keypoints, normalized 0-1)

Optional Stage 4: Multi-person (Phase 2)
  PAF branch: predict 19 limb affinity fields
  Hungarian assignment for person grouping
```

**Estimated model size:**
- Temporal encoder: ~0.5M params
- Spatial encoder: ~1.2M params
- Keypoint decoder: ~0.8M params
- Total: ~2.5M params
- INT8 size: ~2.5 MB
- FP16 size: ~5 MB
- Estimated Pi Zero 2 W inference: 30-60ms per frame

**Rust implementation location:** New module in `wifi-densepose-nn/src/wiflow_pose.rs`

```rust
/// WiFlowPose: Lightweight WiFi CSI to pose estimation model
///
/// Hybrid architecture combining WiFlow's TCN temporal encoder
/// with MultiFormer's dual-token spatial processing and
/// axial self-attention for keypoint decoding.
pub struct WiFlowPoseConfig {
    /// Number of input subcarriers (52 for ESP32 20MHz, 114 for 40MHz)
    pub num_subcarriers: usize,
    /// Temporal window size in frames (default: 20)
    pub temporal_window: usize,
    /// TCN dilation factors (default: [1, 2, 4, 8])
    pub tcn_dilations: Vec<usize>,
    /// Number of output keypoints (default: 17, COCO format)
    pub num_keypoints: usize,
    /// Hidden dimension for spatial encoder (default: 64)
    pub hidden_dim: usize,
    /// Number of attention heads in axial attention (default: 4)
    pub num_attention_heads: usize,
    /// Enable multi-person PAF branch (default: false)
    pub multi_person: bool,
}

impl Default for WiFlowPoseConfig {
    fn default() -> Self {
        Self {
            num_subcarriers: 52,
            temporal_window: 20,
            tcn_dilations: vec![1, 2, 4, 8],
            num_keypoints: 17,
            hidden_dim: 64,
            num_attention_heads: 4,
            multi_person: false,
        }
    }
}
```

### Action 3: Implement Bone Constraint Loss

```
name:           implement_bone_constraint_loss
cost:           2 days
priority:       HIGH
preconditions:  [training_pipeline, lightweight_model]
effects:        [bone_constraint_loss := true]
```

**Loss function following WiFlow:**

```
L_total = L_keypoint + lambda_bone * L_bone + lambda_physics * L_physics

L_keypoint = SmoothL1(pred, gt, beta=0.1)

L_bone = (1/|B|) * sum_{(i,j) in bones} | ||pred_i - pred_j|| - bone_length_{ij} |

L_physics = (1/N) * sum_t max(0, ||pred_t - pred_{t-1}|| - v_max * dt)
```

Where:
- `bones` = 14 COCO bone connections (e.g., left_shoulder-left_elbow)
- `bone_length_{ij}` = average human bone length ratios (normalized to torso length)
- `v_max` = maximum physiologically plausible keypoint velocity (2 m/s for walking, 10 m/s for fast gestures)
- `lambda_bone = 0.2`, `lambda_physics = 0.1`

**Bone length ratios (normalized to torso = shoulder_center to hip_center = 1.0):**

| Bone | Ratio |
|------|-------|
| shoulder-elbow | 0.55 |
| elbow-wrist | 0.50 |
| hip-knee | 0.85 |
| knee-ankle | 0.80 |
| shoulder-hip | 1.00 |
| neck-nose | 0.30 |
| nose-eye | 0.08 |
| eye-ear | 0.12 |

**Implementation location:** `wifi-densepose-train/src/losses.rs` (add `BoneConstraintLoss`)

### Action 4: Implement INT8 Quantization Pipeline

```
name:           implement_int8_quantization
cost:           5 days
priority:       HIGH
preconditions:  [lightweight_model, training_pipeline]
effects:        [int8_quantization := true]
```

**Approach: Post-Training Quantization (PTQ) with calibration**

1. Train model in FP32 using standard pipeline
2. Export to ONNX format
3. Run ONNX Runtime quantization tool with calibration dataset:
   - Collect 1000 representative CSI frames across multiple environments
   - Run calibration to determine per-layer quantization ranges
   - Apply symmetric INT8 quantization for weights, asymmetric for activations
4. Validate quantized model accuracy (target: <2% PCK@20 degradation)

**Quantization-aware considerations:**
- TCN layers: quantize per-channel (dilated convolutions are sensitive to quantization)
- Attention layers: keep attention logits in FP16 (softmax is numerically sensitive)
- Output layer: keep in FP32 (final coordinate regression needs precision)

**Rust implementation:**
```rust
// In wifi-densepose-nn/src/quantize.rs
pub struct QuantizationConfig {
    /// Quantization method
    pub method: QuantMethod, // PTQ, QAT, Dynamic
    /// Per-layer precision overrides
    pub layer_overrides: HashMap<String, Precision>,
    /// Calibration dataset path
    pub calibration_data: PathBuf,
    /// Number of calibration samples
    pub num_calibration_samples: usize,
    /// Target accuracy degradation threshold
    pub max_accuracy_loss: f32,
}

pub enum Precision {
    INT8,
    FP16,
    FP32,
}
```

**ONNX quantization command (for build pipeline):**
```bash
python -m onnxruntime.quantization.quantize \
  --input model_fp32.onnx \
  --output model_int8.onnx \
  --calibrate \
  --calibration_data_reader CsiCalibrationReader \
  --quant_format QDQ \
  --activation_type QUInt8 \
  --weight_type QInt8
```

### Action 5: Build Edge Inference Engine for Pi Zero

```
name:           build_edge_inference_engine
cost:           8 days
priority:       CRITICAL
preconditions:  [lightweight_model, int8_quantization, esp32_pi_protocol]
effects:        [edge_inference_engine := true, pi_zero_deployment := true]
```

**Architecture: Streaming inference with ring buffer**

```
                    UDP/UART
ESP32-S3 ---------> Pi Zero 2 W
                    |
                    v
            +-- RingBuffer<CsiFrame> --+
            |  (capacity: 64 frames)   |
            +------ |  | -------------+
                    v  v
            +-- TemporalWindow --------+
            |  (20 frames, sliding)    |
            +------ | ----------------+
                    v
            +-- WiFlowPose ONNX ------+
            |  (INT8, XNNPACK accel)  |
            +------ | ----------------+
                    v
            +-- PoseTracker -----------+
            |  (Kalman + skeleton)    |
            +------ | ----------------+
                    v
              PoseEstimate output
              (17 keypoints + confidence)
```

**New Rust binary:** `wifi-densepose-cli/src/bin/edge_infer.rs`

```rust
/// Edge inference daemon for Raspberry Pi Zero 2 W
///
/// Receives CSI frames from ESP32 nodes via UDP, maintains a temporal
/// sliding window, runs INT8 ONNX inference, and outputs pose estimates.
///
/// Usage:
///   wifi-densepose edge-infer \
///     --model model_int8.onnx \
///     --listen 0.0.0.0:5555 \
///     --output-port 5556 \
///     --window-size 20 \
///     --max-nodes 4

struct EdgeInferConfig {
    /// Path to INT8 ONNX model
    model_path: PathBuf,
    /// UDP listen address for CSI frames
    listen_addr: SocketAddr,
    /// UDP output address for pose results
    output_addr: Option<SocketAddr>,
    /// Temporal window size
    window_size: usize,
    /// Maximum ESP32 nodes to accept
    max_nodes: usize,
    /// Inference thread count (1-4 on Pi Zero 2 W)
    num_threads: usize,
    /// Enable XNNPACK acceleration
    use_xnnpack: bool,
}
```

**Cross-compilation for Pi Zero 2 W:**

```bash
# Install cross-compilation toolchain
rustup target add aarch64-unknown-linux-gnu
sudo apt install gcc-aarch64-linux-gnu

# Build for Pi Zero 2 W (64-bit Raspberry Pi OS)
cross build --target aarch64-unknown-linux-gnu \
  --release \
  -p wifi-densepose-cli \
  --features edge-inference \
  --no-default-features

# Or for 32-bit Raspberry Pi OS:
# rustup target add armv7-unknown-linux-gnueabihf
# cross build --target armv7-unknown-linux-gnueabihf ...
```

**ONNX Runtime linking for ARM:**
- Use `ort` crate with `download-binaries` feature for automatic aarch64 binary download
- Alternative: build OnnxStream from source for minimal binary size (~2 MB vs ~30 MB for full ONNX Runtime)

### Action 6: Implement CSI Compression on ESP32

```
name:           implement_csi_compression
cost:           5 days
priority:       MEDIUM
preconditions:  [esp32_csi_capture, esp32_pi_protocol]
effects:        [csi_compression := true]
```

**Three compression tiers:**

**Tier 0: No compression (raw CSI)**
- Payload: 52 subcarriers x 2 (I/Q) x 2 bytes = 208 bytes per frame
- Use case: debugging, maximum fidelity

**Tier 1: PCA-16 (run on ESP32)**
- Pre-computed PCA projection matrix (52 -> 16 dimensions)
- Stored in NVS flash during provisioning
- Payload: 16 features x 2 bytes (f16) = 32 bytes per frame
- Compression: 6.5x
- Compute: ~0.1ms on ESP32-S3 (matrix-vector multiply, SIMD)

**Tier 2: PCA-32 (higher fidelity)**
- 52 -> 32 dimensions
- Payload: 32 x 2 = 64 bytes
- Compression: 3.25x

**Tier 3: Learned autoencoder (future)**
- ESP32-S3 has enough compute for a small encoder (~10K params)
- Requires quantized encoder weights in flash
- Most bandwidth-efficient but requires training

**PCA computation (offline, during provisioning):**

```rust
// wifi-densepose-train/src/compression.rs

/// Compute PCA projection matrix from calibration CSI data
pub fn compute_pca_projection(
    calibration_data: &[CsiFrame],
    target_dims: usize,
) -> PcaProjection {
    // 1. Stack all CSI amplitude vectors into matrix [N, S]
    // 2. Center (subtract mean)
    // 3. Compute covariance matrix [S, S]
    // 4. Eigendecomposition, take top `target_dims` eigenvectors
    // 5. Return projection matrix [S, target_dims] and mean vector [S]
    // ...
}

pub struct PcaProjection {
    /// Projection matrix [num_subcarriers, target_dims]
    pub matrix: Vec<f32>,
    /// Mean vector for centering [num_subcarriers]
    pub mean: Vec<f32>,
    /// Number of input subcarriers
    pub input_dims: usize,
    /// Number of output features
    pub output_dims: usize,
}
```

**ESP32 firmware integration:**
- Store PCA matrix in NVS partition (32x52x4 = 6.5 KB for PCA-32)
- Apply projection in CSI callback before UDP transmission
- Selectable via provisioning command

### Action 7: Implement Cross-Environment Adaptation

```
name:           implement_cross_env_adaptation
cost:           8 days
priority:       MEDIUM (Phase 2)
preconditions:  [lightweight_model, training_pipeline, pi_zero_deployment]
effects:        [cross_env_adaptation := true]
```

**Approach: Rapid environment calibration with few-shot adaptation**

Inspired by Arena Physica's template-based design space and MERIDIAN (ADR-027):

1. **Environment fingerprinting (on Pi Zero, at deployment time):**
   - Collect 60 seconds of "empty room" CSI
   - Compute room signature: mean amplitude profile, delay spread, K-factor
   - Match to nearest room template (corridor, office, bedroom, etc.)
   - Load template-specific model weights

2. **Few-shot fine-tuning (optional, on workstation):**
   - Collect 5 minutes of calibration data with known poses
   - Fine-tune last 2 layers of the model (~50K params)
   - Transfer updated model back to Pi Zero

3. **Online adaptation (continuous, on Pi Zero):**
   - Track CSI statistics over time (sliding window mean/variance)
   - Detect distribution shift (KL divergence exceeds threshold)
   - Apply batch normalization statistics update (no gradient computation needed)

**Implementation location:** `wifi-densepose-train/src/rapid_adapt.rs` (extend existing module)

### Action 8: Implement Multi-Person PAF Decoding

```
name:           implement_multi_person_paf
cost:           6 days
priority:       LOW (Phase 2)
preconditions:  [lightweight_model, bone_constraint_loss]
effects:        [multi_person_paf := true]
```

**Architecture (following MultiFormer):**

Add a PAF branch to the WiFlowPose model:

```
Stage 3 features [B, 64, 7]
  |
  +--> Keypoint head: [B, 17, 2] (single-person keypoints)
  |
  +--> PAF head: [B, 38, H, W] (19 limb affinity fields)
  |
  +--> Confidence head: [B, 19, H, W] (part confidence maps)
```

**Multi-person assignment on Pi Zero:**
1. Extract candidate keypoints from confidence maps via NMS
2. Compute PAF integral scores between candidate pairs
3. Solve bipartite matching with Hungarian algorithm
4. Group keypoints into person instances

**Estimated additional cost:** ~1M parameters, ~10ms additional inference time

### Action 9: Implement 3D Pose Lifting

```
name:           implement_3d_pose_lifting
cost:           5 days
priority:       LOW (Phase 3)
preconditions:  [lightweight_model, multi_person_paf, multistatic_fusion]
effects:        [3d_pose_lifting := true]
```

**Approach: Multi-view triangulation + learned depth prior**

With 2+ ESP32 nodes at known positions, compute 3D pose via:

1. Each node pair provides a different viewing angle of the WiFi field
2. 2D pose from each viewpoint is estimated independently
3. Epipolar geometry constrains 3D position from 2D observations
4. Learned depth prior resolves ambiguities (front/back confusion)

This leverages the existing `viewpoint/geometry.rs` module in wifi-densepose-ruvector which already computes GeometricDiversityIndex and Fisher Information for multi-node configurations.

## 3. Hardware Architecture

### 3.1 System Topology

```
                    WiFi AP (existing home router)
                    /         |          \
                   /          |           \
            ESP32-S3 #1   ESP32-S3 #2   ESP32-S3 #3
            (CSI node)    (CSI node)    (CSI node, optional)
                |             |              |
                +------+------+------+-------+
                       | UDP (WiFi)  |
                       v             v
                  Raspberry Pi Zero 2 W
                  (edge inference node)
                       |
                       v
                  Pose output (UDP/MQTT/WebSocket)
                  to display / home automation / API
```

### 3.2 Data Flow Timing

```
T=0ms     ESP32 #1 captures CSI frame (channel 1)
T=2ms     ESP32 #1 applies PCA compression (0.1ms compute)
T=3ms     ESP32 #1 sends UDP packet to Pi Zero (64 bytes)
T=5ms     ESP32 #2 captures CSI frame (channel 6, TDM slot)
T=7ms     ESP32 #2 sends UDP packet to Pi Zero
T=10ms    Pi Zero receives both frames, adds to ring buffer
T=10ms    Pi Zero checks temporal window (20 frames accumulated?)
          If yes: run inference
T=15ms    Temporal encoder processes 20-frame window (5ms)
T=35ms    Spatial encoder + attention (20ms)
T=45ms    Keypoint decoder (10ms)
T=48ms    Kalman filter update + skeleton constraints (3ms)
T=50ms    Pose estimate emitted (17 keypoints + confidence)
```

**Total latency: ~50ms** (well under 150ms target)
**Throughput: 20 Hz** (matching TDMA cycle)

### 3.3 Hardware Bill of Materials

| Component | Unit Cost | Quantity | Total |
|-----------|----------|----------|-------|
| ESP32-S3 DevKit (8MB) | $9 | 2 | $18 |
| Raspberry Pi Zero 2 W | $15 | 1 | $15 |
| MicroSD card (16GB) | $5 | 1 | $5 |
| USB-C power supply | $5 | 1 | $5 |
| **Total** | | | **$43** |

With ESP32-S3 SuperMini ($6 each), total drops to **$37**.

For minimum viable setup (1 ESP32 + 1 Pi Zero): **$24**.

### 3.4 Pi Zero 2 W Specifications

| Parameter | Value |
|-----------|-------|
| SoC | BCM2710A1 (quad-core Cortex-A53 @ 1 GHz) |
| RAM | 512 MB LPDDR2 |
| WiFi | 802.11b/g/n (2.4 GHz only) |
| Bluetooth | BLE 4.2 |
| GPIO | 40-pin header (UART, SPI, I2C) |
| Power | 5V/2A USB micro-B |
| OS | Raspberry Pi OS Lite (64-bit, headless) |

**Memory budget for inference:**

| Component | Memory |
|-----------|--------|
| OS + services | ~100 MB |
| WiFlowPose INT8 model | ~3 MB |
| ONNX Runtime / OnnxStream | ~10-30 MB |
| Ring buffer (64 frames x 4 nodes) | ~1 MB |
| Inference workspace | ~20 MB |
| **Total** | ~134-164 MB |
| **Available** | ~348-378 MB headroom |

Comfortable fit within 512 MB RAM.

## 4. Rust Crate Modifications

### 4.1 Modified Crates

#### wifi-densepose-hardware

**New files:**
- `src/protocol_v2.rs` -- Lightweight ESP32-Pi binary protocol parser/serializer
- `src/pi_zero.rs` -- Pi Zero UDP receiver with ring buffer management

**Modified files:**
- `src/lib.rs` -- Add `pub mod protocol_v2; pub mod pi_zero;`
- `src/aggregator/mod.rs` -- Add support for protocol_v2 frame format

#### wifi-densepose-nn

**New files:**
- `src/wiflow_pose.rs` -- WiFlowPose model definition (TCN + asymmetric conv + axial attention)
- `src/edge_engine.rs` -- Edge-optimized inference engine (streaming, ARM NEON)
- `src/quantize.rs` -- INT8 quantization configuration and validation

**Modified files:**
- `src/lib.rs` -- Add new module exports
- `src/onnx.rs` -- Add XNNPACK execution provider option, INT8 model loading
- `src/translator.rs` -- Add WiFlowPose-compatible input format

#### wifi-densepose-train

**New files:**
- `src/wiflow_pose_trainer.rs` -- Training loop for WiFlowPose architecture
- `src/compression.rs` -- PCA computation for ESP32 CSI compression
- `src/bone_loss.rs` -- Bone constraint and physics consistency losses

**Modified files:**
- `src/losses.rs` -- Add `BoneConstraintLoss`, `PhysicsConsistencyLoss`
- `src/config.rs` -- Add WiFlowPose training configuration options
- `src/dataset.rs` -- Add ESP32-S3 CSI format support (52/114 subcarriers)
- `src/rapid_adapt.rs` -- Add few-shot environment calibration

#### wifi-densepose-signal

**New files:**
- `src/ruvsense/temporal_encoder.rs` -- TCN temporal feature extraction (shared code for ESP32 and Pi)

**Modified files:**
- `src/ruvsense/mod.rs` -- Add `pub mod temporal_encoder;`

#### wifi-densepose-cli

**New files:**
- `src/bin/edge_infer.rs` -- Pi Zero edge inference daemon
- `src/bin/calibrate.rs` -- Environment calibration tool (PCA computation, room fingerprinting)

#### wifi-densepose-core

**Modified files:**
- `src/types.rs` -- Add `CompressedCsiFrame`, `EdgePoseEstimate` types

### 4.2 New Feature Flags

```toml
# wifi-densepose-nn/Cargo.toml
[features]
default = ["onnx"]
onnx = ["ort"]
edge-inference = ["onnx", "xnnpack"]  # NEW: ARM NEON + XNNPACK
candle = ["candle-core", "candle-nn"]
tch-backend = ["tch"]

# wifi-densepose-cli/Cargo.toml
[features]
default = ["full"]
full = ["wifi-densepose-nn/onnx", "wifi-densepose-train/tch-backend"]
edge-inference = ["wifi-densepose-nn/edge-inference"]  # NEW: minimal binary for Pi
```

### 4.3 Cross-Compilation Configuration

```toml
# .cargo/config.toml (add section)
[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"
rustflags = ["-C", "target-cpu=cortex-a53", "-C", "target-feature=+neon"]
```

## 5. ESP32 Firmware Modifications

### 5.1 New Files

- `firmware/esp32-csi-node/main/protocol_v2.h` -- Protocol v2 frame packing
- `firmware/esp32-csi-node/main/pca_compress.h` -- PCA compression for CSI
- `firmware/esp32-csi-node/main/pca_compress.c` -- PCA implementation with ESP32 SIMD
- `firmware/esp32-csi-node/main/pi_zero_mode.c` -- Pi Zero communication mode (lighter than full server mode)

### 5.2 Modified Files

- `firmware/esp32-csi-node/main/csi_handler.c` -- Add compression step in CSI callback
- `firmware/esp32-csi-node/main/nvs_config.c` -- Store PCA matrix in NVS
- `firmware/esp32-csi-node/main/Kconfig.projbuild` -- Add CONFIG_PI_ZERO_MODE, CONFIG_CSI_COMPRESSION options

### 5.3 Provisioning Updates

```bash
# Provision for Pi Zero mode with PCA-16 compression
python firmware/esp32-csi-node/provision.py \
  --port COM7 \
  --ssid "MyWiFi" \
  --password "secret" \
  --target-ip 192.168.1.50 \  # Pi Zero IP
  --target-port 5555 \
  --compression pca-16 \
  --pca-matrix pca_matrix_16.bin
```

## 6. Training Pipeline

### 6.1 Training Workflow

```
Phase 1: Pre-train on public datasets (GPU workstation)
  Dataset: MM-Fi + Wi-Pose (Intel 5300 format, 30 subcarriers)
  Model: WiFlowPose with 30 subcarriers
  Loss: L_keypoint + 0.2 * L_bone + 0.1 * L_physics
  Duration: ~20 hours on single A100

Phase 2: Domain adaptation for ESP32 CSI (GPU workstation)
  Dataset: Self-collected ESP32-S3 data (52 subcarriers)
  Method: Fine-tune all layers with lower learning rate (1e-4)
  Subcarrier interpolation: 30 -> 52 using existing interpolate_subcarriers()
  Duration: ~4 hours

Phase 3: Quantization (CPU workstation)
  Method: Post-training quantization with 1000 calibration samples
  Format: ONNX INT8 (QDQ format)
  Validation: PCK@20 degradation < 2%

Phase 4: Environment calibration (on Pi Zero)
  Method: 60-second empty-room CSI collection
  Output: Room fingerprint + PCA matrix
  Duration: ~2 minutes total
```

### 6.2 Dataset Collection Protocol

For self-collected ESP32 training data:

1. **Setup:** 2 ESP32-S3 nodes at opposite corners of 4x4m room, Pi Zero receiving
2. **Ground truth:** Smartphone camera running MediaPipe Pose (30 FPS), synchronized via NTP
3. **Activities:** Standing, walking, sitting, waving, falling, idle (2 minutes each)
4. **Subjects:** 5+ volunteers with varying body types
5. **Environments:** 3+ rooms (bedroom, office, corridor) for generalization
6. **Total target:** ~100K synchronized CSI-pose frame pairs

**Synchronization approach:**
- ESP32 and Pi Zero synchronized via NTP (< 10ms accuracy on LAN)
- Camera frames timestamped with system clock
- Offline alignment via cross-correlation of movement signals

### 6.3 Transfer Learning Strategy

Following DensePose-WiFi's proven approach:

```
L_total = lambda_pose * L_pose
        + lambda_bone * L_bone
        + lambda_transfer * L_transfer
        + lambda_physics * L_physics

L_transfer = MSE(features_student, features_teacher)
```

Where `features_teacher` come from a pre-trained image-based pose model (HRNet or ViTPose) and `features_student` come from the WiFi CSI model at corresponding intermediate layers.

**Lambda schedule:**
- Epochs 1-20: lambda_transfer = 0.5 (heavy transfer guidance)
- Epochs 20-50: lambda_transfer = 0.2 (moderate guidance)
- Epochs 50-100: lambda_transfer = 0.05 (fine-tuning freedom)

## 7. Timeline and Milestones

### Phase 1: Foundation (Weeks 1-4)

| Week | Actions | Deliverable |
|------|---------|-------------|
| 1 | Action 1 (protocol), ADR-069 draft | Protocol spec + parser tests |
| 2 | Action 2 (model architecture, begin) | WiFlowPose model definition in Rust |
| 2 | Action 3 (bone loss) | Loss functions implemented and tested |
| 3 | Action 2 (model architecture, complete) | Full model with ONNX export |
| 4 | Action 4 (quantization) | INT8 model, accuracy validated |

**Milestone M1:** WiFlowPose model trained on MM-Fi, exported to INT8 ONNX, PCK@20 > 85% on validation set.

### Phase 2: Edge Deployment (Weeks 5-8)

| Week | Actions | Deliverable |
|------|---------|-------------|
| 5 | Action 5 (edge engine, begin) | Cross-compilation working, model loads on Pi |
| 6 | Action 5 (edge engine, complete) | Streaming inference at >= 10 Hz on Pi Zero |
| 6 | Action 6 (CSI compression) | PCA compression on ESP32, verified bandwidth reduction |
| 7 | Integration testing | ESP32 -> Pi Zero full pipeline working |
| 8 | Performance optimization | Latency < 100ms, memory < 200 MB |

**Milestone M2:** End-to-end demo: ESP32 captures CSI, Pi Zero outputs pose at 10+ Hz.

### Phase 3: Accuracy and Adaptation (Weeks 9-12)

| Week | Actions | Deliverable |
|------|---------|-------------|
| 9 | Data collection (ESP32-S3 training data) | 50K+ synchronized CSI-pose frames |
| 10 | Domain adaptation training | ESP32-specific model, MPJPE < 120mm |
| 11 | Action 7 (cross-env adaptation) | Room calibration working |
| 12 | Validation and documentation | ADR-069 finalized, witness bundle |

**Milestone M3:** Single-person MPJPE < 100mm in calibrated environment, cross-environment deployment working with 60-second calibration.

### Phase 4: Multi-Person and 3D (Weeks 13-20)

| Week | Actions | Deliverable |
|------|---------|-------------|
| 13-14 | Action 8 (multi-person PAF) | 2-person pose separation working |
| 15-16 | Action 9 (3D lifting) | Z-axis estimation from multi-node |
| 17-18 | Advanced optimization | Model distillation, QAT |
| 19-20 | Production hardening | OTA updates, monitoring, alerting |

**Milestone M4:** Multi-person 3D pose at 10 Hz on Pi Zero 2 W.

## 8. Risk Analysis

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Pi Zero 2 W inference too slow (> 100ms) | Medium | High | Fall back to activity recognition (smaller model); use Pi 4 instead |
| ESP32-S3 CSI quality insufficient for pose | Low | Critical | Already validated in ADR-028; add directional antennas if needed |
| INT8 quantization degrades accuracy > 5% | Medium | Medium | Use FP16 instead (2x size, ~1.5x slower); apply QAT |
| Cross-environment generalization poor | High | High | Room calibration (Action 7); template-based models; continuous adaptation |
| WiFi interference degrades CSI | Medium | Medium | Coherence gating (already implemented); channel hopping; 5 GHz fallback |
| ONNX Runtime binary too large for Pi Zero | Low | Medium | Use OnnxStream (2 MB) instead of full ONNX Runtime (30 MB) |
| Multi-person association errors | High | Medium | Limit to 2 persons initially; use PAF + Hungarian; AETHER re-ID |

### 8.2 Hardware Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Pi Zero 2 W supply shortage | Medium | Medium | Design also works with Pi 3A+ or Pi 4 |
| ESP32-S3 firmware instability | Low | Medium | Existing firmware battle-tested; OTA rollback |
| WiFi AP interference with CSI | Low | Low | Dedicated 2.4 GHz channel; ESP32 channel hopping |
| Power supply issues (brownout) | Low | Medium | Proper power supply; ESP32 brownout detection |

### 8.3 Research Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| WiFlow results don't reproduce | Medium | High | Fall back to CSI-Former or MultiFormer architecture |
| ESP32 CSI fundamentally different from Intel 5300 | Medium | High | Collect ESP32-specific training data; subcarrier interpolation |
| Bone constraint loss doesn't improve edge accuracy | Low | Low | Remove if no benefit; constraint is simple and cheap |
| PCA compression loses critical CSI information | Low | Medium | Validate with ablation study; fall back to raw CSI if needed |

## 9. Dependency Graph (Action Ordering)

```
                    [esp32_csi_capture] (DONE)
                    /                    \
                   v                      v
    [Action 1: Protocol]          [training_pipeline] (DONE)
           |                      /        |        \
           v                     v         v         v
    [Action 6: Compression] [Action 2: Model] [Action 3: Bone Loss]
           |                     |              |
           |                     +------+-------+
           |                            v
           |                   [Action 4: Quantization]
           |                            |
           +---------------+------------+
                           v
                  [Action 5: Edge Engine]
                           |
                           v
                  [Action 7: Cross-Env] (Phase 2)
                           |
                           v
                  [Action 8: Multi-Person] (Phase 2)
                           |
                           v
                  [Action 9: 3D Lifting] (Phase 3)
```

**Critical path:** Action 1 -> Action 2 -> Action 4 -> Action 5
**Parallel path:** Action 3 can proceed concurrently with Action 2
**Parallel path:** Action 6 can proceed concurrently with Actions 2-4

## 10. Success Criteria

### Phase 1 Exit Criteria

- [ ] WiFlowPose model trains to convergence on MM-Fi dataset
- [ ] PCK@20 >= 85% on MM-Fi validation set
- [ ] INT8 ONNX model size < 5 MB
- [ ] Bone constraint loss reduces physically implausible predictions by > 50%

### Phase 2 Exit Criteria

- [ ] edge_infer binary cross-compiles for aarch64 and runs on Pi Zero 2 W
- [ ] End-to-end latency < 150ms (CSI capture to pose output)
- [ ] Inference rate >= 10 Hz sustained
- [ ] PCA compression reduces bandwidth by >= 3x without > 5% accuracy loss
- [ ] Multi-node support (2 ESP32 nodes + 1 Pi Zero) working

### Phase 3 Exit Criteria

- [ ] Single-person MPJPE < 100mm in calibrated environment
- [ ] Cross-environment deployment works with 60-second calibration
- [ ] System runs continuously for 24 hours without crashes
- [ ] ESP32 OTA firmware update working for CSI compression parameters

### Phase 4 Exit Criteria

- [ ] 2-person pose separation working (MPJPE < 150mm per person)
- [ ] 3D pose estimation from 2+ nodes (Z-axis error < 200mm)
- [ ] Production monitoring and alerting operational

## 11. Relationship to Existing ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-018 | Protocol v2 (Action 1) extends ADR-018 binary frame format |
| ADR-024 | AETHER re-ID embeddings used in multi-person tracking (Action 8) |
| ADR-027 | MERIDIAN cross-env generalization informs Action 7 |
| ADR-028 | ESP32 capability audit validates CSI quality assumptions |
| ADR-029 | RuvSense pipeline stages feed into edge inference (Action 5) |
| ADR-068 | Per-node state pipeline directly used by multi-node inference |

## 12. New ADR Required

**ADR-069: Edge Inference on Raspberry Pi Zero 2 W**

This implementation plan should be formalized as ADR-069 covering:
- Protocol v2 specification
- WiFlowPose architecture selection rationale
- Pi Zero deployment constraints and optimizations
- INT8 quantization strategy
- Cross-compilation approach
- Environment calibration protocol

Status: Proposed, pending this plan's approval.
