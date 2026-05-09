# WeView: WiFi-Based Spatial Intelligence Platform

![WeView Logo](ui/assets/WeView_Logo.png)

<h3 align="center">Real-Time Human Pose Estimation and CSI Sensing Visualization</h3>

<p align="center">
  <strong>A comprehensive framework fusing RF (Radio Frequency) Channel State Information (CSI) with optical inference for robust, non-intrusive spatial telemetry.</strong>
</p>

---

## Abstract

**WeView** is an advanced sensory platform designed to transform ambient WiFi signals into structured spatial intelligence. Traditional computer vision relies heavily on line-of-sight optics, which are vulnerable to occlusions, poor lighting, and privacy constraints. WeView addresses these limitations by extracting Channel State Information (CSI) from commodity edge hardware (ESP32-S3) and fusing it with lightweight optical models (MediaPipe). 

The result is a robust, privacy-preserving, and highly accurate spatial mapping system capable of rendering human poses, tracking occupancy, and monitoring vital signs—even through walls or in zero-light environments.

## Core System Architecture

The platform architecture is divided into three primary layers:

### 1. RF Sensing & Edge Layer (Hardware)
- **CSI Extraction:** Utilizes ESP32-S3 microcontrollers to capture subcarrier amplitude and phase shifts caused by environmental multipath interference.
- **Protocol:** TDM (Time Division Multiplexing) protocol spanning channels 1/6/11 for extended spatial resolution.
- **Edge Pre-processing:** Hampel filtering and SpotFi algorithms run on-edge to suppress high-frequency noise and calibrate phase offsets before transmission.

### 2. Backend & Inference Engine (Rust)
- **Data Ingestion:** A highly concurrent Rust-based WebSocket server ingests raw CSI data at high sampling rates.
- **Feature Extraction:** Transforms physical signal variance into structured tensors suitable for real-time analysis.
- **Topology & Coherence Search:** Evaluates signal coherence against a trained background model to separate static multipath reflections from dynamic human movement.

### 3. Frontend Fusion & Visualization (UI/WebAssembly)
- **Observatory 3D:** A WebGL-accelerated (Three.js) environment that renders human body meshes derived from the sensing data. Includes Realistic, DensePose, and X-Ray rendering pipelines.
- **Dual-Modal Pose Fusion:** A deterministic algorithm that merges optical inference (MediaPipe WASM) with RF telemetry. 
- **Honesty Filter Heuristic:** A novel post-processing layer that evaluates joint confidence. It mathematically suppresses hallucinated lower-body keypoints when bounding boxes indicate only upper-body visibility, relying purely on verified RF-data for occluded joints.

## Technical Novelties & Contributions

For academic and engineering evaluation, WeView introduces several key implementations:

* **Dual-Modal Signal Fusion:** Replaces single-modality bottlenecks by dynamically weighting optical confidence against CSI variance. If a camera is occluded, the system relies heavier on RF data.
* **WASM-Accelerated Inference:** MediaPipe models are executed entirely client-side via WebAssembly, ensuring zero-latency inference without requiring GPU-backed cloud instances.
* **Deterministic Skeleton Suppression:** Prevents standard CNN hallucination by enforcing spatial geometry rules (the "Honesty Filter") when the lower half of the subject is occluded by physical barriers.
* **High-Performance Telemetry UI:** The client dashboard is optimized using `requestAnimationFrame` and GSAP for fluid 60FPS rendering of complex spatial data streams.

## Technology Stack

* **Frontend:** HTML5, Vanilla JavaScript, CSS3 (Glassmorphism UI), GSAP (Animations).
* **3D Rendering:** WebGL, Three.js.
* **Computer Vision:** MediaPipe Pose Landmarker (WASM/GPU).
* **Backend Pipeline:** Rust, Tokio (Async Networking), WebSockets.
* **Embedded Hardware:** Espressif ESP32-S3 (for Orthogonal Frequency-Division Multiplexing (OFDM) CSI extraction).

## Repository Structure

```text
WeView/
├── firmware/              # ESP32-S3 firmware for raw CSI extraction
├── backend/               # Rust-based signal processing & WebSocket server
├── ui/                    # Core visualization platform (Frontend)
│   ├── assets/            # UI assets and logos
│   ├── js/                # Pose fusion logic and MediaPipe configuration
│   ├── observatory-3d.html# WebGL spatial rendering module
│   ├── observatory.html   # 2D heatmap and tracking module
│   ├── pose-fusion.html   # Live dual-modal fusion module
│   └── index.html         # Main application landing page
└── README.md              # Technical documentation
```

## System Requirements & Setup

To reproduce the visualization frontend locally:

1. **Environment:** Any modern web browser supporting WebAssembly and WebGL 2.0 (Chrome 90+, Firefox 88+).
2. **Local Server:** The frontend requires a local HTTP server to bypass CORS restrictions for WASM module loading.
   ```bash
   cd ui
   python3 -m http.server 3000
   ```
3. **Execution:** Navigate to `http://localhost:3000`.

*(Note: Live CSI fusion capabilities require the physical ESP32-S3 mesh network to be active and the Rust backend to be broadcasting on `ws://localhost:8765`).*

## License & Attribution

This project is released under the MIT License. Developed for the Economic Survival 2026 initiative.
