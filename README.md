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
├── v2/                    # Rust-based signal processing & WebSocket server (Rust Sensing)
├── ui/                    # Core visualization platform (Frontend)
│   ├── assets/            # UI assets and logos
│   ├── js/                # Pose fusion logic and MediaPipe configuration
│   ├── observatory-3d.html# WebGL spatial rendering module
│   ├── observatory.html   # 2D heatmap and tracking module
│   ├── pose-fusion.html   # Live dual-modal fusion module
│   └── index.html         # Main application landing page
└── README.md              # Technical documentation
```

## Prerequisites

To compile and run the entire spatial intelligence platform, the following dependencies must be installed:

* **Rust Toolchain:** `rustc 1.70+` and `cargo` (for building the signal processing backend).
* **Python:** `Python 3.8+` (for serving the frontend and evaluating simulation models).
* **Node.js / npm:** (Optional, for building the NVSim Vite dashboard).
* **Browser:** A modern browser with WebGL 2.0 and WebAssembly support (Chrome/Edge 90+, Firefox 88+).
* **Hardware:** ESP32-S3 development boards (Only required for live physical CSI capture. The frontend can run in simulation/webcam-only mode without it).

## Running the Platform

The platform operates in decoupled layers. For full functionality, both the Rust Sensing engine and the UI frontend must be running simultaneously.

### 1. Start the Rust Sensing Engine (v2)

The Rust engine handles the ingestion of raw CSI packets from the ESP32 mesh, computes spatial tensors, and broadcasts them via WebSockets.

```bash
# Navigate to the Rust sensing directory
cd v2/

# Compile and run the Rust server in release mode for maximum performance
cargo run --release
```
*The server will typically bind to `ws://localhost:8765`.*

### 2. Start the Frontend (UI Visualization & Optical Inference)

The frontend uses MediaPipe WASM and WebGL. Due to browser security policies regarding WebAssembly and local files, it must be served via a local HTTP server.

```bash
# Open a new terminal window/tab
cd ui/

# Start a local HTTP server
python3 -m http.server 3000
```

### 3. Access the Dashboards

Navigate to the following URLs in your browser to interact with the modules:
- **Main Landing Page:** `http://localhost:3000`
- **Observatory 3D:** `http://localhost:3000/observatory-3d.html`
- **Pose Fusion:** `http://localhost:3000/pose-fusion.html`

## License & Attribution

This project is released under the MIT License. Developed for the Economic Survival 2026 initiative.
