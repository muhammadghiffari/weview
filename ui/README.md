# WeView: Spatial Intelligence and Pose Estimation Platform

A privacy-preserving spatial intelligence platform that visualizes human presence, movement, and vital signs in real time. The system fuses WiFi Channel State Information (CSI) sensing with browser-based Computer Vision pose estimation to reconstruct human pose within a 3D visualization environment.

No cameras are required for sensing. When connected to an ESP32 hardware mesh, the platform processes WiFi radio scattering patterns to detect presence, motion levels, breathing rate, and heart rate — through walls, in darkness, without any wearable devices.

When hardware is unavailable, the application runs a full demonstration using simulated sensing data or the visitor's webcam via MediaPipe Pose running entirely in-browser.

---

## Live Demo

Open the deployed URL in a browser. No installation is required.

To try the Computer Vision hybrid mode, open Settings and enable the CV Hybrid toggle. The browser will request webcam access and begin real-time pose estimation client-side using WebAssembly.

---

## Features

### 3D Observatory Dashboard

A WebGL rendering environment built with Three.js. The scene renders anatomically proportioned human body meshes, environmental boundaries, a WiFi router model, a floor grid, and a signal field heatmap.

Four render modes are available:
- Realistic — physically based skin materials with subsurface scattering
- DensePose — 24-part UV-mapped body segmentation visualization
- X-Ray — wireframe mesh with emissive glow effect
- Hybrid — solid mesh overlaid with skeletal joint markers

### Computer Vision Hybrid Mode

MediaPipe Pose runs as a WebAssembly module directly in the browser. When enabled, the system overlays a 2D skeleton on the webcam feed and optionally fuses the optical keypoints with WiFi-derived data using a weighted confidence strategy.

No server processing is required for this mode. All computation runs on the client device.

### WiFi Sensing Integration

When a hardware backend is running and reachable, the dashboard connects via WebSocket and receives live sensing telemetry. The payload includes presence classification, motion level, spectral features (motion band power, breathing band power, dominant frequency), and node RSSI measurements.

The frontend infers human position and pose from signal dynamics: RSSI gradients, variance shifts, and motion power drive realistic figure movement within the virtual room.

### Adaptive Fallback

On startup, the application probes for a backend at `ws://[host]:8765/ws/sensing`. If the connection succeeds, the status indicator displays LIVE and rendering is driven by hardware data. If the connection fails or drops, the system falls back to a scripted demonstration containing twelve scenarios, each representing a different sensing context.

---

## Architecture

The frontend is entirely static. It requires no server-side rendering, build step, or runtime environment.

| Layer | Technology |
|---|---|
| 3D Rendering | Three.js (WebGL) |
| Post-Processing | Three.js EffectComposer, UnrealBloomPass, SSAOPass |
| Computer Vision | MediaPipe Pose (WASM, runs in-browser) |
| Data Transport | WebSocket (JSON payloads) |
| Styling | Vanilla CSS with CSS custom properties |

---

## Running Locally

Clone the repository and serve the files with any static HTTP server.

```bash
git clone https://github.com/muhammadghiffari/weview-spatial-intelligence.git
cd weview-spatial-intelligence
python -m http.server 3000
```

Open `http://localhost:3000/observatory-3d.html` in a modern browser.

No build step, package manager, or Node.js installation is required.

---

## Connecting to the Hardware Backend

To use live WiFi sensing data, the Rust sensing server must be running separately.

From the WeView project root:

```bash
cd v2
cargo run -p wifi-densepose-sensing-server -- --source esp32
```

The server will listen on `ws://localhost:8765/ws/sensing`. The dashboard will automatically detect and connect to it on page load.

If no ESP32 hardware is connected, the server falls back to a simulated data source that produces realistic signal patterns for testing.

---

## Deployment

The repository is fully compatible with any static hosting service.

### Vercel

1. Import the repository in the Vercel dashboard.
2. Leave the root directory as the default (repository root).
3. No build command is required.
4. Deploy.

Visitors can use the CV Hybrid mode immediately using their webcam. The WiFi sensing Live mode requires a hardware backend accessible at a reachable WebSocket URL, which can be configured in the Settings panel.

### GitHub Pages

1. Go to repository Settings > Pages.
2. Set the source branch to `main`.
3. Set the folder to `/ (root)`.
4. Save. The site will be available at `https://muhammadghiffari.github.io/weview-spatial-intelligence/`.

---

## Keyboard Controls

| Key | Action |
|---|---|
| R | Cycle render mode (Realistic, DensePose, X-Ray, Hybrid) |
| D | Switch demonstration scenario |
| A | Toggle auto-orbit camera |
| S | Open settings panel |
| Mouse drag | Orbit camera |
| Scroll | Zoom |

---

## Background

This project is based on the WiFi-DensePose research direction originating from Carnegie Mellon University, which demonstrated that 17-keypoint COCO body pose can be estimated from WiFi Channel State Information without cameras. WeView extends this with a production-grade Rust signal processing pipeline, a self-learning contrastive embedding model, and a 3D visualization layer designed for real-world deployment.

Hardware cost starts at $8 per sensing node using an ESP32-S3 microcontroller. A full room deployment with 4 to 6 nodes costs approximately $54 and requires no recurring fees or cloud connectivity.

---

## License

MIT License. See LICENSE for details.
