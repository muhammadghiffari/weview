# WeView: Spatial Intelligence & Multi-Modal Sensing Platform

![WeView Logo](ui/assets/WeView_Logo.png)

<h3 align="center">Real-Time Human Pose Estimation and CSI Sensing Visualization</h3>

<p align="center">
  <strong>A comprehensive framework fusing RF (Radio Frequency) Channel State Information (CSI) with optical inference for robust, non-intrusive spatial telemetry.</strong>
</p>

<p align="center">
  <a href="https://deepwiki.com/muhammadghiffari/weview/9.1-research-papers-and-surveys">
    <img src="https://img.shields.io/badge/DeepWiki-Research%20Papers-0052cc.svg?style=for-the-badge&logo=bookStack&logoColor=white" alt="DeepWiki Research Papers">
  </a>
</p>

---

## Abstract

**WeView (formerly WiFi-DensePose)** is a multi-modal sensing platform that utilizes Channel State Information (CSI) from commodity WiFi hardware (ESP32-S3) to perform human pose estimation, vital sign monitoring, and disaster recovery sensing. The project has evolved from a Python-based research prototype (v1) into a high-performance, domain-driven Rust ecosystem (v2) capable of real-time inference and distributed swarm coordination.

The primary goal of WeView is to "see" through obstacles and in low-light conditions by analyzing how human bodies disturb WiFi signals, mapping these disturbances to the DensePose coordinate system (mapping image pixels to 3D surface coordinates of the human body).

## High-Level Architecture & Major Subsystems

The codebase is organized into four primary pillars:

1. **Sensing Infrastructure:** ESP32-S3 firmware for high-frequency CSI collection and edge processing.
2. **Core Processing (Rust v2):** A workspace of 15+ crates handling signal processing, neural network inference (ONNX/Candle), and spatial-temporal analysis.
3. **Agentic Orchestration (Claude Flow V3):** A self-learning multi-agent swarm that manages code evolution, system optimization, and consensus-driven decision making.
4. **Visualization & UX:** A Tauri-based desktop application, a Three.js-powered pointcloud viewer, and a Glassmorphism web UI for real-time 3D rendering of pose data.

### Version History: v1 (Python) → v2 (Rust)
The system underwent a major architectural shift to improve latency, type safety, and deployment flexibility.

| Feature | v1 Python (Archive) | v2 Rust (Core) |
| --- | --- | --- |
| **Core Logic** | FastAPI / PyTorch | Axum / Candle (ONNX) |
| **Processing** | NumPy / SciPy | `wifi-densepose-signal` (SIMD optimized) |
| **Concurrency** | Asyncio / Multiprocessing | Tokio / Rayon |
| **Edge Logic** | Limited | WASM Edge Intelligence Modules |
| **Orchestration** | Manual | Claude Flow V3 Swarm |

## Intelligence & Orchestration: Claude Flow V3

A unique aspect of this codebase is the integration of **Claude Flow V3**, an agentic framework that manages the system's lifecycle. It uses a hierarchical-mesh topology to coordinate up to 15 specialized AI agents (e.g., `coder`, `security-architect`, `performance-engineer`).

- **SPARC Specification Phase:** Acts as the entry point for domain modeling, using GNN-enhanced search to retrieve similar requirement patterns and ensuring that new features align with existing bounded contexts.
- **Agent-Code Interaction:** Manages system evolution autonomously while enforcing strict Domain-Driven Design constraints.

## Domain-Driven Design (DDD) Models

The WeView project utilizes DDD to manage the high complexity of multi-modal sensing (WiFi, mmWave, and Quantum) and its translation into human pose estimation and vital signs.

| Bounded Context | Type | Responsibility |
| --- | --- | --- |
| **Hardware Platform** | Generic | ESP32 CSI ingestion, radio abstraction, and frame parsing. |
| **Signal Processing** | Core | Phase sanitization, BVP extraction, and Fresnel zone analysis. |
| **Sensing Server** | Core | Orchestration of real-time data flows, model management, and SONA. |
| **RuvSense / RuVector** | Core | Viewpoint attention, CRV signal lines, and multistatic bridge. |
| **WiFi-Mat** | Supporting | Disaster response, survivor triage (START), and Kalman tracking. |
| **Training Pipeline** | Supporting | Dataset loading, virtual augmentation, and rapid adaptation. |
| **CHCI (Claude Flow)** | Supporting | AI agent orchestration, swarm topology, and self-learning hooks. |

## 94-ADR Index (Architecture Decision Records)

The project utilizes the MADR (Markdown Any Decision Records) 3.0 format to maintain consistency and machine-readability. Decisions are organized into primary domains:

- **ADR-002: Modular DDD Architecture** - Established the transition from a monolithic Python structure to a modular Rust workspace.
- **ADR-006: Unified Memory Service** - Defines how CSI data and neural embeddings are stored in a hybrid memory backend.
- **ADR-081: Adaptive Controller (Firmware)** - Governs how the ESP32 firmware dynamically adjusts the CSI sampling rate based on WiFi congestion.

*Engineers can interact with the ADR system using the CLI: `npx claude-flow@v3alpha adr search ...`*

## Visualization & UI Modules (Frontend)

The frontend operates in decoupled layers to interact with the Rust backend:

- **Dual-Modal Pose Fusion Engine:** Uses MediaPipe WASM for optical keypoint extraction, overlaid with CSI data. It introduces an **"Honesty Filter"** algorithm to mathematically suppress hallucinated lower-body keypoints when bounding boxes indicate only upper-body visibility.
- **3D Spatial Observatory:** A fully interactive spatial rendering engine built with WebGL and Three.js supporting Realistic, DensePose, and X-Ray mapping modes.
- **NVSim Magnetometer:** A Vite + Lit TypeScript dashboard designed to visualize NV-diamond magnetometer telemetry.
- **2D Observatory Telemetry:** Canvas engine optimized for rendering top-down occupancy heatmaps.

---

## Getting Started & Build Guide

Before starting the build process, ensure the following toolchains are installed:
- **Rust:** `1.75+ (Stable)` (Core workspace and sensing server)
- **Docker:** `24.0+` (Containerized pipelines and monitoring)
- **Python:** `3.10+` (v1 Archive and ML training scripts)
- **ESP-IDF:** `v5.1+` (ESP32-S3 firmware compilation)
- **QEMU:** `System-xtensa` (Firmware testing without hardware)

### 1. Docker Quick-Start
The fastest way to deploy the sensing infrastructure is via the provided Docker configuration:
```bash
docker-compose up --build
```
*This launches the Rust sensing server, the legacy Python pipeline, and the Prometheus/Grafana monitoring stack.*

### 2. Rust Workspace Build (v2)
```bash
cargo fetch
cargo build --release
cargo test --workspace
```

### 3. Firmware & QEMU Testing
To build for physical ESP32-S3 hardware:
```bash
cd firmware/esp32-csi-node
idf.py set-target esp32s3
idf.py build
```
For headless testing via QEMU:
```bash
./scripts/qemu-test-swarm.sh
```

### 4. The Verification Pipeline (`./verify`)
To ensure the integrity of the sensing proofs and the RuVector witness chain:
```bash
./verify --pipeline all --proof-path ./proofs/latest.json
```

## License & Attribution

This project is released under the MIT License. Developed for Advanced Spatial Intelligence & Real-Time Computer Vision Research. Built for Economic Survival 2026.
