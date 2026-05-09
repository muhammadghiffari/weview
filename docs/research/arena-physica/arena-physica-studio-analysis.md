# Arena Physica Studio Analysis

Research document for wifi-densepose project.
Date: 2026-04-02

---

## 1. What is Arena Physica?

Arena Physica (trading as Arena, arena-ai.com / arenaphysica.com) is a startup pursuing "Electromagnetic Superintelligence" -- building AI foundation models that develop superhuman intuition for how geometry shapes electromagnetic fields.

- **Founded**: 2019
- **Founders**: Pratap Ranade (CEO), Arya Hezarkhani, Claire Pan, Michael Frei, Harish Krishnaswamy
- **Funding**: $30M Series B (April 2025)
- **Offices**: NYC (HQ), SF, LA
- **Customers**: AMD, Anduril Industries, Sivers Semiconductors, Bausch & Lomb
- **Impact claimed**: 35% reduction in engineering man-hours, multi-month acceleration in time-to-market, >3% improvement in product quality

Arena does NOT do WiFi sensing. They build AI-driven tools for RF/electromagnetic hardware design -- antennas, PCBs, filters, RF components. Their relevance to our project is methodological: they demonstrate how to build neural surrogates for Maxwell's equations that run 18,000x to 800,000x faster than traditional solvers.


## 2. Atlas Platform and RF Studio

### 2.1 Atlas (Main Platform)

Atlas is Arena's "agentic platform" for hardware design workflows. It is deployed in production with Fortune 500 companies. Atlas encompasses:

- AI-driven electromagnetic simulation
- Design generation and optimization
- Hardware verification workflows
- Integration with existing engineering tools

### 2.2 Atlas RF Studio (Public Beta)

Atlas RF Studio (https://studio.arenaphysica.com/) is a lightweight public instance of the Atlas platform, released as an "interactive sandbox for AI-driven inverse RF design." It serves as a research preview of their electromagnetic foundation model.

**Current capabilities (Beta):**
- Two-layer RF structures
- 8mm x 8mm maximum dimensions
- Ground vias support
- 3 dielectric material choices
- AI-driven design generation from specifications
- Real-time S-parameter prediction

**Workflow:**
1. User inputs electromagnetic specifications (target S-parameters)
2. Marconi-0 (inverse model) generates candidate geometries via conditional diffusion
3. Heaviside-0 (forward model) evaluates each candidate in 13ms
4. System iterates: generate -> simulate -> refine
5. User receives optimized RF component design

### 2.3 Foundation Models

**Heaviside-0 (Forward Model)**:
- Named after Oliver Heaviside (reformulated Maxwell's equations into modern vector form)
- Predicts: S-parameters (magnitude + phase) and electromagnetic field distributions
- Speed: 13ms single design, 0.3ms batched
- Traditional solver comparison: ~4 minutes (HFSS/FDTD)
- Speedup: 18,000x - 800,000x
- Trained on 3 million designs across 25 expert templates + random structures
- Training data represents 20+ years of combined simulation time
- Accuracy: < 1 dB magnitude-weighted MAE

**Marconi-0 (Inverse Model)**:
- Named after Guglielmo Marconi (radio pioneer)
- Generates physical geometries from target S-parameter specifications
- Uses conditional diffusion process (similar to Stable Diffusion / DALL-E architecture)
- Can produce unconventional geometries that outperform human-designed solutions

### 2.4 Roadmap

Planned extensions include:
- Multi-layer structures
- Silicon integration (tapeout planned by end 2026)
- Multiphysics integration (thermal, mechanical beyond EM)
- Broader frequency ranges and design spaces


## 3. Studio Technical Architecture

### 3.1 Frontend Stack

Based on runtime analysis of https://studio.arenaphysica.com/:

| Component | Technology | Evidence |
|---|---|---|
| Framework | Next.js (App Router, server-side streaming) | `__next_f`, `__next_s` arrays, static chunk loading |
| UI Library | Mantine | Responsive breakpoint utilities (xs, sm, md, lg, xl) |
| Rendering | React (server components + client hydration) | React streaming, component loading |
| Fonts | Custom: Rules (Regular/Medium/Bold), EditionNumericalXXIX, Geist Mono (Google Fonts) | Font declarations in page source |
| Theme | Dark mode default for "rf" domain | `ATLAS_DOMAIN: "rf"` config triggers dark theme |

### 3.2 Backend / API Infrastructure

| Service | Detail |
|---|---|
| API Domain | `https://api.emfm.atlas.arena-ai.com` (Auth0 audience) |
| Organization | `emfmprod` |
| Authentication | Auth0 with custom organization ID |
| Feature Flags | DevCycle SDK (A/B testing) |
| Monitoring | Datadog RUM (Real User Monitoring) |
| 3D Rendering | Unreal Engine server at `https://52.61.97.121` (AWS IP) |
| Terms of Service | Required (`ATLAS_REQUIRE_TOS: true`) |

### 3.3 Configuration Flags (from runtime config)

```json
{
  "AUTH0_AUDIENCE": "https://api.emfm.atlas.arena-ai.com",
  "ATLAS_DOMAIN": "rf",
  "ATLAS_REQUIRE_TOS": true,
  "POLL_FOR_MESSAGES": false,
  "ENABLE_HOTJAR": false,
  "SHOW_DEBUG_LOGS": false
}
```

Key observations:
- `POLL_FOR_MESSAGES: false` -- Messages likely use WebSocket/SSE push rather than polling
- `ENABLE_HOTJAR: false` -- Session replay disabled in production
- `SHOW_DEBUG_LOGS: false` -- Debug mode off
- The `emfm` in the API domain likely stands for "ElectroMagnetic Field Model"

### 3.4 3D Visualization via Unreal Engine

The most technically interesting finding: Studio connects to an Unreal Engine server (IP: 52.61.97.121, AWS us-west region) for 3D electromagnetic field visualization.

**Likely architecture:**
1. User submits design geometry in the Next.js frontend
2. Backend runs Heaviside-0/Marconi-0 inference
3. S-parameter results and field distribution data sent to Unreal Engine instance
4. Unreal Engine renders 3D field visualization (E-field, H-field, current distributions)
5. Pixel streaming sends rendered frames back to browser via WebRTC/WebSocket
6. Interactive controls (rotate, zoom, slice planes) forwarded to Unreal Engine

This is consistent with Unreal Engine's Pixel Streaming technology, which renders on a remote GPU and streams video to a web browser. The `52.61.97.121` IP being hardcoded suggests a dedicated rendering server or fleet.

**Unreal Engine WebSocket Protocol** (standard):
- Signaling server negotiates WebRTC connection
- Control messages: `{ type: "input", data: { ... } }` for mouse/keyboard
- Video stream: H.264/VP8 encoded, streamed via WebRTC data channel
- Bidirectional: user input -> Unreal, rendered frames -> browser

### 3.5 Data Formats (Inferred)

Based on the S-parameter focus:

**Input (Design Specification):**
- Target S-parameters: S11, S21, S12, S22 (magnitude + phase vs frequency)
- Frequency range (likely GHz, given RF focus)
- Material properties (dielectric constant, loss tangent)
- Geometric constraints (layer count, max dimensions)

**Output (Design Result):**
- Geometry: likely a discretized grid (64x64 binary material map based on Not Boring article)
- S-parameters: complex-valued frequency response curves
- Field distributions: 2D/3D electromagnetic field maps
- Performance metrics: return loss, insertion loss, bandwidth

**Probable API format** (speculative, based on EM conventions):
```json
{
  "design": {
    "layers": [
      {
        "geometry": [[0,1,1,0,...], ...],  // Binary material grid
        "material": "FR4",
        "thickness_mm": 0.2
      }
    ],
    "vias": [{"x": 3, "y": 5, "radius_mm": 0.15}],
    "dielectric": "rogers_4003c"
  },
  "simulation": {
    "s_parameters": {
      "frequencies_ghz": [1.0, 1.1, ..., 40.0],
      "s11_mag_db": [-5.2, -5.4, ...],
      "s11_phase_deg": [45.2, 44.8, ...],
      "s21_mag_db": [-0.3, -0.3, ...]
    },
    "field_data": {
      "type": "near_field",
      "grid_size": [64, 64],
      "e_field_magnitude": [[...], ...]
    }
  }
}
```


## 4. UI Components and Features

### 4.1 Observed UI Elements

Based on page source analysis:

- **Dark theme** with custom fonts (Rules family -- geometric sans-serif)
- **Icon system** ("IconMark" component -- likely a custom RF/EM icon set)
- **Responsive design** via Mantine breakpoints
- **ToS gate** requiring acceptance before use
- **Organization-scoped access** (Auth0 org-based multi-tenancy)

### 4.2 Likely Feature Set (inferred from product description and tech stack)

| Feature | Description | UI Component |
|---|---|---|
| Specification Input | Enter target S-parameters, frequency range, constraints | Form with frequency sweep chart |
| Design Canvas | View/edit 2D geometry layers | Interactive grid editor |
| S-parameter Viewer | Plot S11/S21/S12/S22 vs frequency | Interactive chart (likely Recharts or D3) |
| 3D Field Viewer | Visualize E/H field distributions | Unreal Engine pixel-streamed viewport |
| Design History | Browse previous designs and iterations | List/card view with thumbnails |
| Compare View | Side-by-side design comparison | Split-pane layout |
| Export | Download design files (Gerber, GDSII, S-parameter Touchstone) | Download buttons |

### 4.3 Agentic Workflow UI

Atlas RF Studio describes "agentic workflows" that:
1. Accept natural-language or parametric specifications
2. Generate multiple candidate designs
3. Simulate each candidate
4. Present ranked results
5. Allow iterative refinement

This suggests an LLM chat interface (translating intent to specs) alongside the technical EM visualization. The pairing of LLM + LFM (Large Field Model) is explicitly described in their architecture.


## 5. Lessons for Our Sensing Server UI

### 5.1 Architecture Patterns to Adopt

| Arena Physica Pattern | Application to wifi-densepose sensing-server |
|---|---|
| Dark theme default | Already appropriate for a sensing/monitoring dashboard |
| Next.js + Mantine | Consider for our sensing-server UI (currently Axum + vanilla) |
| Auth0 multi-tenancy | Overkill for local deployment; useful for cloud/multi-site |
| Unreal Engine 3D | Too heavy; use Three.js/WebGL for 3D pose visualization |
| WebSocket push (not polling) | Match our real-time CSI streaming needs |
| Feature flags (DevCycle) | Useful for gradual feature rollout |
| Datadog RUM | Consider lightweight alternative (e.g., self-hosted analytics) |

### 5.2 Visualization Approaches

**What Arena visualizes:**
- S-parameters (frequency-domain complex response) -- charts
- Electromagnetic field distributions -- 3D heatmaps
- Design geometry -- 2D grid with material layers

**What we need to visualize:**
- CSI amplitude/phase across subcarriers -- frequency-domain charts (similar to S-parameters)
- Person occupancy heatmap -- 2D/3D voxel grid (similar to field visualization)
- Pose skeleton overlay -- 2D/3D joint rendering
- Vital signs (HR, BR) -- time-series charts
- Node mesh topology -- graph visualization
- Signal quality metrics -- dashboard gauges

**Shared patterns:**
- Both need real-time frequency-domain data visualization
- Both show spatial field/occupancy distributions
- Both benefit from interactive 3D (but at different scales)
- Both require low-latency streaming from computation backend

### 5.3 Data Flow Architecture Comparison

**Arena Physica:**
```
Browser (Next.js) -> API (inference) -> Heaviside-0/Marconi-0 -> Unreal Engine -> Pixel Stream -> Browser
```

**wifi-densepose (recommended):**
```
ESP32 nodes -> sensing-server (Axum) -> WebSocket -> Browser (React/Mantine)
                    |
                    v
              RuvSense pipeline -> pose/vitals -> WebSocket -> Browser
```

Key difference: Arena renders 3D on the server (Unreal Engine) and streams pixels. We should render 3D on the client (Three.js/WebGL) and stream data, because:
- Our 3D scenes are simpler (skeleton + voxels vs. full EM field)
- Client-side rendering avoids GPU server costs
- Lower latency for real-time sensing feedback
- Works offline / on local network

### 5.4 API Design Lessons

**Arena's API pattern** (REST + WebSocket):
- REST for design submission and retrieval
- WebSocket/SSE for live simulation progress and results
- Auth0 JWT for authentication
- Organization-scoped resources

**Recommended for sensing-server:**
- REST endpoints for configuration, history, calibration
- WebSocket for real-time CSI, pose, and vitals streaming
- Optional: SSE as fallback for environments where WebSocket is blocked
- API key or local-only access (no OAuth needed for embedded deployment)

**Proposed WebSocket protocol for sensing-server:**
```json
// Server -> Client: CSI frame
{
  "type": "csi_frame",
  "timestamp_us": 1712000000000,
  "node_id": "esp32-node-1",
  "subcarriers": 56,
  "amplitude": [0.45, 0.52, ...],
  "phase": [-1.23, 0.87, ...]
}

// Server -> Client: Pose update
{
  "type": "pose",
  "timestamp_us": 1712000000000,
  "persons": [
    {
      "id": 0,
      "keypoints": [
        {"name": "nose", "x": 2.3, "y": 1.5, "z": 1.7, "confidence": 0.92},
        ...
      ]
    }
  ]
}

// Server -> Client: Vitals update
{
  "type": "vitals",
  "timestamp_us": 1712000000000,
  "person_id": 0,
  "heart_rate_bpm": 72.5,
  "breathing_rate_rpm": 16.2,
  "presence_score": 0.98
}

// Server -> Client: Occupancy grid
{
  "type": "occupancy",
  "timestamp_us": 1712000000000,
  "nx": 8, "ny": 8, "nz": 4,
  "bounds": [0.0, 0.0, 0.0, 6.0, 6.0, 3.0],
  "densities": [0.0, 0.0, 0.12, ...]
}

// Client -> Server: Configuration
{
  "type": "config",
  "action": "set",
  "key": "tomography.lambda",
  "value": 0.15
}
```

### 5.5 Specific UI Components to Build

Based on Arena Physica's approach and our sensing needs:

**Priority 1 (Core Dashboard):**
1. **Real-time CSI waterfall** -- Subcarrier amplitude over time, color-mapped (similar to spectrogram)
2. **Pose skeleton view** -- 2D/3D rendering of detected keypoints with skeleton connections
3. **Node topology map** -- Show ESP32 mesh with RSSI-colored edges
4. **Vitals panel** -- Heart rate and breathing rate with time-series charts

**Priority 2 (Advanced Visualization):**
5. **Occupancy heatmap** -- 2D top-down view of tomographic voxel grid
6. **Phase coherence indicator** -- Per-link coherence scores (green/yellow/red)
7. **Fresnel zone overlay** -- Show first Fresnel zone on room floor plan per link

**Priority 3 (Configuration/Debug):**
8. **Calibration wizard** -- Guide through empty-room calibration for field_model
9. **Link quality matrix** -- NxN grid showing per-link signal metrics
10. **Raw CSI inspector** -- Select individual link, view amplitude + phase per subcarrier


## 6. Public API Endpoints and Protocols

### 6.1 Confirmed Endpoints

| Endpoint | Protocol | Purpose |
|---|---|---|
| `https://studio.arenaphysica.com` | HTTPS | Main web application (Next.js SSR) |
| `https://api.emfm.atlas.arena-ai.com` | HTTPS | Backend API (Auth0 audience) |
| `https://52.61.97.121` | HTTPS/WSS | Unreal Engine rendering server |

### 6.2 Authentication

- Auth0-based with organization scoping
- Custom audience: `https://api.emfm.atlas.arena-ai.com`
- Organization: `emfmprod`
- Terms of Service required before access

### 6.3 Feature Flags

DevCycle SDK integrated for A/B testing and feature gating. This suggests gradual rollout of new capabilities.

### 6.4 Monitoring

Datadog RUM (Real User Monitoring) for performance tracking. Session replay (Hotjar) is available but disabled in production.

### 6.5 What is NOT Publicly Documented

- REST API endpoints (no public API docs found)
- WebSocket message schemas
- S-parameter data format
- Geometry encoding format
- Rate limits or usage quotas
- Pricing model

Arena Physica appears to operate as a closed platform without public API access. The Studio beta is a controlled preview, not an open API.


## 7. Summary of Findings

### What Arena Physica Is
A $30M-funded startup building neural surrogates for electromagnetic simulation. Their AI predicts S-parameters and field distributions 18,000-800,000x faster than traditional solvers. They serve Fortune 500 hardware companies (AMD, Anduril) for RF component design.

### What Arena Physica Is NOT
They are not a WiFi sensing company. They do not do human pose estimation, CSI analysis, or IoT sensing. The relevance to our project is purely methodological.

### Key Technical Takeaways for wifi-densepose

1. **Neural surrogates for Maxwell's equations work** -- Arena proves that training on millions of simulation examples produces models accurate to < 1 dB MAE running in milliseconds. We could apply the same approach to CSI prediction.

2. **Inverse design via conditional diffusion** -- Marconi-0's approach (generating geometry from target specs) parallels our inverse problem (generating pose from CSI). Conditional diffusion is a viable architecture.

3. **Bidirectional search** -- The generate-evaluate-refine loop is more effective than direct inversion. For real-time sensing, the evaluator (forward model) must be fast.

4. **Domain-specific models beat general LLMs** -- For electromagnetic tasks, specialized architectures substantially outperform GPT-4 / Claude. This validates our approach of building specialized CSI processing rather than relying on general-purpose models.

5. **Studio UI is Next.js + Mantine + Unreal Engine** -- A modern stack, but the Unreal Engine component is overkill for our visualization needs. Three.js/WebGL on the client is more appropriate for our real-time sensing dashboard.

6. **WebSocket push over polling** -- Confirmed by their `POLL_FOR_MESSAGES: false` configuration. Our sensing-server should use WebSocket push for real-time data streaming.


## References

- Arena Physica Homepage: https://www.arenaphysica.com/
- Atlas RF Studio Beta: https://studio.arenaphysica.com/
- Introducing Atlas RF Studio (publication): https://www.arenaphysica.com/publications/rf-studio
- Electromagnetism Secretly Runs the World (Not Boring essay): https://www.notboring.co/p/electromagnetism-secretly-runs-the
- Arena Launches Atlas (press release): https://www.prnewswire.com/news-releases/arena-launches-atlas-to-accelerate-humanitys-rate-of-hardware-innovation-302423412.html
- Arena AI raises $30M (SiliconANGLE): https://siliconangle.com/2025/04/08/arena-ai-raises-30m-accelerate-innovation-hardware-testing-atlas/
- Artificial Intuition (CDFAM presentation): https://www.designforam.com/p/artificial-intuition-building-an
- Pratap Ranade LinkedIn announcement: https://www.linkedin.com/posts/pratap-ranade-7272829_today-im-excited-to-introduce-arena-physica-activity-7442204772725723137-RRtE
- Mantine UI: https://mantine.dev/
- Unreal Engine Pixel Streaming: https://dev.epicgames.com/documentation/en-us/unreal-engine/remote-control-api-websocket-reference-for-unreal-engine
