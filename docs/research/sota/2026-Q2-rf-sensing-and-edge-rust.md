# SOTA Survey — RF Sensing and Edge Rust (2026 Q2)

| Field        | Value                                                                  |
|--------------|------------------------------------------------------------------------|
| **Status**   | Reference / informs `architecture/three-tier-rust-node.md`             |
| **Date**     | 2026-04-25                                                             |
| **Author**   | goal-planner research agent                                            |
| **Scope**    | What's true in 2026, what holds up in the three-tier proposal, what to reconsider |
| **Word target** | ~3,500 words                                                        |

> **Conventions.** Each section answers (a) what's true in 2026, (b) what
> claims in the three-tier proposal hold up, (c) what to reconsider, and
> (d) primary references. Where no primary source could be located, the
> claim is explicitly marked **"no primary source found, mark as
> conjecture."**

---

## 1. WiFi CSI through-wall pose / occupancy estimation

### 1.1 What's true in 2026

The CSI-to-pose literature has matured along three orthogonal axes since
DensePose-from-WiFi (2022) lit the fuse:

- **Lightweight architectures.** WiFlow (Feb 2026) demonstrated a
  spatio-temporal-decoupled network with 4.82 M parameters, 0.47 GFLOPs,
  PCK@20 = 97.0% and MPJPE ≈ 8 mm on the random-split MM-Fi benchmark,
  3–4× smaller than WPformer and ~25× smaller than WiSPPN.
- **Domain generalization.** PerceptAlign (DT-Pose) and the
  cross-environment evaluation in MM-Fi made the cross-subject and
  cross-layout numbers honest. PerceptAlign reports MPJPE 222 mm on Scene
  4 and 317 mm on Scene 5 in cross-layout test, beating prior SOTA by
  >50% — but those are still order-of-magnitude worse than in-domain.
- **Topological priors.** GraphPose-Fi (2025) and topology-constrained
  decoders (DT-Pose) explicitly use the human skeleton as a graph,
  improving plausibility under occlusion.
- **Multistatic geometry.** WeView's own ADR-029/ADR-031 line is the
  practical multistatic story; ISAC-Fi (Aug 2024) and the multistatic
  ISAC-MIMO papers (2024–2025) describe similar geometry as a 6G research
  topic. IEEE 802.11bf-2025 (published 26 September 2025) is the
  standardization vector.

### 1.2 What holds up

The proposal's claim that "3–6 ESP32-S3 nodes can do meaningful pose
work" is consistent with WiFlow's network sizes (4.82 M params, INT8
~5 MB) and with the MM-Fi multi-link benchmark. The CSI pipeline does
not need a Pi *per node* to run inference; one Pi per cluster is
sufficient. WeView's existing ESP32-mesh + sensing-server already
demonstrates the shape.

### 1.3 What to reconsider

- **Through-wall claims are still aggressive.** Published WiFi sensing
  papers focus on line-of-sight or single-wall cases; published
  through-multiple-walls numbers in 2025–2026 are scarce. The
  three-tier proposal's "through-wall" framing should be tempered to
  "through-thin-wall" without primary evidence. *No primary source
  found for through-multiple-walls, mark as conjecture.*
- **Nexmon-on-Pi is not obviously a win.** Nexmon CSI on a Pi 4 captures
  up to 80 MHz BW on Broadcom chips and gives more subcarriers per frame
  than ESP32, but the Pi platform has no equivalent of ESP32 Secure Boot
  V2, and the Broadcom firmware-patch path is fragile across kernel
  releases. WeView's existing ESP32-S3 mesh already beats Nexmon-on-Pi
  on cost, security posture, and provisioning.
- **USRP/SDR is overkill for occupancy and pose**, and is far over the
  proposal's BOM ceiling. It would only become attractive for
  research-grade beamforming or sub-cm ranging.

### 1.4 Primary references

- WiFlow: [arXiv:2602.08661](https://arxiv.org/html/2602.08661) — Feb 2026.
- DT-Pose: [arXiv:2501.09411](https://arxiv.org/abs/2501.09411) — Jan 2025.
- GraphPose-Fi: [arXiv:2511.19105](https://arxiv.org/abs/2511.19105) — Nov 2025.
- Geometry-aware cross-layout HPE: [arXiv:2601.12252](https://arxiv.org/html/2601.12252).
- Nexmon CSI: [seemoo-lab/nexmon_csi](https://github.com/seemoo-lab/nexmon_csi).

---

## 2. IEEE 802.11bf and multistatic ISAC

### 2.1 What's true in 2026

**IEEE Std 802.11bf-2025 was published 26 September 2025** and is the
ratified amendment for WLAN sensing in license-exempt bands 1–7.125 GHz
and >45 GHz. The 3rd SA Ballot Recirculation closed 16 January 2025
with 98% approval. P802.11bf/D8.0 (March 2025) was the last public
draft. The standard defines sensing operation on top of HE/EHT PHYs and
on the DMG/EDMG (60 GHz) PHYs.

3GPP RAN #108 (June 2025) admitted ISAC into the 6G study scope as a
"Day 1" 6G feature. ISAC-Fi (Aug 2024) demonstrated *monostatic* sensing
over commodity WiFi by repurposing the communication waveform.
Multistatic ISAC over cell-free MIMO (2024–2025) is the analytical
direction.

### 2.2 What holds up

The three-tier proposal's framing of "WiFi mesh + multistatic sensing"
is well-aligned with where the standard is moving. ADR-029's existing
multistatic mode and ADR-073's multifrequency mesh scan are the kind of
pre-standard implementations that 802.11bf is now codifying.

### 2.3 What to reconsider

- **802.11bf does not turn an ESP32 into an 802.11bf sensor.** It
  defines a *protocol* for sensing-aware exchanges between APs and
  STAs. Off-the-shelf ESP32-S3 silicon was designed before the standard;
  CSI extraction on ESP32 will keep being a side channel, not a
  standards-blessed feature, until Espressif ships a chip with the
  802.11bf MAC primitives. *No primary source found for an Espressif
  802.11bf-aware product, mark as conjecture.*
- **ISAC-Fi's monostatic-on-commodity-WiFi result** is interesting but
  requires PHY changes; not a path to ESP32 today.
- **The proposal should claim "802.11bf-compatible feature set" rather
  than "802.11bf-compliant"** until silicon exists.

### 2.4 Primary references

- IEEE 802.11bf-2025: [standards.ieee.org](https://standards.ieee.org/ieee/802.11bf/11574/).
- ISAC-Fi: [arXiv:2408.09851](https://arxiv.org/abs/2408.09851).
- IEEE 802.11bf overview paper: [arXiv:2207.04859](https://arxiv.org/pdf/2207.04859).
- NIST overview: [nist.gov/publications/ieee-80211bf](https://www.nist.gov/publications/ieee-80211bf-enabling-widespread-adoption-wi-fi-sensing).

---

## 3. Embedded Rust ecosystem for ESP32-S3 (2026)

### 3.1 What's true in 2026

The esp-rs ecosystem has matured but rebranded:

- **`esp-hal` is at 1.x.** `esp-hal 1.0.0` shipped October 2023; `1.1.0`
  was released April 2024. Stabilized HAL APIs, async drivers, but with
  the constraint that "async drivers can no longer be sent between
  cores and executors."
- **`esp-wifi` was renamed to `esp-radio`** in the 1.x line. The
  scheduler functionality moved to a new crate `esp-rtos`. Existing
  `esp-wifi` references in tutorials are pre-1.x.
- **Embassy on ESP** is split: on no_std ESP-HAL it's a first-class
  citizen, but the Embassy team and Espressif explicitly steer Embassy
  use *toward* `esp-rtos` over time.
- **Embassy on top of `esp-idf-svc` (std)** has a documented gotcha:
  **embassy-executor is not ISR-safe** because it depends on
  `critical-section`, which `esp-idf-hal` implements over FreeRTOS task
  suspension. The recommended std executor is `edge-executor` or the
  built-in `esp-idf-hal` executor.
- **CSI capture on no_std** via `esp-csi-rs` (third-party crate) exists
  but is documented as "still in early development." The
  production-blessed CSI path remains `esp_wifi_set_csi_rx_cb()` in
  ESP-IDF C — exactly what `firmware/esp32-csi-node/main/csi_collector.c`
  uses today.

### 3.2 What holds up

The three-tier proposal's choice to put the **sensor MCU on no_std**
(`esp-hal` + Embassy) avoids the ESP-IDF ISR-safety question entirely,
which is the right architectural answer to a real problem. The proposal
is correct that `heapless` + `postcard` + `embassy-time` is the modern
no_std default.

### 3.3 What to reconsider

- **Update the toolchain names.** The proposal lists `esp-wifi`; in 1.x
  this is `esp-radio`. It lists `embassy-executor` on the comms MCU
  by implication; on the comms MCU the executor must be
  `edge-executor` or `esp-idf-hal`'s built-in executor, not Embassy.
- **CSI maturity is the gating risk.** `esp-csi-rs` is early
  development and the production CSI path is still C. Migrating CSI to
  no_std Rust is a project unto itself, not a free side effect of
  splitting the dies.
- **`esp-idf-svc` parity with C ESP-IDF is good but not 100%.** OTA,
  HTTPS, NVS, BLE provisioning, ESP-WIFI-MESH all have wrappers. Some
  niche ESP-IDF C APIs still need `esp-idf-sys` raw FFI. This is fine
  but means the comms MCU is not "all-Rust" — there's a layer of unsafe
  wrapping at the bottom.

### 3.4 Primary references

- esp-hal releases: [github.com/esp-rs/esp-hal/releases](https://github.com/esp-rs/esp-hal/releases).
- esp-idf-svc CHANGELOG: [github.com/esp-rs/esp-idf-svc/blob/master/CHANGELOG.md](https://github.com/esp-rs/esp-idf-svc/blob/master/CHANGELOG.md).
- Embassy ISR-safety gotcha: [esp-idf-svc#342](https://github.com/esp-rs/esp-idf-svc/issues/342) and esp-idf-svc CHANGELOG.
- esp-csi-rs crate: [crates.io/crates/esp-csi-rs](https://crates.io/crates/esp-csi-rs).
- Embassy Book: [embassy.dev/book](https://embassy.dev/book/).

---

## 4. Edge ML for CSI on ESP32-class hardware

### 4.1 What's true in 2026

- **TFLite Micro on ESP32-S3** is the most-cited path. Reported
  numbers: wake-word inference at 50–60 ms latency, model size ~240 KB
  flash, ~350 KB RAM. INT8 quantization reportedly delivers >6× speedup
  over float on S3. Espressif's `esp-tflite-micro` is the reference
  port.
- **`tract`** (Sonos's pure-Rust ONNX/NNEF runtime) targets std Linux
  primarily; there is no widely-adopted no_std no-alloc port.
- **`candle`** (Hugging Face's Pytorch-flavored Rust ML library) is std
  Linux/macOS/Windows; not designed for MCU class.
- **ONNX Runtime (`ort` Rust binding)** is a wrapper over the C++
  runtime; on ARMv8 (Pi Zero 2W) it works, on Xtensa it does not.
- **ESP-DL** is Espressif's own DL framework for ESP32-S2/S3, optimized
  for the AI extensions of the Xtensa LX7 (which ESP32-S3 has). It is C,
  not Rust.

For a 4.82 M-param INT8 WiFlow at 0.47 GFLOPs:

- On a Pi Zero 2W (Cortex-A53 quad, NEON), inference is plausibly in
  the 50–100 ms range. *No primary measurement found for WiFlow on Pi
  Zero 2W; mark as conjecture.*
- On an ESP32-S3 (Xtensa LX7, 240 MHz, AI extensions), even INT8 4.82M
  is outside the 8 MB flash + 8 MB PSRAM envelope when intermediate
  tensors are counted. WiFlow on S3 would require additional pruning or
  a smaller model class.

### 4.2 What holds up

The proposal's split between "sensor MCU does ISR-clean DSP" and "Pi
runs the model" is the right shape. ML inference at the WiFlow scale is
*not* an ESP32 workload in 2026.

### 4.3 What to reconsider

- **The sensor MCU's ML role should be tiny-feature inference, not
  pose.** Motion classification, presence binary, anomaly thresholding —
  the ADR-039 Tier-0/Tier-1 outputs — fit on ESP32-S3 with TFLite Micro
  or hand-written DSP. They do not fit `tract` or `candle` no_std.
- **For Rust-on-MCU-ML**, the realistic path is hand-rolled INT8
  inference (WeView's `wifi-densepose-nn` already has FFI hooks) or a
  Rust port of a tiny TFLM-style runtime. **No mainstream Rust
  no_std-no_alloc ONNX runtime exists in production at 2026 Q2.**
- **The Pi Zero 2W's 1 GB RAM is fine for WiFlow but tight for larger
  pose models.** A CM4/CM5 with 4 GB unlocks Hugging-Face-class models;
  whether the deployment needs that is a use-case question.

### 4.4 Primary references

- esp-tflite-micro: [github.com/espressif/esp-tflite-micro](https://github.com/espressif/esp-tflite-micro).
- ESP32-S3 TFLite Micro practical guide: [zediot.com](https://zediot.com/blog/esp32-s3-tensorflow-lite-micro/).
- WiFlow architecture (parameters/FLOPs): [arXiv:2602.08661](https://arxiv.org/html/2602.08661).
- ESP32-S3 TinyML INT8 speedup: [zediot.com TinyML optimization](https://zediot.com/blog/esp32-s3-tinyml-optimization/).

---

## 5. QUIC for IoT backhaul

### 5.1 What's true in 2026

- **`quinn` + `rustls` is the production Rust QUIC stack.** Both target
  std Linux, both work fine on ARMv8 (Pi Zero 2W). `rustls` is
  FIPS-validatable via the AWS-LC backend.
- **MQTT-over-QUIC is the emerging IoT pattern.** EMQX 5.x and NanoMQ
  both ship MQTT-over-QUIC; published benchmarks show comparable or
  better tail-latency than MQTT-over-TLS-over-TCP, especially under
  packet loss and mobile-network handoff conditions.
- **For low-rate telemetry** (a few KB at minute granularity), the
  difference between QUIC and TLS-over-TCP is small in steady-state. The
  win is in connection-establishment cost (~1 RTT vs ~3 RTT) and in
  graceful behavior across IP changes.

### 5.2 What holds up

The proposal's choice of `quinn` for the Pi-to-cloud ring is sound and
matches what EMQX, NanoMQ, and Microsoft (MsQuic) are converging on.
`rustls` is a strong default.

### 5.3 What to reconsider

- **Heartbeat-only deployments don't need QUIC.** If the Pi wakes 2
  minutes/day to push aggregated features, an MQTT-over-TLS publish on
  port 8883 is one library, well-supported, and cheaper to operate.
- **QUIC pays off when bidirectional or large-payload traffic is real.**
  Model updates, fleet sync, on-demand video — these are the cases
  where the 1-RTT handshake and connection-migration matter.
- **Don't terminate QUIC inside the comms MCU.** ESP-IDF has no
  production QUIC stack; QUIC belongs on the Pi or gateway, not on the
  MCU.

### 5.4 Primary references

- quinn: [docs.rs/quinn](https://docs.rs/quinn).
- MQTT-over-QUIC IIoT evaluation: [MDPI Sensors 21:5737](https://www.mdpi.com/1424-8220/21/17/5737).
- EMQX MQTT trends: [emqx.com 2025 trends](https://www.emqx.com/en/blog/mqtt-trends-for-2025-and-beyond).

---

## 6. LoRa for sensor mesh fallback

### 6.1 What's true in 2026

- **SX1262** — Semtech's mainstream Gen-2 sub-GHz LoRa transceiver,
  +22 dBm TX, 4.2 mA RX. The default for low-rate, long-range battery
  applications. Mature ecosystem, low BOM cost, supported by `lora-phy`
  and most Meshtastic boards.
- **LR1110** — adds GNSS scan + WiFi scan. Designed for asset-tracking
  workflows where the device opportunistically reports GNSS+WiFi
  fingerprints to a cloud-side resolver.
- **LR1121** — Gen-3, sub-GHz + 2.4 GHz + S/L-band satellite. ~4.5 dB
  better Sub-GHz sensitivity vs SX1262. Cost premium and more system
  complexity.
- **Duty cycles**: EU868 imposes 1% in most sub-bands and 0.1% in the
  863–865 MHz sub-band. US915 uses dwell-time (400 ms) instead of
  duty-cycle limits. Raw-LoRa peer-to-peer must still respect the
  regional regulatory constraint, even though LoRaWAN is not on the
  wire.

For a 20-byte heartbeat at SF7, BW 125 kHz, the airtime is ~40 ms. At
the EU868 1% duty cycle, that's 36 s/hour available — more than 900
heartbeats per hour theoretical max.

### 6.2 What holds up

SX1262 for fallback heartbeats is the correct, well-priced choice. The
proposal's "bytes per minute" framing is well within EU868 1% and US915
dwell-time budgets.

### 6.3 What to reconsider

- **LR1121 is not justified for fallback heartbeats.** The
  satellite/2.4 GHz capabilities are deployment-shape choices, not
  fallback-radio choices.
- **Raw LoRa P2P, not LoRaWAN.** The proposal already implies P2P; this
  should be explicit. LoRaWAN gateways add infrastructure cost without
  improving fallback reliability, and they don't help direct
  node-to-node fallback recovery.
- **LoRa cannot carry CSI features at any meaningful rate.** SF7 BW125
  raw rate is ~5.5 kbps; ADR-081 `rv_feature_state_t` at 5 Hz is 2.4
  kbps gross, 480 B/s, well within budget if compressed and gated.
  Raw ADR-018 frames at 100 KB/s/node are not LoRa-shaped.

### 6.4 Primary references

- Semtech SX1262 datasheet via DigiKey: [forum.digikey.com LoRa breakdown](https://forum.digikey.com/t/lora-hardware-breakdown-key-chips-and-modules-for-iot-applications/52243).
- LR1121 / SX1262 / LR2021 comparison: [nicerf.com](https://www.nicerf.com/news/lr2021-vs-sx1262-vs-lr1121.html).
- TTN duty cycle reference: [thethingsnetwork.org](https://www.thethingsnetwork.org/docs/lorawan/duty-cycle/).
- TTN regional EU863-870: [thethingsnetwork.org regional](https://www.thethingsnetwork.org/docs/lorawan/regional-parameters/eu868/).

---

## 7. Solar + Li-ion power-path for 350 mA bursty IoT loads

### 7.1 What's true in 2026

- **TI BQ24074** — small, simple, linear charger; dual input
  (DC + USB); has the input-voltage-limit feature that crudely
  approximates MPPT for small panels. Adafruit's "Universal" charger
  product is built on it. Low silicon cost, no inductors.
- **TI BQ25798** — newer (2025-class) buck-boost charger with **true
  Voc-sampling MPPT**, dual-input, supports 1–4S Li-ion, 5 A capability,
  3.6–24 V input range. Adafruit launched a development module in May
  2025.
- **Analog Devices LTC4015** — multi-chemistry, two-phase MPPT (15-min
  global sweep + 1-second local dither). High-cost, high-capability;
  overkill for sub-5 W panels.
- **Silergy SPV1050** — purpose-built for sub-watt IoT solar (e.g.
  energy-harvesting sensors). Constant-voltage-ratio MPPT, 70 mA solar
  / 100 mA USB charge limit. Best for *very small* (<1 W) panels and
  micro-energy budgets.

### 7.2 What holds up

For a 2 W panel and a node-average load that bursts to 350 mA, the
BQ24074 (linear) is sufficient. The proposal's choice is fine.

### 7.3 What to reconsider

- **MPPT becomes attractive when panel power × variability is high.**
  At 2 W, the efficiency delta between linear-with-input-voltage-limit
  and true MPPT is on the order of 10–20% in cloudy conditions. For a
  4× harvest-to-load headroom, this is not the binding constraint.
- **If the deployment ever scales to a 5–10 W panel** (e.g., to support
  a Pi that wakes more often than 2 minutes/day), BQ25798's MPPT pays
  off.
- **A super-cap on the input rail** is cheap insurance against the Pi's
  ~350 mA boot inrush; the proposal should consider one.

### 7.4 Primary references

- BQ25798 launch coverage (Adafruit, May 2025): [blog.adafruit.com](https://blog.adafruit.com/2025/05/15/eye-on-npi-ti-bq25798-i2c-controlled-1-to-4-cell-5-a-buck-boost-battery-charger-mppt-for-solar-panels-eyeonnpi-digikey-digikey-adafruit/).
- BQ25798 datasheet: [ti.com](https://www.ti.com/lit/ds/symlink/bq25798.pdf).
- BQ24074 product (Adafruit): [adafruit.com/product/4755](https://www.adafruit.com/product/4755).
- SPV1050 application reference: [DFRobot wiki](https://wiki.dfrobot.com/dfr0579/).

---

## 8. Mesh routing alternatives to ESP-WIFI-MESH

### 8.1 What's true in 2026

- **ESP-WIFI-MESH** documents support up to ~1,000 nodes in 25 layers,
  with a recommended fan-out of 6/node (hardware AP-mode limit is 10).
  Espressif's own newer `esp-mesh-lite` is the lighter, IP-layer-routable
  alternative.
- **Thread / OpenThread** — IPv6-native 802.15.4 mesh, self-healing,
  designed for 250+ node networks per partition. Strong scalability and
  security story. Hardware: ESP32-C6, ESP32-H2, Nordic nRF52840, Silicon
  Labs EFR32.
- **Zigbee** — 802.15.4 like Thread, but with a much older application
  layer. Scales reasonably to ~100 nodes in practice, with congestion
  challenges in dense deployments.
- **BLE Mesh** — managed flooding, optimized for sporadic traffic. Good
  for ~50 nodes; not the right shape for always-on infrastructure.

### 8.2 What holds up

For < 25-node deployments, ESP-WIFI-MESH (or `esp-mesh-lite`) is the
direct continuation of today's WeView mesh and the proposal's choice is
defensible.

### 8.3 What to reconsider

- **For 50–500 node deployments, Thread is the better fit.** It was
  designed for that scale; ESP-WIFI-MESH was not. Using Thread *for the
  control plane* (TIME_SYNC, ROLE_ASSIGN, CHANNEL_PLAN, HEALTH) while
  keeping ADR-018 CSI frames on WiFi is a viable hybrid.
- **The comms MCU choice changes.** ESP-WIFI-MESH stays on ESP32-S3.
  Thread/Zigbee/BLE Mesh prefer ESP32-C6 (which has 802.15.4 + WiFi 6)
  or a separate radio. The proposal's two-S3 die choice forecloses on
  this hybrid; a one-S3 + one-C6 split is worth evaluating.
- **Thread's IPv6-native routing pairs nicely with QUIC.** Both speak
  IP; ESP-WIFI-MESH does not (it uses its own L2-style routing and
  bridges IP).

### 8.4 Primary references

- ESP-WIFI-MESH overview: [docs.espressif.com](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/esp-wifi-mesh.html).
- esp-mesh-lite: [github.com/espressif/esp-mesh-lite](https://github.com/espressif/esp-mesh-lite).
- Silicon Labs benchmarking: [silabs.com mesh-performance](https://www.silabs.com/wireless/multiprotocol/mesh-performance).
- Bluetooth/Thread/Zigbee comparison: [eetimes.com](https://www.eetimes.com/bluetooth-thread-zigbee-mesh-compared/).
- Zigbee vs Matter-over-Thread (2026): [arXiv:2603.04221](https://arxiv.org/html/2603.04221v1).

---

## 9. Pi Zero 2W secure-boot reality

### 9.1 What's true in 2026

- **Raspberry Pi Foundation's official secure-boot path is Pi 4 / Pi 5
  / CM4.** It uses the RPi-bootloader ROM, USB-rooted RSA chain, and
  the `usbboot` tooling. There is no equivalent on the Pi Zero 2W
  (BCM2710A1).
- **Buildroot does support Pi Zero 2W** (April 2025 defconfig update
  uses the same ARM64 `bcm2711_defconfig` as the Pi 4).
- **dm-verity + signed FIT image** is the realistic Pi-Zero-2W path:
  buildroot produces a read-only rootfs, dm-verity covers it with a
  signed Merkle tree, the boot partition has signed kernel/initramfs.
  This delivers integrity but not "secure boot" in the immutable-ROM
  sense.
- **A/B partitions for OTA** is straightforward in buildroot.
  `swupdate` and `RAUC` are the well-known frameworks; both work on Pi
  Zero 2W.

### 9.2 What holds up

The proposal's "buildroot, not Raspberry Pi OS" instinct is correct.
RPi OS does not support secure boot on any Pi.

### 9.3 What to reconsider

- **The "Pi 4 + buildroot is the strongest path" line is true but not a
  Pi Zero 2W story.** If true secure boot with an immutable ROM-rooted
  chain is required, the heavy-compute die should be a CM4 or Pi 5, not
  a Pi Zero 2W.
- **For the proposal's deployment shape** (mostly-off Pi, infrequent
  wake-ups), dm-verity + signed FIT + A/B is probably enough threat
  cover and avoids the cost of a CM4. Document this as an explicit
  tradeoff, not as "the strongest path."
- **`fwupd` is the package-manager-style update agent**; or a
  self-rolled "update-agent" binary signed by the project key. Either
  works; project-style fits with the homogeneous Rust toolchain better.

### 9.4 Primary references

- Raspberry Pi USB-boot secure-boot example: [github.com/raspberrypi/usbboot](https://github.com/raspberrypi/usbboot/blob/master/secure-boot-example/README.md).
- Raspberry Pi forum on secure boot: [forums.raspberrypi.com 352061](https://forums.raspberrypi.com/viewtopic.php?t=352061).
- Buildroot Pi Zero 2W defconfig (April 2025): [lists.buildroot.org](https://lists.buildroot.org/pipermail/buildroot/2025-April/776753.html).

---

## 10. Cross-cutting takeaways

A short list of items that affect more than one section:

1. **The biggest single risk in the proposal is the no_std CSI maturity
   gate.** If `esp-csi-rs` (or whatever replaces it under `esp-radio`)
   does not match `esp_wifi_set_csi_rx_cb` in capture quality and
   ISR-jitter, the sensor-MCU shape collapses back to "C ESP-IDF on the
   sensor MCU too" and the value of the split shrinks.
2. **The cost story improves dramatically if the heavy-compute die is
   shared across nodes.** "One Pi per cluster of 6" is closer to today's
   $9-per-sensor BOM at the per-sensor edge while still adding the
   QUIC/ML/secure-boot story at the cluster level.
3. **IEEE 802.11bf-2025's ratification** changes the regulatory and
   ecosystem landscape but does not change what off-the-shelf ESP32
   silicon can do today. WeView's pre-standard work (ADR-029, ADR-073,
   ADR-081) is well-aligned with the standard's direction; nothing in
   the proposal makes it more or less compatible.
4. **The right "comms MCU" might be ESP32-C6 instead of a second S3.**
   C6 has 802.15.4 (Thread/Zigbee), WiFi 6, and BLE 5.4. For a
   deployment that scales beyond ~25 nodes, the Thread control plane is
   a meaningful upgrade.
5. **Power gating the Pi is the load-bearing power decision.** Soft
   suspend leaks; hard FET cut does not. The proposal's instinct is
   right, but the supercap/transient story has to be designed in.

---

## 11. Items where no primary source was found

This section is required by the project conventions and lists each
non-trivial claim where a primary source could not be located in this
research pass:

- **Through-multiple-walls CSI pose accuracy at room scale.** Published
  papers focus on line-of-sight or single-wall environments. *Mark as
  conjecture for now.*
- **WiFlow inference latency on Pi Zero 2W (Cortex-A53).** Estimated at
  50–100 ms; no measurement found. *Mark as conjecture; benchmark
  before claiming.*
- **Espressif silicon roadmap for 802.11bf-aware MAC primitives.** No
  public announcement from Espressif as of 2026 Q2. *Mark as
  conjecture.*
- **Pi Zero 2W gated cold-boot wake-up time under 5 s with the proposed
  buildroot image.** Mentioned in the proposal as a constraint, no
  measurement found. *Mark as benchmark target.*
- **ESP-WIFI-MESH stable-state tested deployment beyond ~25 nodes.**
  Espressif documents 1,000-node theoretical ceilings but published
  third-party deployment data at scale is sparse. *Mark as conjecture
  pending field test.*

---

## 12. Source list

(Primary references are inlined per-section. This is the unique
domains list for quick reuse.)

- IEEE Standards Association — `standards.ieee.org`
- arXiv — `arxiv.org`
- IEEE Xplore — `ieeexplore.ieee.org`
- Espressif documentation — `docs.espressif.com`
- Espressif GitHub — `github.com/espressif`
- esp-rs project — `github.com/esp-rs`, `crates.io/crates/esp-csi-rs`,
  `docs.rs/esp-idf-hal`
- Embassy project — `embassy.dev`
- The Things Network — `thethingsnetwork.org`
- Texas Instruments — `ti.com`
- Adafruit — `adafruit.com`, `blog.adafruit.com`
- Buildroot — `lists.buildroot.org`
- Silicon Labs — `silabs.com`
- DigiKey forum — `forum.digikey.com`
- NIST — `nist.gov`
- MDPI Sensors — `mdpi.com`
- EMQ technical blog — `emqx.com`
- Raspberry Pi forum / GitHub — `forums.raspberrypi.com`,
  `github.com/raspberrypi/usbboot`
- nicerf comparison guide — `nicerf.com`
- DFRobot wiki — `wiki.dfrobot.com`

---

## Sources

- [WiFlow: A Lightweight WiFi-based Continuous Human Pose Estimation Network](https://arxiv.org/html/2602.08661)
- [Towards Robust and Realistic Human Pose Estimation via WiFi Signals (DT-Pose)](https://arxiv.org/abs/2501.09411)
- [Graph-based 3D Human Pose Estimation using WiFi Signals (GraphPose-Fi)](https://arxiv.org/abs/2511.19105)
- [IEEE 802.11bf-2025](https://standards.ieee.org/ieee/802.11bf/11574/)
- [An Overview on IEEE 802.11bf: WLAN Sensing](https://arxiv.org/pdf/2207.04859)
- [IEEE 802.11bf NIST page](https://www.nist.gov/publications/ieee-80211bf-enabling-widespread-adoption-wi-fi-sensing)
- [ISAC-Fi: Enabling Full-Fledged Monostatic Sensing Over Wi-Fi](https://arxiv.org/abs/2408.09851)
- [Multistatic ISAC Macro–Micro Cooperation](https://www.mdpi.com/1424-8220/24/8/2498)
- [esp-rs/esp-hal releases](https://github.com/esp-rs/esp-hal/releases)
- [esp-idf-svc CHANGELOG](https://github.com/esp-rs/esp-idf-svc/blob/master/CHANGELOG.md)
- [esp-idf-svc Embassy ISR-safety issue #342](https://github.com/esp-rs/esp-idf-svc/issues/342)
- [esp-csi-rs crate](https://crates.io/crates/esp-csi-rs)
- [Embassy Book](https://embassy.dev/book/)
- [esp-tflite-micro](https://github.com/espressif/esp-tflite-micro)
- [ESP32-S3 TFLite Micro practical guide](https://zediot.com/blog/esp32-s3-tensorflow-lite-micro/)
- [ESP32-S3 TinyML Optimization](https://zediot.com/blog/esp32-s3-tinyml-optimization/)
- [quinn QUIC](https://docs.rs/quinn)
- [MQTT-over-QUIC IIoT evaluation (MDPI)](https://www.mdpi.com/1424-8220/21/17/5737)
- [MQTT trends for 2025 (EMQ)](https://www.emqx.com/en/blog/mqtt-trends-for-2025-and-beyond)
- [LoRa SX1262 / LR1121 / LR2021 comparison](https://www.nicerf.com/news/lr2021-vs-sx1262-vs-lr1121.html)
- [LoRa hardware breakdown (DigiKey)](https://forum.digikey.com/t/lora-hardware-breakdown-key-chips-and-modules-for-iot-applications/52243)
- [LoRaWAN duty cycle (TTN)](https://www.thethingsnetwork.org/docs/lorawan/duty-cycle/)
- [LoRaWAN regional EU868 (TTN)](https://www.thethingsnetwork.org/docs/lorawan/regional-parameters/eu868/)
- [BQ25798 launch coverage (Adafruit/DigiKey)](https://blog.adafruit.com/2025/05/15/eye-on-npi-ti-bq25798-i2c-controlled-1-to-4-cell-5-a-buck-boost-battery-charger-mppt-for-solar-panels-eyeonnpi-digikey-digikey-adafruit/)
- [BQ25798 datasheet](https://www.ti.com/lit/ds/symlink/bq25798.pdf)
- [BQ24074 product page](https://www.adafruit.com/product/4755)
- [SPV1050 reference](https://wiki.dfrobot.com/dfr0579/)
- [ESP-WIFI-MESH guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/esp-wifi-mesh.html)
- [esp-mesh-lite](https://github.com/espressif/esp-mesh-lite)
- [Silicon Labs mesh benchmarking](https://www.silabs.com/wireless/multiprotocol/mesh-performance)
- [Bluetooth/Thread/Zigbee comparison (EE Times)](https://www.eetimes.com/bluetooth-thread-zigbee-mesh-compared/)
- [Zigbee vs Matter-over-Thread (arXiv 2603.04221)](https://arxiv.org/html/2603.04221v1)
- [Raspberry Pi USB-boot secure-boot example](https://github.com/raspberrypi/usbboot/blob/master/secure-boot-example/README.md)
- [Raspberry Pi forum: secure boot](https://forums.raspberrypi.com/viewtopic.php?t=352061)
- [Buildroot Pi Zero 2 W defconfig (April 2025)](https://lists.buildroot.org/pipermail/buildroot/2025-April/776753.html)
- [Nexmon CSI](https://github.com/seemoo-lab/nexmon_csi)
