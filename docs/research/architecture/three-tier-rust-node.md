# Three-Tier Rust Node — Exploratory Architecture

| Field        | Value                                                                  |
|--------------|------------------------------------------------------------------------|
| **Status**   | Exploratory / not yet decided                                          |
| **Date**     | 2026-04-25                                                             |
| **Authors**  | ruv (proposal), filed by goal-planner research agent                   |
| **Classifies as** | Speculative architectural alternative to ADR-028 / ADR-081 baseline |
| **Companion**| `docs/research/sota/2026-Q2-rf-sensing-and-edge-rust.md` (SOTA), `docs/research/architecture/decision-tree.md` (decisions) |

> **Reading note.** This document files a long architectural exploration the
> author wrote before any commitment. It is intentionally optimistic in places
> and will be tempered by the SOTA survey filed alongside it. The decision
> tree document maps each load-bearing claim to the evidence that would
> justify acting on it. Nothing in this document supersedes ADR-028 (the
> capability audit) or ADR-081 (the 5-layer adaptive kernel). Both already
> describe a working, single-MCU node; this document describes a
> hypothetical *three-tier* node that would replace it on PCBs that ship
> Pi-class compute next to two ESP32-class radios on a solar-powered HAT.

---

## 1. ADRs this proposal would touch

If pursued, this proposal evolves the following decisions. None are
overturned outright; all need re-read in this light.

- **ADR-028 — ESP32 Capability Audit.** Today's witnessed node is a single
  ESP32-S3 streaming raw ADR-018 frames over UDP. A three-tier node changes
  the audit subject from "one MCU" to "two MCUs + a Pi", with implications
  for the witness bundle, firmware-manifest hashes, and per-node BOM.
- **ADR-081 — Adaptive CSI Mesh Firmware Kernel.** The 5-layer kernel
  already separates radio abstraction (L1), adaptive control (L2), mesh
  plane (L3), feature extraction (L4), and Rust handoff (L5). A three-tier
  node would split L1–L2 onto a no_std sensor MCU, L3 onto an ESP-IDF
  comms MCU, and Layer-5+ Rust workload onto the Pi. The split is
  compatible with the kernel; it is a deployment shape rather than a
  redesign.
- **ADR-018 — ESP32 Dev Implementation.** ADR-018 binary CSI frames remain
  the wire format between the sensor MCU and whoever consumes them. The
  three-tier proposal tightens the contract: ADR-018 frames flow from
  sensor MCU into the comms MCU only, never directly off the node.
- **ADR-029 / ADR-031 — Multistatic and sensing-first RF mode.** A
  hardware-gated Pi Zero 2W enables the sensing-first mode to actually
  hibernate the heavy compute, which ADR-031's power model assumes but the
  current node cannot deliver because heavy compute lives off-node.
- **ADR-032 — Multistatic mesh security hardening.** HMAC-SHA256 beacon
  auth + SipHash-2-4 frame integrity in ADR-032 already cover the
  inter-node bus. The proposal adds Secure Boot V2 + flash encryption
  at-rest on each MCU, and a signed Pi A/B image, which are *complements*
  to ADR-032, not substitutes.

---

## 2. Motivating thesis

A WiFi/RF sensing node has three jobs that prefer three different
runtimes:

1. **Strict-real-time radio capture and DSP** — sub-millisecond ISR
   discipline, no allocator surprises, predictable interrupt latency.
2. **Networking, OTA, mesh, time sync** — TCP/IP, TLS, BLE provisioning,
   ESP-WIFI-MESH, OTA bootloaders, NVS. The full battery of WiFi-stack
   features that come with ESP-IDF and FreeRTOS.
3. **Heavy compute, ML inference, storage, fleet sync** — gigabytes of
   model weights, vision inference, persistent storage, QUIC-based fleet
   sync, optional cloud APIs.

Today's WeView node tries to fit jobs 1 and 2 onto one ESP32-S3, and job 3
either runs on a separate machine (the "sensing-server" host) or is
absent. The thesis of this proposal is that **collapsing all three onto
a single PCB but onto three separate dies** captures most of the
"single node" simplicity without sacrificing the runtime properties of
each layer. Concretely:

- **Sensor MCU** — ESP32-S3, no_std, `esp-hal` + Embassy + `heapless` +
  `postcard`. ISR-driven CSI capture, channel hopping, short-window DSP.
  No WiFi stack of its own (the radio is in the comms MCU); a private
  UART or SPI link to the comms MCU carries serialized frames. *(See SOTA
  survey, §3, for the ISR-safety caveat that tempers this.)*
- **Comms MCU** — second ESP32-S3, ESP-IDF, `esp-idf-svc` + `esp-idf-sys`,
  TLS/HTTPS/OTA/ESP-WIFI-MESH, NVS provisioning, BLE provisioning, LoRa
  fallback. Owns the "outside world."
- **Pi Zero 2W** — *normally power-gated*. Wakes on event from the comms
  MCU, runs heavy ML or fleet-sync work, optionally streams QUIC to a
  gateway, then power-gates again. `tokio` + `quinn` + `rustls` + `axum`.

A single PCB, a single 1S Li-ion + 2 W solar + linear charger, a single
enclosure. Three separate cores each running the runtime they are
actually good at.

---

## 3. Hardware shape (proposed)

### 3.1 Bill of materials (per node, target)

| Slot                | Part                                             | Notes                                             |
|---------------------|--------------------------------------------------|---------------------------------------------------|
| Sensor MCU          | ESP32-S3-WROOM-1 (8 MB flash, 8 MB PSRAM)        | no_std, Embassy, esp-radio. Always-on.            |
| Comms MCU           | ESP32-S3-MINI-1 or -WROOM-1 (4 MB flash)         | ESP-IDF, ESP-WIFI-MESH, OTA, TLS. Mostly-on.      |
| Heavy compute       | Pi Zero 2W (1 GB RAM)                            | Power-gated by default. Wake on event.            |
| LoRa fallback       | Semtech SX1262 module                            | Heartbeat + recovery only. Sub-GHz.               |
| Charger / PMIC      | TI BQ24074 (linear) or BQ25798 (buck-boost MPPT) | See SOTA §7 for trade-off.                         |
| Battery             | 1S Li-ion 18650 (3.0 Ah class)                   | Standard cell, easy to source.                     |
| Solar panel         | ~2 W, 6 V, IP-rated                              | Roof-mount or window-mount.                        |
| Pi power gate       | Logic-level P-FET high-side switch + ESP GPIO    | Hard-cut when idle (350 mA → ~0 mA).               |
| Inter-MCU bus       | UART or SPI between sensor MCU and comms MCU     | Postcard-framed binary on a 4-wire link.           |
| Comms-to-Pi bus     | UART (115200–921600 bps) or SPI                  | Pi-side `tokio-serial`/`spidev`.                   |
| Enclosure           | IP54 or IP65 with antenna pass-through           | -                                                  |
| Estimated BOM       | $40–55                                           | At small build qty; falls with volume.             |

This is roughly 4–6× the ~$9 single-S3 node, which is the largest
single mark against the proposal. See §7.4 for whether the cost makes
sense.

### 3.2 Power-state hierarchy (proposed)

| State          | Sensor MCU       | Comms MCU       | Pi Zero 2W       | Approx draw     |
|----------------|------------------|-----------------|------------------|-----------------|
| Deep idle      | light sleep      | DTIM-modulated  | hard-off         | < 5 mA          |
| Sample window  | active CSI       | passive listen  | hard-off         | ~80 mA          |
| Event publish  | active CSI       | TX burst        | hard-off         | ~150 mA peak    |
| Escalation     | active CSI       | TX + bring-up   | booting          | ~350 mA peak    |
| ML in progress | active CSI       | passive         | inferencing      | ~450 mA         |
| Recovery       | sleep            | LoRa heartbeat  | hard-off         | ~30 mA          |

The Pi is treated as the heavyweight worker that **must** be hard-power-
gated — not soft-suspended — when not in use. ARM SoCs leak in
suspend; a 350 mA "off" leakage destroys solar viability.

### 3.3 Energy budget sketch

- **Daily load** (sketch, *not measured*): ~1.4 Wh/day assuming Pi wakes
  ≤ 2 minutes/day on average, sensor MCU light-sleeps when idle, comms
  MCU DTIM-3 most of the time.
- **Daily harvest**: 2 W panel × 4 PSH × 0.7 system efficiency ≈ 5.6
  Wh/day in the seasonal worst case for mid-latitudes.

Headroom is roughly 4×. If a deployment skews colder/cloudier, or the
inter-MCU bus runs hotter, headroom is 2–3×. SOTA §7 covers whether
the linear-charger + supercap-buffered topology actually delivers this
math, or whether MPPT is needed on a panel this small.

---

## 4. Software shape (proposed)

### 4.1 Sensor MCU — no_std embedded Rust

| Concern              | Crate(s)                                                     |
|----------------------|--------------------------------------------------------------|
| HAL / async runtime  | `esp-hal` 1.x + Embassy executor                             |
| Time / timers        | `embassy-time`                                               |
| Static allocations   | `heapless` (`Vec`, `String`, `Deque`, MPMC channels)         |
| Wire format          | `postcard` over `serde` for compact, schema-stable bytes     |
| CRC                  | `crc` crate (already used host-side for the L4 packet check) |
| RF capture           | `esp-radio` (the rename of `esp-wifi`) — CSI hooks via PR    |
| Inter-MCU bus        | `embassy-uart` or `embedded-hal-async` SPI                   |
| Power management     | `esp-hal::system::sleep::*` + light-sleep wake on GPIO/timer |

Boundary: the sensor MCU does **not** initialize a WiFi stack. It owns
the PHY for CSI capture only. All actual WiFi connectivity is on the
comms MCU. This is the load-bearing simplification of the proposal: it
sidesteps the embassy-on-ESP-IDF ISR-safety question by not running
ESP-IDF on this die at all.

### 4.2 Comms MCU — std + ESP-IDF Rust

| Concern              | Crate(s)                                                                 |
|----------------------|--------------------------------------------------------------------------|
| FreeRTOS bindings    | `esp-idf-sys`                                                            |
| Service abstractions | `esp-idf-svc` (HTTPS, OTA, NVS, mDNS, BLE, MQTT, ESP-NOW)                |
| Async runtime        | `esp-idf-svc::timer::EspTaskTimerService` (NOT Embassy directly — see §6)|
| TLS                  | mbedTLS via `esp-idf-svc`                                                |
| Mesh                 | ESP-WIFI-MESH (or ESP-MESH-LITE — see SOTA §8)                           |
| OTA                  | ESP-IDF native OTA (signed images, A/B partitions)                       |
| LoRa fallback        | `lora-phy` or vendor C driver via `esp-idf-sys`                          |
| Inter-MCU bus        | UART driver (`esp-idf-svc::uart`) framed with postcard                   |
| BLE provisioning     | NimBLE via `esp-idf-svc`                                                 |

The comms MCU is the *only* die that needs the full WiFi-stack security
surface. That makes it the obvious place to enforce Secure Boot V2 +
flash encryption + signed OTA.

### 4.3 Pi Zero 2W — std Rust on Linux

| Concern              | Crate(s)                                                              |
|----------------------|-----------------------------------------------------------------------|
| Async runtime        | `tokio`                                                               |
| QUIC                 | `quinn` + `rustls`                                                    |
| HTTP server (local)  | `axum`                                                                |
| RPC to comms MCU     | `tokio-serial` (UART) or `spidev` (SPI), framed with postcard         |
| ML inference         | `tract` (ONNX), `candle` (Pytorch-flavored), or `ort` (ONNX Runtime)  |
| Persistent storage   | `sled` or `redb`                                                      |
| OS                   | Buildroot-based custom image, A/B partitions, dm-verity, signed       |

Crucial constraint: the Pi runs **buildroot**, not Raspberry Pi OS. The
Raspberry Pi Foundation does not officially support secure boot on the
Pi Zero 2W; the secure-boot path is Pi 4/5-only. The cleanest path on a
Pi Zero 2W is buildroot + signed FIT image + dm-verity on the rootfs +
A/B partitions for OTA. See SOTA §9 for the realistic version of this.

### 4.4 OTA on three dies

| Die          | OTA mechanism                                                         |
|--------------|-----------------------------------------------------------------------|
| Sensor MCU   | `embassy-boot`-style two-slot OTA, signed images, ed25519 verification|
| Comms MCU    | ESP-IDF native OTA, signed by project key, dual app partitions        |
| Pi Zero 2W   | A/B rootfs, signed FIT, fwupd or homemade `update-agent` binary       |

OTA is the area where the three-tier shape is most defensible. Each die's
update is a separate, independently rollback-able artifact. The comms
MCU acts as the *broker* — it pulls signed images for all three dies,
verifies them, and pushes them onto the sensor MCU and Pi over their
respective buses.

---

## 5. Networking shape (proposed)

Three concentric rings:

1. **Inner ring — node-local IPC.** Postcard over UART/SPI between the
   three dies. Length-prefixed, CRC-checked, no encryption (it's on a
   trace, not a wire).
2. **Middle ring — WeView mesh.** ESP-WIFI-MESH (or ESP-MESH-LITE)
   between comms MCUs across nodes, carrying L3 mesh-plane messages
   from ADR-081 (TIME_SYNC, ROLE_ASSIGN, CHANNEL_PLAN, FEATURE_DELTA,
   HEALTH, ANOMALY_ALERT). Authenticated with HMAC-SHA256 per ADR-032.
3. **Outer ring — backhaul.** QUIC from the Pi to a gateway/cloud
   target (`quinn` + `rustls`), with the gateway optionally being
   another node's Pi acting as a fusion-relay. LoRa is the *fallback*
   ring for heartbeats and recovery commands when the WiFi mesh is
   degraded.

LoRa duty-cycle math (EU868 1% in the relevant sub-band, US915 dwell-
time-only) is friendly to "20 bytes every minute" heartbeats; at SF7,
125 kHz, the airtime is ~40 ms per packet — far under the 36 s/hour
EU868 limit. See SOTA §6 for the citation.

---

## 6. Security posture (proposed)

The proposal layers four mechanisms on each MCU:

- **Secure Boot V2** — RSA-3072 or ECDSA signed bootloader, immutable
  primary key digest in eFuse.
- **Flash encryption** — AES-XTS-256 with per-device key burned in eFuse,
  hardware-isolated.
- **Disabled ROM download** — `DIS_DOWNLOAD_MODE` fuse blown after
  provisioning so the device cannot be coerced back into a UART-ROM
  state.
- **Signed OTA images** — separate signing key from the secure-boot key,
  per-image rollback counter, anti-rollback eFuse counter.

On the Pi: dm-verity over a read-only rootfs, signed FIT image with the
RPi-foundation-blessed (where possible) bootcode, A/B partitions, and a
signed manifest of the three dies' image hashes shipped together. The
comms MCU validates the manifest before consuming any image.

This is **complementary** to ADR-032's HMAC-SHA256 + SipHash-2-4 mesh
hardening — those protect frames in flight; Secure Boot + flash
encryption protect images at rest.

---

## 7. Honest critique of this proposal

This section is required by the project conventions. The companion SOTA
survey expands each of these.

### 7.1 The cost story is bad before volume

A single ESP32-S3 node is ~$9 today. A three-tier node is closer to
$40–55. WeView's design point of "many cheap nodes" rewards low BOM. The
three-tier shape is justified only if each node *also* replaces a
sensing-server host (i.e., a Pi or laptop running the sensing pipeline)
that would have cost more than the marginal Pi-on-each-node. In a
deployment with 3 nodes feeding one $80 host, the host already amortizes
across the nodes. In a 50-node deployment, the math changes.

### 7.2 The embassy-on-ESP-IDF ISR-safety question is real

The proposal *avoids* this question by giving the sensor MCU a no_std
runtime instead of putting embassy on top of esp-idf-svc. The reason
this matters: per esp-idf-svc maintainers, **embassy-executor is not
ISR-safe** in the esp-idf-svc setup (it relies on `critical-section`,
which on esp-idf-hal is implemented over FreeRTOS task suspension). On
no_std with `esp-hal`, embassy is fine; on top of ESP-IDF, it is not.
The two-MCU split is the cleanest engineering answer to the question;
the alternative is keeping ESP-IDF on the single MCU (today's design)
and not introducing embassy at all. SOTA §3 documents the citation.

### 7.3 esp-radio replaces esp-wifi, and CSI no_std support is partial

The crate that the sensor MCU would use to capture CSI (in the
`esp-rs/esp-hal` 1.x ecosystem) was renamed to `esp-radio`. Third-party
`esp-csi-rs` exists and targets no_std but is described as
"early development." The 5-layer kernel today runs on top of ESP-IDF
v5.4 in C — a bird in the hand. Migrating CSI capture to no_std is a
distinct project, not a side effect of the three-tier shape. SOTA §2
covers the maturity matrix.

### 7.4 The Pi Zero 2W secure-boot story is weaker than the proposal implies

The Raspberry Pi Foundation's official secure-boot path is **Pi 4 / Pi 5
only**, with a USB-rooted RSA chain. There is no official secure-boot
bring-up document for the Pi Zero 2W. Buildroot + signed FIT + dm-verity
gets you most of the threat surface — but the proposal's "Pi 4 + buildroot
is the strongest path" line is not a Pi Zero 2W story. If true secure
boot matters for the deployment, the heavy-compute die should arguably
be a Pi 4 Compute Module (CM4) and not a Pi Zero 2W. SOTA §9 covers it.

### 7.5 ESP-WIFI-MESH at 50–500 nodes is an open question

Espressif documents up to 1,000 nodes and 25 layers as theoretical limits
for ESP-WIFI-MESH, with a recommended fan-out of 6 per node. There is
limited public evidence of stable 100+ node deployments in adversarial
RF environments. Comms-MCU mesh handling at scale is *not free*: the
mesh stack runs in the comms MCU's main loop, sharing CPU with TLS, OTA,
and BLE. SOTA §8 covers BLE Mesh / Thread / Zigbee comparison. None of
those replace WiFi-stack-sharing for CSI capture, but they could replace
ESP-WIFI-MESH for control-plane traffic if scale becomes a problem.

### 7.6 MPPT vs linear charger at 2 W panel

The proposal's BQ24074-based linear-charger topology is fine for a 2 W
panel; the efficiency loss vs MPPT is real but small at this scale.
At 2 W, the MPPT die (BQ25798) silicon, inductor, and code complexity
costs partly cancel its efficiency gain. SOTA §7 has the math.

### 7.7 The QUIC outer ring is overkill for the heartbeat case

QUIC is a strong choice when the Pi has lots of bursty data and is
behind a NAT or on flaky cellular. For a node that wakes 2 minutes/day
and emits a few KB of summarized features, MQTT-over-TLS or even
plain HTTPS is simpler and adequate. QUIC's value goes up if the Pi
also runs bidirectional model updates or large-batch fleet sync.
SOTA §5.

---

## 8. What evidence would justify acting on this proposal

This section maps to the decision tree in
`docs/research/architecture/decision-tree.md`. The short version:

1. **Per-node cost ceiling.** Decide the BOM ceiling per node. The
   three-tier shape only makes sense above ~$30/node and at deployments
   where the host computer is *not* a separate cost.
2. **CSI no_std maturity gate.** `esp-csi-rs` (or the replacement under
   `esp-radio`) must demonstrate equivalent capture quality to today's
   `esp-wifi-set-csi-rx-cb`-based path on a real ESP32-S3 board, with
   ISR-jitter measured. Until this is verified, the sensor-MCU Rust
   story is risk.
3. **Inter-MCU bus saturation.** Postcard-framed UART/SPI between the
   sensor MCU and comms MCU must carry ADR-018 frames at the target
   capture rate without backpressure-induced drops at the sensor MCU.
4. **Pi power-gate budget.** Measured leakage of the gated Pi Zero 2W,
   with proven cold-boot wake-up under 5 s, is required before the
   energy budget closes.
5. **Mesh scale evidence.** A 12+ node ESP-WIFI-MESH (or alternative)
   field test at sustained 1–10 Hz `rv_feature_state_t` upload is
   required to validate the middle ring at >>3 nodes.
6. **Secure-boot path on Pi Zero 2W.** Either accept that the Pi cannot
   be fully secure-booted, or upgrade the heavy-compute die to a CM4 /
   CM5 / Pi 5 if true secure boot is a deployment requirement.

---

## 9. Open questions

The proposal as written elides answers to these:

- **Why two ESP32-S3 dies and not one ESP32-S3 plus one ESP32-C6?** The
  C6 is RISC-V, has 802.15.4 + WiFi 6, and would let the comms MCU
  handle BLE Mesh / Thread / Zigbee natively. The two-S3 split chose
  homogeneity and Xtensa toolchain; the C6 split chooses richer
  protocol coverage on the comms die.
- **Is the sensor MCU strictly necessary?** Today, the single-MCU node
  (ADR-028 / ADR-081) handles CSI capture and ESP-IDF networking on one
  S3, in C, and works. The two-MCU-on-board case is justified mainly by
  *ISR purity* and *Rust no_std*, not by a missing capability today.
- **Why a Pi Zero 2W rather than the Pi being the gateway?** The
  proposal puts a Pi *on every node*. A more conservative shape is one
  Pi per *site* (or per cluster of 3–6 nodes), with the nodes staying
  single-MCU. That keeps the BOM near today's $9/node for sensors,
  isolates heavy compute, and concentrates secure boot on a smaller
  number of more capable dies. This is the deployment shape implicit in
  ADR-031's sensing-first mode and is worth comparing head-to-head.
- **What does a single 50-node deployment cost** under each of: today's
  shape (one S3 + one host), one-Pi-per-site (one S3 + one Pi per ~6
  nodes), and the proposal (3-die-per-node)? The cost crossover point
  determines which architecture is correct.

---

## 10. Recommendation

This document records the proposal accurately. It does not recommend
adopting it. The recommendation, if a decision is forced, is:

1. **Do not build a three-tier-per-node PCB now.** The current shape
   (single ESP32-S3 + ADR-081 5-layer kernel) is the witnessed system.
2. **Investigate one-Pi-per-site as the cheaper variant** (proposal §9
   bullet 3). It captures most of the heavy-compute and QUIC-backhaul
   benefits at a fraction of the BOM.
3. **Spend the first chunk of effort on the three "evidence" gates from
   §8** — CSI no_std maturity, ESP-WIFI-MESH at scale, and Pi
   secure-boot reality — *before* committing to a hardware re-spin.
4. **Reserve the three-tier shape** for a future "WeView Pro" SKU
   targeting deployments where per-node BOM is not the dominant cost
   and full secure-boot + dm-verity at the edge is mandatory.

The decision tree document codifies these gates as branch points so
they can be checked off independently rather than as one large
all-or-nothing ADR.

---

## 11. Companion documents

- **SOTA survey.** `docs/research/sota/2026-Q2-rf-sensing-and-edge-rust.md`
  — citations, primary sources, what's true in 2026 for each load-bearing
  claim above.
- **Decision tree.** `docs/research/architecture/decision-tree.md` — the
  Mermaid map from each load-bearing decision to its dependencies and
  ADR slot.
- **Existing implementation plan.** `docs/research/architecture/implementation-plan.md`
  — the ESP32-S3 + Pi Zero 2W goal-state plan from 2026-04-02. The
  three-tier proposal is most usefully read as an evolution of *that*
  plan rather than a replacement of ADR-028.
