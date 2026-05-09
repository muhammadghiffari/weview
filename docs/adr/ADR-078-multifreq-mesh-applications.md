# ADR-078: Multi-Frequency Mesh Sensing Applications

| Field       | Value                                      |
|-------------|--------------------------------------------|
| **Status**  | Proposed                                   |
| **Date**    | 2026-04-02                                 |
| **Authors** | ruv                                        |
| **Depends** | ADR-018 (binary frame), ADR-029 (channel hopping), ADR-073 (multi-frequency mesh scan) |

## Context

ADR-073 established multi-frequency mesh scanning: 2 ESP32-S3 nodes hopping across 6 WiFi channels (1, 3, 5, 6, 9, 11) with 9 neighbor WiFi networks as passive illuminators. This ADR defines 5 sensing applications that are **unique to multi-frequency mesh scanning** and impossible with single-channel WiFi sensing.

### Why Multi-Frequency is Required

Single-channel WiFi sensing captures CSI on one frequency (e.g., channel 5 at 2432 MHz). This provides amplitude and phase across ~52-64 OFDM subcarriers within a 20 MHz bandwidth. Multi-frequency mesh scanning extends this to 6 channels spanning 2412-2462 MHz (50 MHz total), with each channel providing independent multipath observations. The applications below exploit the frequency dimension that single-channel sensing cannot access.

### Available Infrastructure

| Resource | Detail |
|----------|--------|
| Node 1 (COM7) | ESP32-S3, channels 1, 6, 11 (non-overlapping), 200ms dwell |
| Node 2 | ESP32-S3, channels 3, 5, 9 (interleaved, near neighbor APs), 200ms dwell |
| Neighbor APs | 9 networks across channels 3, 5, 6, 9, 11 |
| Data transport | UDP port 5006, ADR-018 binary format |
| Recorded data | `data/recordings/overnight-*.csi.jsonl` |

### Neighbor AP Illuminator Table

| SSID | Channel | Freq (MHz) | Signal (%) | Role |
|------|---------|------------|------------|------|
| ruv.net | 5 | 2432 | 100 | Primary illuminator |
| Cohen-Guest | 5 | 2432 | 100 | Co-channel illuminator |
| COGECO-21B20 | 11 | 2462 | 100 | High-freq illuminator |
| HP M255 LaserJet | 5 | 2432 | 94 | Device fingerprinting target |
| conclusion mesh | 3 | 2422 | 44 | Low-freq illuminator |
| NETGEAR72 | 9 | 2452 | 42 | Mid-high illuminator |
| NETGEAR72-Guest | 9 | 2452 | 42 | Co-channel illuminator |
| COGECO-4321 | 11 | 2462 | 30 | Weak high-freq illuminator |
| Innanen | 6 | 2437 | 19 | Weak center-band illuminator |

## Decision

Implement 5 multi-frequency-specific sensing applications, each as a standalone Node.js script in `scripts/`.

---

## Application 1: RF Tomographic Imaging

### Principle

Each WiFi channel "sees" through the room differently because multipath interference patterns are frequency-dependent. A 2 cm path length difference produces a null at 2432 MHz but constructive interference at 2412 MHz. With 6 channels x 2 nodes, we have 12 independent RF path observations through the room.

RF tomography back-projects attenuation along each transmitter-receiver path. Where paths overlap with high attenuation, there is an absorbing object (person, furniture, wall). Where paths show low attenuation, the space is clear.

### Algorithm

```
For each CSI frame:
  1. Compute path attenuation = RSSI_free_space - RSSI_measured
  2. For each cell in a 10x10 room grid:
     a. Compute the cell's distance to the TX->RX line (perpendicular distance)
     b. Weight contribution by 1/distance (cells near the path contribute more)
  3. Accumulate weighted attenuation across all frames, channels, and node pairs
  4. Normalize: cells with high accumulated attenuation = absorbers (people/objects)
```

Uses the Algebraic Reconstruction Technique (ART) for iterative refinement, or simple backprojection for real-time display.

### Resolution

- Theoretical: ~lambda/2 = 6 cm (at 2.4 GHz)
- Practical with 2 nodes: ~20 cm (limited by node geometry)
- Frequency diversity gain: sqrt(6) improvement over single-channel = ~2.4x

### Why Single-Channel Cannot Do This

Single-channel provides only 1 frequency observation per path. Frequency-selective fading means a single channel may show zero attenuation through a person (if the path happens to be at a constructive interference point). Multiple channels provide independent attenuation measurements through the same spatial path, enabling reliable detection.

### Script

`scripts/rf-tomography.js`

---

## Application 2: Passive Bistatic Radar

### Principle

Neighbor WiFi APs transmit continuously and uncontrollably. The ESP32 nodes capture CSI from these transmissions, which includes phase and amplitude modulated by objects in the room. Each neighbor AP acts as a free "illuminator of opportunity" at a known position and frequency.

This is the same principle used by military passive radar systems (e.g., the Ukrainian Kolchuga, Czech VERA-NG) that use FM radio and TV transmitters to detect aircraft without emitting any signals themselves. Here we use WiFi APs instead of broadcast towers, and detect people instead of aircraft.

### Algorithm

```
For each neighbor AP (identified by BSSID/channel):
  1. Track CSI phase progression across consecutive frames
  2. Compute Doppler shift: fd = d(phase)/dt / (2*pi)
     - Positive Doppler = target moving toward the AP
     - Negative Doppler = target moving away
  3. Compute range from subcarrier phase slope:
     - tau = d(phase)/d(subcarrier_freq) / (2*pi)
     - range = c * tau (where c = speed of light)
  4. Build range-Doppler map per AP
  5. Fuse multi-static detections:
     - Each AP provides a range ellipse (locus of constant TX->target->RX delay)
     - Intersection of 3+ ellipses = target position
```

### Multi-Static Geometry

With 3+ neighbor APs as transmitters and 2 ESP32 receivers, we have 6+ bistatic pairs. Each pair constrains the target to an ellipse. The intersection provides 2D position.

```
         AP1 (ch5)        AP2 (ch11)
           \                /
            \   TARGET    /
             \   /|\    /
              \ / | \ /
    ESP32_1 ---*--+--*--- ESP32_2
              / \ | / \
             /   \|/    \
            /   TARGET   \
           /                \
         AP3 (ch3)        AP4 (ch9)
```

### Why Single-Channel Cannot Do This

Single-channel only captures CSI from APs on that one channel. With channel 5, you see ruv.net and Cohen-Guest, but miss COGECO-21B20 (ch11), conclusion mesh (ch3), NETGEAR72 (ch9). Multi-frequency scanning captures illumination from all 9 APs across 6 channels, providing the geometric diversity needed for position triangulation.

### Script

`scripts/passive-radar.js`

---

## Application 3: Frequency-Selective Material Classification

### Principle

Different materials interact with 2.4 GHz WiFi signals differently, and critically, their absorption/reflection varies with frequency:

| Material | Attenuation Pattern | Frequency Dependence |
|----------|--------------------|--------------------|
| Metal | Total reflection, deep null | Frequency-flat (blocks all equally) |
| Water/Human body | Strong absorption | Increases with frequency (dielectric loss ~ f^2) |
| Wood | Mild attenuation | Increases with frequency (moisture content) |
| Glass | Low attenuation | Nearly frequency-flat |
| Drywall | Low-moderate attenuation | Slight frequency dependence |
| Concrete | Moderate-high attenuation | Increases with frequency |

### Algorithm

```
For each subcarrier index i across all channels:
  1. Measure attenuation A(i, ch) on each channel
  2. Compute frequency selectivity:
     - Flat ratio = std(A across channels) / mean(A across channels)
     - Slope = linear regression of A vs frequency
  3. Classify:
     - Flat ratio < 0.1 AND high attenuation -> Metal
     - Flat ratio < 0.1 AND low attenuation -> Glass/Air
     - Positive slope (A increases with freq) AND high A -> Water/Human
     - Positive slope AND moderate A -> Wood
     - High variance across channels -> Complex scatterer
```

### Physics Basis

At 2.4 GHz, water's complex permittivity is epsilon_r = 77 - j10. The imaginary component (loss) increases with frequency within the WiFi band. Metal is a perfect conductor regardless of frequency. Glass (epsilon_r ~ 6 - j0.1) has negligible loss at all WiFi frequencies.

The 50 MHz span (2412-2462 MHz) is only ~2% of the carrier frequency, but this is sufficient to detect the frequency-dependent absorption signature of water-bearing materials (human body, wet wood, potted plants) versus frequency-flat materials (metal, glass).

### Why Single-Channel Cannot Do This

Material classification requires measuring how attenuation varies with frequency. A single channel provides only one frequency point -- there is no frequency axis to measure against. Multi-frequency scanning provides 6 frequency points spanning 50 MHz, enabling slope and variance computation.

### Script

`scripts/material-classifier.js`

---

## Application 4: Through-Wall Motion Detection

### Principle

Lower WiFi frequencies penetrate walls better than higher frequencies. At 2.4 GHz, wall attenuation for a standard drywall+stud partition is approximately:

| Channel | Freq (MHz) | Drywall Loss (dB) | Concrete Loss (dB) |
|---------|------------|-------------------|-------------------|
| 1 | 2412 | 2.5 | 8.0 |
| 6 | 2437 | 2.6 | 8.3 |
| 11 | 2462 | 2.7 | 8.6 |

The absolute differences are small (~0.2 dB), but with 6 channels we can:

1. **Baseline the wall's frequency-dependent attenuation profile** during a calibration period (no one behind the wall)
2. **Detect changes above baseline** that indicate motion behind the wall
3. **Weight lower channels more heavily** since they have better through-wall SNR
4. **Cross-validate** across channels: real through-wall motion appears on all channels (with frequency-dependent amplitude), while interference/noise typically appears on only one channel

### Algorithm

```
Calibration phase (60 seconds, no motion behind wall):
  For each channel ch:
    baseline_mean[ch] = mean(CSI amplitude over calibration)
    baseline_std[ch] = std(CSI amplitude over calibration)

Detection phase:
  For each frame on channel ch:
    1. Compute deviation = |current_amplitude - baseline_mean[ch]| / baseline_std[ch]
    2. Channel weight = f(penetration_quality[ch])
    3. Per-channel score = deviation * weight
  
  Fused score = weighted sum across channels
  Alert if fused_score > threshold for N consecutive frames
```

### Why Single-Channel Cannot Do This

Single-channel through-wall detection suffers from high false-positive rates because it cannot distinguish wall effects from motion. With multi-frequency, we can:

1. Characterize the wall's frequency response during calibration
2. Subtract the wall effect per channel
3. Cross-validate detections across channels (real motion is coherent across frequencies; noise is not)

The frequency diversity provides a ~2.4x improvement in detection SNR (sqrt(6) independent observations).

### Script

`scripts/through-wall-detector.js`

---

## Application 5: Device Fingerprinting via RF Emissions

### Principle

Every electronic device has unique RF characteristics visible in the WiFi spectrum. When a device transmits (or even when its internal oscillators radiate EMI), it modulates nearby WiFi signals in device-specific ways:

- **WiFi APs**: each AP has unique transmit power, phase noise, and clock drift characteristics
- **Printers**: the HP M255 LaserJet creates specific subcarrier patterns when printing (motor EMI)
- **Microwave ovens**: 2.45 GHz magnetron radiates across channels 8-11, creating distinctive wideband interference
- **Bluetooth devices**: 2.4 GHz frequency-hopping creates transient spikes across channels

### Algorithm

```
Learning phase:
  For each known device (from WiFi scan SSID/BSSID correlation):
    1. Record CSI patterns when device is active vs inactive
    2. Compute per-channel signature:
       - Mean amplitude profile across subcarriers
       - Variance profile (active devices increase variance on specific subcarriers)
       - Phase noise characteristics
    3. Store signature as device fingerprint

Detection phase:
  For each analysis window:
    1. Compute current CSI profile per channel
    2. Correlate against stored fingerprints
    3. Report device activity: "HP printer active (confidence 0.87)"
```

### Multi-Frequency Advantage

Different devices affect different channels:

- HP printer (ch5): affects subcarriers 20-40 on channel 5 during print jobs
- NETGEAR72 router (ch9): creates clock-drift correlated phase patterns on channel 9
- Microwave: broadband interference strongest on channels 9-11

Single-channel sensing only sees devices that affect that one channel. Multi-frequency scanning observes the full 2412-2462 MHz band, detecting device activity regardless of which channel the device operates on.

### Script

`scripts/device-fingerprint.js`

---

## Implementation

### Shared Infrastructure

All 5 scripts share common infrastructure:

| Component | Detail |
|-----------|--------|
| Packet format | ADR-018 binary (UDP) or .csi.jsonl (replay) |
| IQ parsing | `parseIqHex()` for JSONL, `parseCSIFrame()` for binary UDP |
| Channel assignment | From binary freq field, or simulated round-robin for legacy JSONL |
| Node positions | Configurable, default: Node 1 at (0,0), Node 2 at (3,0) meters |
| Visualization | ASCII Unicode block characters and box drawing |

### Scripts

| Script | Application | Lines | Key Algorithm |
|--------|------------|-------|---------------|
| `scripts/rf-tomography.js` | RF Tomographic Imaging | ~500 | ART backprojection |
| `scripts/passive-radar.js` | Passive Bistatic Radar | ~500 | Range-Doppler + multi-static fusion |
| `scripts/material-classifier.js` | Material Classification | ~450 | Frequency-selective attenuation analysis |
| `scripts/through-wall-detector.js` | Through-Wall Detection | ~400 | Baselined multi-channel anomaly detection |
| `scripts/device-fingerprint.js` | Device Fingerprinting | ~450 | Per-channel signature correlation |

### Data Requirements

- **Live mode**: UDP port 5006, 2 ESP32 nodes channel-hopping per ADR-073
- **Replay mode**: `--replay <file.csi.jsonl>` with overnight recordings
- **Calibration**: through-wall detector requires 60s calibration with `--calibrate`

## Performance Targets

| Application | Latency | Update Rate | Accuracy Target |
|-------------|---------|-------------|-----------------|
| RF Tomography | <100ms per frame | 1 Hz image update | 20 cm spatial resolution |
| Passive Radar | <200ms per frame | 2 Hz range-Doppler | 1 m range, 0.1 m/s velocity |
| Material Classification | <500ms per window | 0.5 Hz classification | 70% correct material ID |
| Through-Wall Detection | <100ms per frame | 2 Hz detection | 90% true positive, <10% false positive |
| Device Fingerprinting | <1s per window | 0.2 Hz activity update | 80% correct device ID |

## Risks

### Limited Frequency Span

The 50 MHz span (2412-2462 MHz) is only 2% of the carrier frequency. Material classification accuracy depends on the attenuation slope being measurable within this narrow range. Mitigation: use long averaging windows (5-10 seconds) to improve SNR of frequency-dependent measurements.

### Node Geometry

2 nodes provide limited spatial diversity for tomographic imaging. The backprojection is essentially 1D along the node-to-node axis, with poor resolution perpendicular to it. Mitigation: neighbor APs provide additional geometric diversity for passive radar mode.

### Legacy Data Compatibility

Overnight recordings (`data/recordings/overnight-*.csi.jsonl`) were captured before multi-frequency scanning was deployed and lack channel/frequency fields. Scripts simulate channel assignment for replay. Full multi-frequency data requires re-recording with channel hopping enabled.

### Phase Calibration

Passive radar requires accurate phase tracking across consecutive frames. ESP32 CSI phase includes a random offset per channel hop that must be removed. Mitigation: use phase-difference between consecutive frames rather than absolute phase.

## Alternatives Considered

1. **5 GHz multi-frequency**: rejected -- no 5 GHz APs visible in environment, no free illuminators.
2. **UWB (ultra-wideband)**: rejected -- ESP32-S3 does not support UWB. Would require additional hardware (DW1000/DW3000 modules).
3. **Dedicated radar hardware**: rejected -- multi-frequency WiFi sensing achieves similar capabilities using existing infrastructure at zero additional cost.

## References

- Wilson, J. & Patwari, N. (2010). "Radio Tomographic Imaging with Wireless Networks." IEEE Trans. Mobile Computing.
- Colone, F. et al. (2012). "WiFi-Based Passive Bistatic Radar: Data Processing Schemes and Experimental Results." IEEE Trans. Aerospace and Electronic Systems.
- Adib, F. & Katabi, D. (2013). "See Through Walls with WiFi!" ACM SIGCOMM.
- Banerjee, A. et al. (2014). "RF-based material identification using WiFi signals." ACM MobiCom.
