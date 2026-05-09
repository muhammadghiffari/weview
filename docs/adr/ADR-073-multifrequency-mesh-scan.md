# ADR-073: Multi-Frequency Mesh Scanning

| Field       | Value                                      |
|-------------|--------------------------------------------|
| **Status**  | Proposed                                   |
| **Date**    | 2026-04-02                                 |
| **Authors** | ruv                                        |
| **Depends** | ADR-018 (binary frame), ADR-029 (channel hopping), ADR-039 (edge processing), ADR-060 (channel override) |

## Context

The current WiFi-DensePose deployment uses 2 ESP32-S3 nodes operating on a single WiFi channel (channel 5, 2432 MHz). A scan of the office environment reveals 9 WiFi networks across 6 distinct channels (1, 3, 5, 6, 9, 11), each broadcasting continuously. These neighbor networks are free RF illuminators whose signals pass through the room and interact with objects, people, and walls.

**Current single-channel limitations:**

1. **19% null subcarriers** — metal objects (desk, monitor frame, filing cabinet) create frequency-selective fading that blocks specific subcarriers on channel 5. These nulls are permanent blind spots in the RF map.

2. **No frequency diversity** — objects that are transparent at 2432 MHz may be opaque at 2412 MHz or 2462 MHz, and vice versa. A metal mesh that blocks one wavelength (122.5 mm at 2432 MHz) may pass another (124.0 mm at 2412 MHz) due to the mesh aperture-to-wavelength ratio.

3. **Single-perspective CSI** — both nodes see the same 52-64 subcarriers on the same channel. The subcarrier indices map to the same frequency bins, providing no spectral diversity.

4. **Neighbor illuminator waste** — 6 other APs broadcast continuously in the room. Their signals pass through walls, furniture, and people, creating CSI-measurable reflections that we currently ignore because we only listen on channel 5.

## Decision

Implement interleaved multi-frequency channel hopping across the 2 ESP32-S3 nodes, scanning 6 WiFi channels to build a wideband RF map of the room.

### Channel Allocation Strategy

The 2.4 GHz ISM band has 3 non-overlapping 20 MHz channels (1, 6, 11) and several partially-overlapping channels between them. We allocate channels to maximize both spectral coverage and illuminator exploitation:

```
Node 1: ch 1, 6, 11  (non-overlapping, full band coverage)
Node 2: ch 3, 5, 9   (interleaved, near neighbor APs)
```

**Rationale for this split:**

| Channel | Freq (MHz) | Node | Neighbor Illuminators                        | Purpose                           |
|---------|------------|------|----------------------------------------------|-----------------------------------|
| 1       | 2412       | 1    | (none visible, but lower freq = better penetration) | Low-frequency penetration          |
| 3       | 2422       | 2    | conclusion mesh (signal 44)                  | Exploit neighbor AP as illuminator |
| 5       | 2432       | 2    | ruv.net (100), Cohen-Guest (100), HP LaserJet (94) | Primary channel, strongest illuminators |
| 6       | 2437       | 1    | Innanen (signal 19)                          | Center band, non-overlapping       |
| 9       | 2452       | 2    | NETGEAR72 (42), NETGEAR72-Guest (42)         | Exploit dual NETGEAR illuminators  |
| 11      | 2462       | 1    | COGECO-21B20 (100), COGECO-4321 (30)         | High-frequency, strong illuminators |

Each node dwells on a channel for 250 ms (configurable), collects 3-4 CSI frames, then hops to the next. The 3-channel rotation completes in 750 ms, giving ~1.3 full rotations per second.

### Physics Basis

At 2.4 GHz, WiFi wavelength ranges from 122.0 mm (ch 14, 2484 MHz) to 124.0 mm (ch 1, 2412 MHz). While this is a narrow range (~2%), the effect on multipath is significant:

1. **Frequency-selective fading**: multipath reflections create constructive/destructive interference patterns that vary with frequency. A 2 cm path length difference produces a null at 2432 MHz but constructive interference at 2412 MHz.

2. **Diffraction around objects**: Huygens-Fresnel diffraction depends on wavelength. Objects smaller than ~lambda/2 (61 mm) scatter differently across the band. Common office objects (monitor bezels, chair legs, cable bundles) are in this range.

3. **Material transparency**: some materials (wire mesh, perforated metal, PCB ground planes) have frequency-dependent transmission. A monitor's EMI shielding mesh with 5 mm apertures blocks 2.4 GHz signals but the exact attenuation varies with frequency due to slot antenna effects.

4. **Subcarrier orthogonality**: OFDM subcarriers on different channels are in different frequency bins. A null on subcarrier 15 of channel 5 does not imply a null on subcarrier 15 of channel 1, because they map to different absolute frequencies.

### Null Diversity Mechanism

```
Channel 5 subcarriers:  ▅▆█▇▅▃▁_▁▃▅▆█▇▅▃▁_▁▃▅▆█▇▅▃
                                 ^ null (metal desk)
Channel 1 subcarriers:  ▃▅▆█▇▅▃▅▆█▇▅▃▅▆█▇▅▃▅▆█▇▅▃▅▃
                                 ^ resolved! Different freq = different null pattern

Channel 11 subcarriers: ▅▃▁_▁▃▅▆█▇▅▃▅▆▅▃▁_▁▃▅▆█▇▅▃▅
                              ^ null here instead (shifted by frequency offset)
```

By fusing subcarrier data across channels, nulls that exist on one channel are filled by non-null data from other channels. The remaining nulls (present on ALL channels) represent truly opaque objects — large metal surfaces that block all 2.4 GHz frequencies.

### Wideband View

Single channel: ~52-64 subcarriers (20 MHz bandwidth)
Multi-channel (6 channels): ~312-384 effective subcarrier observations (120 MHz coverage)

This is not simply 6x the resolution (the subcarrier spacing within each channel is the same), but it provides:
- 6x the spectral diversity for null mitigation
- 6x the illuminator variety (different APs = different signal paths)
- Frequency-dependent scattering signatures for material classification

## Integration

### Firmware (already supported)

The channel hopping infrastructure is already implemented in the ESP32 firmware (ADR-029):

```c
// csi_collector.h — already exists
void csi_collector_set_hop_table(const uint8_t *channels, uint8_t hop_count, uint32_t dwell_ms);
void csi_collector_start_hop_timer(void);
```

The ADR-018 binary frame header already includes the channel/frequency field at bytes [8..11], so the server-side parser can distinguish frames from different channels without any firmware changes.

### Provisioning Commands

```bash
# Node 1 (COM7): non-overlapping channels 1, 6, 11
python firmware/esp32-csi-node/provision.py --port COM7 \
  --ssid "ruv.net" --password "..." --target-ip 192.168.1.20 \
  --hop-channels 1,6,11 --hop-dwell-ms 250

# Node 2 (COM_): interleaved channels 3, 5, 9
python firmware/esp32-csi-node/provision.py --port COM_ \
  --ssid "ruv.net" --password "..." --target-ip 192.168.1.20 \
  --hop-channels 3,5,9 --hop-dwell-ms 250
```

Note: `--hop-channels` and `--hop-dwell-ms` require provision.py support for writing these values to NVS. If not yet implemented, the firmware's `csi_collector_set_hop_table()` can be called directly from the main init code with compile-time constants.

### Server-Side Processing

Three new Node.js scripts consume the multi-channel CSI data:

| Script | Purpose |
|--------|---------|
| `scripts/rf-scan.js` | Single-channel live RF room scanner with ASCII spectrum |
| `scripts/rf-scan-multifreq.js` | Multi-channel scanner with null diversity analysis |
| `scripts/benchmark-rf-scan.js` | Quantitative benchmark of multi-channel performance |

All scripts parse the ADR-018 binary UDP format and use the frequency field to separate frames by channel.

### Cognitum Seed Integration

The Cognitum Seed vector store (ADR-069) currently stores 1,605 vectors from single-channel CSI. With multi-frequency scanning:

1. **Per-channel feature vectors**: store separate 8-dim feature vectors for each channel, tagged with channel number. This increases the vector count to ~9,630 (6 channels x 1,605).

2. **Wideband feature vector**: concatenate or average per-channel features into a 48-dim wideband vector for richer kNN search. Objects that are ambiguous on one channel may be clearly distinguishable in the wideband representation.

3. **Null-aware embeddings**: encode null subcarrier patterns as part of the feature vector. The null pattern itself is informative — a consistent null at subcarrier 15 across all channels indicates a large metal object, while a null only on channel 5 indicates a frequency-dependent scatterer.

## Performance Targets

| Metric | Single-Channel Baseline | Multi-Channel Target | Method |
|--------|------------------------|---------------------|--------|
| Subcarrier count | ~52-64 | ~312-384 (6x) | 6 channels x 52-64 subcarriers |
| Null gap | 19% | <5% | Null diversity across channels |
| Position resolution | ~30 cm | ~15 cm | sqrt(6) improvement from independent observations |
| Per-channel FPS | 12 fps | ~4 fps | 250 ms dwell x 3 channels = 750 ms rotation |
| Total FPS (all channels) | 12 fps | ~12 fps per node (4 fps x 3 channels) |
| Wideband rotation | N/A | ~1.3 Hz | Full 3-channel rotation in 750 ms |

## Risks

### Per-Channel Sample Rate Reduction

Channel hopping reduces the per-channel sample rate from 12 fps (single channel) to approximately 4 fps per channel (250 ms dwell, 3 channels). This affects:

- **Vitals extraction**: breathing rate (0.1-0.5 Hz) requires at least 2 fps (Nyquist). At 4 fps per channel, this is met. Heart rate (0.8-2.0 Hz) requires at least 4 fps, which is marginal. Mitigation: keep one channel as "primary" with longer dwell for vitals, or fuse phase data across channels.

- **Motion tracking**: 4 fps is sufficient for walking speed (<2 m/s) but insufficient for fast gestures. If gesture recognition is needed, reduce to 2-channel hopping or increase dwell rate.

### Channel Hopping Latency

`esp_wifi_set_channel()` takes ~1-5 ms on ESP32-S3. During the transition, no CSI frames are captured. At 250 ms dwell, this is <2% overhead.

### AP Disconnection

Channel hopping may cause the ESP32 to lose connection to the home AP (ruv.net on channel 5) when dwelling on other channels. The STA reconnects automatically, but there may be brief UDP packet loss. Mitigation: the firmware already handles this gracefully — CSI collection works in promiscuous mode regardless of STA connection state.

### Increased Server Load

2 nodes x 3 channels x 4 fps = 24 frames/second total UDP traffic. Each frame is ~150-200 bytes (20-byte header + 64 subcarriers x 2 bytes I/Q). Total: ~4.8 KB/s — negligible.

## Alternatives Considered

1. **5 GHz channels**: ESP32-S3 supports 5 GHz CSI, and the shorter wavelength (60 mm) provides better spatial resolution. Rejected because: (a) no 5 GHz APs visible in the current environment, so no free illuminators; (b) 5 GHz has worse wall penetration, reducing the effective sensing volume.

2. **More nodes**: adding a 3rd or 4th ESP32 node would increase spatial diversity without channel hopping. Rejected for now due to cost, but this is complementary — more nodes + channel hopping would give both spatial and spectral diversity.

3. **Wider bandwidth (HT40)**: using 40 MHz channels doubles subcarrier count per channel. Rejected because: (a) HT40 requires a secondary channel, reducing available channels for hopping; (b) many neighbor APs use HT20, so their illumination only covers 20 MHz.

## SNN Integration (ADR-074)

Multi-frequency scanning produces subcarrier data across 6 channels, creating temporal patterns that are well-suited for spiking neural network processing. ADR-074 introduces an SNN with STDP learning that consumes the multi-channel CSI stream.

**Key interactions with multi-frequency data:**

1. **Null diversity as SNN input**: subcarriers that are null on one channel but active on another produce a distinctive spike pattern (spikes only during certain channel dwells). STDP learns to associate these cross-channel patterns with specific objects or zones — something a single-channel SNN cannot do.

2. **Channel-interleaved temporal coding**: because each node dwells on 3 channels in a 750ms rotation, the SNN receives subcarrier data in a repeating temporal pattern (ch1 → ch2 → ch3 → ch1 ...). The SNN's LIF membrane dynamics integrate spikes across the rotation, naturally performing cross-channel fusion through temporal summation. A hidden neuron that receives spikes from subcarrier 15 on channel 1 AND subcarrier 15 on channel 6 will fire more strongly than one receiving either alone.

3. **Expanded input mode**: on the server (not constrained by ESP32 memory), the SNN can use 384 input neurons (6 channels x 64 subcarriers) instead of 128. This provides maximum spectral diversity per frame but requires ~150 KB of weight storage. The `snn-csi-processor.js` script supports this via the `--hidden` flag to scale the network.

4. **Illuminator fingerprinting**: different neighbor APs have different beamforming patterns and power levels. The SNN learns which subcarrier patterns belong to which illuminator, enabling it to distinguish AP-specific signatures from human-caused perturbations. This is especially useful for the NETGEAR dual-AP setup on channel 9, where two illuminators from different positions create stereo-like RF coverage.

## References

- ADR-018: CSI binary frame format
- ADR-029: Channel hopping infrastructure
- ADR-039: Edge processing pipeline
- ADR-060: Channel override provisioning
- ADR-069: Cognitum Seed CSI pipeline
- ADR-074: Spiking neural network for CSI sensing
- IEEE 802.11-2020, Section 21 (OFDM PHY)
- ESP-IDF CSI Guide: https://docs.espressif.com/projects/esp-idf/en/v5.4/esp32s3/api-guides/wifi.html#wi-fi-channel-state-information
