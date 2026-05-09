# ADR-077: Novel RF Sensing Applications

**Status:** Accepted  
**Date:** 2026-04-02  
**Authors:** ruv  
**Depends on:** ADR-018 (CSI binary protocol), ADR-073 (multifrequency mesh scan), ADR-075 (MinCut person separation), ADR-076 (CSI spectrogram embeddings)

## Context

The existing ESP32 CSI + Cognitum Seed infrastructure collects rich multi-modal data:
- 2 ESP32-S3 nodes streaming CSI at ~22 fps each (64-128 subcarriers, channel hopping ch 1/3/5/6/9/11)
- Vitals extraction: breathing rate, heart rate, motion energy, presence score (1 Hz per node)
- 8-dimensional feature vectors per frame
- Cognitum Seed with BME280 (temp/humidity/pressure), PIR, reed switch, vibration sensor

No new hardware is required. All 6 applications below derive novel insights from data already being collected via the ADR-018 binary protocol over UDP port 5006.

## Decision

Implement 6 novel RF sensing applications as standalone Node.js scripts that process live UDP or replayed `.csi.jsonl` recordings.

---

## Application 1: Sleep Quality Monitoring

### Input
Breathing rate (BR) and heart rate (HR) time series from vitals packets (0xC5110002), sampled at ~1 Hz per node over 6-8 hours.

### Algorithm
Sliding window analysis (5-minute windows, 1-minute stride) classifying sleep stages:

| Stage | BR (BPM) | BR Variance | HR Pattern | Motion |
|-------|----------|-------------|------------|--------|
| **Deep (N3)** | 6-12 | Very low (<2.0) | Slow, regular | None |
| **Light (N1/N2)** | 12-18 | Moderate (2.0-8.0) | Normal | Minimal |
| **REM** | 15-25 | High (>8.0), irregular | Elevated | Eyes only (low CSI motion) |
| **Awake** | >18 or <6 | Any | Variable | Moderate-high |

Each 5-minute window is scored by:
1. Compute BR mean and variance within the window
2. Compute HR mean and coefficient of variation (CV)
3. Compute motion energy mean (from vitals `motion_energy` field)
4. Classify stage using threshold hierarchy: Awake > REM > Light > Deep

### Output
- Real-time sleep stage classification
- ASCII hypnogram (time vs. stage)
- Summary: total sleep time, sleep efficiency (TST / time in bed), time per stage
- Optional JSON for health app integration

### Validation
Overnight recording (`overnight-1775217646.csi.jsonl`, 113k frames, ~40 min) should show:
- Transition from active (awake) to resting states
- Decreased motion energy over time
- BR stabilization in sleeping segments

### Clinical Relevance
Consumer-grade sleep tracking without wearables. RF-based sensing avoids compliance issues (forgotten wristbands, dead batteries). Not diagnostic; informational only.

---

## Application 2: Breathing Disorder Screening (Apnea Detection)

### Input
Breathing rate time series from vitals packets at ~1 Hz.

### Algorithm
Detect respiratory events in the BR time series:

| Event | Definition | Duration |
|-------|-----------|----------|
| **Apnea** | BR drops below 3 BPM (effective cessation) | >= 10 seconds |
| **Hypopnea** | BR drops > 50% from 5-min rolling baseline | >= 10 seconds |

Scoring:
1. Maintain 5-minute rolling baseline BR (exponential moving average)
2. Flag apnea when BR < 3 BPM for >= 10 consecutive seconds
3. Flag hypopnea when BR < 50% of baseline for >= 10 consecutive seconds
4. Compute AHI (Apnea-Hypopnea Index) = total events / hours monitored

| AHI | Severity |
|-----|----------|
| < 5 | Normal |
| 5-15 | Mild |
| 15-30 | Moderate |
| > 30 | Severe |

### Output
- Per-event log: type (apnea/hypopnea), start time, duration, BR during event
- Hourly AHI and overall AHI
- Severity classification
- Alert on severe events (consecutive apneas > 30s)

### Clinical Relevance
Pre-screening tool for obstructive sleep apnea (OSA). Provides motivation for clinical polysomnography referral. Not a diagnostic device; informational pre-screen only.

---

## Application 3: Emotional State / Stress Detection

### Input
Heart rate time series from vitals packets at ~1 Hz.

### Algorithm
Heart Rate Variability (HRV) analysis:

1. **RMSSD** (Root Mean Square of Successive Differences):
   - Compute successive HR differences within 5-minute windows
   - RMSSD = sqrt(mean(diff^2))
   - High RMSSD = high vagal tone = relaxed
   - Low RMSSD = sympathetic dominance = stressed

2. **LF/HF Ratio** (via FFT on 5-minute HR windows):
   - LF band: 0.04-0.15 Hz (sympathetic + parasympathetic)
   - HF band: 0.15-0.40 Hz (parasympathetic)
   - High LF/HF (> 2.0) = stressed
   - Low LF/HF (< 1.0) = relaxed

3. **Stress Score** (0-100):
   - `score = 50 * (1 - RMSSD_norm) + 50 * LF_HF_norm`
   - Where `RMSSD_norm` = RMSSD / max_expected_RMSSD (capped at 1.0)
   - And `LF_HF_norm` = min(LF_HF / 4.0, 1.0)

### Output
- Real-time stress score (0-100)
- RMSSD and LF/HF ratio per window
- ASCII trend chart over hours
- Activity context correlation (motion level vs. stress)

### Validation
- Periods of activity (walking, working) should correlate with higher stress scores
- Quiet rest should show lower scores
- Sleeping should show lowest scores (high HRV, low LF/HF)

---

## Application 4: Gait Analysis / Movement Disorder Detection

### Input
- Motion energy time series from vitals packets
- CSI phase variance from raw CSI frames (0xC5110001)
- Cross-node RSSI from vitals packets

### Algorithm

1. **Cadence Extraction**: FFT on motion_energy within 5-second sliding windows
   - Walking cadence: dominant frequency 0.8-2.0 Hz (normal: ~1.0 Hz = 120 steps/min)
   - Running: > 2.0 Hz
   - Stationary: no dominant peak

2. **Stride Regularity**: Autocorrelation of motion_energy
   - Regular walking: strong autocorrelation peak at step period
   - Irregularity score = 1 - (peak_height / baseline)

3. **Asymmetry Detection**: Compare motion energy oscillation between two ESP32 nodes
   - Symmetric gait: both nodes see similar oscillation period and amplitude
   - Asymmetry index = |period_node1 - period_node2| / mean_period

4. **Tremor Detection**: High-frequency phase variance analysis
   - Compute phase variance per subcarrier in 2-second windows
   - Tremor band: 3-8 Hz component in phase variance time series
   - Parkinsonian tremor: 4-6 Hz, resting
   - Essential tremor: 5-8 Hz, action

### Output
- Cadence (steps/min)
- Stride regularity score (0-1)
- Asymmetry index (0 = symmetric, 1 = highly asymmetric)
- Tremor score and dominant frequency
- Walking vs. stationary classification

### Validation
Overnight data should show clear stationary periods with no cadence detected. Any walking segments should show cadence in the 0.8-2.0 Hz range.

---

## Application 5: Material/Object Change Detection

### Input
Per-subcarrier amplitude from raw CSI frames (0xC5110001).

### Algorithm

1. **Baseline Establishment** (first 10 minutes or configurable):
   - Record mean amplitude per subcarrier (Welford online mean)
   - Record null pattern: which subcarriers are below null threshold (amplitude < 2.0)

2. **Change Detection** (sliding 30-second windows):
   - Compare current null pattern to baseline
   - New nulls appearing = new metal object blocking RF path
   - Existing nulls disappearing = metal object removed
   - Null position shifted = object moved
   - Amplitude change without null change = non-metal material (wood, water, glass)

3. **Material Classification** heuristic:
   - Metal: sharp null (amplitude drops to near 0 on specific subcarriers)
   - Water/human: broad amplitude reduction across many subcarriers
   - Wood/plastic: minimal amplitude change, mostly phase shift
   - Glass: frequency-selective (affects higher subcarriers more)

### Output
- Change events with timestamp, type (add/remove/move), affected subcarrier range
- Estimated material category
- Null pattern delta visualization (ASCII)
- Event timeline for monitoring

### Validation
Overnight data has 19% null baseline. Changes in null pattern over the recording period indicate environment changes (doors opening/closing, person entering/leaving).

---

## Application 6: Room Environment Fingerprinting

### Input
- 8-dimensional feature vectors from feature packets (0xC5110003)
- Motion energy and presence score from vitals packets

### Algorithm

1. **Online Clustering** using running k-means (k=5, updateable centroids):
   - Each incoming 8-dim feature vector is assigned to nearest centroid
   - Centroid updated via exponential moving average (alpha=0.01)
   - New cluster created if distance to all centroids exceeds threshold

2. **State Labeling** (heuristic from vitals correlation):
   - Cluster with lowest motion_energy = "empty/sleeping"
   - Cluster with highest motion_energy = "active/walking"
   - Intermediate clusters = "resting", "working", "transitional"

3. **Transition Tracking**:
   - Build state transition matrix (from_state -> to_state counts)
   - Detect anomalous transitions (rare in historical data)

4. **Daily Profile**:
   - Aggregate state durations per hour
   - Compare across days for routine detection

### Output
- Current room state and confidence
- State timeline (ASCII)
- Transition matrix
- Daily pattern profile
- Anomaly score (deviation from established daily pattern)

### Validation
Overnight recording should show 2-3 stable clusters corresponding to activity periods at different times. Transitions should be infrequent and correspond to real behavioral changes.

---

## Implementation

All scripts share common infrastructure:
- ADR-018 binary packet parsing (same as rf-scan.js, mincut-person-counter.js)
- JSONL replay via readline interface
- Live UDP via dgram
- Pure Node.js, no external dependencies
- CLI: `--replay <file>` for offline, `--port <N>` for live, `--json` for programmatic output

| Script | Primary Packets | Key Algorithm |
|--------|----------------|---------------|
| `sleep-monitor.js` | vitals (0xC5110002) | BR/HR window classification |
| `apnea-detector.js` | vitals (0xC5110002) | BR pause detection, AHI scoring |
| `stress-monitor.js` | vitals (0xC5110002) | HRV RMSSD + FFT LF/HF |
| `gait-analyzer.js` | vitals + raw CSI | FFT cadence + phase tremor |
| `material-detector.js` | raw CSI (0xC5110001) | Null pattern baseline + delta |
| `room-fingerprint.js` | feature (0xC5110003) + vitals | Online k-means clustering |

## Consequences

### Positive
- 6 new sensing applications from existing hardware (zero additional cost)
- All offline-capable via JSONL replay (no live hardware needed for development)
- Pure JS, no native dependencies, runs on any platform with Node.js
- Each script is standalone and composable

### Negative
- Vitals accuracy depends on ESP32 CSI quality (RSSI, multipath)
- HRV analysis at 1 Hz HR sampling is coarse compared to ECG
- Material classification is heuristic, not definitive
- Sleep staging without EEG is approximate (consumer-grade accuracy)

### Risks
- Users may misinterpret health-related outputs as clinical diagnoses
- Mitigation: all scripts include disclaimers in output headers
