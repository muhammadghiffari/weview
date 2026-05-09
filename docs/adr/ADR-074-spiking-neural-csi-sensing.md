# ADR-074: Spiking Neural Network for CSI Sensing

| Field       | Value                                      |
|-------------|--------------------------------------------|
| **Status**  | Proposed                                   |
| **Date**    | 2026-04-02                                 |
| **Authors** | ruv                                        |
| **Depends** | ADR-018 (binary frame), ADR-029 (channel hopping), ADR-069 (Cognitum Seed), ADR-073 (multi-frequency mesh) |

## Context

The current WiFi-DensePose CSI sensing pipeline uses two approaches for interpreting subcarrier data:

1. **Static thresholds** — presence detection fires when subcarrier variance exceeds a fixed value. This works in calibrated environments but fails when the RF landscape changes (furniture moved, new objects, temperature drift). Recalibration requires manual intervention or batch retraining.

2. **Batch-trained FC encoder** — the neural network in `wifi-densepose-nn` maps CSI frames to 8-dimensional feature vectors. It requires labeled training data, offline training epochs, and model deployment. The encoder cannot adapt to a new environment without collecting new data and retraining.

Neither approach handles online adaptation. When an ESP32 node is deployed in a new room, the first hours produce noisy, unreliable output until the thresholds are tuned or a model is trained. In disaster scenarios (ADR MAT), there is no time for calibration.

**Spiking Neural Networks (SNNs)** offer an alternative. Unlike traditional ANNs that process continuous values in batch mode, SNNs communicate through discrete spike events and learn online via Spike-Timing-Dependent Plasticity (STDP). This is a natural fit for CSI data:

- CSI subcarrier amplitudes are temporal signals sampled at 12-22 fps
- Amplitude changes (not absolute values) carry the information about motion, breathing, and presence
- STDP learns temporal correlations between subcarriers without labels
- Event-driven processing means idle rooms (no motion) consume near-zero compute

The `@ruvector/spiking-neural` package (vendored at `vendor/ruvector/npm/packages/spiking-neural/`) provides production-ready LIF neurons, STDP learning, lateral inhibition, and SIMD-optimized vector math in pure JavaScript with zero dependencies.

## Decision

Integrate `@ruvector/spiking-neural` into the CSI sensing pipeline as an online unsupervised pattern learner that runs alongside the existing FC encoder. The SNN provides real-time adaptation while the FC encoder provides stable baseline predictions.

### Network Architecture

```
CSI Frame (128 subcarriers)
        |
        v
[ Rate Encoding ] -----> 128 input neurons (one per subcarrier)
        |                   amplitude delta -> spike rate
        v
[ LIF Hidden Layer ] ---> 64 hidden neurons (tau=20ms)
        |                   STDP learns subcarrier correlations
        |                   lateral inhibition -> sparse codes
        v
[ LIF Output Layer ] ---> 8 output neurons
        |
        v
  presence | motion | breathing | heart_rate | phase_var | persons | fall | rssi
```

**Layer parameters:**

| Layer | Neurons | tau (ms) | v_thresh (mV) | Function |
|-------|---------|----------|---------------|----------|
| Input | 128 | N/A | N/A | Rate-coded spike generation from subcarrier deltas |
| Hidden | 64 | 20.0 | -50.0 | STDP learns correlated subcarrier groups |
| Output | 8 | 25.0 | -50.0 | Each neuron specializes in one sensing modality |

**Synapse parameters:**

| Connection | Count | a_plus | a_minus | w_init | Lateral Inhibition |
|------------|-------|--------|---------|--------|-------------------|
| Input -> Hidden | 8,192 | 0.005 | 0.005 | 0.3 | No |
| Hidden -> Output | 512 | 0.003 | 0.003 | 0.2 | Yes (strength=15.0) |

Total synapses: 8,704. At 4 bytes per weight, this is 34 KB — fits in ESP32 SRAM.

### Input Encoding

CSI amplitudes are converted to spike rates using rate coding:

1. Compute per-subcarrier amplitude: `amp[i] = sqrt(I[i]^2 + Q[i]^2)` from the ADR-018 binary frame
2. Compute amplitude delta from previous frame: `delta[i] = |amp[i] - prev_amp[i]|`
3. Normalize deltas to [0, 1] range: `norm[i] = min(delta[i] / max_delta, 1.0)`
4. Feed `norm` to `rateEncoding(norm, dt, max_rate)` which produces Poisson spikes

Higher amplitude changes produce more spikes. Static subcarriers (no motion) produce few or no spikes. This is the key energy advantage: an empty room generates almost no spikes, so the SNN does almost no work.

### STDP Learning Rule

STDP strengthens connections between neurons that fire together (within a time window) and weakens connections between neurons that fire out of sync:

- **LTP (Long-Term Potentiation)**: if a presynaptic neuron fires before a postsynaptic neuron within 20ms, the weight increases by `a_plus * exp(-dt/tau_stdp)`
- **LTD (Long-Term Depression)**: if a postsynaptic neuron fires before a presynaptic neuron, the weight decreases by `a_minus * exp(-dt/tau_stdp)`

Over time, this causes the hidden layer neurons to specialize. Subcarriers that consistently change together (e.g., subcarriers 10-20 affected by a person walking through zone A) become strongly connected to the same hidden neuron. Different motion patterns activate different hidden neuron clusters.

### Lateral Inhibition (Winner-Take-All)

The output layer uses lateral inhibition with strength 15.0. When one output neuron fires, it suppresses all others. This forces each output neuron to specialize in a distinct pattern:

- Output 0: presence (any subcarrier activity above baseline)
- Output 1: motion (widespread subcarrier changes, high spike rate)
- Output 2: breathing (periodic 0.1-0.5 Hz modulation on chest-area subcarriers)
- Output 3: heart rate (periodic 0.8-2.0 Hz modulation, lower amplitude than breathing)
- Output 4: phase variance (phase instability across subcarriers)
- Output 5: person count (number of distinct active subcarrier clusters)
- Output 6: fall (sudden high-amplitude burst followed by silence)
- Output 7: RSSI trend (overall signal strength change)

The neuron-to-label mapping is not fixed by training. Instead, the mapping is discovered by observing which output neuron fires most for each known condition during an optional calibration phase. If no calibration is available, the output is reported as raw spike counts per output neuron, and downstream consumers (Cognitum Seed, SONA) interpret the patterns.

### Integration with Existing Pipeline

The SNN does not replace the FC encoder. It runs in parallel:

```
CSI Frame ----+----> FC Encoder --------> 8-dim feature vector (stable, trained)
              |
              +----> SNN (STDP) --------> 8-dim spike rate vector (adaptive, online)
              |
              +----> SONA Adapter -------> Weighted fusion of both signals
```

SONA (Self-Optimizing Neural Architecture) receives both signals and learns which source is more reliable for each output dimension. In a new environment where the FC encoder has not been retrained, SONA automatically weights the SNN output higher because it adapts faster. As the FC encoder is retrained on local data, SONA shifts weight back toward it.

### Energy and Compute Budget

| Metric | FC Encoder | SNN (STDP) | Ratio |
|--------|-----------|------------|-------|
| Compute per frame (idle room) | 8,192 MACs | ~50 spike events | ~160x less |
| Compute per frame (active room) | 8,192 MACs | ~500 spike events | ~16x less |
| Memory | 34 KB weights | 34 KB weights | Equal |
| Adaptation | Offline retraining | Online, continuous | SNN wins |
| Stability | High (frozen weights) | Lower (weights drift) | FC wins |
| Latency to first useful output | Hours (needs training data) | ~30 seconds | SNN wins |

The SNN's event-driven nature means it processes only spikes, not every subcarrier on every frame. In an idle room with no motion, subcarrier deltas are near zero, spike rates drop to near zero, and the SNN consumes negligible compute. This is ideal for battery-powered or thermally constrained deployments (ESP32, Cognitum Seed Pi Zero).

### Deployment Targets

| Platform | Runtime | Notes |
|----------|---------|-------|
| Node.js server | `require('@ruvector/spiking-neural')` | Primary. Receives UDP frames, runs SNN. |
| Cognitum Seed (Pi Zero) | Node.js ARM | 34 KB model fits. ~0.06ms per step at 100 neurons. |
| ESP32-S3 (WASM) | wasm3 interpreter | Optional. SNN weights exported as flat Float32Array. |
| Browser | WebAssembly or JS | Via `wifi-densepose-wasm` crate's JS bindings. |

### Multi-Channel SNN (ADR-073 Integration)

With multi-frequency mesh scanning (ADR-073), the SNN input expands:

- **Single-channel mode**: 128 input neurons (64 subcarriers x 2 for I/Q or amplitude/phase)
- **Multi-channel mode**: 128 input neurons, but the subcarrier index rotates across channels. Each channel's subcarriers map to the same neuron indices, but at different time slots. The SNN's temporal dynamics naturally integrate cross-channel information because STDP operates across time.

Alternatively, for maximum spectral diversity, a wider SNN (384 input neurons for 6 channels x 64 subcarriers) can be used on the server where memory is not constrained.

## Performance Targets

| Metric | Target | Method |
|--------|--------|--------|
| SNN step latency | <0.1ms | 128-64-8 network, ~8,700 synapses |
| STDP convergence | <30 seconds | ~360 frames at 12 fps, patterns stabilize |
| Output accuracy (after adaptation) | >80% | Compared to manually labeled ground truth |
| Memory footprint | <50 KB | Weights + neuron state |
| Idle room spike rate | <10 spikes/frame | Event-driven: near-zero compute when nothing moves |
| Adaptation to new environment | <2 minutes | STDP relearns subcarrier correlations |

## Risks

### Weight Drift

STDP learning never stops. In a stable environment, weights can slowly drift as the network over-fits to the current RF landscape. Mitigation: implement weight decay (multiply all weights by 0.999 per second) and clamp weights to [w_min, w_max].

### Output Neuron Reassignment

If the RF environment changes significantly (new furniture, different room), output neurons may reassign their specialization. The mapping from output neuron index to label (presence, motion, etc.) may change. Mitigation: periodically log the output neuron activity and detect reassignment events. Downstream consumers should use the spike pattern, not the neuron index, for classification.

### Interference with FC Encoder

If SONA naively averages the SNN and FC encoder outputs, a poorly adapted SNN could degrade overall accuracy. Mitigation: SONA uses confidence-weighted fusion. The SNN output includes a confidence signal (total spike count / expected spike count). Low confidence = low weight.

### STDP Learning Rate Sensitivity

If `a_plus` and `a_minus` are too high, the SNN oscillates and never converges. If too low, adaptation takes too long. The default values (0.005 and 0.003) are conservative. The script includes a `--learning-rate` flag for tuning.

## Alternatives Considered

1. **Online gradient descent on FC encoder** — backprop through the FC network with each new frame. Rejected because: (a) requires a loss function, which requires labels; (b) continuous gradient updates on a small model lead to catastrophic forgetting of the pretrained representations.

2. **Adaptive thresholds only** — replace fixed thresholds with exponentially-weighted moving averages. Rejected because: (a) single-variable thresholds cannot capture multi-subcarrier correlations; (b) no representation learning — each subcarrier is still processed independently.

3. **Reservoir computing (Echo State Network)** — use a fixed random recurrent network as a temporal feature extractor. Partially viable, but: (a) requires a linear readout layer trained with labels; (b) the random reservoir does not adapt to the specific RF environment.

4. **Train SNN with supervision** — use surrogate gradient methods to train the SNN on labeled data. Rejected because: (a) defeats the purpose of online unsupervised learning; (b) the `@ruvector/spiking-neural` package does not implement surrogate gradients.

## Implementation

The integration is implemented in `scripts/snn-csi-processor.js`, a standalone Node.js script that:

1. Receives live CSI frames via UDP (port 5006, ADR-018 binary format)
2. Decodes subcarrier I/Q data and computes amplitude deltas
3. Feeds deltas through rate encoding into the SNN
4. Applies STDP learning on every frame (online, unsupervised)
5. Maps output neuron spike counts to sensing labels
6. Prints real-time ASCII visualization of SNN activity
7. Optionally forwards learned patterns to Cognitum Seed

## References

- ADR-018: CSI binary frame format
- ADR-029: Channel hopping infrastructure
- ADR-069: Cognitum Seed CSI pipeline
- ADR-073: Multi-frequency mesh scanning
- Maass, W. (1997). "Networks of spiking neurons: The third generation of neural network models." Neural Networks, 10(9), 1659-1671.
- Bi, G. & Poo, M. (1998). "Synaptic modifications in cultured hippocampal neurons: Dependence on spike timing." Journal of Neuroscience, 18(24), 10464-10472.
- `@ruvector/spiking-neural` v1.0.1 — LIF, STDP, lateral inhibition, SIMD
