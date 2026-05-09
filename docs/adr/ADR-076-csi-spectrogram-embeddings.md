# ADR-076: CSI Spectrogram Embeddings via CNN + Graph Transformer

| Field       | Value                                      |
|-------------|--------------------------------------------|
| **Status**  | Proposed                                   |
| **Date**    | 2026-04-02                                 |
| **Authors** | ruv                                        |
| **Depends** | ADR-018 (binary frame), ADR-024 (AETHER contrastive embeddings), ADR-029 (RuvSense), ADR-069 (Cognitum Seed bridge), ADR-073 (multi-frequency mesh scan) |

## Context

The current CSI processing pipeline extracts an 8-dimensional hand-crafted feature vector per frame: mean amplitude, amplitude variance, max amplitude, mean phase, phase variance, bandwidth, spectral centroid, and RSSI. These features are effective for basic presence detection and room fingerprinting but discard the rich spatial-frequency structure present in the raw subcarrier data.

A single CSI frame from an ESP32-S3 contains 64 subcarriers (or 128 in HT40 mode), each with I/Q components. When stacked over time, 20 consecutive frames form a **64x20 subcarrier-by-time matrix** — effectively a grayscale spectrogram image. This matrix encodes:

1. **Frequency-selective fading** — metal objects create persistent null zones at specific subcarrier indices (visible as dark vertical stripes)
2. **Doppler signatures** — human motion produces time-varying amplitude patterns across subcarriers (visible as horizontal wave patterns)
3. **Multipath structure** — room geometry creates characteristic interference patterns unique to each environment
4. **Activity fingerprints** — walking, sitting, breathing, and falling produce distinct 2D texture patterns in the subcarrier-time matrix

These 2D structural patterns are invisible to the 8-dim feature vector, which collapses all subcarrier information into scalar statistics. A CNN embedding can preserve this spatial structure.

### Existing Vendor Libraries

**@ruvector/cnn** (v0.1.0) provides:
- WASM-based CNN feature extraction (~5ms per 224x224 image, ~900KB model)
- Configurable embedding dimension (default 512, we use 128 for compact storage)
- L2-normalized embeddings with cosine similarity search
- Contrastive training via InfoNCE and triplet loss
- SIMD-optimized layer operations (batch norm, global average pooling, ReLU)
- Works in both Node.js and browser environments

**ruvector-graph-transformer** provides:
- Sublinear O(n log n) graph attention via LSH bucketing and PPR sampling
- Proof-gated mutation substrate for verified computations
- Temporal causal attention with Granger causality (relevant for CSI time series)
- Manifold attention on product spaces S^n x H^m x R^k

**@ruvector/graph-wasm** (v2.0.2) provides:
- Neo4j-compatible property graph database in WASM
- Node/edge creation with arbitrary properties and embeddings
- Hyperedge support for multi-node relationships
- Cypher query language

### Current Limitations of 8-dim Features

| Limitation | Impact |
|------------|--------|
| No subcarrier-level information | Cannot distinguish frequency-selective vs broadband fading |
| No temporal pattern encoding | Walking gait (periodic) looks identical to random motion (aperiodic) |
| No 2D structure | Room fingerprint reduced to 8 numbers; two rooms with similar statistics are indistinguishable |
| No cross-subcarrier correlation | Cannot detect standing waves, node patterns, or multipath clusters |
| Poor kNN discrimination | 8 dimensions provides limited hypersphere surface area for separating environments |

## Decision

Treat the CSI subcarrier-by-time matrix as a grayscale spectrogram image and apply CNN embedding to produce a 128-dimensional representation that preserves 2D spatial-frequency structure. Use a graph transformer to fuse embeddings across multiple ESP32 nodes.

### Architecture

```
ESP32 Node 1          ESP32 Node 2
     |                      |
     v                      v
  UDP 5006              UDP 5006
     |                      |
     v                      v
 [64 subcarriers]      [64 subcarriers]
 [20-frame window]     [20-frame window]
     |                      |
     v                      v
 64x20 amplitude       64x20 amplitude
 matrix (grayscale)    matrix (grayscale)
     |                      |
     v                      v
 @ruvector/cnn         @ruvector/cnn
 CnnEmbedder           CnnEmbedder
     |                      |
     v                      v
 128-dim vector        128-dim vector
     |                      |
     +-------+  +----------+
             |  |
             v  v
    Graph Transformer (2-node graph)
    Edge weight = cross-node correlation
             |
             v
    Fused 128-dim vector
             |
     +-------+-------+
     |               |
     v               v
  Cognitum Seed   kNN Search
  (128-dim store) (similar rooms)
```

### Step 1: CSI-to-Spectrogram Conversion

Each ESP32 transmits CSI frames via UDP in ADR-018 binary format. The `iq_hex` field contains I/Q pairs for each subcarrier (2 bytes per subcarrier: I + Q as unsigned 8-bit values).

```
Amplitude[sc] = sqrt(I[sc]^2 + Q[sc]^2)
```

A sliding window of 20 frames produces a 64x20 matrix. Normalization to 0-255 grayscale:

```
pixel[sc][t] = clamp(255 * (amplitude[sc][t] - min) / (max - min), 0, 255)
```

Where `min` and `max` are computed over the entire 64x20 window for per-window contrast normalization. This ensures the CNN sees the relative structure regardless of absolute signal strength (which varies with distance, TX power, and environmental absorption).

### Step 2: CNN Embedding

The 64x20 grayscale matrix is resized to the CNN's expected input size (224x224 via nearest-neighbor upsampling, since we want to preserve the discrete subcarrier structure rather than blur it with bilinear interpolation). The input is replicated across 3 channels (RGB) since @ruvector/cnn expects RGB input.

Configuration:
- **Input**: 224x224x3 (upsampled from 64x20, grayscale replicated to RGB)
- **Embedding dimension**: 128 (reduced from default 512 for compact storage and faster kNN)
- **Normalization**: L2-enabled (cosine similarity = dot product on unit sphere)
- **Latency**: ~5ms per window on modern hardware

The 128-dim embedding encodes the 2D structure of the spectrogram: null zones, Doppler patterns, multipath signatures, and activity textures.

### Step 3: Graph Transformer for Multi-Node Fusion

With 2 ESP32 nodes (generalizable to N), we construct a graph:

```
Nodes: {Node_1, Node_2}
Edges: {(Node_1, Node_2, weight=cross_correlation)}
Node features: 128-dim CNN embedding per node
```

The graph attention mechanism learns which node is more informative for each prediction:

1. **Query/Key/Value** from each node's 128-dim embedding
2. **Edge weight** = Pearson cross-correlation between the two nodes' raw amplitude vectors (captures how much their CSI observations agree)
3. **Attention score** = softmax(Q_i * K_j / sqrt(d) + edge_weight_bias)
4. **Output** = weighted sum of value vectors

This produces a fused 128-dim vector that combines both nodes' perspectives, automatically weighting the node with cleaner signal (higher SNR, less fading) more heavily.

**Generalization to 3+ nodes**: Adding a third ESP32 adds one node and 2 edges to the graph. The attention mechanism handles variable-size graphs without architecture changes.

### Step 4: Storage and Search

The fused 128-dim embedding is stored in Cognitum Seed (ADR-069) alongside the existing 8-dim features:

| Store | Dimension | Content | Use Case |
|-------|-----------|---------|----------|
| `csi-features` | 8-dim | Hand-crafted statistics | Fast presence detection |
| `csi-spectrograms` | 128-dim | CNN spectrogram embedding | Environment fingerprinting, anomaly detection |
| `csi-spectrograms-fused` | 128-dim | Graph-fused multi-node embedding | Cross-viewpoint room signature |

kNN search on the 128-dim store finds past spectrograms that "look like" the current one:
- **Environment fingerprinting**: "What room does this RF pattern match?"
- **Cross-room transfer**: "Which training room is most similar to this deployment room?"
- **Anomaly detection**: Low similarity to all known patterns = unknown environment or novel activity
- **Temporal segmentation**: Similarity drops = activity transition boundaries

### Comparison: 8-dim vs 128-dim vs Combined

| Property | 8-dim hand-crafted | 128-dim CNN | Combined |
|----------|-------------------|-------------|----------|
| Subcarrier structure | Lost | Preserved | Both available |
| Temporal patterns | Lost | Preserved (20-frame window) | Both |
| Computation | ~0.1ms | ~5ms | ~5ms |
| Storage per vector | 32 bytes | 512 bytes | 544 bytes |
| kNN discrimination | Low (8-dim curse) | High (128-dim surface) | Highest |
| Interpretability | High (named features) | Low (learned) | Mixed |
| Training required | No | Optional (pre-trained works) | Optional |
| Multi-node fusion | Average/max | Graph attention | Graph attention |

### Contrastive Training (Optional Enhancement)

The CNN embedding works out-of-the-box with the pre-trained weights. For domain-specific improvements, contrastive training with CSI data:

1. **Positive pairs**: Same room, different time windows (should embed similarly)
2. **Negative pairs**: Different rooms or different activities (should embed differently)
3. **Loss**: InfoNCE with temperature 0.07 (standard SimCLR)
4. **Augmentation**: Time-shift (slide window by 1-5 frames), subcarrier dropout (zero 10% of rows), amplitude jitter (multiply by uniform [0.8, 1.2])

This teaches the CNN that "same room at different times" should produce similar embeddings, while "different rooms" should produce different embeddings.

## Consequences

### Positive

1. **Richer representation**: 128 dimensions capture 2D structure that 8 dimensions cannot
2. **Environment fingerprinting**: kNN on spectrograms can distinguish rooms that look identical in 8-dim feature space
3. **Activity detection**: Temporal patterns (gait periodicity, breathing frequency) are encoded in the spectrogram texture
4. **Multi-node fusion**: Graph attention automatically weights the most informative node, improving robustness to single-node occlusion or interference
5. **Incremental adoption**: 128-dim store operates alongside 8-dim store; no migration needed
6. **Browser-compatible**: WASM-based CNN runs in the sensing-server UI for live visualization

### Negative

1. **5ms latency per window**: Acceptable for 1.3 Hz update rate (750ms rotation from ADR-073), but constrains real-time applications
2. **900KB model download**: One-time cost, cached after first load
3. **128-dim storage**: 16x more bytes per vector than 8-dim; mitigated by the fact that we store one embedding per 20-frame window (not per frame)
4. **Opaque embeddings**: Unlike named 8-dim features, CNN embeddings are not human-interpretable
5. **Input size mismatch**: 64x20 matrix must be upsampled to 224x224; nearest-neighbor preserves structure but wastes computation on padded regions

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| CNN embeddings not discriminative enough for CSI | Contrastive fine-tuning on CSI spectrograms; fall back to 8-dim if 128-dim kNN recall is worse |
| Graph transformer overhead for 2-node graph | Lightweight attention (single head, no MLP); O(1) for 2 nodes |
| Upsampling artifacts from 64x20 to 224x224 | Nearest-neighbor preserves discrete structure; consider training a smaller CNN on native 64x20 input |
| WASM initialization delay | Call `init()` at server startup, not per-request |

## Implementation

### Files

| File | Purpose |
|------|---------|
| `scripts/csi-spectrogram.js` | CSI-to-spectrogram pipeline with CNN embedding, ASCII visualization, Cognitum Seed ingest |
| `scripts/mesh-graph-transformer.js` | Multi-node graph attention fusion using @ruvector/graph-wasm |
| `docs/adr/ADR-076-csi-spectrogram-embeddings.md` | This ADR |

### Dependencies

| Package | Version | Source |
|---------|---------|--------|
| `@ruvector/cnn` | 0.1.0 | `vendor/ruvector/npm/packages/ruvector-cnn/` |
| `@ruvector/graph-wasm` | 2.0.2 | `vendor/ruvector/npm/packages/graph-wasm/` |

### Data Format

CSI JSONL frames from `data/recordings/pretrain-1775182186.csi.jsonl`:

```json
{
  "timestamp": 1775182186.123,
  "node_id": 1,
  "magic": 3289481217,
  "size": 148,
  "rssi": -45,
  "type": "CSI",
  "iq_hex": "00000f030d030e040d030d030d030c020d020d01...",
  "subcarriers": 64
}
```

`iq_hex` encoding: 2 hex characters per byte, 4 hex characters per subcarrier (I byte + Q byte). Total length = `subcarriers * 4` hex characters.

## References

- ADR-018: Binary CSI frame format
- ADR-024: AETHER contrastive CSI embeddings (Rust-side)
- ADR-029: RuvSense multistatic sensing mode
- ADR-069: Cognitum Seed RVF ingest bridge
- ADR-073: Multi-frequency mesh scanning
- SimCLR: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (2020)
- GATv2: Brody et al., "How Attentive are Graph Attention Networks?" (2021)
