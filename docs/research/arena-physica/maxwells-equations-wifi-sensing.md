# Maxwell's Equations in WiFi/RF Sensing

Research document for wifi-densepose project.
Date: 2026-04-02

---

## 1. Maxwell's Equations and CSI Extraction

### 1.1 Foundational Electromagnetic Theory

All WiFi-based sensing ultimately derives from Maxwell's four partial differential equations governing electromagnetic field behavior:

```
(1) Gauss's Law (Electric):       nabla . E = rho / epsilon_0
(2) Gauss's Law (Magnetic):       nabla . B = 0
(3) Faraday's Law:                nabla x E = -dB/dt
(4) Ampere-Maxwell Law:           nabla x B = mu_0 * J + mu_0 * epsilon_0 * dE/dt
```

In free space with no charges or currents (the indoor propagation case), these simplify to the wave equation:

```
nabla^2 E - mu_0 * epsilon_0 * d^2 E / dt^2 = 0
```

yielding plane wave solutions `E(r, t) = E_0 * exp(j(k . r - omega * t))` where `k = 2*pi / lambda` is the wavenumber. At 2.4 GHz WiFi, `lambda ~ 12.5 cm`; at 5 GHz, `lambda ~ 6 cm`.

### 1.2 From Maxwell to Channel State Information

Channel State Information (CSI) is the frequency-domain representation of the wireless channel's impulse response. The derivation from Maxwell's equations proceeds through several simplification layers:

**Layer 1: Full Maxwell's equations** -- Exact but computationally intractable for room-scale environments at GHz frequencies.

**Layer 2: High-frequency ray optics (Geometrical Optics / Uniform Theory of Diffraction)** -- When object dimensions >> lambda (walls, furniture), Maxwell's equations reduce to ray tracing. Each ray follows Snell's law at interfaces, with Fresnel reflection/transmission coefficients computed from the dielectric contrast.

**Layer 3: Multipath channel model** -- The channel impulse response aggregates all propagation paths:

```
h(t) = sum_{n=1}^{N} alpha_n * exp(-j * phi_n) * delta(t - tau_n)
```

where for each path n:
- `alpha_n` = complex attenuation (from free-space path loss, reflection, diffraction)
- `phi_n = 2*pi*f*tau_n` = phase shift
- `tau_n = d_n / c` = propagation delay (distance / speed of light)

**Layer 4: Channel Frequency Response (CFR) = CSI** -- The Fourier transform of h(t):

```
H(f_k) = sum_{n=1}^{N} alpha_n * exp(-j * 2*pi * f_k * tau_n)
```

Each OFDM subcarrier k at frequency f_k provides one complex CSI measurement:

```
H(f_k) = |H(f_k)| * exp(j * angle(H(f_k)))
```

With 802.11n/ac providing 56-256 subcarriers and 802.11ax up to 512 subcarriers across 160 MHz bandwidth, CSI captures a frequency-sampled version of the channel's multipath structure.

**Key insight for sensing**: When a human moves in the environment, paths reflecting off the body change their `alpha_n`, `tau_n`, and `phi_n`, modulating the CSI. The sensing problem is to invert this relationship -- recover body state from CSI changes.

### 1.3 The Two CSI Models

The Tsinghua WiFi Sensing Tutorial (tns.thss.tsinghua.edu.cn) identifies two mainstream models:

**Ray-Tracing Model**: Establishes explicit geometric relationships between signal paths and CSI. The received signal is:

```
V = sum_{n=1}^{N} |V_n| * exp(-j * phi_n)
```

This model enables extraction of geometric parameters (distances, reflection points, angles of arrival) from CSI data. It underpins localization and tracking applications.

**Scattering Model**: Decomposes CSI into static and dynamic contributions:

```
H(f,t) = sum_{o in Omega_s} H_o(f,t) + sum_{p in Omega_d} H_p(f,t)
```

Dynamic scatterers (moving bodies) contribute through angular integration:

```
H_p(f,t) = integral_0^{2pi} integral_0^{pi} h_p(alpha, beta, f, t) * exp(-j*k*v_p*cos(alpha)*t) d_alpha d_beta
```

The scattering model yields the CSI autocorrelation:

```
rho_H(f, tau) ~ sinc(k * v * tau)
```

enabling speed extraction from autocorrelation peak analysis:

```
v = x_0 * lambda / (2 * pi * tau_0)
```

where `x_0` is the first sinc extremum location and `tau_0` is the corresponding time lag.

### 1.4 Practical Simplifications Used in WiFi Sensing

| Approximation | Physical Basis | Used When | Accuracy |
|---|---|---|---|
| Ray tracing (GO/UTD) | High-frequency limit of Maxwell | Objects >> lambda | Good for LOS + major reflections |
| Fresnel zone model | Wave diffraction | Target near TX-RX line | Excellent for presence/respiration |
| Born approximation | Weak scattering (small perturbation) | Low-contrast objects | Breaks down for human body |
| Rytov approximation | Phase perturbation expansion | Moderate scattering | Better for lossy media |
| Free-space path loss | 1/r^2 power decay | Coarse attenuation models | Adequate for RSSI-based sensing |

**Relevance to wifi-densepose**: Our `field_model.rs` implements the eigenstructure approach (Layer 2.5 -- between full ray tracing and statistical models), decomposing the channel covariance via SVD to separate environmental modes from body perturbation. Our `tomography.rs` implements the voxel-based inverse at Layer 3 using L1-regularized least squares.


## 2. Physics-Informed Neural Networks (PINNs) for RF Sensing

### 2.1 PINN Architecture for Wireless Channels

Physics-Informed Neural Networks embed physical laws as constraints in the loss function or network architecture. For RF sensing, PINNs encode electromagnetic propagation principles:

**Standard PINN loss for RF propagation:**

```
L_total = L_data + lambda_physics * L_physics + lambda_boundary * L_boundary

where:
  L_data = (1/N) * sum |H_pred(f_k) - H_meas(f_k)|^2     (CSI measurement fit)
  L_physics = (1/M) * sum |nabla^2 E + k^2 * E|^2          (Helmholtz equation residual)
  L_boundary = (1/B) * sum |E_pred - E_bc|^2                (boundary conditions)
```

The Helmholtz equation `nabla^2 E + k^2 * n^2(r) * E = 0` (time-harmonic Maxwell) constrains the solution space, where `n(r)` is the spatially varying refractive index.

### 2.2 Key Papers and Approaches

**PINN + GNN for RF Map Construction** (arXiv 2507.22513):
- Combines Physics-Informed Neural Networks with Graph Neural Networks
- Physical constraints from EM propagation laws guide learning
- Parameterizes multipath signals into received power, delay, and angle of arrival
- Integrates spatial dependencies for accurate prediction

**PINN for Wireless Channel Estimation** (NeurIPS 2025, OpenReview r3plaU6DvW):
- Synergistically combines model-based channel estimation with deep network
- Exploits prior information about environmental propagation
- Critical for next-gen wireless systems: precoding, interference reduction, sensing

**ReVeal: High-Fidelity Radio Propagation** (DySPAN 2025):
- Physics-informed approach for radio environment mapping
- Achieves high fidelity with limited measurement data

**Physics-Informed Generative Model for Passive RF Sensing** (arXiv 2310.04173, Savazzi et al.):
- Variational Auto-Encoder integrating EM body diffraction
- Forward model: predicts CSI perturbation from body position/pose
- Validated against classical diffraction-based EM tools AND real RF measurements
- Enables real-time processing where traditional EM is too slow

**Multi-Modal Foundational Model** (arXiv 2602.04016, February 2026):
- Foundation model for AI-driven physical-layer wireless systems
- Physics-guided pretraining grounded in EM propagation principles
- Treats wireless as inherently multimodal physical system

**Generative AI for Wireless Sensing** (arXiv 2509.15258, September 2025):
- Physics-informed diffusion models for data augmentation
- Channel prediction and environment modeling
- Conditional mechanisms constrained by EM laws

### 2.3 PINN Architecture for CSI-Based Sensing

```
Algorithm: Physics-Informed CSI Sensing Network

Input: CSI tensor H[time, subcarrier, antenna] of shape (T, K, M)
Output: Body state estimate (pose, position, or occupancy)

1. PREPROCESSING (physics-guided):
   a. Remove carrier frequency offset (CFO): H_clean = H * exp(-j*2*pi*delta_f*t)
   b. Conjugate multiply across antenna pairs to cancel common phase noise
   c. Compute CSI-ratio: H_ratio(f,t) = H_dynamic(f,t) / H_static(f,t)

2. PHYSICS ENCODER:
   a. Embed Fresnel zone geometry as positional encoding
   b. Apply multi-head attention with frequency-aware kernels
   c. Enforce causality: attention mask respects propagation delay ordering

3. PHYSICS-CONSTRAINED DECODER:
   a. Predict body state x_hat
   b. Forward-simulate expected CSI from x_hat using ray-tracing differentiable renderer
   c. Compute physics loss: L_phys = ||H_simulated(x_hat) - H_measured||^2

4. TRAINING LOSS:
   L = L_pose_supervision + alpha * L_phys + beta * L_temporal_smoothness
```

### 2.4 Relevance to wifi-densepose

Our RuvSense pipeline already implements physics-guided preprocessing (phase alignment, coherence gating, Fresnel zone awareness). The next step would be to:

1. Add a differentiable ray-tracing forward model as a physics constraint during NN training
2. Use the field model eigenstructure (from `field_model.rs`) as an informed prior
3. Embed Fresnel zone geometry from link topology as architectural bias


## 3. Inverse Electromagnetic Scattering for Body Reconstruction

### 3.1 The Inverse Problem

The forward problem: given a known body position/shape and room geometry, predict the CSI.

```
Forward:  body_state -> Maxwell/ray-tracing -> H(f,t)     [well-posed]
Inverse:  H(f,t) -> ??? -> body_state                     [ill-posed]
```

WiFi sensing is fundamentally an inverse scattering problem. A WiFi antenna receives signal as 1D amplitude/phase -- the spatial information of the 3D scene is collapsed to a single CSI complex number per subcarrier per antenna pair. Reconstructing fine-grained spatial information from this compressed observation is severely ill-posed.

### 3.2 Linearized Inverse Scattering: Born and Rytov Approximations

**Helmholtz equation with scatterer:**

```
nabla^2 E(r) + k^2 * (1 + O(r)) * E(r) = 0
```

where `O(r) = epsilon_r(r) - 1` is the object function (dielectric contrast of the body relative to free space).

**Born approximation** (first-order): Assumes the field inside the scatterer equals the incident field:

```
E_scattered(r) ~ k^2 * integral O(r') * E_incident(r') * G(r, r') dr'
```

where `G(r, r')` is the free-space Green's function. This is valid when `O(r)` is small and the object is electrically small. For the human body at 2.4 GHz (`epsilon_r ~ 40-60` for muscle tissue), the Born approximation is grossly violated.

**Rytov approximation**: Expands the complex phase rather than the field:

```
E_total(r) = E_incident(r) * exp(psi(r))

psi(r) ~ (k^2 / E_incident(r)) * integral O(r') * E_incident(r') * G(r, r') dr'
```

The Rytov approximation handles larger phase accumulation than Born but still assumes weak scattering. It works better for lossy media where absorption limits multiple scattering.

**Extended Phaseless Rytov Approximation (xPRA-LM)** (Dubey et al., arXiv 2110.03211):
- First linear phaseless inverse scattering approximation with large validity range
- Demonstrated with 2.4 GHz WiFi nodes for indoor imaging
- Handles objects with `epsilon_r` up to 15+j1.5 (20x wavelength size)
- At `epsilon_r = 77+j7` (water/tissue), shape reconstruction still accurate

### 3.3 Iterative Nonlinear Methods

For high-contrast scatterers like the human body, iterative methods are required:

**Distorted Born Iterative Method (DBIM):**

```
Algorithm: DBIM for WiFi Body Imaging

Input: Measured scattered field E_s at receiver locations
Output: Object function O(r) (dielectric map of scene)

1. Initialize: O_0(r) = 0 (empty room)
2. For iteration i = 0, 1, 2, ...:
   a. Solve forward problem: compute total field E_i(r) in medium with O_i(r)
   b. Compute Green's function G_i(r, r') for medium O_i(r)
   c. Linearize: delta_E_s = K_i * delta_O   (Frechet derivative)
   d. Solve: delta_O = K_i^+ * (E_s_measured - E_s_computed(O_i))
   e. Update: O_{i+1} = O_i + delta_O
   f. Check convergence: ||E_s_measured - E_s_computed(O_{i+1})|| < epsilon
```

**Challenges for WiFi sensing:**
- WiFi provides sparse spatial sampling (few antenna pairs vs. full aperture)
- Phase is often unavailable (RSSI-only) or corrupted by hardware imperfections
- Real-time requirement conflicts with iterative forward solves
- Human body is a strong, moving scatterer

### 3.4 Radio Tomographic Imaging (RTI)

RTI (Wilson & Patwari, 2010) simplifies the inverse scattering problem by:
1. Using only RSS (received signal strength) -- phaseless
2. Assuming a voxelized scene with additive attenuation model
3. Linearizing: measured attenuation = sum of voxel attenuations along path

**Forward model:**

```
y = W * x + n

where:
  y = [y_1, ..., y_L]^T   attenuation measurements (L links)
  x = [x_1, ..., x_V]^T   voxel occupancy values (V voxels)
  W = [w_{l,v}]             weight matrix (link-voxel intersection)
  n = measurement noise
```

**Weight model (elliptical):**

```
w_{l,v} = { 1 / sqrt(d_l)   if d_{l,v}^tx + d_{l,v}^rx < d_l + lambda_w
           { 0               otherwise

where:
  d_l = distance between TX_l and RX_l
  d_{l,v}^tx = distance from TX_l to voxel v center
  d_{l,v}^rx = distance from RX_l to voxel v center
  lambda_w = excess path length parameter (typically ~lambda/4)
```

**Inverse solution (Tikhonov-regularized):**

```
x_hat = (W^T W + alpha * C^{-1})^{-1} * W^T * y
```

where `C` is the spatial covariance matrix and `alpha` controls regularization.

**Our implementation** (`tomography.rs`) uses ISTA (Iterative Shrinkage-Thresholding Algorithm) with L1 regularization for sparsity:

```
Algorithm: ISTA for RF Tomography (as in tomography.rs)

Input: Weight matrix W, observations y, lambda (L1 weight)
Output: Sparse voxel densities x

1. Initialize x = 0
2. step_size = 1 / ||W^T * W||_spectral
3. For iter = 1 to max_iterations:
   a. gradient = W^T * (W * x - y)
   b. x_candidate = x - step_size * gradient
   c. x = soft_threshold(x_candidate, lambda * step_size)
      where soft_threshold(z, t) = sign(z) * max(|z| - t, 0)
   d. residual = ||W * x - y||
   e. if residual < tolerance: break
```

### 3.5 Reconciling RTI with Inverse Scattering

Dubey, Li & Murch (arXiv 2311.09633) reconciled empirical RTI with formal inverse scattering theory:
- RTI's additive attenuation model corresponds to a first-order Born approximation of the scattered field amplitude
- Their enhanced method reconstructs both shape AND material properties
- Validated at 2.4 GHz with WiFi transceivers indoors

### 3.6 State-of-the-Art: Deep Learning Approaches

**DensePose From WiFi** (Geng, Huang, De la Torre, arXiv 2301.00250, CMU):
- Maps WiFi CSI amplitude+phase to UV coordinates across 24 body regions
- Uses 3 TX + 3 RX antennas, 56 subcarriers per link
- Teacher-student training: camera-based DensePose provides labels
- Performance comparable to image-based approaches
- Works through walls and in darkness

**RF-Pose** (Zhao et al., CVPR 2018, MIT CSAIL):
- Through-wall human pose estimation using radio signals
- Cross-modal supervision: vision model trains RF model
- Generalizes to through-wall scenarios with no through-wall training data

**Person-in-WiFi** (Wang et al., ICCV 2019, CMU):
- End-to-end body segmentation and pose from WiFi
- Standard 802.11n signals, off-the-shelf hardware

**3D WiFi Pose Estimation** (arXiv 2204.07878):
- Free-form and moving activities
- 3D joint position estimation from CSI

**HoloCSI** (2025-2026):
- Holographic tomography pipeline coupling physics-guided projection with adaptive top-k sparse transformer
- Preprocesses: CFO rectification, Doppler compensation, antenna-pair normalization
- Sparse multi-head attention prunes low-magnitude query-key pairs (quadratic -> near-linear complexity)
- Results: +2.9 dB PSNR, +3.6% SSIM, +12.4% mesh IoU vs baselines
- 25 fps on RTX-4070-mobile at 5% sparsity; 7 fps on Raspberry Pi 5 with attention-GRU variant


## 4. Computational Electromagnetics for WiFi Sensing

### 4.1 FDTD (Finite-Difference Time-Domain)

FDTD discretizes Maxwell's curl equations on a Yee grid and marches forward in time:

```
Algorithm: FDTD Update (2D TM mode, simplified)

Grid: dx = dy = lambda/20 (minimum 10 cells per wavelength)
Time step: dt = dx / (c * sqrt(2))  [Courant condition]

For each time step n:
  1. Update H fields:
     H_z^{n+1/2}(i,j) = H_z^{n-1/2}(i,j) + (dt/mu_0) * [
       (E_x^n(i,j+1) - E_x^n(i,j)) / dy -
       (E_y^n(i+1,j) - E_y^n(i,j)) / dx
     ]

  2. Update E fields:
     E_x^{n+1}(i,j) = E_x^n(i,j) + (dt / epsilon(i,j)) * [
       (H_z^{n+1/2}(i,j) - H_z^{n+1/2}(i,j-1)) / dy
     ]
```

**For WiFi at 2.4 GHz:**
- Wavelength: 12.5 cm
- Grid cell: ~6 mm (20 cells/lambda)
- Room 6m x 6m x 3m: 1000 x 1000 x 500 = 500M cells
- Memory: ~24 GB (6 field components * 4 bytes * 500M)
- Time steps: ~10,000 for steady state

**Key references for WiFi FDTD:**
- Lauer & Ertel (2003), "Using Large-Scale FDTD for Indoor WLAN" -- Full FDTD at 2.45 GHz in office environments
- Lui et al. (2018), "Human Body Shadowing" -- FDTD human body model for ray-tracing calibration (Hindawi IJAP 9084830)
- Martinez-Gonzalez et al. (2008), "FDTD Assessment Human Exposure WiFi/Bluetooth" -- SAR computation with anatomical body models

**Practical limitations**: FDTD is too slow for real-time sensing but valuable for:
- Generating training data for neural networks
- Validating approximate models
- Understanding near-field body-wave interaction

### 4.2 Method of Moments (MoM)

MoM converts Maxwell's integral equations into matrix equations by expanding fields in basis functions:

```
[Z] * [I] = [V]

where:
  Z_{mn} = integral integral G(r_m, r_n) * f_m(r) * f_n(r') dS dS'
  I_n = unknown current coefficients
  V_m = incident field excitation
```

**Application**: MoM excels for antenna analysis and is used to model WiFi antenna patterns. Less practical for full room simulation due to O(N^2) memory and O(N^3) solve time.

### 4.3 FEM (Finite Element Method)

FEM handles complex geometries and material interfaces more naturally than FDTD:

```
Weak form of Helmholtz equation:
integral nabla x E_test . (1/mu_r * nabla x E) dV - k_0^2 * integral E_test . epsilon_r * E dV
= -j * omega * integral E_test . J_s dV
```

**Application**: HFSS (Ansys) and COMSOL use FEM for electromagnetic simulation. Arena Physica's Heaviside-0 model was trained against such commercial FEM solvers.

### 4.4 Comparison for WiFi Sensing Applications

| Method | Speed | Accuracy | Body Modeling | Room Scale | Real-Time |
|---|---|---|---|---|---|
| FDTD | Hours | Full-wave exact | Excellent | Feasible (GPU) | No |
| MoM | Hours | Exact for surfaces | Good (surface) | Impractical | No |
| FEM | Hours | Exact | Excellent | Feasible | No |
| Ray tracing | Seconds | GO/UTD approximation | Coarse | Easy | Near real-time |
| RTI (ISTA) | Milliseconds | Linear approximation | Voxelized | Easy | Yes |
| Neural surrogate | Milliseconds | Trained accuracy | Implicit | Trained domain | Yes |

### 4.5 Hybrid Approaches: Neural Surrogates Trained on CEM

The most promising direction combines full-wave accuracy with real-time speed:

1. **Offline**: Run thousands of FDTD/FEM simulations with different body positions
2. **Train**: Neural network learns the mapping from body state to CSI
3. **Deploy**: Neural surrogate runs in milliseconds for real-time inference

This is exactly Arena Physica's approach (Section 5), applied to RF component design rather than sensing. The same methodology applies to WiFi sensing: train a neural forward model on FDTD data, then use it as a differentiable physics constraint during inverse model training.


## 5. Arena Physica's Approach

### 5.1 Company Overview

Arena Physica (arena-ai.com / arenaphysica.com) pursues "Electromagnetic Superintelligence" -- building foundation models that develop superhuman intuition for how geometry shapes electromagnetic fields. Founded by Pratap Ranade (CEO), Arya Hezarkhani, Claire Pan, Michael Frei, and Harish Krishnaswamy. Offices in NYC (HQ), SF, LA.

Raised $30M Series B (April 2025). Deployed with AMD, Anduril Industries, Sivers Semiconductors, Bausch & Lomb. Claims 35% reduction in engineering man-hours and multi-month acceleration in time-to-market.

### 5.2 Technical Architecture

Arena's Atlas platform uses two foundation models:

**Heaviside-0 (Forward Model)**:
- Input: PCB/RF geometry (discretized as grid)
- Output: S-parameters (magnitude + phase) and field distributions
- Speed: 13ms per design (single), 0.3ms batched
- Comparison: Traditional solver (HFSS/FDTD) takes ~4 minutes
- Speedup: 18,000x to 800,000x

**Marconi-0 (Inverse Model)**:
- Input: Target S-parameter specification
- Output: Physical geometry that achieves the specification
- Method: Conditional diffusion process (similar to image generation)
- Generates unconventional geometries no human designer would conceive

**Training data**: 3 million simulated designs across 25 expert templates + random structures, totaling 20+ years of combined simulation time. Incorporates both S-parameter data and electromagnetic field distributions.

**Validation**: Predictions validated against commercial numerical field solvers (likely HFSS). Internal testing shows < 1 dB magnitude-weighted MAE (RF engineers operate in 20-30 dB ranges).

### 5.3 Relationship to Maxwell's Equations

Arena does NOT solve Maxwell's equations directly. Instead:

1. **Training phase**: Maxwell's equations are solved by conventional solvers (FDTD/FEM/MoM) millions of times to generate training data
2. **Inference phase**: Neural surrogate approximates Maxwell's solutions in milliseconds
3. **Design loop**: Generator proposes geometry -> Evaluator predicts EM behavior -> Iterate

As Pratap Ranade states: the model "learns the syntax of physics" inductively from examples, rather than deductively from equations. This trades precision for speed -- acceptable when searching design space where "speed and direction matter more than precision."

### 5.4 The "Large Field Model" (LFM) Concept

Arena's LFM is distinct from Large Language Models:
- LLMs learn linguistic patterns from text
- LFMs learn electromagnetic field patterns from simulation data
- The input is geometry (not text); the output is field distributions (not tokens)
- Domain-specific architecture substantially outperforms general LLMs on EM tasks

### 5.5 Relevance to WiFi Sensing

Arena Physica focuses on RF component design (antennas, PCBs, filters), not WiFi sensing. However, their approach is directly transferable:

| Arena Physica (Design) | WiFi Sensing (Our Case) |
|---|---|
| Forward: geometry -> S-parameters | Forward: body pose -> CSI |
| Inverse: S-parameters -> geometry | Inverse: CSI -> body pose |
| Train on FDTD/FEM simulations | Train on ray-tracing / FDTD simulations |
| 13ms inference | Real-time CSI inference |
| Conditional diffusion for generation | Conditional generation for pose prediction |

**Key lesson for wifi-densepose**: Building a neural forward model (body_pose -> expected_CSI) trained on electromagnetic simulation data, then using it as a differentiable physics constraint during inverse model training, could significantly improve our pose estimation accuracy and generalization. This is the "physics-informed" approach with the computational burden shifted to offline training.


## 6. Connections to wifi-densepose Codebase

### 6.1 Existing Physics-Based Modules

| Module | Physical Model | Maxwell Connection |
|---|---|---|
| `field_model.rs` | SVD eigenstructure decomposition | Eigenmode basis of room's EM field |
| `tomography.rs` | L1-regularized RTI (ISTA solver) | Linearized inverse scattering |
| `multistatic.rs` | Attention-weighted cross-node fusion | Exploits geometric diversity of multiple TX/RX |
| `phase_align.rs` | LO phase offset estimation | Corrects hardware-induced phase corruption |
| `coherence.rs` | Z-score coherence scoring | Statistical test on EM field stability |
| `coherence_gate.rs` | Accept/Reject decisions | Quality control on EM measurements |
| `adversarial.rs` | Physical impossibility detection | Enforces EM consistency constraints |

### 6.2 Potential Enhancements Based on This Research

1. **Differentiable ray-tracing forward model**: Train a neural surrogate on ray-tracing simulations of CSI for various body poses in the deployment room. Use as physics constraint in pose estimation.

2. **Fresnel zone integration**: Augment the attention mechanism in `multistatic.rs` with Fresnel zone geometry -- links where the body falls within the first Fresnel zone should receive higher attention weight.

3. **xPRA-LM inverse scattering**: For higher-resolution body imaging than RTI, implement the Extended Phaseless Rytov Approximation. Our tomography module currently uses the simpler additive attenuation model.

4. **HoloCSI-style sparse transformer**: Replace the dense attention in cross-viewpoint fusion with top-k sparse attention for efficiency on ESP32-constrained deployments.

5. **Physics-informed training loss**: When training the DensePose model, add a loss term penalizing physically impossible CSI patterns (e.g., signals that would require faster-than-light propagation or negative attenuation).


## 7. References

### Core WiFi Sensing Surveys
- WiFi Sensing with Channel State Information: A Survey. ACM Computing Surveys, 2019. https://dl.acm.org/doi/fullHtml/10.1145/3310194
- Cross-Domain WiFi Sensing with Channel State Information: A Survey. ACM Computing Surveys, 2022. https://dl.acm.org/doi/10.1145/3570325
- Wireless sensing applications with Wi-Fi CSI, preprocessing techniques, and detection algorithms: A survey. Computer Communications, 2024. https://www.sciencedirect.com/science/article/abs/pii/S0140366424002214
- Understanding CSI (Tsinghua Tutorial). https://tns.thss.tsinghua.edu.cn/wst/docs/pre/

### Physics-Informed Neural Networks for RF
- PINN and GNN-based RF Map Construction. arXiv 2507.22513
- Physics-Informed Neural Networks for Wireless Channel Estimation. NeurIPS 2025, OpenReview r3plaU6DvW
- ReVeal: High-Fidelity Radio Propagation. DySPAN 2025. https://wici.iastate.edu/wp-content/uploads/2025/03/ReVeal-DySPAN25.pdf
- Physics-informed generative model for passive RF sensing. Savazzi et al., arXiv 2310.04173
- Multi-Modal Foundational Model for Wireless Communication and Sensing. arXiv 2602.04016
- Generative AI Meets Wireless Sensing: Towards Wireless Foundation Model. arXiv 2509.15258
- Physics-Informed Neural Networks for Sensing Radio Spectrum. IJRTE v14i3, 2025

### Inverse Scattering and Body Reconstruction
- DensePose From WiFi. Geng, Huang, De la Torre. arXiv 2301.00250
- Through-Wall Human Pose Estimation Using Radio Signals. Zhao et al., CVPR 2018. https://rfpose.csail.mit.edu/
- Person-in-WiFi: Fine-grained Person Perception. Wang et al., ICCV 2019
- 3D Human Pose Estimation for Free-from Activities Using WiFi. arXiv 2204.07878
- EM-POSE: 3D Human Pose from Sparse Electromagnetic Trackers. ICCV 2021
- Reconciling Radio Tomographic Imaging with Phaseless Inverse Scattering. Dubey, Li, Murch. arXiv 2311.09633
- Accurate Indoor RF Imaging using Extended Rytov Approximation. Dubey et al., arXiv 2110.03211
- Phaseless Extended Rytov Approximation for Strongly Scattering Low-Loss Media. IEEE, 2022. https://ieeexplore.ieee.org/document/9766313/
- Distorted Wave Extended Phaseless Rytov Iterative Method. arXiv 2205.12578
- 3D Full Convolution Electromagnetic Reconstruction Neural Network (3D-FCERNN). PMC 9689780

### Radio Tomographic Imaging
- Radio Tomographic Imaging with Wireless Networks. Wilson & Patwari, 2010. https://span.ece.utah.edu/uploads/RTI_version_3.pdf
- Compressive Sensing Based Radio Tomographic Imaging with Spatial Diversity. PMC 6386865
- Passive Localization Based on Radio Tomography Images with CNN. Nature Scientific Reports, 2025
- Enhancing Accuracy of WiFi Tomographic Imaging Using Human-Interference Model. 2018

### Fresnel Zone Models
- WiFi CSI-based device-free sensing: from Fresnel zone model to CSI-ratio model. CCF Trans. Pervasive Computing, 2021. https://link.springer.com/article/10.1007/s42486-021-00077-z
- Towards a Dynamic Fresnel Zone Model for WiFi-based Human Activity Recognition. ACM IMWUT, 2023. https://dl.acm.org/doi/10.1145/3596270
- CSI-based human sensing using model-based approaches: a survey. JCDE, 2021. https://academic.oup.com/jcde/article/8/2/510/6137731

### Computational Electromagnetics
- Using Large-Scale FDTD for Indoor WLAN. ResearchGate. https://www.researchgate.net/publication/42637096
- Human Body Shadowing -- FDTD and UTD. Hindawi IJAP, 2018. https://www.hindawi.com/journals/ijap/2018/9084830/
- FDTD Assessment Human Exposure WiFi/Bluetooth. ResearchGate. https://www.researchgate.net/publication/23400115
- Simulation of Wireless LAN Indoor Propagation Using FDTD. IEEE, 2007. https://ieeexplore.ieee.org/document/4396450
- Waveguide Models of Indoor Channels: FDTD Insights. ResearchGate. https://www.researchgate.net/publication/4368711
- XFdtd 3D EM Simulation Software. Remcom. https://www.remcom.com/xfdtd-3d-em-simulation-software
- Wireless InSite Ray Tracing. Remcom. https://www.remcom.com/wireless-insite-em-propagation-software/

### Arena Physica
- Introducing Atlas RF Studio. https://www.arenaphysica.com/publications/rf-studio
- Electromagnetism Secretly Runs the World. Not Boring (Packy McCormick). https://www.notboring.co/p/electromagnetism-secretly-runs-the
- Arena Launches Atlas (Press Release). https://www.prnewswire.com/news-releases/arena-launches-atlas-to-accelerate-humanitys-rate-of-hardware-innovation-302423412.html
- Arena AI raises $30M. SiliconANGLE. https://siliconangle.com/2025/04/08/arena-ai-raises-30m-accelerate-innovation-hardware-testing-atlas/
- Artificial Intuition: Building an AI Mind for EM Design. CDFAM NYC 2025. https://www.designforam.com/p/artificial-intuition-building-an

### Holographic / Advanced
- HoloCSI: Holographic tomography pipeline with physics-guided projection and sparse transformer. 2025-2026
- CSI-Bench: Large-Scale In-the-Wild Dataset for Multi-task WiFi Sensing. arXiv 2505.21866
- RFBoost: Understanding and Boosting Deep WiFi Sensing via Physical Data Augmentation. arXiv 2410.07230
- Vision Reimagined: AI-Powered Breakthroughs in WiFi Indoor Imaging. arXiv 2401.04317
- Electromagnetic Information Theory for 6G. arXiv 2401.08921
