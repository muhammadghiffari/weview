/**
 * WeView — Dual-Modal Pose Estimation (v5 — MediaPipe integration)
 * Real webcam pose via MediaPipe Pose Landmarker + simulated CSI fusion.
 */

import { CsiSimulator } from './csi-simulator.js?v=13';
import { FusionEngine } from './fusion-engine.js?v=13';
import { CanvasRenderer } from './canvas-renderer.js?v=13';
import { KEYPOINT_NAMES, SKELETON_CONNECTIONS } from './pose-decoder.js?v=13';

// === MediaPipe CDN ===
const MP_VISION_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MP_WASM = `${MP_VISION_CDN}/wasm`;
const MP_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task';

// === State ===
let mode = 'dual';
let isRunning = false;
let isPaused = false;
let startTime = 0;
let frameCount = 0;
let fps = 0;
let lastFpsTime = 0;
let confidenceThreshold = 0.3;
let poseLandmarker = null;
let mpLoading = false;

const latency = { video: 0, csi: 0, fusion: 0, total: 0 };

// === Components ===
const csiSimulator = new CsiSimulator({ subcarriers: 52, timeWindow: 56 });
const fusionEngine = new FusionEngine(128);
const renderer = new CanvasRenderer();

// === DOM Elements ===
const webcamEl = document.getElementById('webcam');
const skeletonCanvas = document.getElementById('skeleton-canvas');
const skeletonCtx = skeletonCanvas.getContext('2d');
const csiCanvas = document.getElementById('csi-canvas');
const csiCtx = csiCanvas.getContext('2d');
const embeddingCanvas = document.getElementById('embedding-canvas');
const embeddingCtx = embeddingCanvas.getContext('2d');

const modeSelect = document.getElementById('mode-select');
const statusDot = document.getElementById('status-dot');
const statusLabel = document.getElementById('status-label');
const fpsDisplay = document.getElementById('fps-display');
const cameraPrompt = document.getElementById('camera-prompt');
const startCameraBtn = document.getElementById('start-camera-btn');
const pauseBtn = document.getElementById('pause-btn');
const confSlider = document.getElementById('confidence-slider');
const confValue = document.getElementById('confidence-value');
const wsUrlInput = document.getElementById('ws-url');
const connectWsBtn = document.getElementById('connect-ws-btn');

const videoBar = document.getElementById('video-bar');
const csiBar = document.getElementById('csi-bar');
const fusedBar = document.getElementById('fused-bar');
const videoBarVal = document.getElementById('video-bar-val');
const csiBarVal = document.getElementById('csi-bar-val');
const fusedBarVal = document.getElementById('fused-bar-val');
const latVideoEl = document.getElementById('lat-video');
const latCsiEl = document.getElementById('lat-csi');
const latFusionEl = document.getElementById('lat-fusion');
const latTotalEl = document.getElementById('lat-total');
const crossModalEl = document.getElementById('cross-modal-sim');

const rssiBarEl = document.getElementById('rssi-bar');
const rssiValueEl = document.getElementById('rssi-value');
const rssiQualityEl = document.getElementById('rssi-quality');
const rssiSparkCanvas = document.getElementById('rssi-sparkline');
const rssiSparkCtx = rssiSparkCanvas ? rssiSparkCanvas.getContext('2d') : null;
const rssiHistory = [];
const RSSI_HISTORY_MAX = 80;

// Last detected MediaPipe keypoints (normalized 0-1)
let lastMpKeypoints = null;
let lastMpConfidence = 0;
let cameraActive = false;

// ── MediaPipe Loader ──────────────────────────────────────────────────────────

async function loadMediaPipe() {
  if (poseLandmarker || mpLoading) return;
  mpLoading = true;
  statusLabel.textContent = 'LOADING MP...';

  try {
    const vision = await import(/* @vite-ignore */ `${MP_VISION_CDN}/vision_bundle.mjs`);
    const { FilesetResolver, PoseLandmarker } = vision;
    const fileset = await FilesetResolver.forVisionTasks(MP_WASM);
    poseLandmarker = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MP_MODEL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numPoses: 1,
    });
    console.log('[PoseFusion] MediaPipe Pose Landmarker ready');
    const backendEl = document.getElementById('cnn-backend');
    if (backendEl) backendEl.textContent = 'MediaPipe Pose (WASM+GPU)';
  } catch (e) {
    console.error('[PoseFusion] MediaPipe load failed:', e);
    const backendEl = document.getElementById('cnn-backend');
    if (backendEl) backendEl.textContent = 'MediaPipe failed — motion fallback';
  } finally {
    mpLoading = false;
  }
}

// ── COCO 17 keypoint names mapping from MediaPipe 33 landmarks ────────────────
// MediaPipe returns 33 landmarks; we map to COCO 17 keypoints.
const MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

// Lower body joint indices (COCO): 11=l_hip, 12=r_hip, 13=l_knee, 14=r_knee, 15=l_ankle, 16=r_ankle
const LOWER_BODY = new Set([11, 12, 13, 14, 15, 16]);

function mediapipeToKeypoints(landmarks) {
  if (!landmarks || landmarks.length < 29) return null;

  // First pass: check if the camera actually shows the lower body.
  // If shoulder landmarks (MP indices 11,12) are in the bottom 40% of frame,
  // that means the person is close / only upper body visible.
  const lShoulder = landmarks[11];
  const rShoulder = landmarks[12];
  const shoulderY = ((lShoulder?.y || 0.5) + (rShoulder?.y || 0.5)) / 2;
  // If shoulders are in lower half of frame, lower body is likely NOT visible
  const lowerBodyVisible = shoulderY < 0.55;

  const kps = [];
  for (let i = 0; i < 17; i++) {
    const lm = landmarks[MP_TO_COCO[i]];
    if (!lm) { kps.push({ x: 0.5, y: 0.5, confidence: 0, name: KEYPOINT_NAMES[i] }); continue; }

    let conf = lm.visibility ?? 0;

    // Suppress lower body joints when they're clearly not visible in frame
    if (LOWER_BODY.has(i) && !lowerBodyVisible) {
      conf = 0; // Force to zero — these are hallucinated by the model
    }

    kps.push({
      x: 1 - lm.x,  // mirror for front-facing camera
      y: lm.y,
      confidence: conf,
      name: KEYPOINT_NAMES[i],
    });
  }
  return kps;
}

// ── Camera ────────────────────────────────────────────────────────────────────

async function startCamera() {
  cameraPrompt.style.display = 'none';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
    webcamEl.srcObject = stream;
    await webcamEl.play();
    cameraActive = true;
    statusDot.classList.remove('offline');
    statusLabel.textContent = 'LIVE';
    resizeCanvases();
    // Load MediaPipe in background
    loadMediaPipe();
  } catch (err) {
    console.error('[PoseFusion] Camera failed:', err);
    cameraPrompt.style.display = 'flex';
    cameraPrompt.querySelector('p').textContent = 'Camera access denied. Try CSI-only mode.';
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────

function init() {
  console.log('[PoseFusion] init v5 — MediaPipe integration');
  resizeCanvases();
  window.addEventListener('resize', resizeCanvases);

  modeSelect.addEventListener('change', (e) => { mode = e.target.value; updateModeUI(); });
  startCameraBtn.addEventListener('click', startCamera);
  pauseBtn.addEventListener('click', () => {
    isPaused = !isPaused;
    pauseBtn.textContent = isPaused ? '▶ Resume' : '⏸ Pause';
  });
  confSlider.addEventListener('input', (e) => {
    confidenceThreshold = parseFloat(e.target.value);
    confValue.textContent = confidenceThreshold.toFixed(2);
  });

  connectWsBtn.addEventListener('click', async () => {
    const url = wsUrlInput.value.trim();
    if (!url) return;
    connectWsBtn.textContent = 'Connecting...';
    const ok = await csiSimulator.connectLive(url);
    connectWsBtn.textContent = ok ? '✓ Connected' : 'Connect';
    if (ok) connectWsBtn.classList.add('active');
  });

  // Auto-connect to local sensing server
  const defaultWsUrl = 'ws://localhost:8765/ws/sensing';
  if (wsUrlInput) wsUrlInput.value = defaultWsUrl;
  csiSimulator.connectLive(defaultWsUrl).then(ok => {
    if (ok && connectWsBtn) {
      connectWsBtn.textContent = '✓ Live ESP32';
      connectWsBtn.classList.add('active');
      statusLabel.textContent = 'LIVE CSI';
      statusDot.classList.remove('offline');
    }
  });

  updateModeUI();
  startTime = performance.now() / 1000;
  isRunning = true;
  requestAnimationFrame(mainLoop);
}

function updateModeUI() {
  const needsVideo = mode !== 'csi';
  cameraPrompt.style.display = (needsVideo && !cameraActive) ? 'flex' : 'none';
  const labelMap = { dual: 'DUAL FUSION', video: 'VIDEO ONLY', csi: 'CSI ONLY' };
  const modeLabel = document.getElementById('mode-label');
  const promptLabel = document.getElementById('prompt-mode-label');
  if (modeLabel) modeLabel.textContent = labelMap[mode] || mode;
  if (promptLabel) promptLabel.textContent = labelMap[mode] || mode;
}

function resizeCanvases() {
  const videoPanel = document.querySelector('.video-panel');
  if (videoPanel) {
    const rect = videoPanel.getBoundingClientRect();
    skeletonCanvas.width = rect.width;
    skeletonCanvas.height = rect.height;
  }
  csiCanvas.width = Math.max(200, csiCanvas.parentElement.clientWidth);
  csiCanvas.height = 120;
  embeddingCanvas.width = Math.max(200, embeddingCanvas.parentElement.clientWidth);
  embeddingCanvas.height = 140;
}

// ── Main Loop ─────────────────────────────────────────────────────────────────

let _lastMpTs = 0;
let _loopErrorShown = false;

function mainLoop(timestamp) {
  if (!isRunning) return;
  requestAnimationFrame(mainLoop);
  if (isPaused) return;

  try {
    const elapsed = performance.now() / 1000 - startTime;
    const totalStart = performance.now();

    // ── Video Pipeline (MediaPipe) ──
    let videoKeypoints = null;
    let videoConfidence = 0;

    if (mode !== 'csi' && cameraActive && poseLandmarker && webcamEl.readyState >= 2) {
      const now = performance.now();
      if (now - _lastMpTs > 33) {  // ~30fps cap
        _lastMpTs = now;
        const t0 = performance.now();
        try {
          const result = poseLandmarker.detectForVideo(webcamEl, now);
          if (result.landmarks && result.landmarks.length > 0) {
            videoKeypoints = mediapipeToKeypoints(result.landmarks[0]);
            if (videoKeypoints) {
              videoConfidence = videoKeypoints.reduce((s, k) => s + k.confidence, 0) / videoKeypoints.length;
              lastMpKeypoints = videoKeypoints;
              lastMpConfidence = videoConfidence;

              // Feed to CSI simulator for correlated demo
              const noseKp = videoKeypoints[0];
              csiSimulator.updatePersonState(
                videoConfidence > 0.3 ? 1.0 : 0,
                noseKp.x, noseKp.y,
                videoConfidence
              );
            }
          }
        } catch (e) { /* skip frame */ }
        latency.video = performance.now() - t0;
      } else {
        // Reuse last result
        videoKeypoints = lastMpKeypoints;
        videoConfidence = lastMpConfidence;
      }
    }

    // ── CSI Pipeline ──
    let csiKeypoints = null;
    let csiConfidence = 0;

    if (mode !== 'video') {
      const t0 = performance.now();
      const csiFrame = csiSimulator.nextFrame(elapsed);

      // Generate synthetic CSI-based keypoints (simulated through-wall pose)
      const csiPresence = csiSimulator.personPresence || 0;
      if (csiPresence > 0.1) {
        csiKeypoints = generateCsiKeypoints(elapsed, csiPresence);
        csiConfidence = csiPresence * 0.7;
      }

      // Draw CSI heatmap
      const heatmap = csiSimulator.getHeatmapData();
      renderer.drawCsiHeatmap(csiCtx, heatmap, csiCanvas.width, csiCanvas.height);

      latency.csi = performance.now() - t0;
    }

    // ── Fusion ──
    const t0f = performance.now();
    let finalKeypoints = null;

    if ((mode === 'dual' || mode === 'video') && videoKeypoints) {
      // HONEST MODE: Only draw what the camera actually sees.
      // If joints have confidence 0 (suppressed lower body), they stay invisible.
      // CSI can slightly refine VISIBLE joints only (dual mode).
      if (mode === 'dual' && csiKeypoints) {
        finalKeypoints = videoKeypoints.map((vk, i) => {
          if (vk.confidence < 0.15) {
            // This joint is not visible — keep it invisible, ignore CSI
            return { x: vk.x, y: vk.y, confidence: 0, name: vk.name };
          }
          const ck = csiKeypoints[i];
          if (!ck) return vk;
          return {
            x: vk.x * 0.92 + ck.x * 0.08,
            y: vk.y * 0.92 + ck.y * 0.08,
            confidence: vk.confidence,
            name: vk.name,
          };
        });
      } else {
        finalKeypoints = videoKeypoints;
      }
    } else if (mode === 'dual' && !videoKeypoints && csiKeypoints) {
      finalKeypoints = csiKeypoints;
    } else if (mode === 'csi') {
      finalKeypoints = csiKeypoints;
    }

    latency.fusion = performance.now() - t0f;

    // ── Render Skeleton ──
    const labelMap = { dual: 'DUAL FUSION', video: 'VIDEO ONLY', csi: 'CSI ONLY' };
    renderer.drawSkeleton(skeletonCtx, finalKeypoints || [], skeletonCanvas.width, skeletonCanvas.height, {
      minConfidence: confidenceThreshold,
      color: mode === 'csi' ? 'amber' : 'green',
      label: labelMap[mode],
    });

    // ── Render Embedding Space ──
    updateEmbeddingViz(embeddingCtx, videoKeypoints, csiKeypoints, finalKeypoints);

    // ── Update UI ──
    latency.total = performance.now() - totalStart;

    frameCount++;
    if (timestamp - lastFpsTime > 500) {
      fps = Math.round(frameCount * 1000 / (timestamp - lastFpsTime));
      lastFpsTime = timestamp;
      frameCount = 0;
      fpsDisplay.textContent = `${fps} FPS`;
    }

    const vc = videoConfidence;
    const cc = csiConfidence;
    const fc = finalKeypoints ? finalKeypoints.reduce((s, k) => s + k.confidence, 0) / finalKeypoints.length : 0;
    videoBar.style.width = `${vc * 100}%`;
    csiBar.style.width = `${cc * 100}%`;
    fusedBar.style.width = `${fc * 100}%`;
    videoBarVal.textContent = `${Math.round(vc * 100)}%`;
    csiBarVal.textContent = `${Math.round(cc * 100)}%`;
    fusedBarVal.textContent = `${Math.round(fc * 100)}%`;

    latVideoEl.textContent = `${latency.video.toFixed(1)}ms`;
    latCsiEl.textContent = `${latency.csi.toFixed(1)}ms`;
    latFusionEl.textContent = `${latency.fusion.toFixed(1)}ms`;
    latTotalEl.textContent = `${latency.total.toFixed(1)}ms`;

    // Cross-modal similarity (cosine between video and CSI keypoint vectors)
    if (videoKeypoints && csiKeypoints) {
      const sim = computeKeypointSimilarity(videoKeypoints, csiKeypoints);
      crossModalEl.textContent = sim.toFixed(3);
    }

    // RuVector attention stats (simulated for UI)
    const rvEnergyEl = document.getElementById('rv-energy');
    const rvRefineEl = document.getElementById('rv-refine');
    const rvImpactEl = document.getElementById('rv-impact');
    if (rvEnergyEl) rvEnergyEl.textContent = (fc * 1.2).toFixed(2);
    if (rvRefineEl) rvRefineEl.textContent = (latency.fusion * 0.8).toFixed(1) + 'px';
    if (rvImpactEl) rvImpactEl.textContent = Math.round(fc * 100) + '%';
    if (fc > 0.1) document.querySelectorAll('.rv-stage').forEach(el => el.classList.add('active'));

    // RSSI
    updateRssi(csiSimulator.rssiDbm);

  } catch (err) {
    if (!_loopErrorShown) {
      _loopErrorShown = true;
      console.error('[MainLoop]', err);
    }
  }
}

// ── CSI Keypoint Generator (simulated through-wall pose) ──────────────────────

function generateCsiKeypoints(elapsed, presence) {
  const sway = Math.sin(elapsed * 0.8) * 0.03;
  const breathe = Math.sin(elapsed * 1.5) * 0.005;
  const walk = Math.sin(elapsed * 1.2) * 0.02;
  const cx = 0.5 + sway;
  const cy = 0.45;
  const s = 0.25; // scale

  const kps = [
    { x: cx, y: cy - s * 0.45, confidence: presence * 0.6 },           // nose
    { x: cx - 0.015, y: cy - s * 0.47, confidence: presence * 0.5 },   // l_eye
    { x: cx + 0.015, y: cy - s * 0.47, confidence: presence * 0.5 },   // r_eye
    { x: cx - 0.03, y: cy - s * 0.44, confidence: presence * 0.4 },    // l_ear
    { x: cx + 0.03, y: cy - s * 0.44, confidence: presence * 0.4 },    // r_ear
    { x: cx - s * 0.22, y: cy - s * 0.25 + breathe, confidence: presence * 0.7 }, // l_shoulder
    { x: cx + s * 0.22, y: cy - s * 0.25 + breathe, confidence: presence * 0.7 }, // r_shoulder
    { x: cx - s * 0.28, y: cy + walk, confidence: presence * 0.55 },    // l_elbow
    { x: cx + s * 0.28, y: cy + walk, confidence: presence * 0.55 },    // r_elbow
    { x: cx - s * 0.3, y: cy + s * 0.15 - walk, confidence: presence * 0.45 },  // l_wrist
    { x: cx + s * 0.3, y: cy + s * 0.15 + walk, confidence: presence * 0.45 },  // r_wrist
    { x: cx - s * 0.12, y: cy + s * 0.25, confidence: presence * 0.65 },  // l_hip
    { x: cx + s * 0.12, y: cy + s * 0.25, confidence: presence * 0.65 },  // r_hip
    { x: cx - s * 0.14 + walk, y: cy + s * 0.5, confidence: presence * 0.55 },  // l_knee
    { x: cx + s * 0.14 - walk, y: cy + s * 0.5, confidence: presence * 0.55 },  // r_knee
    { x: cx - s * 0.12 + walk * 1.2, y: cy + s * 0.72, confidence: presence * 0.5 }, // l_ankle
    { x: cx + s * 0.12 - walk * 1.2, y: cy + s * 0.72, confidence: presence * 0.5 }, // r_ankle
  ];

  for (let i = 0; i < kps.length; i++) kps[i].name = KEYPOINT_NAMES[i];
  return kps;
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function computeKeypointSimilarity(kpsA, kpsB) {
  let dotP = 0, magA = 0, magB = 0;
  const n = Math.min(kpsA.length, kpsB.length);
  for (let i = 0; i < n; i++) {
    dotP += kpsA[i].x * kpsB[i].x + kpsA[i].y * kpsB[i].y;
    magA += kpsA[i].x ** 2 + kpsA[i].y ** 2;
    magB += kpsB[i].x ** 2 + kpsB[i].y ** 2;
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom > 0 ? dotP / denom : 0;
}

// Simple embedding viz from keypoint positions
const embHistory = { video: [], csi: [], fused: [] };

function updateEmbeddingViz(ctx, vkps, ckps, fkps) {
  const project = (kps) => {
    if (!kps || kps.length < 5) return null;
    // PCA-like: use first 2 principal components of keypoint positions
    let sx = 0, sy = 0;
    for (const k of kps) { sx += k.x * k.confidence; sy += k.y * k.confidence; }
    return [sx / kps.length - 0.5, sy / kps.length - 0.5];
  };

  const vp = project(vkps), cp = project(ckps), fp = project(fkps);
  if (vp) { embHistory.video.push(vp); if (embHistory.video.length > 60) embHistory.video.shift(); }
  if (cp) { embHistory.csi.push(cp); if (embHistory.csi.length > 60) embHistory.csi.shift(); }
  if (fp) { embHistory.fused.push(fp); if (embHistory.fused.length > 60) embHistory.fused.shift(); }

  renderer.drawEmbeddingSpace(ctx, embHistory, embeddingCanvas.width, embeddingCanvas.height);
}

// ── RSSI ──────────────────────────────────────────────────────────────────────

function updateRssi(dbm) {
  if (!rssiBarEl) return;
  const clamped = Math.max(-100, Math.min(-30, dbm));
  const pct = ((clamped + 100) / 70) * 100;
  rssiBarEl.style.width = `${pct}%`;
  rssiValueEl.textContent = `${Math.round(clamped)} dBm`;
  let quality;
  if (clamped > -50) quality = 'Excellent';
  else if (clamped > -60) quality = 'Good';
  else if (clamped > -70) quality = 'Fair';
  else if (clamped > -80) quality = 'Weak';
  else quality = 'Poor';
  rssiQualityEl.textContent = quality;
  if (clamped > -60) rssiValueEl.style.color = 'var(--green-glow)';
  else if (clamped > -75) rssiValueEl.style.color = 'var(--amber)';
  else rssiValueEl.style.color = 'var(--red-alert)';
  rssiHistory.push(clamped);
  if (rssiHistory.length > RSSI_HISTORY_MAX) rssiHistory.shift();
  drawRssiSparkline();
}

function drawRssiSparkline() {
  if (!rssiSparkCtx || rssiHistory.length < 2) return;
  const w = rssiSparkCanvas.width, h = rssiSparkCanvas.height, ctx = rssiSparkCtx;
  ctx.clearRect(0, 0, w, h);
  const len = rssiHistory.length;
  const step = w / (RSSI_HISTORY_MAX - 1);
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, 'rgba(0,210,120,0.3)');
  grad.addColorStop(1, 'rgba(0,210,120,0)');
  ctx.beginPath();
  for (let i = 0; i < len; i++) {
    const x = (RSSI_HISTORY_MAX - len + i) * step;
    const y = h - ((rssiHistory[i] + 100) / 70) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  const lastX = (RSSI_HISTORY_MAX - 1) * step;
  const firstX = (RSSI_HISTORY_MAX - len) * step;
  ctx.lineTo(lastX, h); ctx.lineTo(firstX, h); ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();
  ctx.beginPath();
  for (let i = 0; i < len; i++) {
    const x = (RSSI_HISTORY_MAX - len + i) * step;
    const y = h - ((rssiHistory[i] + 100) / 70) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = '#00d878'; ctx.lineWidth = 1.5; ctx.stroke();
  const ly = h - ((rssiHistory[len - 1] + 100) / 70) * h;
  const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 300);
  ctx.beginPath(); ctx.arc(lastX, ly, 2 + pulse, 0, Math.PI * 2);
  ctx.fillStyle = '#00d878'; ctx.fill();
}

// Boot
document.addEventListener('DOMContentLoaded', init);
