/**
 * CVHybridManager — Webcam pose estimation + WiFi CSI fusion
 *
 * Uses the browser's built-in ML / MediaPipe Pose (loaded via CDN) when camera
 * is available. Falls back to WiFi-only mode seamlessly.
 *
 * Emits fused 17-keypoint COCO arrays and confidence scores that the
 * Observatory 3D main loop can consume.
 *
 * Fusion strategy:
 *   - Video confidence derived from landmark visibility average
 *   - WiFi confidence from CSI classification.confidence
 *   - Weighted blend: fused = alpha*video + (1-alpha)*wifi
 *     where alpha = videoConf / (videoConf + wifiConf)
 */

const MEDIAPIPE_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/pose.js';

// MediaPipe → COCO-17 index remapping
// MediaPipe has 33 landmarks; we extract the 17 COCO equivalents
const MP_TO_COCO = [
  0,   // 0  nose
  2,   // 1  left_eye
  5,   // 2  right_eye
  7,   // 3  left_ear
  8,   // 4  right_ear
  11,  // 5  left_shoulder
  12,  // 6  right_shoulder
  13,  // 7  left_elbow
  14,  // 8  right_elbow
  15,  // 9  left_wrist
  16,  // 10 right_wrist
  23,  // 11 left_hip
  24,  // 12 right_hip
  25,  // 13 left_knee
  26,  // 14 right_knee
  27,  // 15 left_ankle
  28,  // 16 right_ankle
];

// Room coordinate mapping: video pixel → 3D room space
// Video coords: x in [0,1] L→R, y in [0,1] T→B
// Room coords:  x in [-5,5], z in [-4,4]
function videoToRoom(vx, vy) {
  return [
    (vx - 0.5) * 9,   // x: center→sides
    0,                  // y: will be overridden from height estimate
    (vy - 0.5) * 6,   // z: top→bottom of frame = front→back of room
  ];
}

export class CVHybridManager {
  constructor(videoEl, skeletonCanvas) {
    this._video  = videoEl;
    this._canvas = skeletonCanvas;
    this._ctx    = skeletonCanvas.getContext('2d');
    this._pose   = null;       // MediaPipe Pose instance
    this._stream = null;
    this._animId = null;
    this._cameraActive = false;
    this._mediapipeLoaded = false;

    // Output state
    this.videoKeypoints  = null;  // raw 17×3 from camera (room coords)
    this.videoConf       = 0;
    this.wifiKeypoints   = null;  // 17×3 from WiFi/demo
    this.wifiConf        = 0;
    this.fusedKeypoints  = null;
    this.fusedConf       = 0;
    this.mode = 'hybrid'; // 'hybrid' | 'cv-only' | 'wifi-only'

    // Callbacks
    this.onFused = null; // (fusedKps, videoConf, wifiConf, fusedConf) => void
  }

  // ---- Camera ----

  async startCamera() {
    try {
      this._stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false,
      });
      this._video.srcObject = this._stream;
      await new Promise(res => { this._video.onloadedmetadata = res; });
      this._video.play();
      this._cameraActive = true;
      await this._loadMediaPipe();
      this._startDetectionLoop();
      return true;
    } catch (e) {
      console.warn('[CVHybrid] Camera unavailable:', e.message);
      this._cameraActive = false;
      return false;
    }
  }

  stopCamera() {
    if (this._stream) {
      this._stream.getTracks().forEach(t => t.stop());
      this._stream = null;
    }
    this._cameraActive = false;
    if (this._animId) { cancelAnimationFrame(this._animId); this._animId = null; }
  }

  // ---- MediaPipe ----

  async _loadMediaPipe() {
    if (this._mediapipeLoaded) return;
    return new Promise((resolve) => {
      // Try loading MediaPipe; if unavailable, fall back gracefully
      const script = document.createElement('script');
      script.src = MEDIAPIPE_URL;
      script.onload = async () => {
        try {
          const Pose = window.Pose;
          if (!Pose) { resolve(); return; }
          this._pose = new Pose({
            locateFile: (file) =>
              `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`,
          });
          this._pose.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            enableSegmentation: false,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
          });
          this._pose.onResults((results) => this._onMediaPipeResults(results));
          await this._pose.initialize();
          this._mediapipeLoaded = true;
          console.log('[CVHybrid] MediaPipe Pose loaded ✓');
        } catch (e) {
          console.warn('[CVHybrid] MediaPipe init failed, using WiFi-only:', e.message);
        }
        resolve();
      };
      script.onerror = () => {
        console.warn('[CVHybrid] MediaPipe CDN unavailable, using fallback skeleton');
        resolve();
      };
      document.head.appendChild(script);
    });
  }

  _startDetectionLoop() {
    const detect = async () => {
      if (!this._cameraActive) return;
      if (this._pose && this._video.readyState >= 2) {
        try {
          await this._pose.send({ image: this._video });
        } catch (_) {}
      } else if (this._cameraActive) {
        // MediaPipe not available: draw raw video-based skeleton approx
        this._drawFallbackSkeleton();
      }
      this._animId = requestAnimationFrame(detect);
    };
    this._animId = requestAnimationFrame(detect);
  }

  _onMediaPipeResults(results) {
    const lms = results.poseLandmarks;
    if (!lms || lms.length < 29) {
      this.videoKeypoints = null;
      this.videoConf = 0;
      this._updateFusion();
      return;
    }

    // Extract COCO-17 keypoints in room coordinates
    const kps = MP_TO_COCO.map(mpIdx => {
      const lm = lms[mpIdx];
      const roomXZ = videoToRoom(lm.x, lm.y);
      // Estimate Y from image position: y=0 at ankle level, y~1.7 at head
      // We'll override with WiFi or estimate from shoulder height
      return [roomXZ[0], 0, roomXZ[2]];
    });

    // Estimate real Y heights from body proportions
    // Assume standing ~1.7m, use shoulder landmark as anchor
    const shoulderY_vid = (lms[11].y + lms[12].y) / 2;
    const ankleY_vid    = (lms[27].y + lms[28].y) / 2;
    const headY_vid     = lms[0].y;
    const totalH_vid    = Math.max(0.01, ankleY_vid - headY_vid);
    const realH         = 1.72; // assume 1.72m person
    const scale         = realH / totalH_vid;

    for (let i = 0; i < 17; i++) {
      const mpIdx = MP_TO_COCO[i];
      const lm    = lms[mpIdx];
      // Y: invert (image top=high, image bottom=low), scale to real height
      const relY  = ankleY_vid - lm.y; // positive = above ankle
      kps[i][1]   = Math.max(0, relY * scale);
    }

    this.videoKeypoints = kps;
    // Confidence: average visibility of key landmarks
    const keyLms = [0, 5, 6, 11, 12, 15, 16];
    this.videoConf = keyLms.reduce((s, i) => s + (lms[MP_TO_COCO[i]]?.visibility || 0), 0) / keyLms.length;

    this._drawCVSkeleton(lms);
    this._updateFusion();
  }

  // ---- Fusion ----

  setWifiKeypoints(kps, conf) {
    this.wifiKeypoints = kps;
    this.wifiConf = conf || 0;
    this._updateFusion();
  }

  setMode(mode) {
    this.mode = mode;
    this._updateFusion();
  }

  _updateFusion() {
    if (this.mode === 'cv-only') {
      this.fusedKeypoints = this.videoKeypoints;
      this.fusedConf      = this.videoConf;
    } else if (this.mode === 'wifi-only') {
      this.fusedKeypoints = this.wifiKeypoints;
      this.fusedConf      = this.wifiConf;
    } else {
      // Hybrid weighted blend
      const vc = this.videoConf;
      const wc = this.wifiConf;
      const total = vc + wc;

      if (total < 0.001) {
        this.fusedKeypoints = this.wifiKeypoints;
        this.fusedConf      = 0;
      } else if (!this.videoKeypoints) {
        this.fusedKeypoints = this.wifiKeypoints;
        this.fusedConf      = wc;
      } else if (!this.wifiKeypoints) {
        this.fusedKeypoints = this.videoKeypoints;
        this.fusedConf      = vc;
      } else {
        const alpha = vc / total;
        this.fusedKeypoints = this.wifiKeypoints.map((wk, i) => {
          const vk = this.videoKeypoints[i];
          if (!vk) return wk;
          return [
            wk[0] * (1-alpha) + vk[0] * alpha,
            wk[1] * (1-alpha) + vk[1] * alpha,
            wk[2] * (1-alpha) + vk[2] * alpha,
          ];
        });
        this.fusedConf = Math.min(1, vc * 0.6 + wc * 0.4 + (vc > 0.5 && wc > 0.5 ? 0.1 : 0));
      }
    }

    if (this.onFused) {
      this.onFused(this.fusedKeypoints, this.videoConf, this.wifiConf, this.fusedConf);
    }
  }

  // ---- CV Canvas Drawing ----

  _drawCVSkeleton(mpLandmarks) {
    const w = this._canvas.width  = this._video.videoWidth  || 280;
    const h = this._canvas.height = this._video.videoHeight || 157;
    const ctx = this._ctx;
    ctx.clearRect(0, 0, w, h);

    const PAIRS = [
      [0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],
      [5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],
    ];

    const pt = (mpIdx) => ({
      x: mpLandmarks[mpIdx].x * w,
      y: mpLandmarks[mpIdx].y * h,
      vis: mpLandmarks[mpIdx].visibility || 0,
    });

    // Draw bones
    ctx.lineWidth = 2;
    for (const [a, b] of PAIRS) {
      const pA = pt(MP_TO_COCO[a]);
      const pB = pt(MP_TO_COCO[b]);
      if (pA.vis < 0.3 || pB.vis < 0.3) continue;
      const alpha = Math.min(pA.vis, pB.vis);
      ctx.strokeStyle = `rgba(0,216,120,${alpha * 0.85})`;
      ctx.shadowColor = '#00d878';
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.moveTo(pA.x, pA.y);
      ctx.lineTo(pB.x, pB.y);
      ctx.stroke();
    }

    // Draw joints
    ctx.shadowBlur = 8;
    for (let i = 0; i < 17; i++) {
      const p = pt(MP_TO_COCO[i]);
      if (p.vis < 0.3) continue;
      const isKey = [0, 5, 6, 11, 12].includes(i);
      ctx.fillStyle = isKey ? `rgba(255,64,96,${p.vis})` : `rgba(0,200,255,${p.vis * 0.8})`;
      ctx.shadowColor = isKey ? '#ff4060' : '#00c8ff';
      ctx.beginPath();
      ctx.arc(p.x, p.y, isKey ? 5 : 3, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.shadowBlur = 0;
  }

  _drawFallbackSkeleton() {
    // When MediaPipe unavailable: draw a simple overlay showing "CV unavailable"
    const ctx = this._ctx;
    ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);
    ctx.fillStyle = 'rgba(0,200,255,0.5)';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillText('WiFi-only mode', 8, 20);
  }

  // ---- DensePose UV canvas ----

  renderDensePoseCanvas(canvas, kps) {
    if (!kps) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Project 3D keypoints onto 2D canvas (front view, Y-up)
    // Find bounding box
    let minY = Infinity, maxY = -Infinity, minX = Infinity, maxX = -Infinity;
    for (const p of kps) {
      minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
    }
    const rangeX = Math.max(0.5, maxX - minX);
    const rangeY = Math.max(1.0, maxY - minY);
    const pad = 15;

    const px = (i) => pad + ((kps[i][0] - minX) / rangeX) * (w - pad*2);
    const py = (i) => h - pad - ((kps[i][1] - minY) / rangeY) * (h - pad*2);

    // Draw body segments as filled shapes
    const SEGMENTS = [
      { pts:[5,6,12,11], color:'#ff4444', name:'torso' },
      { pts:[5,7,9], color:'#44aaff', name:'larm' },
      { pts:[6,8,10], color:'#2288dd', name:'rarm' },
      { pts:[11,13,15], color:'#44ff88', name:'lleg' },
      { pts:[12,14,16], color:'#22dd66', name:'rleg' },
    ];

    for (const seg of SEGMENTS) {
      ctx.beginPath();
      ctx.moveTo(px(seg.pts[0]), py(seg.pts[0]));
      for (let i = 1; i < seg.pts.length; i++) {
        ctx.lineTo(px(seg.pts[i]), py(seg.pts[i]));
      }
      ctx.closePath();
      ctx.fillStyle = seg.color + 'aa';
      ctx.strokeStyle = seg.color;
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();
    }

    // Head ellipse
    const hx = px(0), hy = py(0);
    const rx = (w - pad*2) * 0.12, ry = rx * 1.3;
    ctx.beginPath();
    ctx.ellipse(hx, hy, rx, ry, 0, 0, Math.PI*2);
    ctx.fillStyle = '#ffaa44aa';
    ctx.strokeStyle = '#ffaa44';
    ctx.lineWidth = 1.5;
    ctx.fill(); ctx.stroke();

    // Labels
    ctx.fillStyle = 'rgba(232,236,224,0.35)';
    ctx.font = '8px JetBrains Mono, monospace';
    ctx.fillText('FRONT VIEW', 4, h-4);
  }
}
