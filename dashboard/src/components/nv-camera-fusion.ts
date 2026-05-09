/**
 * nv-camera-fusion.ts — Webcam + MediaPipe Pose Estimation Component
 *
 * Automatically starts the webcam on mount (auto-consent with getUserMedia),
 * runs MediaPipe Pose via CDN, converts landmarks to FusionPerson signals,
 * and updates cameraPersons / cameraConfidence / cameraActive in the store.
 *
 * The video feed is hidden by default (privacy) but can be toggled visible.
 * The component is invisible — it has no visual output of its own; the
 * nv-pose-overlay component reads the store signals to draw the stickman.
 */
import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import {
  cameraPersons, cameraConfidence, cameraActive, pushLog,
  type FusionPerson, type FusionKeypoint,
} from '../store/appStore';
import { KP_NAMES } from '../store/fusionEngine';

// ── MediaPipe types (loaded from CDN) ────────────────────────────────────────

interface Landmark { x: number; y: number; z: number; visibility?: number; }
interface PoseLandmarkerResult { landmarks: Landmark[][]; }
interface MPPoseLandmarker {
  detectForVideo(video: HTMLVideoElement, ts: number): PoseLandmarkerResult;
  close(): void;
}

declare global {
  interface Window {
    __mpPoseLandmarker?: MPPoseLandmarker;
    __mpPoseLoading?: boolean;
  }
}

const MP_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MP_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task';

async function loadMediaPipe(): Promise<MPPoseLandmarker | null> {
  if (window.__mpPoseLandmarker) return window.__mpPoseLandmarker;
  if (window.__mpPoseLoading) return null;
  window.__mpPoseLoading = true;

  try {
    // Dynamically import from CDN — avoids bundling the 4 MB WASM
    // @ts-ignore
    const module = await import(
      /* @vite-ignore */
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
    ) as Record<string, unknown>;

    const { FilesetResolver, PoseLandmarker } = module as {
      FilesetResolver: { forVisionTasks(url: string): Promise<unknown> };
      PoseLandmarker: { createFromOptions(fs: unknown, opts: unknown): Promise<MPPoseLandmarker> };
    };

    const fs = await FilesetResolver.forVisionTasks(MP_CDN);
    const landmarker = await PoseLandmarker.createFromOptions(fs, {
      baseOptions: {
        modelAssetPath: MP_MODEL,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numPoses: 3,
    });

    window.__mpPoseLandmarker = landmarker;
    return landmarker;
  } catch (e) {
    console.error('[nv-camera-fusion] MediaPipe load failed:', e);
    return null;
  } finally {
    window.__mpPoseLoading = false;
  }
}

// ── Component ─────────────────────────────────────────────────────────────────

@customElement('nv-camera-fusion')
export class NvCameraFusion extends LitElement {
  @state() private _camError = '';
  @state() private _showPreview = false;
  @state() private _mpReady = false;

  private _stream: MediaStream | null = null;
  private _rafId = 0;
  private _landmarker: MPPoseLandmarker | null = null;
  private _lastTs = 0;

  static styles = css`
    :host { display: block; }
    .preview-wrap {
      position: relative;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: var(--bg-0);
    }
    video {
      width: 100%;
      display: block;
      transform: scaleX(-1); /* Mirror for self-view */
    }
    .cam-badge {
      position: absolute;
      top: 6px; left: 6px;
      font-size: 10px;
      font-family: var(--mono);
      background: rgba(0,0,0,0.6);
      color: var(--ok);
      padding: 2px 6px;
      border-radius: 4px;
    }
    .error {
      font-size: 11px;
      color: var(--warn);
      padding: 4px 0;
    }
  `;

  override async connectedCallback(): Promise<void> {
    super.connectedCallback();
    await this._startCamera();
  }

  override disconnectedCallback(): void {
    super.disconnectedCallback();
    this._stopCamera();
  }

  private async _startCamera(): Promise<void> {
    try {
      this._stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false,
      });
      cameraActive.value = true;
      this._camError = '';
      pushLog('ok', '🎥 Camera started — loading MediaPipe…');

      // Attach stream to video element after render
      await this.updateComplete;
      const video = this.renderRoot.querySelector('video') as HTMLVideoElement;
      if (video) {
        video.srcObject = this._stream;
        await video.play();
      }

      // Load MediaPipe (may take a few seconds)
      this._landmarker = await loadMediaPipe();
      if (this._landmarker) {
        this._mpReady = true;
        pushLog('ok', '🤖 MediaPipe Pose ready — hybrid fusion active');
        this._startDetectionLoop();
      } else {
        pushLog('warn', '⚠️ MediaPipe unavailable — WiFi-only mode');
        cameraConfidence.value = 0;
      }
    } catch (e) {
      this._camError = (e instanceof Error) ? e.message : 'Camera access denied';
      cameraActive.value = false;
      cameraConfidence.value = 0;
      pushLog('warn', `📷 Camera unavailable: ${this._camError} — using WiFi mode`);
    }
  }

  private _stopCamera(): void {
    cancelAnimationFrame(this._rafId);
    this._stream?.getTracks().forEach(t => t.stop());
    this._stream = null;
    this._landmarker?.close();
    this._landmarker = null;
    cameraActive.value = false;
    cameraConfidence.value = 0;
    cameraPersons.value = [];
  }

  private _startDetectionLoop(): void {
    const video = this.renderRoot.querySelector('video') as HTMLVideoElement;
    if (!video || !this._landmarker) return;

    const detect = (): void => {
      if (!this._landmarker || video.readyState < 2) {
        this._rafId = requestAnimationFrame(detect);
        return;
      }

      const now = performance.now();
      if (now - this._lastTs < 50) { // ~20 fps cap for performance
        this._rafId = requestAnimationFrame(detect);
        return;
      }
      this._lastTs = now;

      try {
        const result = this._landmarker.detectForVideo(video, now);
        this._processResult(result);
      } catch {
        // Video frame not ready yet — skip
      }

      this._rafId = requestAnimationFrame(detect);
    };

    this._rafId = requestAnimationFrame(detect);
  }

  private _processResult(result: PoseLandmarkerResult): void {
    const poses = result.landmarks ?? [];

    // Calculate overall camera confidence from average landmark visibility
    let totalVis = 0;
    let totalKps = 0;
    const persons: FusionPerson[] = poses.slice(0, 3).map((landmarks, personIdx) => {
      const keypoints: FusionKeypoint[] = landmarks.slice(0, 17).map((lm, i) => {
        const vis = lm.visibility ?? 0;
        totalVis += vis;
        totalKps++;
        return {
          name: KP_NAMES[i] ?? `kp_${i}`,
          // MediaPipe returns normalized 0..1; scale to 640×480 viewport
          x: (1 - lm.x) * 640, // mirror x for front-facing
          y: lm.y * 480,
          z: lm.z,
          confidence: vis,
        };
      });

      return {
        id: personIdx + 1,
        source: 'camera' as const,
        confidence: keypoints.reduce((s, k) => s + k.confidence, 0) / keypoints.length,
        keypoints,
      };
    });

    cameraPersons.value = persons;
    cameraConfidence.value = totalKps > 0
      ? Math.min(totalVis / totalKps, 1)
      : 0;
  }

  togglePreview(): void {
    this._showPreview = !this._showPreview;
  }

  override render() {
    return html`
      <video
        style="display: none"
        playsinline
        muted
        autoplay
      ></video>

      ${this._showPreview ? html`
        <div class="preview-wrap">
          <video playsinline muted autoplay
            style="width:100%; display:block; transform:scaleX(-1);"
            @loadedmetadata=${(e: Event) => {
              const v = e.target as HTMLVideoElement;
              if (this._stream) v.srcObject = this._stream;
            }}
          ></video>
          <span class="cam-badge">
            ${this._mpReady ? '● LIVE' : '● Loading…'}
          </span>
        </div>
      ` : ''}

      ${this._camError ? html`<div class="error">⚠ ${this._camError}</div>` : ''}
    `;
  }
}
