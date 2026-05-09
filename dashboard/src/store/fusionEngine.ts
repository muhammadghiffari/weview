/**
 * fusionEngine.ts — Hybrid Camera + WiFi CSI Fusion Logic
 *
 * Merges MediaPipe camera pose data with WiFi-derived person detections.
 * Automatically selects the best source per confidence level:
 *   - camera conf > 0.70  → camera primary (accurate keypoints)
 *   - camera conf < 0.30  → wifi primary (through walls / dark)
 *   - between             → weighted blend
 *
 * Also sends camera hints to the backend so Rust can bias its person count.
 */

import { effect } from '@preact/signals-core';
import {
  cameraPersons, cameraConfidence, cameraActive,
  wifiPersons, wifiPersonCount,
  fusionMode, fusedPersons,
  type FusionPerson, type FusionKeypoint,
} from './appStore';

// ── Thresholds ───────────────────────────────────────────────────────────────

const CAMERA_PRIMARY_THRESHOLD  = 0.70;  // above → camera wins
const CAMERA_FALLBACK_THRESHOLD = 0.30;  // below → wifi wins
const HINT_SEND_INTERVAL_MS     = 500;   // how often to POST camera-hint to backend

// ── COCO-17 bone connections for keypoint interpolation ──────────────────────

export const BONE_PAIRS: [number, number][] = [
  [5, 7], [7, 9],   // left arm
  [6, 8], [8, 10],  // right arm
  [5, 6],           // shoulders
  [5, 11], [6, 12], // torso sides
  [11, 12],         // hips
  [11, 13], [13, 15], // left leg
  [12, 14], [14, 16], // right leg
  [0, 5], [0, 6],   // neck to shoulders (approx)
];

export const KP_NAMES = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
];

// ── Hint sender ──────────────────────────────────────────────────────────────

let _lastHintSent = 0;
let _backendUrl = '';

export function setBackendUrl(url: string): void {
  _backendUrl = url;
}

async function sendCameraHint(personCount: number, confidence: number): Promise<void> {
  const now = Date.now();
  if (now - _lastHintSent < HINT_SEND_INTERVAL_MS) return;
  _lastHintSent = now;

  const base = _backendUrl || `http://${window.location.hostname}:8080`;
  try {
    await fetch(`${base}/api/v1/camera-hint`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ person_count: personCount, confidence, source: 'mediapipe' }),
    });
  } catch {
    // Backend may not be running; ignore silently
  }
}

// ── Keypoint blend ───────────────────────────────────────────────────────────

function blendKeypoint(a: FusionKeypoint, b: FusionKeypoint, alpha: number): FusionKeypoint {
  return {
    name: a.name,
    x: a.x * alpha + b.x * (1 - alpha),
    y: a.y * alpha + b.y * (1 - alpha),
    z: a.z * alpha + b.z * (1 - alpha),
    confidence: a.confidence * alpha + b.confidence * (1 - alpha),
  };
}

function blendPersons(cam: FusionPerson, wifi: FusionPerson, alpha: number): FusionPerson {
  const blended: FusionKeypoint[] = cam.keypoints.map((kp, i) => {
    const wkp = wifi.keypoints[i];
    if (!wkp) return kp;
    return blendKeypoint(kp, wkp, alpha);
  });
  return {
    id: cam.id,
    source: 'blend',
    confidence: cam.confidence * alpha + wifi.confidence * (1 - alpha),
    keypoints: blended,
  };
}

// ── Core fusion logic ─────────────────────────────────────────────────────────

function runFusion(): void {
  const camConf = cameraConfidence.value;
  const camPersons = cameraPersons.value;
  const wifPersons = wifiPersons.value;
  const camOn = cameraActive.value;

  let mode: 'camera' | 'wifi' | 'blend';
  let result: FusionPerson[];

  if (!camOn || camConf < CAMERA_FALLBACK_THRESHOLD) {
    // Camera off or very low confidence → pure WiFi
    mode = 'wifi';
    result = wifPersons.map(p => ({ ...p, source: 'wifi' as const }));

  } else if (camConf >= CAMERA_PRIMARY_THRESHOLD) {
    // Camera reliable → use camera as ground truth
    mode = 'camera';
    result = camPersons.map(p => ({ ...p, source: 'camera' as const }));

    // Send strong hint to backend
    void sendCameraHint(camPersons.length, camConf);

  } else {
    // Blend zone: alpha = how much to trust camera
    mode = 'blend';
    const alpha = (camConf - CAMERA_FALLBACK_THRESHOLD) /
                  (CAMERA_PRIMARY_THRESHOLD - CAMERA_FALLBACK_THRESHOLD);

    // Match camera persons to wifi persons by index
    const maxPersons = Math.max(camPersons.length, wifPersons.length);
    result = [];
    for (let i = 0; i < maxPersons; i++) {
      const cam = camPersons[i];
      const wif = wifPersons[i];
      if (cam && wif) {
        result.push(blendPersons(cam, wif, alpha));
      } else if (cam) {
        result.push({ ...cam, source: 'blend' });
      } else if (wif) {
        result.push({ ...wif, source: 'blend' });
      }
    }

    // Send blended hint to backend
    void sendCameraHint(result.length, camConf);
  }

  // Cap at 3 persons
  result = result.slice(0, 3);

  fusionMode.value = mode;
  fusedPersons.value = result;
}

// ── Auto-run fusion whenever inputs change ───────────────────────────────────

let _started = false;
export function startFusionEngine(): void {
  if (_started) return;
  _started = true;
  effect(runFusion);
}

// ── WiFi backend WebSocket person parser ─────────────────────────────────────

/** Parse persons from a backend SensingUpdate WebSocket message. */
export function parseWifiPersons(data: unknown): void {
  if (typeof data !== 'object' || data === null) return;
  const msg = data as Record<string, unknown>;

  // Parse estimated_persons count
  const count = typeof msg.estimated_persons === 'number' ? msg.estimated_persons : 0;
  wifiPersonCount.value = Math.min(count, 3);

  // Parse keypoint persons array if present
  const rawPersons = Array.isArray(msg.persons) ? msg.persons : [];
  const parsed: FusionPerson[] = rawPersons.slice(0, 3).map((p: unknown, i: number) => {
    const person = p as Record<string, unknown>;
    const kps: FusionKeypoint[] = Array.isArray(person.keypoints)
      ? (person.keypoints as unknown[]).slice(0, 17).map((kp: unknown) => {
          const k = kp as Record<string, unknown>;
          return {
            name: typeof k.name === 'string' ? k.name : KP_NAMES[i] ?? 'unknown',
            x: typeof k.x === 'number' ? k.x : 320,
            y: typeof k.y === 'number' ? k.y : 240,
            z: typeof k.z === 'number' ? k.z : 0,
            confidence: typeof k.confidence === 'number' ? k.confidence : 0.5,
          };
        })
      : generateFallbackKeypoints(i, count);

    return {
      id: typeof person.id === 'number' ? person.id : i + 1,
      source: 'wifi' as const,
      confidence: typeof person.confidence === 'number' ? person.confidence : 0.5,
      keypoints: kps,
    };
  });

  wifiPersons.value = parsed;
}

/** Generate synthetic keypoints when WiFi has count but no keypoint data. */
function generateFallbackKeypoints(personIdx: number, totalPersons: number): FusionKeypoint[] {
  const spacing = 640 / (totalPersons + 1);
  const baseX = spacing * (personIdx + 1);
  const offsets: [number, number][] = [
    [0, -80], [-8, -88], [8, -88], [-16, -82], [16, -82],
    [-30, -50], [30, -50], [-45, -15], [45, -15],
    [-50, 20], [50, 20], [-20, 20], [20, 20],
    [-22, 70], [22, 70], [-24, 120], [24, 120],
  ];
  return KP_NAMES.map((name, i) => ({
    name,
    x: baseX + (offsets[i]?.[0] ?? 0),
    y: 240 + (offsets[i]?.[1] ?? 0),
    z: 0,
    confidence: 0.4,
  }));
}
