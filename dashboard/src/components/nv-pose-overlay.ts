/**
 * nv-pose-overlay.ts — Stickman Skeleton Renderer
 *
 * Renders fused person detections as animated stick figures on an SVG canvas.
 * Color-coded by data source:
 *   🟢 oklch green  = camera (MediaPipe, high confidence)
 *   🔵 oklch cyan   = WiFi CSI (through-wall)
 *   🟡 oklch amber  = blend (mixed sources)
 *
 * Reads fusedPersons + fusionMode signals from the store.
 * Designed to be placed as an overlay on top of the main scene canvas.
 */
import { LitElement, html, css, svg } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { effect } from '@preact/signals-core';
import { fusedPersons, fusionMode, cameraActive, cameraConfidence, wifiPersonCount } from '../store/appStore';
import { BONE_PAIRS } from '../store/fusionEngine';

// Source colors
const COLOR: Record<string, string> = {
  camera: 'oklch(0.78 0.18 145)',  // green
  wifi:   'oklch(0.78 0.14 195)',  // cyan
  blend:  'oklch(0.78 0.16 70)',   // amber
};

@customElement('nv-pose-overlay')
export class NvPoseOverlay extends LitElement {
  /** Viewport width the keypoints are scaled to (matches camera 640×480) */
  @property({ type: Number }) viewW = 640;
  @property({ type: Number }) viewH = 480;

  @state() private _tick = 0;

  static styles = css`
    :host {
      display: block;
      position: absolute;
      inset: 0;
      pointer-events: none;
    }
    svg {
      width: 100%;
      height: 100%;
      overflow: visible;
    }
    .kp-circle {
      transition: cx 0.08s ease, cy 0.08s ease;
    }
    .bone-line {
      transition: x1 0.08s ease, y1 0.08s ease, x2 0.08s ease, y2 0.08s ease;
    }
    .person-label {
      font-family: var(--mono, monospace);
      font-size: 12px;
      font-weight: 600;
      paint-order: stroke fill;
    }
    .source-badge {
      font-family: var(--mono, monospace);
      font-size: 9px;
      opacity: 0.8;
    }
    .status-bar {
      position: absolute;
      bottom: 8px;
      left: 50%;
      transform: translateX(-50%);
      display: flex;
      gap: 6px;
      align-items: center;
      background: rgba(13,17,23,0.75);
      backdrop-filter: blur(6px);
      border: 1px solid var(--line, #333);
      border-radius: 999px;
      padding: 4px 12px;
      font-family: var(--mono, monospace);
      font-size: 10px;
      color: var(--ink-2, #aaa);
      pointer-events: none;
    }
    .dot {
      width: 7px; height: 7px;
      border-radius: 50%;
      flex-shrink: 0;
    }
  `;

  override connectedCallback(): void {
    super.connectedCallback();
    effect(() => {
      fusedPersons.value;
      fusionMode.value;
      cameraActive.value;
      cameraConfidence.value;
      wifiPersonCount.value;
      this._tick++;
      this.requestUpdate();
    });
  }

  /** Scale keypoint coords from 640×480 space to SVG viewBox */
  private _sx(x: number): number {
    return (x / this.viewW) * 1000;
  }
  private _sy(y: number): number {
    return (y / this.viewH) * 600;
  }

  private _renderPerson(person: any, idx: number) {
    const color = COLOR[person.source] ?? COLOR.wifi;
    const kps = person.keypoints;
    if (!kps.length) return '';

    // Head position (nose = kp 0)
    const head = kps[0] ?? kps[5];
    const headX = this._sx(head.x);
    const headY = this._sy(head.y);

    const personLabel = `P${person.id}`;
    const sourceLabel = person.source === 'camera' ? '🎥' : person.source === 'wifi' ? '📡' : '🔀';
    const confPct = Math.round(person.confidence * 100);

    return svg`
      <g class="person-group" data-person-id=${person.id}>

        <!-- Bone connections -->
        ${BONE_PAIRS.map(([a, b]) => {
          const kpA = kps[a];
          const kpB = kps[b];
          if (!kpA || !kpB) return '';
          const minConf = Math.min(kpA.confidence, kpB.confidence);
          if (minConf < 0.15) return '';
          return svg`
            <line class="bone-line"
              x1=${this._sx(kpA.x)} y1=${this._sy(kpA.y)}
              x2=${this._sx(kpB.x)} y2=${this._sy(kpB.y)}
              stroke=${color}
              stroke-width=${1.5 + minConf}
              stroke-opacity=${0.4 + minConf * 0.5}
              stroke-linecap="round"
            />
          `;
        })}

        <!-- Keypoint circles -->
        ${kps.map((kp: any, i: number) => {
          if (kp.confidence < 0.15) return '';
          const r = i < 5 ? 4 : 3; // head kps slightly bigger
          return svg`
            <circle class="kp-circle"
              cx=${this._sx(kp.x)} cy=${this._sy(kp.y)}
              r=${r}
              fill=${color}
              fill-opacity=${0.6 + kp.confidence * 0.4}
              stroke="rgba(0,0,0,0.4)"
              stroke-width="0.8"
            />
          `;
        })}

        <!-- Person label -->
        <text
          class="person-label"
          x=${headX} y=${headY - 18}
          text-anchor="middle"
          fill=${color}
          stroke="rgba(0,0,0,0.6)"
          stroke-width="3"
        >${personLabel}</text>
        <text
          class="source-badge"
          x=${headX} y=${headY - 6}
          text-anchor="middle"
          fill=${color}
        >${sourceLabel} ${confPct}%</text>
      </g>
    `;
  }

  override render() {
    const persons = fusedPersons.value;
    const mode = fusionMode.value;
    const camOn = cameraActive.value;
    const camConf = cameraConfidence.value;
    const wifiCount = wifiPersonCount.value;

    const modeColor = mode === 'camera' ? COLOR.camera : mode === 'blend' ? COLOR.blend : COLOR.wifi;
    const modeLabel = mode === 'camera' ? '🎥 Camera' : mode === 'blend' ? '🔀 Blend' : '📡 WiFi';

    return html`
      <svg viewBox="0 0 1000 600" preserveAspectRatio="xMidYMid meet" id="pose-overlay-svg">
        <defs>
          <filter id="pose-glow">
            <feGaussianBlur stdDeviation="2" result="b"/>
            <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        ${persons.map((p, i) => this._renderPerson(p, i))}
      </svg>

      <!-- Status bar -->
      <div class="status-bar" id="fusion-status-bar">
        <span class="dot" style="background:${modeColor}; box-shadow: 0 0 6px ${modeColor};"></span>
        <span style="color:${modeColor}; font-weight:600;">${modeLabel}</span>
        <span style="color:var(--ink-3);">·</span>
        <span>${persons.length} person${persons.length !== 1 ? 's' : ''}</span>
        ${camOn ? html`
          <span style="color:var(--ink-3);">·</span>
          <span>cam ${Math.round(camConf * 100)}%</span>
        ` : ''}
        ${wifiCount > 0 ? html`
          <span style="color:var(--ink-3);">·</span>
          <span>wifi ${wifiCount}</span>
        ` : ''}
      </div>
    `;
  }
}
