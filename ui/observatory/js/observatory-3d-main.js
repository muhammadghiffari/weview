/**
 * Observatory 3D — Realistic DensePose Main Orchestrator
 * Reuses existing modules: DemoDataGenerator, PoseSystem, ScenarioProps,
 * PostProcessing, NebulaBackground, HudController.
 * Adds: RealisticBody meshes + CVHybridManager fusion.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { DemoDataGenerator } from './demo-data.js';
import { NebulaBackground } from './nebula-background.js';
import { PostProcessing } from './post-processing.js';
import { PoseSystem } from './pose-system.js';
import { ScenarioProps } from './scenario-props.js';
import { HudController, DEFAULTS, SETTINGS_VERSION, SCENARIO_NAMES } from './hud-controller.js';
import { RealisticBody } from './realistic-body.js';
import { CVHybridManager } from './cv-hybrid-manager.js';

const MAX_BODIES = 4;

class Observatory3D {
  constructor() {
    this._canvas = document.getElementById('observatory-canvas');
    this.settings = { ...DEFAULTS, renderMode: 'realistic', bodyOpacity: 0.95, skeletonOverlay: 0.2, muscleDetail: 0.6 };
    try {
      const v = localStorage.getItem('weview-settings-version');
      if (v === SETTINGS_VERSION) {
        const s = localStorage.getItem('weview-observatory-settings');
        if (s) Object.assign(this.settings, JSON.parse(s));
      }
    } catch {}

    // Renderer
    this._renderer = new THREE.WebGLRenderer({ canvas: this._canvas, antialias: true, powerPreference: 'high-performance' });
    this._renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this._renderer.setSize(window.innerWidth, window.innerHeight);
    this._renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this._renderer.toneMappingExposure = this.settings.exposure;
    this._renderer.shadowMap.enabled = true;
    this._renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    // Scene
    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color(0x060810);
    this._scene.fog = new THREE.FogExp2(0x060810, 0.004);

    // Camera
    this._camera = new THREE.PerspectiveCamera(this.settings.fov, window.innerWidth / window.innerHeight, 0.1, 300);
    this._camera.position.set(5, 4, 7);
    this._camera.lookAt(0, 1, 0);

    // Controls
    this._controls = new OrbitControls(this._camera, this._canvas);
    this._controls.enableDamping = true;
    this._controls.dampingFactor = 0.08;
    this._controls.minDistance = 2;
    this._controls.maxDistance = 25;
    this._controls.maxPolarAngle = Math.PI * 0.88;
    this._controls.target.set(0, 1, 0);
    this._controls.update();

    this._clock = new THREE.Clock();

    // Data
    this._demoData = new DemoDataGenerator();
    this._demoData.setCycleDuration(this.settings.cycle || 30);
    if (this.settings.scenario && this.settings.scenario !== 'auto') this._demoData.setScenario(this.settings.scenario);
    this._currentData = null;

    // Build scene
    this._setupLighting();
    this._nebula = new NebulaBackground(this._scene);
    this._buildRoom();
    this._buildRouter();
    this._poseSystem = new PoseSystem();
    this._scenarioProps = new ScenarioProps(this._scene);
    this._buildSignalField();

    // Realistic bodies (replaces wireframe FigurePool)
    this._bodies = [];
    for (let i = 0; i < MAX_BODIES; i++) {
      this._bodies.push(new RealisticBody(this._scene, this.settings.renderMode || 'realistic'));
    }

    // Post-processing
    this._postProcessing = new PostProcessing(this._renderer, this._scene, this._camera);
    this._applyPostSettings();

    // HUD
    this._hud = new HudController(this);
    // Patch: HudController expects _figurePool — provide no-op
    this._figurePool = { applyColors: () => {}, update: () => {} };

    // CV Hybrid
    this._cvManager = new CVHybridManager(
      document.getElementById('webcam'),
      document.getElementById('cv-skeleton-canvas')
    );
    this._cvManager.onFused = (kps, vc, wc, fc) => this._onCVFusion(vc, wc, fc);
    this._initCVUI();

    // Per-person smoothed state trackers
    this._personStates = [];
    for (let i = 0; i < MAX_BODIES; i++) {
      this._personStates.push({
        smoothX: 0, smoothY: 0, smoothZ: 0,    // smoothed position
        smoothFacing: 0,                        // smoothed facing angle
        currentPose: 'standing',                // current pose label
        targetPose: 'standing',                 // target pose (from data)
        poseBlend: 1.0,                         // 0..1 blend toward target
        visible: false,                         // was visible last frame?
        fadeIn: 0,                              // 0..1 opacity fade
        lastMotionScore: 0,                     // smoothed motion score
        initialized: false,
      });
    }

    // State
    this._autopilot = false;
    this._autoAngle = 0;
    this._fpsFrames = 0;
    this._fpsTime = 0;
    this._fpsValue = 60;
    this._showFps = false;
    this._qualityLevel = 2;

    // WebSocket
    this._ws = null;
    this._liveData = null;

    // Input
    this._initKeyboard();
    // NOTE: We don't call _hud.initSettings() because observatory-3d.html
    // has a different settings layout. We init our own settings safely.
    this._initOwnSettings();
    this._hud.initQuickSelect();
    this._initExtraSettings();
    window.addEventListener('resize', () => this._onResize());

    this._animate();

    // Try to auto-connect to live backend after a short delay
    // (lets the page render first; falls back silently if server not running)
    setTimeout(() => this._autoDetectLive(), 1500);
  }

  // ---- Our own safe settings init (replaces HudController.initSettings) ----
  _initOwnSettings() {
    const overlay = document.getElementById('settings-overlay');
    const btn     = document.getElementById('settings-btn');
    const closeBtn= document.getElementById('settings-close');
    if (btn) btn.addEventListener('click', () => this._hud.toggleSettings());
    if (closeBtn) closeBtn.addEventListener('click', () => this._hud.toggleSettings());
    if (overlay) overlay.addEventListener('click', (e) => { if (e.target === overlay) this._hud.toggleSettings(); });

    // Tab switching
    document.querySelectorAll('.stab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.stab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.stab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(`stab-${tab.dataset.stab}`).classList.add('active');
      });
    });

    const s = this.settings;
    const self = this;

    // Safe range binder (skips missing elements)
    const bindRange = (id, key, applyFn) => {
      const el = document.getElementById(id);
      const valEl = document.getElementById(`${id}-val`);
      if (!el) return;
      el.value = s[key] != null ? s[key] : el.value;
      if (valEl) valEl.textContent = s[key] != null ? s[key] : el.value;
      el.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        s[key] = v;
        if (valEl) valEl.textContent = v;
        if (applyFn) applyFn(v);
        self._hud.saveSettings();
      });
    };

    // Rendering tab
    bindRange('opt-bloom', 'bloom', v => { this._postProcessing._bloomPass.strength = v; });
    bindRange('opt-exposure', 'exposure', v => { this._renderer.toneMappingExposure = v; });
    bindRange('opt-vignette', 'vignette', v => { this._postProcessing._vignettePass.uniforms.uVignetteStrength.value = v; });

    // Scene tab
    bindRange('opt-ambient', 'ambient', v => { this._ambient.intensity = v * 4.0; });
    bindRange('opt-reflect', 'reflect', v => {
      this._floorMat.roughness = 1.0 - v * 0.7;
      this._floorMat.metalness = v * 0.5;
    });
    bindRange('opt-fov', 'fov', v => {
      this._camera.fov = v;
      this._camera.updateProjectionMatrix();
    });

    // Checkboxes
    const roomEl = document.getElementById('opt-room');
    if (roomEl) {
      roomEl.checked = s.room !== false;
      roomEl.addEventListener('change', e => {
        s.room = e.target.checked;
        this._roomWire.visible = e.target.checked;
      });
    }
    const fieldEl = document.getElementById('opt-signal-field');
    if (fieldEl) {
      fieldEl.checked = true;
      fieldEl.addEventListener('change', e => {
        this._fieldPoints.visible = e.target.checked;
      });
    }

    // Scenario select
    const scenarioSel = document.getElementById('opt-scenario');
    if (scenarioSel) {
      scenarioSel.value = s.scenario || 'auto';
      scenarioSel.addEventListener('change', e => {
        s.scenario = e.target.value;
        this._demoData.setScenario(e.target.value);
      });
    }

    // Data source
    const dsSel = document.getElementById('opt-data-source');
    if (dsSel) {
      dsSel.value = s.dataSource || 'demo';
      dsSel.addEventListener('change', e => {
        s.dataSource = e.target.value;
        const wsRow = document.getElementById('ws-url-row');
        if (wsRow) wsRow.style.display = e.target.value === 'ws' ? 'flex' : 'none';
        if (e.target.value === 'ws') {
          const url = s.wsUrl || `ws://${location.hostname}:8765/ws/sensing`;
          this._connectWS(url);
        } else {
          this._disconnectWS();
          this._hud.updateSourceBadge('demo', null);
        }
      });
    }

    // WS URL input
    const wsInput = document.getElementById('opt-ws-url');
    if (wsInput) {
      wsInput.value = s.wsUrl || `ws://${location.hostname}:8765/ws/sensing`;
      const doConnect = () => {
        s.wsUrl = wsInput.value.trim();
        if (s.wsUrl) this._connectWS(s.wsUrl);
      };
      wsInput.addEventListener('change', doConnect);
      wsInput.addEventListener('keydown', e => { if (e.key === 'Enter') doConnect(); });
    }

    // Buttons
    const resetCamBtn = document.getElementById('btn-reset-camera');
    if (resetCamBtn) resetCamBtn.addEventListener('click', () => {
      this._camera.position.set(5, 4, 7);
      this._controls.target.set(0, 1, 0);
      this._controls.update();
    });
    const resetBtn = document.getElementById('btn-reset-settings');
    if (resetBtn) resetBtn.addEventListener('click', () => {
      Object.assign(s, DEFAULTS);
      this._applyPostSettings();
      this._renderer.toneMappingExposure = s.exposure;
      this._camera.fov = s.fov;
      this._camera.updateProjectionMatrix();
    });

    this._grid.visible = s.grid !== false;
    this._roomWire.visible = s.room !== false;
  }

  // ---- Lighting ----
  _setupLighting() {
    this._ambient = new THREE.AmbientLight(0xddeeff, this.settings.ambient * 4.0);
    this._scene.add(this._ambient);
    const hemi = new THREE.HemisphereLight(0x8899bb, 0x203040, 1.5);
    this._scene.add(hemi);
    const key = new THREE.DirectionalLight(0xfff0dd, 1.8);
    key.position.set(4, 8, 3);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 0.5; key.shadow.camera.far = 25;
    key.shadow.camera.left = -10; key.shadow.camera.right = 10;
    key.shadow.camera.top = 10; key.shadow.camera.bottom = -10;
    this._scene.add(key);
    const fill = new THREE.DirectionalLight(0x99aacc, 0.8);
    fill.position.set(-5, 5, -3);
    this._scene.add(fill);
    const rim = new THREE.DirectionalLight(0x7799bb, 0.5);
    rim.position.set(0, 7, -6);
    this._scene.add(rim);
    const overhead = new THREE.PointLight(0xaabbcc, 1.5, 20, 1);
    overhead.position.set(0, 3.8, 0);
    this._scene.add(overhead);
  }

  // ---- Room ----
  _buildRoom() {
    this._grid = new THREE.GridHelper(12, 24, 0x1a4830, 0x0c2818);
    this._grid.material.opacity = 0.4;
    this._grid.material.transparent = true;
    this._scene.add(this._grid);
    const boxGeo = new THREE.BoxGeometry(12, 4, 10);
    const edges = new THREE.EdgesGeometry(boxGeo);
    this._roomWire = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x0a6b3a, opacity: 0.25, transparent: true }));
    this._roomWire.position.y = 2;
    this._scene.add(this._roomWire);
    const floorGeo = new THREE.PlaneGeometry(12, 10);
    this._floorMat = new THREE.MeshStandardMaterial({ color: 0x141c16, roughness: 1 - this.settings.reflect * 0.7, metalness: this.settings.reflect * 0.5, emissive: 0x030605, emissiveIntensity: 0.06 });
    const floor = new THREE.Mesh(floorGeo, this._floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this._scene.add(floor);
    // Walls: subtle back wall
    const wallMat = new THREE.MeshStandardMaterial({ color: 0x141818, roughness: 0.85, emissive: 0x040606, emissiveIntensity: 0.04, side: THREE.BackSide });
    const backWall = new THREE.Mesh(new THREE.PlaneGeometry(12, 4), wallMat);
    backWall.position.set(0, 2, -5);
    backWall.receiveShadow = true;
    this._scene.add(backWall);
    // Table
    const tableMat = new THREE.MeshStandardMaterial({ color: 0x6b5840, roughness: 0.55, emissive: 0x1a1408, emissiveIntensity: 0.25 });
    const table = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.6, 0.5), tableMat);
    table.position.set(-4, 0.3, -3);
    table.castShadow = true;
    this._scene.add(table);
  }

  _buildRouter() {
    this._routerGroup = new THREE.Group();
    this._routerGroup.position.set(-4, 0.92, -3);
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x505060, roughness: 0.2, metalness: 0.7 });
    this._routerGroup.add(new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.12, 0.35), bodyMat));
    for (let i = -1; i <= 1; i++) {
      const ant = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.35), bodyMat);
      ant.position.set(i * 0.2, 0.24, 0);
      ant.rotation.z = i * 0.15;
      this._routerGroup.add(ant);
    }
    this._routerLed = new THREE.Mesh(new THREE.SphereGeometry(0.025), new THREE.MeshBasicMaterial({ color: 0x00d878 }));
    this._routerLed.position.set(0.22, 0.07, 0.18);
    this._routerGroup.add(this._routerLed);
    this._routerLight = new THREE.PointLight(0x2090ff, 1.2, 8);
    this._routerLight.position.set(0, 0.3, 0);
    this._routerGroup.add(this._routerLight);
    this._scene.add(this._routerGroup);
  }

  _buildSignalField() {
    const gs = 20, count = gs * gs;
    const positions = new Float32Array(count * 3);
    this._fieldColors = new Float32Array(count * 3);
    this._fieldSizes = new Float32Array(count);
    for (let iz = 0; iz < gs; iz++) for (let ix = 0; ix < gs; ix++) {
      const idx = iz * gs + ix;
      positions[idx * 3] = (ix - gs / 2) * 0.6;
      positions[idx * 3 + 1] = 0.02;
      positions[idx * 3 + 2] = (iz - gs / 2) * 0.5;
      this._fieldSizes[idx] = 8;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(this._fieldColors, 3));
    geo.setAttribute('size', new THREE.BufferAttribute(this._fieldSizes, 1));
    this._fieldMat = new THREE.PointsMaterial({ size: 0.35, vertexColors: true, transparent: true, opacity: this.settings.field, blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true });
    this._fieldPoints = new THREE.Points(geo, this._fieldMat);
    this._scene.add(this._fieldPoints);
  }

  // ---- Apply settings ----
  _applyPostSettings() {
    const pp = this._postProcessing;
    pp._bloomPass.strength = this.settings.bloom;
    pp._bloomPass.radius = this.settings.bloomRadius;
    pp._bloomPass.threshold = this.settings.bloomThresh;
    pp._vignettePass.uniforms.uVignetteStrength.value = this.settings.vignette;
    pp._vignettePass.uniforms.uGrainStrength.value = this.settings.grain;
    pp._vignettePass.uniforms.uChromaticStrength.value = this.settings.chromatic;
  }
  _applyColors() {} // No wireframe colors needed

  // ---- CV UI ----
  _initCVUI() {
    const startBtn = document.getElementById('cv-start-camera');
    const toggleBtn = document.getElementById('cv-toggle-btn');
    const inset = document.getElementById('cv-inset');
    const noCamera = document.getElementById('cv-no-camera');
    if (startBtn) startBtn.addEventListener('click', async () => {
      const ok = await this._cvManager.startCamera();
      if (ok && noCamera) noCamera.style.display = 'none';
    });
    if (toggleBtn) toggleBtn.addEventListener('click', () => inset.classList.toggle('collapsed'));
    document.querySelectorAll('.cv-mode-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.cv-mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this._cvManager.setMode(btn.dataset.mode);
      });
    });
  }
  _onCVFusion(vc, wc, fc) {
    const pct = v => Math.round(v * 100) + '%';
    const set = (id, w, t) => { const e = document.getElementById(id); if (e) e.style.width = pct(w); const v = document.getElementById(id.replace('bar', 'val')); if (v) v.textContent = pct(w); };
    set('fusion-video-bar', vc);
    set('fusion-wifi-bar', wc);
    set('fusion-fused-bar', fc);
  }

  // ---- Extra settings for 3D ----
  _initExtraSettings() {
    const rmSel = document.getElementById('opt-render-mode');
    if (rmSel) rmSel.addEventListener('change', e => {
      this.settings.renderMode = e.target.value;
      for (const b of this._bodies) b.setRenderMode(e.target.value);
      document.getElementById('render-mode-label').textContent = e.target.value.toUpperCase();
    });
    this._bindRange3D('opt-body-opacity', 'bodyOpacity', v => { for (const b of this._bodies) b.setBodyOpacity(v); });
    this._bindRange3D('opt-skeleton-overlay', 'skeletonOverlay', v => { for (const b of this._bodies) b.setSkeletonOverlay(v); });
    this._bindRange3D('opt-ssao', 'ssao');
    this._bindRange3D('opt-muscle-detail', 'muscleDetail');
  }
  _bindRange3D(id, key, fn) {
    const el = document.getElementById(id);
    const val = document.getElementById(id + '-val');
    if (!el) return;
    el.value = this.settings[key] || el.value;
    if (val) val.textContent = el.value;
    el.addEventListener('input', e => {
      const v = parseFloat(e.target.value);
      this.settings[key] = v;
      if (val) val.textContent = v;
      if (fn) fn(v);
    });
  }

  // ---- Keyboard ----
  _initKeyboard() {
    window.addEventListener('keydown', e => {
      if (this._hud.settingsOpen) return;
      switch (e.key.toLowerCase()) {
        case 'a': this._autopilot = !this._autopilot; this._controls.enabled = !this._autopilot; break;
        case 'd': this._demoData.cycleScenario(); break;
        case 'f': this._showFps = !this._showFps; document.getElementById('fps-counter').style.display = this._showFps ? 'block' : 'none'; break;
        case 's': this._hud.toggleSettings(); break;
        case 'r': {
          const modes = ['realistic', 'densepose', 'xray', 'hybrid'];
          const cur = modes.indexOf(this.settings.renderMode);
          this.settings.renderMode = modes[(cur + 1) % modes.length];
          for (const b of this._bodies) b.setRenderMode(this.settings.renderMode);
          document.getElementById('render-mode-label').textContent = this.settings.renderMode.toUpperCase();
          const sel = document.getElementById('opt-render-mode');
          if (sel) sel.value = this.settings.renderMode;
          break;
        }
        case 'v': document.getElementById('cv-inset')?.classList.toggle('collapsed'); break;
        case ' ': e.preventDefault(); this._demoData.paused = !this._demoData.paused; break;
      }
    });
  }

  // ---- WebSocket Live Data ----

  _autoDetectLive() {
    const host = window.location.hostname || 'localhost';
    const wsUrl = `ws://${host}:8765/ws/sensing`;
    console.log('[Observatory3D] Auto-connecting to', wsUrl);
    this.settings.wsUrl = wsUrl;
    this._connectWS(wsUrl);
    const wsInp = document.getElementById('opt-ws-url');
    if (wsInp) wsInp.value = wsUrl;
  }

  _connectWS(url) {
    this._disconnectWS();
    if (!url || url.trim() === '') return;
    const wsUrl = url.startsWith('ws') ? url : url.replace(/^http/, 'ws') + '/ws/sensing';
    console.log('[Observatory3D] Connecting WebSocket →', wsUrl);
    try {
      this._ws = new WebSocket(wsUrl);
      this._ws.onopen = () => {
        console.log('[Observatory3D] ✓ WebSocket LIVE');
        this.settings.dataSource = 'ws';
        this._hud.updateSourceBadge('ws', this._ws);
        const dsSel = document.getElementById('opt-data-source');
        if (dsSel) dsSel.value = 'ws';
        // Reset all person state so stale demo positions don't bleed in
        for (const st of this._personStates) st.initialized = false;
      };
      this._ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          const msgType = msg.msg_type || msg.type || '';
          if (msgType !== 'sensing_update') return;
          this._liveData = this._transformSensingUpdate(msg);
        } catch (e) {
          console.warn('[Observatory3D] WS parse error', e);
        }
      };
      this._ws.onclose = () => {
        console.log('[Observatory3D] WebSocket closed → demo mode');
        this._ws = null;
        this._liveData = null;
        this.settings.dataSource = 'demo';
        this._hud.updateSourceBadge('demo', null);
        const dsSel = document.getElementById('opt-data-source');
        if (dsSel) dsSel.value = 'demo';
      };
      this._ws.onerror = (e) => {
        console.warn('[Observatory3D] WebSocket error', e.message || e);
      };
    } catch(e) {
      console.warn('[Observatory3D] Failed to open WebSocket', e);
    }
  }

  _disconnectWS() {
    if (this._ws) { this._ws.close(); this._ws = null; }
    this._liveData = null;
  }

  /**
   * Transform backend sensing_update → observatory data format.
   *
   * WiFi sensing provides: motion_level, motion_band_power, breathing_band_power,
   * mean_rssi, variance, dominant_freq_hz — but NO body keypoints.
   *
   * We infer realistic figure behavior from signal dynamics:
   *   - Position shifts based on RSSI/variance changes over time
   *   - Pose mapped from motion_level + motion_band_power intensity
   *   - Breathing driven by breathing_band_power
   *   - Facing direction inferred from position delta
   */

  // Signal history for position inference (initialized once)
  _initSignalHistory() {
    if (this._sigHist) return;
    this._sigHist = {
      rssiHistory: [],        // last N mean_rssi values
      varianceHistory: [],    // last N variance values
      motionHistory: [],      // last N motion_band_power values
      posX: 0,                // inferred position X
      posZ: 0,                // inferred position Z
      velX: 0,                // velocity X
      velZ: 0,                // velocity Z
      facing: 0,              // inferred facing
      wanderPhase: 0,         // phase for idle micro-wander
      lastTimestamp: 0,
      frameCount: 0,
    };
  }

  _transformSensingUpdate(msg) {
    this._initSignalHistory();
    const sh = this._sigHist;

    const cls   = msg.classification || {};
    const feat  = msg.features || {};
    const nodes = msg.nodes || [];

    const isPresent   = cls.presence === true;
    const motionLevel = cls.motion_level || 'absent';
    const confidence  = cls.confidence  || 0;
    const now = msg.timestamp || Date.now() / 1000;
    const dt  = sh.lastTimestamp > 0 ? Math.min(now - sh.lastTimestamp, 2) : 0.5;
    sh.lastTimestamp = now;
    sh.frameCount++;

    // ---- Track signal history (keep last 20 samples) ----
    const rssi     = feat.mean_rssi || -50;
    const variance = feat.variance  || 0;
    const motionBP = feat.motion_band_power || 0;

    sh.rssiHistory.push(rssi);
    sh.varianceHistory.push(variance);
    sh.motionHistory.push(motionBP);
    if (sh.rssiHistory.length > 20) sh.rssiHistory.shift();
    if (sh.varianceHistory.length > 20) sh.varianceHistory.shift();
    if (sh.motionHistory.length > 20) sh.motionHistory.shift();

    // ---- Infer position from signal dynamics ----
    // RSSI change → person moved toward/away from router
    // Variance spike → person is actively moving
    const rssiDelta = sh.rssiHistory.length > 1
      ? rssi - sh.rssiHistory[sh.rssiHistory.length - 2]
      : 0;
    const varMean = sh.varianceHistory.reduce((a, b) => a + b, 0) / sh.varianceHistory.length;
    const motionAvg = sh.motionHistory.reduce((a, b) => a + b, 0) / sh.motionHistory.length;

    // Movement speed proportional to motion_band_power
    const speed = Math.min(0.8, motionBP * 4);

    if (isPresent && motionLevel === 'present_moving') {
      // Walking: use RSSI gradient for radial direction, variance for lateral
      const radialPush = rssiDelta * 0.15;   // RSSI increase = closer to router
      const lateralPush = (variance - varMean) * 0.08;

      sh.velX += (radialPush + lateralPush * Math.cos(sh.wanderPhase)) * dt;
      sh.velZ += (lateralPush * Math.sin(sh.wanderPhase) - radialPush * 0.3) * dt;

      // Add purposeful wander (smooth Lissajous-like path)
      sh.wanderPhase += dt * (0.3 + motionBP * 0.5);
      sh.velX += Math.cos(sh.wanderPhase * 0.7) * speed * 0.04 * dt;
      sh.velZ += Math.sin(sh.wanderPhase * 1.1) * speed * 0.04 * dt;

    } else if (isPresent) {
      // Standing still: only micro-weight-shifts
      sh.wanderPhase += dt * 0.1;
      sh.velX += Math.sin(sh.wanderPhase * 2.3) * 0.001 * dt;
      sh.velZ += Math.cos(sh.wanderPhase * 1.7) * 0.001 * dt;
    }

    // Apply velocity with damping
    const damping = isPresent && motionLevel === 'present_moving' ? 0.92 : 0.97;
    sh.velX *= damping;
    sh.velZ *= damping;
    sh.posX += sh.velX;
    sh.posZ += sh.velZ;

    // Clamp to room bounds (-4..4, -3..3)
    sh.posX = Math.max(-4, Math.min(4, sh.posX));
    sh.posZ = Math.max(-3, Math.min(3, sh.posZ));

    // ---- Facing from velocity direction ----
    const speedMag = Math.sqrt(sh.velX * sh.velX + sh.velZ * sh.velZ);
    if (speedMag > 0.003) {
      sh.facing = Math.atan2(sh.velX, sh.velZ);
    }

    // ---- Pose from motion level + intensity ----
    let pose = 'standing';
    let motionScore = 10;
    if (!isPresent) {
      pose = 'standing';
      motionScore = 0;
    } else if (motionLevel === 'present_moving') {
      motionScore = Math.min(100, motionBP * 300);
      pose = motionScore > 60 ? 'walking' : 'walking'; // always walk when moving
    } else {
      // present_still: subtle idle
      motionScore = Math.min(20, motionBP * 80);
      pose = 'standing';
    }

    // ---- Vital signs from spectral features ----
    const breathFreqHz = feat.dominant_freq_hz || 0;
    const breathBpm = (breathFreqHz > 0.1 && breathFreqHz < 0.8)
      ? breathFreqHz * 60
      : (feat.breathing_band_power > 0.02 ? 15 : 0);
    const breathConf = Math.min(1, (feat.breathing_band_power || 0) * 10);
    const heartRateBpm = breathBpm > 0 ? 65 + (feat.spectral_power || 0) * 20 : 0;

    // ---- Build persons ----
    const persons = [];
    if (isPresent) {
      persons.push({
        id: 'live_0',
        position: [sh.posX, 0, sh.posZ],
        pose,
        motion_score: motionScore,
        facing: sh.facing,
        confidence,
        keypoints: [],
      });
    }

    return {
      type: 'sensing_update',
      timestamp: now,
      source: msg.source || 'live',
      classification: {
        presence: isPresent,
        motion_level: motionLevel,
        confidence,
        fall_detected: false,
      },
      features: {
        mean_rssi:            rssi,
        variance:             variance,
        motion_band_power:    motionBP,
        breathing_band_power: feat.breathing_band_power || 0,
        dominant_freq_hz:     breathFreqHz,
        spectral_power:       feat.spectral_power || 0,
      },
      vital_signs: {
        breathing_rate_bpm:    breathBpm,
        heart_rate_bpm:        heartRateBpm,
        breathing_confidence:  breathConf,
        heart_rate_confidence: 0.3,
      },
      signal_field: msg.signal_field || null,
      nodes,
      persons,
      estimated_persons: isPresent ? 1 : 0,
    };
  }

  // ---- Animation Loop ----
  _animate() {
    requestAnimationFrame(() => this._animate());
    const dt = Math.min(this._clock.getDelta(), 0.1);
    const elapsed = this._clock.getElapsedTime();

    // Data source: prefer live WS, fall back to demo
    if (this.settings.dataSource === 'ws' && this._liveData) {
      this._currentData = this._liveData;
      // Still tick demo (keeps HUD scenario label happy) but don't advance time
      this._demoData.update(0);
    } else {
      this._currentData = this._demoData.update(dt);
    }
    const data = this._currentData;

    // Updates
    this._nebula.update(dt, elapsed);
    this._scenarioProps.update(data, this._demoData.currentScenario);
    this._updateBodies(data, elapsed);
    this._updateSignalField(data);
    this._hud.updateHUD(data, this._demoData);
    this._hud.updateSparkline(data);
    this._updateDensePoseCanvas(data, elapsed);

    // Router LED pulses faster when live data arriving
    const ledFreq = this.settings.dataSource === 'ws' ? 12 : 8;
    this._routerLed.material.opacity = 0.5 + 0.5 * Math.sin(elapsed * ledFreq);
    this._routerLight.intensity = 0.3 + 0.2 * Math.sin(elapsed * 3);

    // Autopilot
    if (this._autopilot) {
      this._autoAngle += dt * this.settings.orbitSpeed;
      const r = 9;
      this._camera.position.set(Math.sin(this._autoAngle) * r, 4 + Math.sin(this._autoAngle * 0.5), Math.cos(this._autoAngle) * r);
      this._controls.target.set(0, 1, 0);
      this._controls.update();
    }
    this._controls.update();
    this._postProcessing.update(elapsed);
    this._postProcessing.render();

    this._updateFPS(dt);
  }

  // ---- Body Updates (with heavy smoothing for realistic motion) ----
  _updateBodies(data, elapsed) {
    const persons = data?.persons || [];
    const vs = data?.vital_signs || {};
    const isPresent = data?.classification?.presence || false;
    const breathBpm = vs.breathing_rate_bpm || 0;
    const breathPulse = breathBpm > 0 ? Math.sin(elapsed * Math.PI * 2 * (breathBpm / 60)) * 0.012 : 0;

    for (let i = 0; i < this._bodies.length; i++) {
      const st = this._personStates[i];

      if (i < persons.length && isPresent) {
        const p = persons[i];
        const rawX = p.position?.[0] || 0;
        const rawY = p.position?.[1] || 0;
        const rawZ = p.position?.[2] || 0;
        const rawFacing = p.facing || 0;
        const rawPose = p.pose || 'standing';
        const rawMotion = p.motion_score || 0;

        // ---- Position smoothing ----
        // Use very heavy lerp (0.03) for natural, purposeful movement
        // Higher values for first frame / teleport detection
        if (!st.initialized) {
          st.smoothX = rawX;
          st.smoothY = rawY;
          st.smoothZ = rawZ;
          st.smoothFacing = rawFacing;
          st.currentPose = rawPose;
          st.targetPose = rawPose;
          st.lastMotionScore = rawMotion;
          st.initialized = true;
          st.fadeIn = 0;
        }

        // Adaptive smoothing: faster when motion_score is high, slower when still
        const motionFactor = Math.min(1, rawMotion / 100); // 0..1
        const posLerp = 0.02 + motionFactor * 0.04; // 0.02 (still) .. 0.06 (fast)

        // Detect teleport (position jump > 2m) → snap instead of lerp
        const dx = rawX - st.smoothX;
        const dz = rawZ - st.smoothZ;
        const jumpDist = Math.sqrt(dx*dx + dz*dz);
        if (jumpDist > 2.0) {
          st.smoothX = rawX;
          st.smoothZ = rawZ;
        } else {
          st.smoothX += (rawX - st.smoothX) * posLerp;
          st.smoothZ += (rawZ - st.smoothZ) * posLerp;
        }
        st.smoothY += (rawY - st.smoothY) * posLerp;

        // ---- Facing smoothing (wrap-safe angle lerp) ----
        let facingDiff = rawFacing - st.smoothFacing;
        // Normalize to [-PI, PI]
        while (facingDiff > Math.PI) facingDiff -= Math.PI * 2;
        while (facingDiff < -Math.PI) facingDiff += Math.PI * 2;
        st.smoothFacing += facingDiff * 0.04;

        // ---- Motion score smoothing ----
        st.lastMotionScore += (rawMotion - st.lastMotionScore) * 0.05;

        // ---- Pose transition ----
        if (rawPose !== st.targetPose) {
          st.targetPose = rawPose;
          st.poseBlend = 0;
        }
        st.poseBlend = Math.min(1, st.poseBlend + 0.025); // ~40 frames transition
        if (st.poseBlend >= 0.5) {
          st.currentPose = st.targetPose;
        }

        // ---- Fade in ----
        if (!st.visible) st.fadeIn = 0;
        st.fadeIn = Math.min(1, st.fadeIn + 0.03);
        st.visible = true;

        // Build smoothed person object for PoseSystem
        const smoothPerson = {
          ...p,
          position: [st.smoothX, st.smoothY, st.smoothZ],
          facing: st.smoothFacing,
          pose: st.currentPose,
          motion_score: st.lastMotionScore,
        };

        const kps = this._poseSystem.generateKeypoints(smoothPerson, elapsed, breathPulse);
        this._bodies[i].show(true);
        // Keypoint lerp: heavier smoothing (0.12) for realistic feel
        this._bodies[i].applyKeypoints(kps, 0.12);

        // Feed WiFi keypoints to CV manager for fusion
        if (i === 0) {
          this._cvManager.setWifiKeypoints(kps, data?.classification?.confidence || 0.5);
        }
      } else {
        // Fade out
        if (st.visible) {
          st.fadeIn = Math.max(0, st.fadeIn - 0.02);
          if (st.fadeIn <= 0) {
            st.visible = false;
            st.initialized = false;
            this._bodies[i].show(false);
          }
        } else {
          this._bodies[i].show(false);
        }
      }
    }
  }

  _updateDensePoseCanvas(data, elapsed) {
    const dpCanvas = document.getElementById('densepose-canvas');
    if (!dpCanvas) return;
    const persons = data?.persons || [];
    if (persons.length > 0 && data?.classification?.presence) {
      // Use smoothed state for DensePose canvas too
      const st = this._personStates[0];
      const smoothPerson = {
        ...persons[0],
        position: st.initialized ? [st.smoothX, st.smoothY, st.smoothZ] : persons[0].position,
        facing: st.initialized ? st.smoothFacing : persons[0].facing,
        pose: st.initialized ? st.currentPose : persons[0].pose,
      };
      const kps = this._poseSystem.generateKeypoints(smoothPerson, elapsed, 0);
      this._cvManager.renderDensePoseCanvas(dpCanvas, kps);
    }
  }

  _updateSignalField(data) {
    const field = data?.signal_field?.values;
    if (!field) return;
    const count = Math.min(field.length, 400);
    for (let i = 0; i < count; i++) {
      const v = field[i] || 0;
      let r, g, b;
      if (v < 0.3) { r = 0; g = v * 1.5; b = v * 0.3; }
      else if (v < 0.6) { const t = (v - 0.3) / 0.3; r = t * 0.3; g = 0.45 + t * 0.4; b = 0.09 - t * 0.05; }
      else { const t = (v - 0.6) / 0.4; r = 0.3 + t * 0.7; g = 0.85 - t * 0.2; b = 0.04; }
      this._fieldColors[i * 3] = r;
      this._fieldColors[i * 3 + 1] = g;
      this._fieldColors[i * 3 + 2] = b;
      this._fieldSizes[i] = 5 + v * 15;
    }
    this._fieldPoints.geometry.attributes.color.needsUpdate = true;
    this._fieldPoints.geometry.attributes.size.needsUpdate = true;
  }

  _updateFPS(dt) {
    this._fpsFrames++;
    this._fpsTime += dt;
    if (this._fpsTime >= 1) {
      this._fpsValue = Math.round(this._fpsFrames / this._fpsTime);
      this._fpsFrames = 0;
      this._fpsTime = 0;
      if (this._showFps) document.getElementById('fps-counter').textContent = `${this._fpsValue} FPS`;
    }
  }

  _onResize() {
    const w = window.innerWidth, h = window.innerHeight;
    this._camera.aspect = w / h;
    this._camera.updateProjectionMatrix();
    this._renderer.setSize(w, h);
    this._postProcessing.resize(w, h);
  }
}

new Observatory3D();
