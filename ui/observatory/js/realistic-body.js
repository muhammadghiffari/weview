/**
 * RealisticBody — Procedural anatomical 3D human figure
 *
 * Builds a multi-segment body using primitives that approximate real anatomy:
 * head, neck, torso (upper/lower), pelvis, arms (upper/forearm/hand),
 * legs (thigh/shin/foot). Each segment has DensePose-style UV coloring.
 *
 * Render modes:
 *   'realistic'  — skin-toned PBR mesh with subsurface scatter approx
 *   'densepose'  — colored by body part (24 UV segments)
 *   'xray'       — semi-transparent wireframe + skeleton glow
 *   'hybrid'     — mesh + overlaid skeleton bones
 */
import * as THREE from 'three';

// DensePose 24-part color palette
export const DP_COLORS = {
  head:         0xffaa44,
  neck:         0xffbb55,
  torso_front:  0xff4444,
  torso_back:   0xdd3333,
  upper_arm_l:  0x44aaff,
  upper_arm_r:  0x2288dd,
  lower_arm_l:  0x66ccff,
  lower_arm_r:  0x44aadd,
  hand_l:       0x88ddff,
  hand_r:       0x66bbee,
  thigh_l:      0x44ff88,
  thigh_r:      0x22dd66,
  shin_l:       0x66ffaa,
  shin_r:       0x44ee88,
  foot_l:       0x88ffcc,
  foot_r:       0x66eeaa,
  pelvis:       0xff6644,
};

// Skin tone (subsurface approximation)
const SKIN_COLOR   = 0xd4956a;
const SKIN_EMISSIVE= 0x3a1a08;
const CLOTH_COLOR  = 0x2a3a5a;
const CLOTH_EMISSIVE=0x080c18;

function makeMat(mode, dpColor) {
  if (mode === 'densepose') {
    return new THREE.MeshStandardMaterial({
      color: dpColor, emissive: dpColor,
      emissiveIntensity: 0.12,
      roughness: 0.65, metalness: 0.0,
      transparent: false,
    });
  }
  if (mode === 'xray') {
    return new THREE.MeshStandardMaterial({
      color: 0x00d878, emissive: 0x00d878,
      emissiveIntensity: 0.4,
      transparent: true, opacity: 0.18,
      wireframe: false,
      roughness: 0.3, metalness: 0.0,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
  }
  // realistic / hybrid
  const isTorso = dpColor === DP_COLORS.torso_front || dpColor === DP_COLORS.torso_back || dpColor === DP_COLORS.pelvis;
  const color   = isTorso ? CLOTH_COLOR : SKIN_COLOR;
  const emissive= isTorso ? CLOTH_EMISSIVE : SKIN_EMISSIVE;
  return new THREE.MeshStandardMaterial({
    color, emissive, emissiveIntensity: 0.08,
    roughness: isTorso ? 0.8 : 0.55,
    metalness: 0.0,
    transparent: mode === 'hybrid',
    opacity: mode === 'hybrid' ? 0.9 : 1.0,
  });
}

// Tapered cylinder helper: points from p0 to p1 with given radii
function taperedCylinder(p0, p1, r0, r1, segs=8) {
  const dir = new THREE.Vector3().subVectors(p1, p0);
  const len = dir.length();
  const geo = new THREE.CylinderGeometry(r1, r0, len, segs, 1);
  const mesh = new THREE.Mesh(geo);
  const mid  = new THREE.Vector3().addVectors(p0, p1).multiplyScalar(0.5);
  mesh.position.copy(mid);
  mesh.quaternion.setFromUnitVectors(
    new THREE.Vector3(0,1,0),
    dir.normalize()
  );
  return mesh;
}

export class RealisticBody {
  constructor(scene, mode = 'realistic') {
    this._scene = scene;
    this._mode  = mode;
    this._group = new THREE.Group();
    this._group.visible = false;
    scene.add(this._group);

    this._segments = []; // { mesh, dpColor, name }
    this._bones    = []; // thin cylinders for skeleton overlay
    this._joints   = []; // spheres at key joints
    this._shadowMesh = null;

    this._kps = null;       // latest keypoints [17][3]
    this._prevKps = null;
    this._opacity  = 1.0;
    this._skeletonOpacity = 0.0;

    // Build skeleton overlay (always built, visibility controlled)
    this._skeletonGroup = new THREE.Group();
    this._group.add(this._skeletonGroup);

    this._build();
  }

  get group() { return this._group; }

  // ---- Build ----

  _build() {
    const m = this._mode;

    // We store geometry specs; actual meshes updated each frame from keypoints.
    // Instead of static geometry, we create placeholder meshes per segment and
    // reposition them via applyKeypoints().

    // Head (sphere at kp 0)
    this._head = this._addSeg('head', new THREE.SphereGeometry(0.12, 16, 12), DP_COLORS.head);
    // Neck (kp 0 → midpoint shoulders)
    this._neck = this._addCylSeg('neck', 0.04, 0.04, 0.15, DP_COLORS.neck);
    // Upper torso (shoulders bar kp 5-6 + spine to hip midpoint)
    this._upperTorso = this._addBoxSeg('torso_front', 0.44, 0.28, 0.18, DP_COLORS.torso_front);
    // Lower torso
    this._lowerTorso = this._addBoxSeg('torso_back', 0.36, 0.24, 0.16, DP_COLORS.torso_back);
    // Pelvis
    this._pelvis = this._addBoxSeg('pelvis', 0.32, 0.16, 0.16, DP_COLORS.pelvis);

    // Arms
    this._ua_l = this._addCylSeg('upper_arm_l', 0.055, 0.044, 0.3, DP_COLORS.upper_arm_l);
    this._ua_r = this._addCylSeg('upper_arm_r', 0.055, 0.044, 0.3, DP_COLORS.upper_arm_r);
    this._la_l = this._addCylSeg('lower_arm_l', 0.042, 0.034, 0.27, DP_COLORS.lower_arm_l);
    this._la_r = this._addCylSeg('lower_arm_r', 0.042, 0.034, 0.27, DP_COLORS.lower_arm_r);
    // Hands
    this._hand_l = this._addSeg('hand_l', new THREE.SphereGeometry(0.038,8,6), DP_COLORS.hand_l);
    this._hand_r = this._addSeg('hand_r', new THREE.SphereGeometry(0.038,8,6), DP_COLORS.hand_r);

    // Legs
    this._th_l = this._addCylSeg('thigh_l', 0.085, 0.065, 0.42, DP_COLORS.thigh_l);
    this._th_r = this._addCylSeg('thigh_r', 0.085, 0.065, 0.42, DP_COLORS.thigh_r);
    this._sh_l = this._addCylSeg('shin_l',  0.062, 0.045, 0.38, DP_COLORS.shin_l);
    this._sh_r = this._addCylSeg('shin_r',  0.062, 0.045, 0.38, DP_COLORS.shin_r);
    // Feet
    this._foot_l = this._addBoxSeg('foot_l', 0.1, 0.06, 0.2, DP_COLORS.foot_l);
    this._foot_r = this._addBoxSeg('foot_r', 0.1, 0.06, 0.2, DP_COLORS.foot_r);

    // Skeleton overlay (shared across modes for hybrid/xray)
    this._buildSkeletonOverlay();

    // Ground shadow blob
    const shadowGeo = new THREE.CircleGeometry(0.3, 16);
    const shadowMat = new THREE.MeshBasicMaterial({
      color: 0x000000, transparent: true, opacity: 0.25,
      depthWrite: false,
    });
    this._shadowMesh = new THREE.Mesh(shadowGeo, shadowMat);
    this._shadowMesh.rotation.x = -Math.PI / 2;
    this._shadowMesh.position.y = 0.002;
    this._group.add(this._shadowMesh);
  }

  _addSeg(name, geo, dpColor) {
    const mat  = makeMat(this._mode, dpColor);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    this._group.add(mesh);
    this._segments.push({ mesh, mat, dpColor, name });
    return mesh;
  }

  _addCylSeg(name, rTop, rBot, len, dpColor) {
    const geo  = new THREE.CylinderGeometry(rTop, rBot, len, 10, 1);
    const mesh = this._addSeg(name, geo, dpColor);
    mesh.userData._origLen = len; // store base height for proper rescaling
    return mesh;
  }

  _addBoxSeg(name, w, h, d, dpColor) {
    const geo = new THREE.BoxGeometry(w, h, d, 2, 2, 1);
    return this._addSeg(name, geo, dpColor);
  }

  _buildSkeletonOverlay() {
    const PAIRS = [
      [0,1],[0,2],[1,3],[2,4],
      [5,6],[5,7],[7,9],[6,8],[8,10],
      [5,11],[6,12],[11,12],
      [11,13],[13,15],[12,14],[14,16],
    ];
    const boneMat = new THREE.MeshBasicMaterial({
      color: 0x00d878, transparent: true,
      opacity: this._skeletonOpacity,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this._boneMat = boneMat;
    this._boneLines = [];
    for (const [a, b] of PAIRS) {
      const geo = new THREE.CylinderGeometry(0.012, 0.012, 1, 6, 1);
      geo.translate(0, 0.5, 0);
      geo.rotateX(Math.PI / 2);
      const m = new THREE.Mesh(geo, boneMat);
      m.castShadow = false;
      this._skeletonGroup.add(m);
      this._boneLines.push({ mesh: m, a, b });
    }
    // Joint spheres
    this._jointMeshes = [];
    const jointMat = new THREE.MeshBasicMaterial({
      color: 0xff4060, transparent: true,
      opacity: this._skeletonOpacity,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this._jointMat = jointMat;
    for (let i = 0; i < 17; i++) {
      const geo = new THREE.SphereGeometry(0.022, 8, 8);
      const m   = new THREE.Mesh(geo, jointMat);
      this._skeletonGroup.add(m);
      this._jointMeshes.push(m);
    }
  }

  // ---- Public API ----

  setRenderMode(mode) {
    this._mode = mode;
    // Rebuild materials
    for (const seg of this._segments) {
      seg.mesh.material.dispose();
      seg.mesh.material = makeMat(mode, seg.dpColor);
      seg.mat = seg.mesh.material;
    }
    const skelOpacity = (mode === 'xray') ? 0.9 : (mode === 'hybrid') ? 0.5 : 0.0;
    this._boneMat.opacity  = skelOpacity;
    this._jointMat.opacity = skelOpacity;
    this._skeletonOpacity  = skelOpacity;
    // In xray mode make body segments very transparent
    if (mode === 'xray') {
      for (const seg of this._segments) {
        seg.mesh.material.opacity = 0.12;
        seg.mesh.material.transparent = true;
        seg.mesh.material.depthWrite = false;
      }
    }
  }

  setBodyOpacity(v) {
    for (const seg of this._segments) {
      seg.mesh.material.opacity = v;
      seg.mesh.material.transparent = v < 1;
    }
  }

  setSkeletonOverlay(v) {
    this._boneMat.opacity  = v;
    this._jointMat.opacity = v;
  }

  show(visible) {
    this._group.visible = visible;
  }

  /**
   * Apply 17-keypoint COCO array [x,y,z] to all body segments.
   * lerpT: 0=snap, 1=use previous (no movement)
   */
  applyKeypoints(kps, lerpT = 0.18) {
    if (!kps || kps.length < 17) return;

    if (this._prevKps && lerpT > 0) {
      const blended = kps.map((p, i) => {
        const pr = this._prevKps[i];
        return [
          pr[0] + (p[0] - pr[0]) * lerpT,
          pr[1] + (p[1] - pr[1]) * lerpT,
          pr[2] + (p[2] - pr[2]) * lerpT,
        ];
      });
      this._kps = blended;
    } else {
      this._kps = kps;
    }
    this._prevKps = this._kps;

    const k = this._kps;
    const v = (i) => new THREE.Vector3(k[i][0], k[i][1], k[i][2]);

    // Helper: position+orient cylinder mesh between two keypoints
    const orientSeg = (mesh, pA, pB) => {
      const dir = new THREE.Vector3().subVectors(pB, pA);
      const len = dir.length();
      const mid = new THREE.Vector3().addVectors(pA, pB).multiplyScalar(0.5);
      mesh.position.copy(mid);
      const origLen = mesh.userData._origLen || 0.3;
      mesh.scale.y = len / origLen;
      const dirN = dir.clone().normalize();
      if (dirN.lengthSq() > 0.001) {
        mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0), dirN);
      }
    };

    // ---- Head ----
    this._head.position.set(k[0][0], k[0][1] + 0.06, k[0][2]);

    // ---- Neck ----
    const noseP    = v(0);
    const lShouP   = v(5);
    const rShouP   = v(6);
    const shoulderMid = new THREE.Vector3().addVectors(lShouP, rShouP).multiplyScalar(0.5);
    const neckBase = shoulderMid.clone().lerp(noseP, 0.2);
    orientSeg(this._neck, neckBase, noseP.clone().add(new THREE.Vector3(0,-0.1,0)));

    // ---- Upper torso ----
    const lHipP  = v(11);
    const rHipP  = v(12);
    const hipMid = new THREE.Vector3().addVectors(lHipP, rHipP).multiplyScalar(0.5);
    const spineCenter = shoulderMid.clone().lerp(hipMid, 0.5);
    this._upperTorso.position.copy(shoulderMid.clone().lerp(spineCenter, 0.5));
    const torsoDir = new THREE.Vector3().subVectors(shoulderMid, hipMid).normalize();
    this._upperTorso.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0), torsoDir);
    const torsoH = shoulderMid.distanceTo(hipMid) * 0.55;
    this._upperTorso.scale.y = torsoH / 0.28;
    const torsoW = lShouP.distanceTo(rShouP) * 0.9;
    this._upperTorso.scale.x = torsoW / 0.44;

    // ---- Lower torso ----
    this._lowerTorso.position.copy(hipMid.clone().lerp(spineCenter, 0.4));
    this._lowerTorso.quaternion.copy(this._upperTorso.quaternion);
    this._lowerTorso.scale.set(torsoW / 0.36 * 0.85, torsoH / 0.24 * 0.5, 1);

    // ---- Pelvis ----
    this._pelvis.position.copy(hipMid.clone().add(new THREE.Vector3(0, -0.06, 0)));
    this._pelvis.scale.x = lHipP.distanceTo(rHipP) / 0.32;

    // ---- Arms ----
    const lElbP = v(7); const lWriP = v(9);
    const rElbP = v(8); const rWriP = v(10);
    orientSeg(this._ua_l, lShouP, lElbP);
    orientSeg(this._ua_r, rShouP, rElbP);
    orientSeg(this._la_l, lElbP, lWriP);
    orientSeg(this._la_r, rElbP, rWriP);
    this._hand_l.position.set(k[9][0], k[9][1], k[9][2]);
    this._hand_r.position.set(k[10][0], k[10][1], k[10][2]);

    // ---- Legs ----
    const lKneP = v(13); const lAnkP = v(15);
    const rKneP = v(14); const rAnkP = v(16);
    orientSeg(this._th_l, lHipP, lKneP);
    orientSeg(this._th_r, rHipP, rKneP);
    orientSeg(this._sh_l, lKneP, lAnkP);
    orientSeg(this._sh_r, rKneP, rAnkP);
    // Feet
    this._foot_l.position.copy(v(15)).add(new THREE.Vector3(0, 0.025, 0.05));
    this._foot_r.position.copy(v(16)).add(new THREE.Vector3(0, 0.025, 0.05));

    // Shadow at foot midpoint
    const footMidX = (k[15][0] + k[16][0]) / 2;
    const footMidZ = (k[15][2] + k[16][2]) / 2;
    this._shadowMesh.position.set(footMidX, 0.002, footMidZ);

    // ---- Skeleton overlay ----
    for (const bone of this._boneLines) {
      const pA = v(bone.a), pB = v(bone.b);
      const dir = new THREE.Vector3().subVectors(pB, pA);
      const len = dir.length();
      bone.mesh.position.copy(pA);
      bone.mesh.scale.z = len;
      bone.mesh.lookAt(pB);
    }
    for (let i = 0; i < 17; i++) {
      this._jointMeshes[i].position.set(k[i][0], k[i][1], k[i][2]);
    }
  }

  dispose() {
    for (const seg of this._segments) {
      seg.mesh.geometry.dispose();
      seg.mat.dispose();
    }
    this._scene.remove(this._group);
  }
}
