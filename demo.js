import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RIRTracer, makeSurfelTextures } from './rirTracer.js';
import { RIRAuralizer } from './rirAuralizer.js';

const $ = (id) => document.getElementById(id);
const status = (s) => { $('status').textContent = s; };

/* ---------------------------------------------------------------- room (shoebox, spans [0..size]) */
const ROOM = new THREE.Vector3(6, 3, 4);
const TEX = 64;                                  // 64x64 = 4096 surfels

function shoeboxSurfels(size, maxCount) {
  const { x: X, y: Y, z: Z } = size;
  const walls = [
    { o: [0, 0, 0], u: [1, 0, 0], v: [0, 0, 1], n: [0, 1, 0],  lu: X, lv: Z },   // floor
    { o: [0, Y, 0], u: [1, 0, 0], v: [0, 0, 1], n: [0, -1, 0], lu: X, lv: Z },   // ceiling
    { o: [0, 0, 0], u: [1, 0, 0], v: [0, 1, 0], n: [0, 0, 1],  lu: X, lv: Y },
    { o: [0, 0, Z], u: [1, 0, 0], v: [0, 1, 0], n: [0, 0, -1], lu: X, lv: Y },
    { o: [0, 0, 0], u: [0, 0, 1], v: [0, 1, 0], n: [1, 0, 0],  lu: Z, lv: Y },
    { o: [X, 0, 0], u: [0, 0, 1], v: [0, 1, 0], n: [-1, 0, 0], lu: Z, lv: Y },
  ];
  const area = 2 * (X * Y + Y * Z + X * Z);
  let a = Math.sqrt(area / maxCount), counts;
  for (;;) {                                      // largest grid that still fits in the texture
    counts = walls.map((w) => [Math.max(1, Math.round(w.lu / a)), Math.max(1, Math.round(w.lv / a))]);
    if (counts.reduce((s, c) => s + c[0] * c[1], 0) <= maxCount) break;
    a *= 1.01;
  }
  const positions = new Float32Array(maxCount * 3).fill(1e6);   // unused texels: far away, zero normal
  const normals = new Float32Array(maxCount * 3);
  let idx = 0, halfDiag = 0;
  walls.forEach((w, wi) => {
    const [nx, ny] = counts[wi], cu = w.lu / nx, cv = w.lv / ny;
    halfDiag = Math.max(halfDiag, 0.5 * Math.hypot(cu, cv));
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++, idx++) {
      for (let c = 0; c < 3; c++) {
        positions[idx * 3 + c] = w.o[c] + w.u[c] * (i + 0.5) * cu + w.v[c] * (j + 0.5) * cv;
        normals[idx * 3 + c] = w.n[c];
      }
    }
  });
  return { positions, normals, surfelRadius: halfDiag * 1.05, count: idx };   // 5% margin closes cell-corner gaps
}

/* ---------------------------------------------------------------- room (arbitrary triangle mesh, e.g. glTF) */
function meshTriangles(root) {                  // world-space triangles of every mesh under root
  root.updateMatrixWorld(true);
  const tris = [];
  root.traverse((m) => {
    if (!m.isMesh || !m.geometry.attributes.position) return;
    const pos = m.geometry.attributes.position, index = m.geometry.index;
    const n = index ? index.count : pos.count;
    const vert = (i) => new THREE.Vector3().fromBufferAttribute(pos, index ? index.getX(i) : i).applyMatrix4(m.matrixWorld);
    for (let i = 0; i + 2 < n; i += 3) {
      const a = vert(i), b = vert(i + 1), c = vert(i + 2);
      const nrm = new THREE.Vector3().crossVectors(b.clone().sub(a), c.clone().sub(a));
      const area = 0.5 * nrm.length();
      if (area < 1e-9) continue;               // degenerate
      tris.push({ a, b, c, n: nrm.normalize(), area });
    }
  });
  return tris;
}

function meshSurfels(tris, maxCount) {
  const area = tris.reduce((s, t) => s + t.area, 0);
  const step = area / maxCount;                 // surface area per surfel
  const positions = new Float32Array(maxCount * 3).fill(1e6);
  const normals = new Float32Array(maxCount * 3);
  const G = 1.32471795724474602596, a1 = 1 / G, a2 = 1 / (G * G);   // R2 low-discrepancy sequence
  let idx = 0, next = Math.random() * step, cum = 0;
  for (const t of tris) {                       // systematic sampling: surfel count proportional to area
    cum += t.area;
    const r0 = Math.random(), r1 = Math.random();
    for (let j = 0; next < cum && idx < maxCount; j++, idx++, next += step) {
      let u = (r0 + j * a1) % 1, v = (r1 + j * a2) % 1;
      if (u + v > 1) { u = 1 - u; v = 1 - v; } // fold the unit square onto the triangle
      for (let c = 0; c < 3; c++) {
        const k = 'xyz'[c];
        positions[idx * 3 + c] = t.a[k] + u * (t.b[k] - t.a[k]) + v * (t.c[k] - t.a[k]);
        normals[idx * 3 + c] = t.n[k];
      }
    }
  }
  // samples are irregular, so the discs need more overlap than the shoebox grid to keep rays from leaking
  return { positions, normals, surfelRadius: 1.0 * Math.sqrt(step), count: idx };
}

/* ---------------------------------------------------------------- three.js scene (visual feedback only) */
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(devicePixelRatio);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.5));   // lights glTF (PBR) room materials
let roomVisual = new THREE.Group();
roomVisual.add(new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(ROOM.x, ROOM.y, ROOM.z)),
  new THREE.LineBasicMaterial({ color: 0x4488ff })).translateX(ROOM.x / 2).translateY(ROOM.y / 2).translateZ(ROOM.z / 2));
roomVisual.add(new THREE.GridHelper(Math.max(ROOM.x, ROOM.z), 12, 0x333333, 0x222222).translateX(ROOM.x / 2).translateZ(ROOM.z / 2));
scene.add(roomVisual);
const roomBox = new THREE.Box3(new THREE.Vector3(), ROOM.clone());

const sourcePos = new THREE.Vector3(4.5, 1.4, 2.8);
const sourceMesh = new THREE.Mesh(new THREE.SphereGeometry(0.12), new THREE.MeshBasicMaterial({ color: 0xff8844 }));
sourceMesh.position.copy(sourcePos);
scene.add(sourceMesh);

const camera = new THREE.PerspectiveCamera(75, 1, 0.05, 50);
camera.rotation.order = 'YXZ';
camera.position.set(1.5, 1.6, 1.2);
let yaw = -0.6, pitch = 0;

function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

let dragging = false;
canvas.addEventListener('pointerdown', (e) => { dragging = true; canvas.setPointerCapture(e.pointerId); canvas.style.cursor = 'grabbing'; });
canvas.addEventListener('pointerup', () => { dragging = false; canvas.style.cursor = 'grab'; });
canvas.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  yaw -= e.movementX * 0.004;
  pitch = THREE.MathUtils.clamp(pitch - e.movementY * 0.004, -1.5, 1.5);
});
addEventListener('keydown', (e) => {
  const step = 0.25, f = new THREE.Vector3(-Math.sin(yaw), 0, -Math.cos(yaw)), r = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
  if (e.key === 'w') camera.position.addScaledVector(f, step);
  if (e.key === 's') camera.position.addScaledVector(f, -step);
  if (e.key === 'd') camera.position.addScaledVector(r, step);
  if (e.key === 'a') camera.position.addScaledVector(r, -step);
  camera.position.clamp(roomBox.min.clone().addScalar(0.3), roomBox.max.clone().subScalar(0.3));
});

/* ---------------------------------------------------------------- tracer + auralizer */
const aur = new RIRAuralizer({ speakers: 'cube' });
let tracer = null;

async function init() {
  const frag = await (await fetch('./rirTrace.frag')).text();
  const room = shoeboxSurfels(ROOM, TEX * TEX);
  const [tex1, tex2] = makeSurfelTextures(room.positions, room.normals, TEX, TEX);
  tracer = new RIRTracer(renderer, frag, {
    tex1, tex2, camera,
    source: sourcePos,
    surfelRadius: room.surfelRadius,
    cameraRelative: false,            // directions in WORLD axes -> head tracking via AudioListener, no re-trace
    rayTexSize: 128, batches: 4, maxBounces: 64,
    absorption: new THREE.Vector4(0.20, 0.25, 0.30, 0.35),
  });
  aur.loadAnechoic(makeTestSignal(aur.ctx));
  $('trace').disabled = false;
  $('room').disabled = false;
  status(`ready — ${room.count} surfels`);
  const url = new URLSearchParams(location.search).get('room');
  if (url) await loadRoom(url, url.split('/').pop());
}

/* ---------------------------------------------------------------- glTF rooms */
async function loadRoom(src, name) {           // src: URL string or File
  status('loading ' + name + '…');
  const loader = new GLTFLoader();
  const gltf = typeof src === 'string'
    ? await loader.loadAsync(src)
    : await loader.parseAsync(await src.arrayBuffer(), '');   // .glb or .gltf with embedded buffers
  const tris = meshTriangles(gltf.scene);
  if (!tris.length) throw new Error(name + ' contains no triangles');
  const room = meshSurfels(tris, TEX * TEX);

  const u = tracer.traceMat.uniforms;
  u.uTex1.value.dispose(); u.uTex2.value.dispose();
  [u.uTex1.value, u.uTex2.value] = makeSurfelTextures(room.positions, room.normals, TEX, TEX);
  u.uSurfelRadius.value = room.surfelRadius;

  scene.remove(roomVisual);
  roomVisual = gltf.scene;
  scene.add(roomVisual);
  roomBox.setFromObject(roomVisual);
  // same relative placement as in the default shoebox
  const rel = (x, y, z) => new THREE.Vector3(x, y, z).multiply(roomBox.getSize(new THREE.Vector3())).add(roomBox.min);
  sourcePos.copy(rel(0.75, 0.47, 0.7));       // uSource holds this vector, so the tracer sees the change
  sourceMesh.position.copy(sourcePos);
  camera.position.copy(rel(0.25, 0.53, 0.3));
  camera.far = Math.max(50, 2 * roomBox.getSize(new THREE.Vector3()).length());
  camera.updateProjectionMatrix();

  status(`${name}: ${tris.length} triangles → ${room.count} surfels, r = ${room.surfelRadius.toFixed(3)} m — trace again`);
}

$('room').onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  $('trace').disabled = true;
  try { await loadRoom(f, f.name); } catch (err) { console.error(err); status('error: ' + err.message); }
  $('trace').disabled = false;
};

$('trace').onclick = async () => {
  $('trace').disabled = true;
  status('ray tracing…');
  await new Promise((r) => setTimeout(r, 30));
  const t0 = performance.now();
  tracer.run();
  status('synthesizing impulse response…');
  await aur.setRIRFromTracer(tracer);           // reads back the GPU result, builds IRs, swaps convolvers
  drawEchogram(tracer.readOmni(), tracer.opts.numBins);
  status(`RIR ready (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
  $('trace').disabled = false;
  $('play').disabled = false;
};

let playing = false;
$('play').onclick = async () => {
  if (playing) { aur.stop(); playing = false; return; }
  await aur.play({ loop: true });
  playing = true;
};

$('file').onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  await aur.loadAnechoic(f);
  if (playing) await aur.play({ loop: true });
  status('loaded ' + f.name);
};

/* ---------------------------------------------------------------- helpers */
function makeTestSignal(ctx) {                  // dry claps + tone bursts, so the room is easy to hear
  const fs = ctx.sampleRate, buf = ctx.createBuffer(1, fs * 6, fs), d = buf.getChannelData(0);
  [0.2, 1.7, 3.2, 4.7].forEach((t0, idx) => {
    const n0 = Math.floor(t0 * fs);
    for (let i = 0; i < 0.15 * fs; i++) {
      const env = Math.exp(-i / (0.008 * fs));
      d[n0 + i] += (idx % 2 === 0)
        ? (Math.random() * 2 - 1) * env * 0.8                                   // clap
        : Math.sin(2 * Math.PI * 800 * i / fs) * Math.exp(-i / (0.03 * fs)) * 0.6; // blip
    }
  });
  return buf;
}

function drawEchogram(omni, nBins) {
  const c = $('echo'), g = c.getContext('2d'), W = c.width, H = c.height;
  g.clearRect(0, 0, W, H);
  let max = 1e-12;
  for (let i = 0; i < omni.length; i++) max = Math.max(max, omni[i]);
  const colors = ['#f44', '#fa4', '#4f4', '#48f'], range = 70;       // dB
  for (let b = 0; b < 4; b++) {
    g.strokeStyle = colors[b]; g.beginPath();
    for (let x = 0; x < W; x++) {
      const i0 = Math.floor(x / W * nBins), i1 = Math.max(i0 + 1, Math.floor((x + 1) / W * nBins));
      let m = 0;
      for (let i = i0; i < i1; i++) m = Math.max(m, omni[i * 4 + b]);
      const db = 10 * Math.log10(m / max + 1e-12);
      const y = H * (1 - (db + range) / range);
      x ? g.lineTo(x, Math.min(H, y)) : g.moveTo(x, Math.min(H, y));
    }
    g.stroke();
  }
}

/* ---------------------------------------------------------------- loop */
renderer.setAnimationLoop(() => {
  camera.rotation.set(pitch, yaw, 0);
  renderer.render(scene, camera);
  aur.setListenerFromCamera(camera);            // head tracking
});

init().catch((e) => { console.error(e); status('error: ' + e.message); });
