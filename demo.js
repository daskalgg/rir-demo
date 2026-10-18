import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RIRTracer, makeSurfelTextures, BANDS } from './rirTracer.js';
import { RIRAuralizer } from './rirAuralizer.js';

const $ = (id) => document.getElementById(id);
const status = (s) => { $('status').textContent = s; };

/* ---------------------------------------------------------------- room materials */
const TEX = 64;                                  // 64x64 = 4096 surfels
// absorption per octave band (BANDS: 125 Hz .. 16 kHz) and scattering, applied to every surface; values follow
// the usual 125 Hz - 4 kHz tables, with 8 / 16 kHz continuing each material's trend (tables rarely go that high)
const DEFAULT_MATERIAL = { absorption: [0.20, 0.22, 0.25, 0.27, 0.30, 0.33, 0.35, 0.37], scattering: 0.2 };
const MATERIALS = [
  //                                                                 125   250   500   1k    2k    4k    8k    16k
  { id: 'concrete', label: 'Concrete (smooth)',            absorption: [0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.04, 0.05], scattering: 0.05 },
  { id: 'brick',    label: 'Brick (unglazed)',             absorption: [0.03, 0.03, 0.03, 0.04, 0.05, 0.07, 0.08, 0.09], scattering: 0.10 },
  { id: 'marble',   label: 'Marble / glazed tile',         absorption: [0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03], scattering: 0.05 },
  { id: 'plaster',  label: 'Plasterboard on studs',        absorption: [0.29, 0.10, 0.05, 0.04, 0.07, 0.09, 0.10, 0.11], scattering: 0.05 },
  { id: 'wood',     label: 'Wood panelling',               absorption: [0.28, 0.22, 0.17, 0.09, 0.10, 0.11, 0.12, 0.13], scattering: 0.10 },
  { id: 'glass',    label: 'Glass (window)',               absorption: [0.35, 0.25, 0.18, 0.12, 0.07, 0.04, 0.03, 0.03], scattering: 0.05 },
  { id: 'carpet',   label: 'Carpet on concrete',           absorption: [0.02, 0.06, 0.14, 0.37, 0.60, 0.65, 0.68, 0.70], scattering: 0.20 },
  { id: 'curtain',  label: 'Heavy curtains',               absorption: [0.14, 0.35, 0.55, 0.72, 0.70, 0.65, 0.65, 0.65], scattering: 0.40 },
  { id: 'audience', label: 'Audience / upholstered seats', absorption: [0.60, 0.74, 0.88, 0.96, 0.93, 0.85, 0.80, 0.75], scattering: 0.70 },
  { id: 'foam',     label: 'Acoustic foam (anechoic)',     absorption: [0.90, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99], scattering: 0.50 },
];
const bandLabel = (f) => (f >= 1000 ? `${f / 1000}k` : `${f}`);
const BAND_COLORS = BANDS.map((_, b) => `hsl(${(b / (BANDS.length - 1)) * 270}, 90%, 62%)`);   // red (low) -> violet (high)
let roomMaterial = DEFAULT_MATERIAL;            // the loaded room's own material (glTF scene extras), see loadRoom

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
  // samples are irregular, so the discs need generous overlap to keep rays from leaking through gaps
  return { positions, normals, surfelRadius: 1.0 * Math.sqrt(step), count: idx };
}

/* ---------------------------------------------------------------- three.js scene (visual feedback only) */
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(devicePixelRatio);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.5));   // lights glTF (PBR) room materials
let roomVisual = new THREE.Group();            // replaced by the loaded glTF scene
scene.add(roomVisual);
const roomBox = new THREE.Box3();               // empty until a room is loaded

const sourcePos = new THREE.Vector3();
const sourceMesh = new THREE.Mesh(new THREE.SphereGeometry(0.12), new THREE.MeshBasicMaterial({ color: 0xff8844 }));
sourceMesh.position.copy(sourcePos);
scene.add(sourceMesh);

const camera = new THREE.PerspectiveCamera(75, 1, 0.05, 50);
camera.rotation.order = 'YXZ';
let yaw = 0, pitch = 0;

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
    if (e.target.closest?.('input, select, textarea') || roomBox.isEmpty()) return;
    const step = 0.25, f = new THREE.Vector3(-Math.sin(yaw), 0, -Math.cos(yaw)), r = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
    if (e.key === 'w') camera.position.addScaledVector(f, step);
    if (e.key === 's') camera.position.addScaledVector(f, -step);
    if (e.key === 'd') camera.position.addScaledVector(r, step);
    if (e.key === 'a') camera.position.addScaledVector(r, -step);
    camera.position.clamp(roomBox.min.clone().addScalar(0.3), roomBox.max.clone().subScalar(0.3));

    if (e.key == 'ArrowUp') sourceMesh.position.addScaledVector(f, step);
    if (e.key == 'ArrowDown') sourceMesh.position.addScaledVector(f, -step);
    if (e.key == 'ArrowRight') sourceMesh.position.addScaledVector(r, step);
    if (e.key == 'ArrowLeft') sourceMesh.position.addScaledVector(r, -step);
    sourceMesh.position.clamp(roomBox.min.clone().addScalar(0.3), roomBox.max.clone().subScalar(0.3));
    sourcePos.copy(sourceMesh.position);        // uSource holds sourcePos

    if (['w', 'a', 's', 'd', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) requestTrace();
});

/* ---------------------------------------------------------------- tracer + auralizer */
const aur = new RIRAuralizer({ speakers: 'cube', crossfade: 0.4 });
let tracer = null, traceFrag = null;            // the tracer is created by the first loadRoom()

async function init() {
  traceFrag = await (await fetch('./rirTrace.frag')).text();
  aur.loadAnechoic(makeTestSignal(aur.ctx));    // "Test signal" is the first entry of the sound dropdown

  // anechoic sounds in sounds/ (index.json is written by sounds/build-index.mjs)
  let sounds = [];
  try { sounds = await (await fetch('./sounds/index.json')).json(); } catch (e) { console.warn('no sounds/index.json', e); }
  for (const snd of sounds) $('soundSelect').append(new Option(snd.name, 'sounds/' + snd.file));

  // rooms in models/ (index.json is written by the generators / models/build-index.mjs), plus ?room=url
  let rooms = [];
  try { rooms = await (await fetch('./models/index.json')).json(); } catch (e) { console.warn('no models/index.json', e); }
  const sel = $('roomSelect');
  for (const r of rooms) sel.append(new Option(r.name, 'models/' + r.file));
  const param = new URLSearchParams(location.search).get('room');
  if (param && ![...sel.options].some((o) => o.value === param)) sel.append(new Option(param.split('/').pop(), param));
  $('room').disabled = false;
  if (!sel.options.length) { status('no rooms in models/ — open a glTF file'); return; }
  sel.value = param || sel.options[0].value;
  sel.disabled = false;
  await selectRoom();
}

/* ---------------------------------------------------------------- glTF rooms */
let loadToken = 0;
async function loadRoom(src, name) {           // src: URL string or File
  const token = ++loadToken;
  status('loading ' + name + '…');
  const loader = new GLTFLoader();
  const gltf = typeof src === 'string'
    ? await loader.loadAsync(src)
    : await loader.parseAsync(await src.arrayBuffer(), '');   // .glb or .gltf with embedded buffers
  if (token !== loadToken) return;              // another room was picked meanwhile
  const tris = meshTriangles(gltf.scene);
  if (!tris.length) throw new Error(name + ' contains no triangles');
  const room = meshSurfels(tris, TEX * TEX);

  const [tex1, tex2] = makeSurfelTextures(room.positions, room.normals, TEX, TEX);
  if (!tracer) {
    tracer = new RIRTracer(renderer, traceFrag, {
      tex1, tex2, camera,
      source: sourcePos,
      surfelRadius: room.surfelRadius,
      cameraRelative: false,          // directions in WORLD axes -> head tracking via AudioListener, no re-trace
      rayTexSize: RAY_TEX, batches: +$('rays').value, maxBounces: +$('bounces').value,
      absorption: DEFAULT_MATERIAL.absorption, scattering: DEFAULT_MATERIAL.scattering,
    });
    $('trace').disabled = false;
  } else {
    const u = tracer.traceMat.uniforms;
    u.uTex1.value.dispose(); u.uTex2.value.dispose();
    [u.uTex1.value, u.uTex2.value] = [tex1, tex2];
    u.uSurfelRadius.value = room.surfelRadius;
  }
  // optional room material from the glTF scene's extras: { "absorption": [one per band in BANDS], "scattering": s }
  const { absorption: abs, scattering: sc } = gltf.scene.userData;
  const customAbs = Array.isArray(abs) && abs.length === BANDS.length && abs.every((a) => a >= 0 && a <= 1);
  if (abs !== undefined && !customAbs) console.warn(`${name}: extras.absorption needs ${BANDS.length} values in 0..1 (${BANDS.join(', ')} Hz)`, abs);
  const customSc = typeof sc === 'number' && sc >= 0 && sc <= 1;
  roomMaterial = {
    absorption: customAbs ? abs : DEFAULT_MATERIAL.absorption,
    scattering: customSc ? sc : DEFAULT_MATERIAL.scattering,
  };
  selectMaterial('room');                       // a new room starts with its own material

  scene.remove(roomVisual);
  roomVisual = gltf.scene;
  scene.add(roomVisual);
  roomBox.setFromObject(roomVisual);
  // start positions: empty nodes named "Source" / "Listener" if the file has them, else fixed fractions of the
  // bounding box
  const rel = (x, y, z) => new THREE.Vector3(x, y, z).multiply(roomBox.getSize(new THREE.Vector3())).add(roomBox.min);
  const marker = (name) => {
    let found = null;
    gltf.scene.traverse((o) => { if (!found && o.name.toLowerCase() === name) found = o; });
    return found && found.getWorldPosition(new THREE.Vector3());
  };
  sourcePos.copy(marker('source') || rel(0.75, 0.47, 0.7));   // uSource holds this vector, so the tracer sees the change
  sourceMesh.position.copy(sourcePos);
  camera.position.copy(marker('listener') || rel(0.25, 0.53, 0.3));
  yaw = Math.atan2(camera.position.x - sourcePos.x, camera.position.z - sourcePos.z);   // face the source
  pitch = 0;
  camera.far = Math.max(50, 2 * roomBox.getSize(new THREE.Vector3()).length());
  camera.updateProjectionMatrix();

  status(`${name}: ${tris.length} triangles → ${room.count} surfels, r = ${room.surfelRadius.toFixed(3)} m` +
    (customAbs ? `, absorption ${abs.join(' / ')}` : '') + (customSc ? `, scattering ${sc}` : ''));
  requestTrace(0);
}

/* ---------------------------------------------------------------- settings panel */
// rays are added in batches of RAY_TEX^2: a bigger per-pass texture risks the GPU watchdog (each ray scans every surfel)
const RAY_TEX = 128;
function updateSettings() {
  const batches = +$('rays').value, bounces = +$('bounces').value;
  $('raysOut').textContent = (batches * RAY_TEX * RAY_TEX).toLocaleString();
  $('bouncesOut').textContent = bounces;
  if (tracer) { tracer.setBatches(batches); tracer.setMaxBounces(bounces); }
}
$('rays').oninput = $('bounces').oninput = updateSettings;
$('rays').onchange = $('bounces').onchange = () => requestTrace();   // on release, not on every drag step
updateSettings();

// material: a preset fills in the coefficient fields; editing a field switches to "Custom"
const coefInput = () => Object.assign(document.createElement('input'), { type: 'number', min: 0, max: 1, step: 0.01 });
const coefLabel = (text, cls) => Object.assign(document.createElement('span'), { textContent: text, className: cls || '' });
const absInputs = BANDS.map(coefInput);
$('coefs').append(
  coefLabel('Hz'), ...BANDS.map((f) => coefLabel(bandLabel(f), 'head')),
  coefLabel('Absorption'), ...absInputs,
  coefLabel('Scattering'), Object.assign(coefInput(), { id: 'scattering' }),
);
$('material').append(
  new Option('Room default', 'room'),
  ...MATERIALS.map((m) => new Option(m.label, m.id)),
  new Option('Custom', 'custom'),
);

function selectMaterial(id) {
  const m = id === 'room' ? roomMaterial : MATERIALS.find((x) => x.id === id);
  $('material').value = id;
  if (m) {
    absInputs.forEach((el, b) => { el.value = m.absorption[b]; });
    $('scattering').value = m.scattering;
  }
  applyMaterial();
}

function applyMaterial() {
  const clamp01 = (el) => {
    const v = THREE.MathUtils.clamp(Number.isFinite(+el.value) && el.value !== '' ? +el.value : 0, 0, 1);
    el.value = v;
    return v;
  };
  if (!tracer) return;
  tracer.setAbsorption(absInputs.map(clamp01));
  tracer.setScattering(clamp01($('scattering')));
}

$('material').onchange = () => { selectMaterial($('material').value); requestTrace(); };
for (const el of [...absInputs, $('scattering')]) {
  el.onchange = () => { $('material').value = 'custom'; applyMaterial(); requestTrace(); };
}
selectMaterial('room');

// room dropdown: URLs from models/index.json (+ ?room=), and files opened from disk ("local:<n>")
const localRooms = [];
async function selectRoom() {
  const opt = $('roomSelect').selectedOptions[0];
  if (!opt) return;
  const src = opt.value.startsWith('local:') ? localRooms[+opt.value.slice(6)] : opt.value;
  try { await loadRoom(src, opt.textContent); } catch (err) { console.error(err); status('error: ' + err.message); }
}
$('roomSelect').onchange = selectRoom;

$('room').onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  localRooms.push(f);
  $('roomSelect').append(new Option(`${f.name} (local file)`, `local:${localRooms.length - 1}`));
  $('roomSelect').value = `local:${localRooms.length - 1}`;
  $('roomSelect').disabled = false;
  e.target.value = '';                          // picking the same file again still fires change
  selectRoom();
};

/* ---------------------------------------------------------------- (re)tracing */
// Changes are debounced, and at most one trace runs at a time: a change during a trace queues exactly one
// more, which picks up the latest state. The auralizer cross-fades to each new RIR.
let tracing = false, traceQueued = false, traceTimer = 0;

function requestTrace(delay = 250) {
  if (!tracer || !$('auto').checked) return;
  clearTimeout(traceTimer);
  traceTimer = setTimeout(trace, delay);
}

async function trace() {
  if (tracing) { traceQueued = true; return; }
  tracing = true;
  try {
    status('ray tracing…');
    await new Promise((r) => setTimeout(r, 30));  // let the status repaint
    const t0 = performance.now();
    tracer.run();
    status('synthesizing impulse response…');
    const ir = await aur.setRIRFromTracer(tracer);   // reads back the GPU result, builds IRs, cross-fades convolvers
    drawEchogram(tracer.readOmni(), tracer.opts.numBins);
    setDebugData(tracer.readDirectional(), tracer.directPath());
    status(`RIR ready (${((performance.now() - t0) / 1000).toFixed(1)} s)` +
      (ir.level < 1 ? ` · turned down ${(-20 * Math.log10(ir.level)).toFixed(1)} dB to avoid clipping` : ''));
    $('play').disabled = false;
  } catch (e) {
    console.error(e); status('error: ' + e.message);
  } finally {
    tracing = false;
  }
  if (traceQueued) { traceQueued = false; trace(); }
}

$('trace').onclick = () => trace();
$('auto').onchange = () => requestTrace(0);

let playing = false;
$('play').onclick = async () => {
  if (playing) { aur.stop(); playing = false; return; }
  await aur.play({ loop: true });
  playing = true;
};

// sound dropdown: the test signal, URLs from sounds/index.json, and files opened from disk ("local:<n>")
const localSounds = [];
let soundToken = 0;
async function selectSound() {
  const opt = $('soundSelect').selectedOptions[0], token = ++soundToken;
  try {
    let buf;
    if (opt.value === 'test') {
      buf = makeTestSignal(aur.ctx);
    } else {
      status('loading ' + opt.textContent + '…');
      const src = opt.value.startsWith('local:') ? localSounds[+opt.value.slice(6)] : opt.value;
      let bytes;
      if (typeof src === 'string') {
        const res = await fetch(src);
        if (!res.ok) throw new Error(`${src}: HTTP ${res.status}`);
        bytes = await res.arrayBuffer();
      } else {
        bytes = await src.arrayBuffer();
      }
      buf = await aur.ctx.decodeAudioData(bytes);   // resampled to the context rate
    }
    if (token !== soundToken) return;           // another sound was picked meanwhile
    await aur.loadAnechoic(buf);
    if (playing) await aur.play({ loop: true });
    const secs = Math.round(buf.duration);
    status(`sound: ${opt.textContent} (${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, '0')})`);
  } catch (err) {
    console.error(err);
    if (token === soundToken) status('error: ' + err.message);
  }
}
$('soundSelect').onchange = selectSound;

$('file').onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  localSounds.push(f);
  $('soundSelect').append(new Option(`${f.name} (local file)`, `local:${localSounds.length - 1}`));
  $('soundSelect').value = `local:${localSounds.length - 1}`;
  e.target.value = '';                          // picking the same file again still fires change
  selectSound();
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
  const range = 70, B = BANDS.length;          // dB
  for (let b = 0; b < B; b++) {
    g.strokeStyle = BAND_COLORS[b]; g.beginPath();
    for (let x = 0; x < W; x++) {
      const i0 = Math.floor(x / W * nBins), i1 = Math.max(i0 + 1, Math.floor((x + 1) / W * nBins));
      let m = 0;
      for (let i = i0; i < i1; i++) m = Math.max(m, omni[i * B + b]);
      const db = 10 * Math.log10(m / max + 1e-12);
      const y = H * (1 - (db + range) / range);
      x ? g.lineTo(x, Math.min(H, y)) : g.moveTo(x, Math.min(H, y));
    }
    g.stroke();
  }
  echo = { image: g.getImageData(0, 0, W, H), duration: nBins * tracer.opts.binDuration, range };
}

// hover: crosshair. Vertical line -> arrival time (and the path length sound travels in that time),
// horizontal line -> level in dB relative to the loudest bin of any band (the top of the plot)
let echo = null;                                // { image, duration, range } of the last drawn echogram
$('echo').addEventListener('pointermove', (e) => {
  if (!echo) return;
  const c = e.currentTarget, g = c.getContext('2d'), r = c.getBoundingClientRect();
  const x = THREE.MathUtils.clamp((e.clientX - r.left) / r.width * c.width, 0, c.width - 1);
  const y = THREE.MathUtils.clamp((e.clientY - r.top) / r.height * c.height, 0, c.height - 1);
  const t = x / c.width * echo.duration, db = -echo.range * y / c.height;
  g.putImageData(echo.image, 0, 0);
  g.strokeStyle = '#fff8'; g.lineWidth = 1;
  g.beginPath();
  g.moveTo(Math.round(x) + 0.5, 0); g.lineTo(Math.round(x) + 0.5, c.height);
  g.moveTo(0, Math.round(y) + 0.5); g.lineTo(c.width, Math.round(y) + 0.5);
  g.stroke();

  g.font = '12px system-ui, sans-serif';
  const tag = (text, lx, ly) => {               // text box with its top-left at (lx, ly)
    const w = g.measureText(text).width;
    g.fillStyle = '#000b'; g.fillRect(lx - 3, ly, w + 6, 17);
    g.fillStyle = '#fff'; g.fillText(text, lx, ly + 13);
  };
  const tLabel = `${(t * 1000).toFixed(1)} ms · ${(t * tracer.opts.speedOfSound).toFixed(1)} m`;
  const tw = g.measureText(tLabel).width;
  tag(tLabel, x + 6 + tw < c.width ? x + 6 : x - 6 - tw, 3);                      // flip near the right edge

  const dLabel = `${db.toFixed(1)} dB`, dw = g.measureText(dLabel).width;
  const dx = x < c.width / 2 ? c.width - dw - 6 : 6;                            // opposite side to the pointer
  const dy = y - 20 > 22 ? y - 20 : Math.min(y + 4, c.height - 18);             // above the line, clear of the time tag
  tag(dLabel, dx, dy);
});
$('echo').addEventListener('pointerleave', () => { if (echo) $('echo').getContext('2d').putImageData(echo.image, 0, 0); });

/* ---------------------------------------------------------------- debug view of the directional RIR texture */
const DBG_RANGE = 60;                           // dB shown
const INFERNO = [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]];
const LUT = Array.from({ length: 256 }, (_, i) => {
  const x = i / 255 * (INFERNO.length - 1), k = Math.min(Math.floor(x), INFERNO.length - 2), f = x - k;
  return INFERNO[k].map((c, ch) => Math.round(c + f * (INFERNO[k + 1][ch] - c)));
});
let dbg = null;                                 // { dir, direct } from the last trace

function setDebugData(dir, direct) {
  dbg = { dir, direct };
  $('dbgSlice').max = tracer.opts.numSlices - 1;
  drawRirTexture();
}

// energy per steradian of tile pixel (i, j) in slice s (j = 0 is the bottom row, as in the texture)
function dbgValue(s, i, j, band) {
  const o = tracer.opts, idx = ((Math.floor(s / o.atlasCols) * o.sliceHeight + j) * o.sliceWidth * o.atlasCols
                                + (s % o.atlasCols) * o.sliceWidth + i) * BANDS.length;
  const d = dbg.dir;
  let e = 0;
  if (band >= 0) e = d[idx + band];
  else for (let b = 0; b < BANDS.length; b++) e += d[idx + b];
  return e / tracer.pixelSolidAngle(j);
}

// one panorama: the selected slice, or the sum of all slices (slice < 0)
function dbgPanorama(slice, band) {
  const o = tracer.opts, W = o.sliceWidth, H = o.sliceHeight, out = new Float32Array(W * H);
  const slices = slice < 0 ? [...Array(o.numSlices).keys()] : [slice];
  for (const s of slices) for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) out[j * W + i] += dbgValue(s, i, j, band);
  return out;
}

function drawRirTexture() {
  if ($('debug').hidden || !dbg) return;
  const o = tracer.opts, W = o.sliceWidth, H = o.sliceHeight;
  const band = +$('dbgBand').value, slice = +$('dbgSlice').value;
  const pano = dbgPanorama(slice, band);

  // single slices are normalised to the brightest pixel of ANY slice, so stepping through them shows the decay
  let max = 1e-30;
  if (slice < 0) for (const v of pano) max = Math.max(max, v);
  else for (let s = 0; s < o.numSlices; s++) for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) max = Math.max(max, dbgValue(s, i, j, band));

  const c = $('dbgCanvas'), g = c.getContext('2d');
  c.width = W; c.height = H;
  const img = g.createImageData(W, H);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const db = 10 * Math.log10(pano[j * W + i] / max + 1e-30);
    const p = ((H - 1 - j) * W + i) * 4;       // "up" at the top
    img.data.set(LUT[Math.round(255 * THREE.MathUtils.clamp(1 + db / DBG_RANGE, 0, 1))], p);
    img.data[p + 3] = 255;
  }
  g.putImageData(img, 0, 0);

  // direct sound: should sit on a bright spot (shown when its slice is selected, or in the combined view)
  const { distance, direction: d } = dbg.direct, ds = Math.floor(distance / o.speedOfSound / o.sliceDuration);
  if (slice < 0 || slice === ds) {
    const u = Math.atan2(d.x, -d.z) / (2 * Math.PI) + 0.5, v = Math.asin(THREE.MathUtils.clamp(d.y, -1, 1)) / Math.PI + 0.5;
    g.strokeStyle = '#4cf'; g.lineWidth = 0.75;
    g.beginPath(); g.arc(u * W, (1 - v) * H, 3, 0, 2 * Math.PI); g.stroke();
  }
  dbg.view = { pano, max };
}

$('showDebug').onchange = () => { $('debug').hidden = !$('showDebug').checked; drawRirTexture(); };
$('dbgBand').append(...BANDS.map((f, b) => new Option(f >= 1000 ? `${f / 1000} kHz` : `${f} Hz`, b)));
$('dbgBand').onchange = drawRirTexture;
$('dbgSlice').oninput = () => {
  const s = +$('dbgSlice').value;
  $('dbgSliceOut').textContent = s < 0 ? `all: ${sliceLabel(0, tracer.opts.numSlices)}` : `${s}: ${sliceLabel(s)}`;
  drawRirTexture();
};

function sliceLabel(s, count = 1) {
  const ms = tracer.opts.sliceDuration * 1000;
  return `${(s * ms).toFixed(0)}–${((s + count) * ms).toFixed(0)} ms`;
}

$('dbgCanvas').addEventListener('pointermove', (e) => {
  if (!dbg?.view) return;
  const o = tracer.opts, W = o.sliceWidth, H = o.sliceHeight, { pano, max } = dbg.view;
  const r = e.currentTarget.getBoundingClientRect();
  const i = THREE.MathUtils.clamp(Math.floor((e.clientX - r.left) / r.width * W), 0, W - 1);
  const j = THREE.MathUtils.clamp(H - 1 - Math.floor((e.clientY - r.top) / r.height * H), 0, H - 1);
  const dir = tracer.pixelDirection(i, j);
  const az = THREE.MathUtils.radToDeg(Math.atan2(dir.x, -dir.z)), el = THREE.MathUtils.radToDeg(Math.asin(dir.y));
  const db = 10 * Math.log10(pano[j * W + i] / max + 1e-30);
  $('dbgInfo').textContent = `az ${az.toFixed(0)}°  el ${el.toFixed(0)}°  ${db < -300 ? '−∞' : db.toFixed(1)} dB`;
});

/* ---------------------------------------------------------------- loop */
renderer.setAnimationLoop(() => {
  camera.rotation.set(pitch, yaw, 0);
  renderer.render(scene, camera);
  aur.setListenerFromCamera(camera);            // head tracking
});

init().catch((e) => { console.error(e); status('error: ' + e.message); });
