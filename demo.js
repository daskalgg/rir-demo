import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RIRTracer, makeBVHTextures, BANDS, airAbsorption } from './rirTracer.js';
import { RIRAuralizer } from './rirAuralizer.js';

const $ = (id) => document.getElementById(id);
const status = (s) => { $('status').textContent = s; };

const DEFAULT_MATERIAL = { absorption: [0.20, 0.22, 0.25, 0.27, 0.30, 0.33, 0.35, 0.37], scattering: 0.2 };
const MATERIALS = [
  { id: 'abs0',     label: 'Uniform, no absorption (0)',   absorption: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], scattering: 0.20 },
  { id: 'abs20',    label: 'Uniform, absorption 0.2',      absorption: [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20], scattering: 0.20 },
  { id: 'abs50',    label: 'Uniform, absorption 0.5',      absorption: [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50], scattering: 0.20 },
  { id: 'abs70',    label: 'Uniform, absorption 0.7',      absorption: [0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70], scattering: 0.20 },
  { id: 'abs100',   label: 'Uniform, full absorption (1)', absorption: [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00], scattering: 0.20 },
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
const BAND_COLORS = BANDS.map((_, b) => `hsl(${(b / (BANDS.length - 1)) * 270}, 90%, 62%)`);
let roomMaterial = DEFAULT_MATERIAL;

function meshTriangles(root) {
  root.updateMatrixWorld(true);
  const out = [], v = new THREE.Vector3();
  root.traverse((m) => {
    if (!m.isMesh || !m.geometry.attributes.position) return;
    const pos = m.geometry.attributes.position, index = m.geometry.index;
    const n = index ? index.count : pos.count;
    for (let i = 0; i + 2 < n; i += 3) {
      for (let k = 0; k < 3; k++) {
        v.fromBufferAttribute(pos, index ? index.getX(i + k) : i + k).applyMatrix4(m.matrixWorld);
        out.push(v.x, v.y, v.z);
      }
    }
  });
  return new Float32Array(out);
}

const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.5));
let roomVisual = new THREE.Group();
scene.add(roomVisual);
const roomBox = new THREE.Box3();

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

let drag = null;
canvas.addEventListener('pointerdown', (e) => {
  if (drag) return;
  drag = { id: e.pointerId, x: e.clientX, y: e.clientY };
  canvas.setPointerCapture(e.pointerId); canvas.style.cursor = 'grabbing';
});
const endDrag = (e) => { if (drag?.id === e.pointerId) { drag = null; canvas.style.cursor = 'grab'; } };
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);
canvas.addEventListener('pointermove', (e) => {
  if (drag?.id !== e.pointerId) return;
  yaw -= (e.clientX - drag.x) * 0.004;
  pitch = THREE.MathUtils.clamp(pitch - (e.clientY - drag.y) * 0.004, -1.5, 1.5);
  drag.x = e.clientX; drag.y = e.clientY;
});

function moveBy(target, forward, right, up = 0) {
  const f = new THREE.Vector3(-Math.sin(yaw), 0, -Math.cos(yaw)), r = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
  const obj = target === 'source' ? sourceMesh : camera;
  obj.position.addScaledVector(f, forward).addScaledVector(r, right);
  obj.position.y += up;
  obj.position.clamp(roomBox.min.clone().addScalar(0.3), roomBox.max.clone().subScalar(0.3));
  sourcePos.copy(sourceMesh.position);
}

const KEYS = { w: ['listener', 1, 0, 0], s: ['listener', -1, 0, 0], d: ['listener', 0, 1, 0], a: ['listener', 0, -1, 0],
  e: ['listener', 0, 0, 1], q: ['listener', 0, 0, -1],
  ArrowUp: ['source', 1, 0, 0], ArrowDown: ['source', -1, 0, 0], ArrowRight: ['source', 0, 1, 0], ArrowLeft: ['source', 0, -1, 0],
  PageUp: ['source', 0, 0, 1], PageDown: ['source', 0, 0, -1],
  'Shift+ArrowUp': ['source', 0, 0, 1], 'Shift+ArrowDown': ['source', 0, 0, -1] };
addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  const move = (e.shiftKey && KEYS['Shift+' + key]) || KEYS[key];
  if (e.target.closest?.('input, select, textarea') || roomBox.isEmpty() || !move) return;
  e.preventDefault();
  const [target, f, r, u] = move, step = 0.25;
  moveBy(target, f * step, r * step, u * step);
  requestTrace();
});

const PAD_SPEED = 1.5;
let padTarget = 'listener', padDir = null;
for (const b of document.querySelectorAll('#pad [data-target]')) {
  b.onclick = () => {
    padTarget = b.dataset.target;
    for (const o of document.querySelectorAll('#pad [data-target]')) o.classList.toggle('on', o === b);
  };
}
for (const b of document.querySelectorAll('#pad [data-dir]')) {
  b.addEventListener('pointerdown', (e) => {
    b.setPointerCapture(e.pointerId); b.classList.add('held');
    padDir = b.dataset.dir.split(',').map(Number);
  });
  const release = () => { b.classList.remove('held'); padDir = null; };
  b.addEventListener('pointerup', release);
  b.addEventListener('pointercancel', release);
  b.addEventListener('contextmenu', (e) => e.preventDefault());
}

const aur = new RIRAuralizer({ crossfade: 0.4 });
let tracer = null, traceFrag = null;

async function init() {
  traceFrag = await (await fetch('./rirTrace.frag')).text();
  aur.loadAnechoic(makeTestSignal(aur.ctx));

  let sounds = [];
  try { sounds = await (await fetch('./sounds/index.json')).json(); } catch (e) { console.warn('no sounds/index.json', e); }
  for (const snd of sounds) $('soundSelect').append(new Option(snd.name, 'sounds/' + snd.file));

  await applyRenderMode();

  let rooms = [];
  try { rooms = await (await fetch('./models/index.json')).json(); } catch (e) { console.warn('no models/index.json', e); }
  const sel = $('roomSelect');
  for (const r of rooms) sel.append(new Option(r.name, 'models/' + r.file));
  const param = new URLSearchParams(location.search).get('room');
  if (param && ![...sel.options].some((o) => o.value === param)) sel.append(new Option(param.split('/').pop(), param));
  $('room').disabled = false;
  if (!sel.options.length) { status('no rooms in models/ — open a glTF file'); return; }
  const DEFAULT_ROOM = 'models/cube-10m.glb';
  sel.value = param || ([...sel.options].some((o) => o.value === DEFAULT_ROOM) ? DEFAULT_ROOM : sel.options[0].value);
  sel.disabled = false;
  await selectRoom();
}

let loadToken = 0;
const startPos = { source: new THREE.Vector3(), listener: new THREE.Vector3() };
function resetPositions() {
  sourcePos.copy(startPos.source);
  sourceMesh.position.copy(sourcePos);
  camera.position.copy(startPos.listener);
  yaw = Math.atan2(camera.position.x - sourcePos.x, camera.position.z - sourcePos.z);
  pitch = 0;
}
$('reset').addEventListener('click', () => { if (roomBox.isEmpty()) return; resetPositions(); requestTrace(0); });

async function loadRoom(src, name) {
  const token = ++loadToken;
  status('loading ' + name + '…');
  const loader = new GLTFLoader();
  const gltf = typeof src === 'string'
    ? await loader.loadAsync(src)
    : await loader.parseAsync(await src.arrayBuffer(), '');
  if (token !== loadToken) return;
  const t0 = performance.now();
  const bvh = makeBVHTextures(meshTriangles(gltf.scene));
  const buildMs = performance.now() - t0;

  if (!tracer) {
    tracer = new RIRTracer(renderer, traceFrag, {
      bvh, camera,
      source: sourcePos,
      cameraRelative: false,
      rayTexSize: RAY_TEX, rays: rayCount(), maxBounces: +$('bounces').value,
      absorption: DEFAULT_MATERIAL.absorption, scattering: DEFAULT_MATERIAL.scattering,
      airAbsorption: airCoefficients(),
    });
    $('trace').disabled = false;
  } else {
    tracer.setGeometry(bvh);
  }
  const { absorption: abs, scattering: sc } = gltf.scene.userData;
  const customAbs = Array.isArray(abs) && abs.length === BANDS.length && abs.every((a) => a >= 0 && a <= 1);
  if (abs !== undefined && !customAbs) console.warn(`${name}: extras.absorption needs ${BANDS.length} values in 0..1 (${BANDS.join(', ')} Hz)`, abs);
  const customSc = typeof sc === 'number' && sc >= 0 && sc <= 1;
  roomMaterial = {
    absorption: customAbs ? abs : DEFAULT_MATERIAL.absorption,
    scattering: customSc ? sc : DEFAULT_MATERIAL.scattering,
  };
  selectMaterial('room');

  scene.remove(roomVisual);
  roomVisual = gltf.scene;
  scene.add(roomVisual);
  roomBox.setFromObject(roomVisual);
  const rel = (x, y, z) => new THREE.Vector3(x, y, z).multiply(roomBox.getSize(new THREE.Vector3())).add(roomBox.min);
  const marker = (name) => {
    let found = null;
    gltf.scene.traverse((o) => { if (!found && o.name.toLowerCase() === name) found = o; });
    return found && found.getWorldPosition(new THREE.Vector3());
  };
  startPos.source.copy(marker('source') || rel(0.75, 0.47, 0.7));
  startPos.listener.copy(marker('listener') || rel(0.25, 0.53, 0.3));
  resetPositions();
  camera.far = Math.max(50, 2 * roomBox.getSize(new THREE.Vector3()).length());
  camera.updateProjectionMatrix();

  status(`${name}: ${bvh.triangles} triangles, BVH of ${bvh.nodeCount} nodes (depth ${bvh.depth}, ${buildMs.toFixed(0)} ms)` +
    (customAbs ? `, absorption ${abs.join(' / ')}` : '') + (customSc ? `, scattering ${sc}` : ''));
  requestTrace(0);
}

const RAY_TEX = 128, RAY_STEP = RAY_TEX * RAY_TEX / 2;
const RAY_COUNTS = [2048, 4096, ...Array.from({ length: 64 }, (_, i) => (i + 1) * RAY_STEP)];
$('rays').max = RAY_COUNTS.length - 1;
const rayCount = () => RAY_COUNTS[+$('rays').value];
function updateSettings() {
  const rays = rayCount(), bounces = +$('bounces').value;
  $('raysOut').textContent = rays.toLocaleString();
  $('bouncesOut').textContent = bounces;
  if (tracer) { tracer.setRayCount(rays); tracer.setMaxBounces(bounces); }
}
$('rays').oninput = $('bounces').oninput = updateSettings;
$('rays').onchange = $('bounces').onchange = () => requestTrace();
updateSettings();

function airCoefficients() {
  return $('air').checked ? airAbsorption(+$('airTemp').value, +$('airRH').value) : BANDS.map(() => 0);
}
function updateAir() {
  $('airTempOut').textContent = `${$('airTemp').value} °C`;
  $('airRHOut').textContent = `${$('airRH').value} %`;
  $('airTemp').disabled = $('airRH').disabled = !$('air').checked;
  if (tracer) tracer.setAirAbsorption(airCoefficients());
}
$('airTemp').oninput = $('airRH').oninput = updateAir;
$('air').onchange = $('airTemp').onchange = $('airRH').onchange = () => { updateAir(); requestTrace(); };
updateAir();

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
  e.target.value = '';
  selectRoom();
};

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
    await new Promise((r) => setTimeout(r, 30));
    const t0 = performance.now();
    const { bounces } = tracer.run();
    status('synthesizing impulse response…');
    const omni = tracer.readOmni(), directional = tracer.readDirectional();
    const ir = await aur.setRIRFromTracer(tracer, { omni, directional });
    drawEchogram(omni, tracer.opts.numBins);
    const direct = tracer.directPath();
    setDebugData(directional, direct);
    status("RIR ready");
    $('play').disabled = false;
    $('export').disabled = false;
  } catch (e) {
    console.error(e); status('error: ' + e.message);
  } finally {
    tracing = false;
  }
  if (traceQueued) { traceQueued = false; trace(); }
}

$('trace').onclick = () => trace();
$('auto').onchange = () => requestTrace(0);

async function applyRenderMode() {
  $('render').disabled = true;
  try {
    await aur.setOrder(+$('render').value);
    status(`rendering: ${$('render').selectedOptions[0].textContent}`);
  } catch (err) {
    console.error(err);
    status('ambisonic renderer unavailable: ' + (err.message || err));
  }
  $('render').disabled = false;
}
$('render').onchange = applyRenderMode;

$('export').onclick = () => {
  const { wav, order } = aur.exportAmbisonicWav();
  const room = ($('roomSelect').selectedOptions[0]?.textContent || 'room').replace(/[^\w.-]+/g, '_');
  const url = URL.createObjectURL(new Blob([wav], { type: 'audio/wav' }));
  const a = Object.assign(document.createElement('a'), { href: url, download: `${room}-ARIR-${order}OA-AmbiX-${aur.ctx.sampleRate}Hz.wav` });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  status(`exported ${a.download} (${(order + 1) ** 2} channels, ACN / SN3D, absolute level)`);
};

$('bypass').onchange = () => aur.setBypass($('bypass').checked);
aur.setBypass($('bypass').checked);

let playing = false;
$('play').onclick = async () => {
  if (playing) { aur.stop(); playing = false; return; }
  await aur.play({ loop: true });
  playing = true;
};

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
      buf = await aur.ctx.decodeAudioData(bytes);
    }
    if (token !== soundToken) return;
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
  e.target.value = '';
  selectSound();
};

function makeTestSignal(ctx) {
  const fs = ctx.sampleRate, buf = ctx.createBuffer(1, fs * 6, fs), d = buf.getChannelData(0);
  [0.2, 1.7, 3.2, 4.7].forEach((t0, idx) => {
    const n0 = Math.floor(t0 * fs);
    for (let i = 0; i < 0.15 * fs; i++) {
      const env = Math.exp(-i / (0.008 * fs));
      d[n0 + i] += (idx % 2 === 0)
        ? (Math.random() * 2 - 1) * env * 0.8
        : Math.sin(2 * Math.PI * 800 * i / fs) * Math.exp(-i / (0.03 * fs)) * 0.6;
    }
  });
  return buf;
}

function drawEchogram(omni, nBins) {
  const c = $('echo'), g = c.getContext('2d'), W = c.width, H = c.height;
  g.clearRect(0, 0, W, H);
  let max = 1e-12;
  for (let i = 0; i < omni.length; i++) max = Math.max(max, omni[i]);
  const range = 70, B = BANDS.length;
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

let echo = null;
function echoHover(e) {
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
  const tag = (text, lx, ly) => {
    const w = g.measureText(text).width;
    g.fillStyle = '#000b'; g.fillRect(lx - 3, ly, w + 6, 17);
    g.fillStyle = '#fff'; g.fillText(text, lx, ly + 13);
  };
  const tLabel = `${(t * 1000).toFixed(1)} ms · ${(t * tracer.opts.speedOfSound).toFixed(1)} m`;
  const tw = g.measureText(tLabel).width;
  tag(tLabel, x + 6 + tw < c.width ? x + 6 : x - 6 - tw, 3);

  const dLabel = `${db.toFixed(1)} dB`, dw = g.measureText(dLabel).width;
  const dx = x < c.width / 2 ? c.width - dw - 6 : 6;
  const dy = y - 20 > 22 ? y - 20 : Math.min(y + 4, c.height - 18);
  tag(dLabel, dx, dy);
}
$('echo').addEventListener('pointerdown', echoHover);
$('echo').addEventListener('pointermove', echoHover);
$('echo').addEventListener('pointerleave', (e) => {
  if (echo && e.pointerType !== 'touch') $('echo').getContext('2d').putImageData(echo.image, 0, 0);
});

const DBG_RANGE = 60;
const dbgMaxDistance = () => tracer.opts.numSlices * tracer.opts.sliceDuration * tracer.opts.speedOfSound;
let dbg = null;

function setDebugData(dir, direct) {
  dbg = { dir, direct };
  $('dbgSlice').max = tracer.opts.numSlices - 1;
  drawRirTexture();
}

function dbgValue(s, i, j, band) {
  const o = tracer.opts, idx = ((Math.floor(s / o.atlasCols) * o.sliceHeight + j) * o.sliceWidth * o.atlasCols
                                + (s % o.atlasCols) * o.sliceWidth + i) * BANDS.length;
  const d = dbg.dir;
  let e = 0;
  if (band >= 0) e = d[idx + band];
  else for (let b = 0; b < BANDS.length; b++) e += d[idx + b];
  return e / tracer.pixelSolidAngle(j);
}

function dbgPanorama(slice, band) {
  const o = tracer.opts, W = o.sliceWidth, H = o.sliceHeight;
  const energy = new Float32Array(W * H), distance = new Float32Array(W * H);
  const slices = slice < 0 ? [...Array(o.numSlices).keys()] : [slice];
  for (const s of slices) {
    const d = (s + 0.5) * o.sliceDuration * o.speedOfSound;
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const e = dbgValue(s, i, j, band);
      energy[j * W + i] += e;
      distance[j * W + i] += e * d;
    }
  }
  for (let p = 0; p < W * H; p++) distance[p] = energy[p] > 0 ? distance[p] / energy[p] : 0;
  return { energy, distance };
}

function drawRirTexture() {
  if ($('debug').hidden || !dbg) return;
  const o = tracer.opts, W = o.sliceWidth, H = o.sliceHeight;
  const band = +$('dbgBand').value, slice = +$('dbgSlice').value;
  const pano = dbgPanorama(slice, band), maxDist = dbgMaxDistance();

  let max = 1e-30;
  if (slice < 0) for (const v of pano.energy) max = Math.max(max, v);
  else for (let s = 0; s < o.numSlices; s++) for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) max = Math.max(max, dbgValue(s, i, j, band));

  const c = $('dbgCanvas'), g = c.getContext('2d');
  c.width = W; c.height = H;
  const img = g.createImageData(W, H);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const q = j * W + i, p = ((H - 1 - j) * W + i) * 4;
    const level = THREE.MathUtils.clamp(1 + 10 * Math.log10(pano.energy[q] / max + 1e-30) / DBG_RANGE, 0, 1);
    const red = Math.round(255 * level);
    img.data[p] = red;
    img.data[p + 1] = red > 0 ? Math.round(255 * Math.min(1, pano.distance[q] / maxDist)) : 0;
    img.data[p + 2] = 0;
    img.data[p + 3] = 255;
  }
  g.putImageData(img, 0, 0);

  const { distance, direction: d, visible } = dbg.direct, ds = Math.floor(distance / o.speedOfSound / o.sliceDuration);
  if (visible && (slice < 0 || slice === ds)) {
    const u = Math.atan2(d.x, -d.z) / (2 * Math.PI) + 0.5, v = Math.asin(THREE.MathUtils.clamp(d.y, -1, 1)) / Math.PI + 0.5;
    g.strokeStyle = '#fff'; g.lineWidth = 0.75;
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

function dbgHover(e) {
  if (!dbg?.view) return;
  const o = tracer.opts, W = o.sliceWidth, H = o.sliceHeight, { pano, max } = dbg.view;
  const r = e.currentTarget.getBoundingClientRect();
  const i = THREE.MathUtils.clamp(Math.floor((e.clientX - r.left) / r.width * W), 0, W - 1);
  const j = THREE.MathUtils.clamp(H - 1 - Math.floor((e.clientY - r.top) / r.height * H), 0, H - 1);
  const dir = tracer.pixelDirection(i, j);
  const az = THREE.MathUtils.radToDeg(Math.atan2(dir.x, -dir.z)), el = THREE.MathUtils.radToDeg(Math.asin(dir.y));
  const q = j * W + i, db = 10 * Math.log10(pano.energy[q] / max + 1e-30);
  $('dbgInfo').textContent = `az ${az.toFixed(0)}°  el ${el.toFixed(0)}°  energy ${db < -300 ? '−∞' : db.toFixed(1)} dB` +
    (pano.energy[q] > 0 ? `  distance ${pano.distance[q].toFixed(1)} m (${(pano.distance[q] / o.speedOfSound * 1000).toFixed(1)} ms)` : '');
}
$('dbgCanvas').addEventListener('pointerdown', dbgHover);
$('dbgCanvas').addEventListener('pointermove', dbgHover);

const narrow = matchMedia('(max-width: 820px), (pointer: coarse)');
function setPanelOpen(open) {
  $('more').hidden = !open;
  document.body.classList.toggle('panel-open', open);
  $('toggleUi').textContent = open ? 'Hide' : 'Settings';
  $('toggleUi').setAttribute('aria-expanded', open);
}
$('toggleUi').onclick = () => setPanelOpen($('more').hidden);
function placeDebug() {
  (narrow.matches ? $('more') : document.body).insertBefore($('debug'), narrow.matches ? null : $('pad'));
}
narrow.addEventListener('change', placeDebug);
placeDebug();
setPanelOpen(!narrow.matches);

let lastFrame = 0;
renderer.setAnimationLoop((t) => {
  const dt = Math.min((t - lastFrame) / 1000, 0.1); lastFrame = t;
  if (padDir && !roomBox.isEmpty()) {
    moveBy(padTarget, ...padDir.map((v) => v * PAD_SPEED * dt));
    requestTrace();
  }
  camera.rotation.set(pitch, yaw, 0);
  renderer.render(scene, camera);
  aur.setListenerFromCamera(camera);
});

init().catch((e) => { console.error(e); status('error: ' + e.message); });
