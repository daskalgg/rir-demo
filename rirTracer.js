// rirTracer.js  — Three.js (r165+, WebGL2) driver for rirTrace.frag
//
// The listener sits at the camera position. Arrival directions are expressed in the
// camera's own frame (turn the camera -> the image scrolls), or in world frame if
// cameraRelative:false.
//
// Energy is tracked in 8 octave bands (BANDS, 125 Hz .. 16 kHz). The GPU targets are RGBA32F, so each holds two
// groups of 4 bands side by side; readOmni() / readDirectional() return them interleaved, 8 floats per entry.
//   tracer.directional : atlas of equirectangular images, one tile per time slice
//                        tile i -> col = i % cols, row = floor(i / cols)   (row 0 = bottom of texture)
//                        bands 0-3 in the bottom half of the texture, bands 4-7 stacked above
//                        each pixel = total ENERGY arriving from that pixel's patch of the sphere
//                        (divide by pixel solid angle for radiance; see pixelSolidAngle())
//   tracer.omni        : fine-resolution energy histogram (sum over all directions), sized for maxDuration and
//                        wrapped into rows of OMNI_W bins: bin i of band group g at (i % OMNI_W, g * omniRows +
//                        floor(i / OMNI_W)). Only the first numBins bins (this run's window, see plan()) are filled.
//
// The window adapts to the room: before each trace, plan() casts a few CPU rays from the source to measure the mean
// free path (and how many rays leave an open room), and from that plus the absorption works out when the slowest
// band has decayed by minEnergyDb (Eyring-style). That sets the window (<= maxDuration) and the bounces traced.
//
// Usage:
//   const frag = await (await fetch('./rirTrace.frag')).text();
//   const bvh = makeBVHTextures(triangles);         // Float32Array, 9 floats (3 vertices) per triangle
//   const tracer = new RIRTracer(renderer, frag, {
//       bvh, camera,
//       source: new THREE.Vector3(2, 1.5, 3),
//   });
//   tracer.run();                                   // call again whenever camera/source moves
//   material.map = tracer.directional;              // or sample a tile via tracer.sliceTransform(i)

import * as THREE from 'three';

/** Octave band centre frequencies (Hz). */
export const BANDS = [125, 250, 500, 1000, 2000, 4000, 8000, 16000];

/** Energy attenuation of air per band (BANDS), 1/m, after ISO 9613-1 (pure tone at each band centre, sea-level
 *  pressure). tempC in °C, humidity = relative humidity in %. 20 °C / 50 % gives the default airAbsorption. */
export function airAbsorption(tempC, humidity) {
  const T = tempC + 273.15, T0 = 293.15, T01 = 273.16;
  const h = humidity * 10 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151);   // molar concentration of water vapour, %
  const frO = 24 + 4.04e4 * h * (0.02 + h) / (0.391 + h);              // relaxation frequencies of O2 / N2, Hz
  const frN = (T / T0) ** -0.5 * (9 + 280 * h * Math.exp(-4.170 * ((T / T0) ** (-1 / 3) - 1)));
  return BANDS.map((f) => {
    const dBPerM = 8.686 * f * f * (1.84e-11 * (T / T0) ** 0.5 + (T / T0) ** -2.5 * (
      0.01275 * Math.exp(-2239.1 / T) / (frO + f * f / frO) + 0.1068 * Math.exp(-3352 / T) / (frN + f * f / frN)));
    return dBPerM / (10 * Math.LOG10E);         // dB -> energy (Neper-like) coefficient
  });
}

const FULLSCREEN_VERT = /* glsl */`
void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

// shared by both accumulate vertex shaders
const ACCUM_COMMON = /* glsl */`
uniform sampler2D uHit;                  // [u, v, pathLength, bounces]  (bounces < 0 -> invalid)
uniform float uSpeedOfSound;
uniform vec4  uAbsorption[2];            // energy absorption per band (0..1): bands 0-3, 4-7
uniform vec4  uAirAbsorption[2];         // 1/m per band (energy)
uniform int   uGroup;                    // band group accumulated by this pass (0 = bands 0-3, 1 = bands 4-7)
uniform float uRayWeight;                // 1 / (totalRays * PI * listenerRadius^2)
out vec4 vEnergy;

vec4 fetchHit() {
    ivec2 sz = textureSize(uHit, 0);
    return texelFetch(uHit, ivec2(gl_VertexID % sz.x, gl_VertexID / sz.x), 0);
}
vec4 bandEnergy(vec4 h) {
    vec4 refl = pow(max(vec4(1.0) - uAbsorption[uGroup], vec4(1e-6)), vec4(h.w));
    return refl * exp(-uAirAbsorption[uGroup] * h.z) * uRayWeight;
}
void clipPoint() { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vEnergy = vec4(0.0); gl_PointSize = 1.0; }
`;

// 1D fine time histogram (all directions summed)
const OMNI_VERT = ACCUM_COMMON + /* glsl */`
uniform float uBinDuration;
uniform int   uNumBins;                  // bins in this run's window (<= the target's capacity)
uniform int   uOmniWidth;                // bins per row
uniform int   uOmniRows;                 // rows per band group; group 1 sits above group 0
void main() {
    vec4 h = fetchHit();
    float bin = floor(h.z / uSpeedOfSound / uBinDuration);
    if (h.w < 0.0 || bin >= float(uNumBins)) { clipPoint(); return; }
    vEnergy = bandEnergy(h);
    gl_PointSize = 1.0;
    int i = int(bin);
    vec2 p = vec2(float(i % uOmniWidth), float(uGroup * uOmniRows + i / uOmniWidth)) + 0.5;
    gl_Position = vec4(p / vec2(float(uOmniWidth), float(2 * uOmniRows)) * 2.0 - 1.0, 0.0, 1.0);
}
`;

// directional atlas: pixel = tile(timeSlice) + equirect(u,v)
const DIR_VERT = ACCUM_COMMON + /* glsl */`
uniform float uSliceDuration;
uniform int   uNumSlices;
uniform int   uCols;
uniform vec2  uSliceSize;                // pixels per tile (w, h)
uniform vec2  uAtlasSize;                // pixels (w, h) of ONE band group; the target is twice as high
void main() {
    vec4 h = fetchHit();
    float slice = floor(h.z / uSpeedOfSound / uSliceDuration);
    if (h.w < 0.0 || slice >= float(uNumSlices)) { clipPoint(); return; }
    int s = int(slice);
    vec2 tileOrigin = vec2(float(s % uCols), float(s / uCols)) * uSliceSize + vec2(0.0, float(uGroup) * uAtlasSize.y);
    vec2 inTile = min(floor(h.xy * uSliceSize), uSliceSize - 1.0);
    vec2 p = tileOrigin + inTile + 0.5;
    vEnergy = bandEnergy(h);
    gl_PointSize = 1.0;
    gl_Position = vec4(p / (uAtlasSize * vec2(1.0, 2.0)) * 2.0 - 1.0, 0.0, 1.0);
}
`;

const ACCUM_FRAG = /* glsl */`
in vec4 vEnergy;
layout(location = 0) out vec4 outColor;
void main() { outColor = vEnergy; }
`;

const floatTarget = (w, h, count = 1) => new THREE.WebGLRenderTarget(w, h, {
  count,
  type: THREE.FloatType, format: THREE.RGBAFormat,
  minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
  depthBuffer: false, stencilBuffer: false, generateMipmaps: false,
});

export class RIRTracer {
  constructor(renderer, traceFragSource, opts) {
    this.renderer = renderer;
    if (!renderer.capabilities.isWebGL2) throw new Error('WebGL2 required');
    if (!renderer.extensions.has('EXT_color_buffer_float')) throw new Error('EXT_color_buffer_float required');
    if (!renderer.extensions.has('EXT_float_blend')) throw new Error('EXT_float_blend required (additive float blending)');

    const o = this.opts = Object.assign({
      camera: null,               // listener = this camera (position + orientation)
      cameraRelative: true,       // false -> arrival directions in world axes
      listener: new THREE.Vector3(),   // used only when no camera is given
      source: new THREE.Vector3(),
      rayTexSize: 128,            // 128x128 = 16384 rays per batch
      batches: 8,                 // total rays = batches * rayTexSize^2, unless `rays` is given
      rays: undefined,            // total rays, in whole rows of rayTexSize (the last batch may be partial)
      maxBounces: 2000,           // upper limit on the bounces of one run (see plan(), setMaxBounces())
      minEnergyDb: 80,            // the window ends once every band has decayed this much (see plan())
      speedOfSound: 343,
      listenerRadius: 0.15,
      scattering: 0.2,
      absorption:    [0.10, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25],   // per band (BANDS), energy
      // energy attenuation of air, 1/m (ISO 9613-1, 20 °C, 50 % RH)
      airAbsorption: [1.0e-4, 3.0e-4, 6.3e-4, 1.1e-3, 2.3e-3, 6.8e-3, 2.4e-2, 8.6e-2],
      // fine omni histogram
      maxDuration: 3, binDuration: 0.0005,                        // longest window (s), histogram bin (s)
      // directional atlas
      sliceWidth: 128, sliceHeight: 64,                           // equirect tile (2:1)
      numSlices: 32, sliceDuration: 0.004,                        // 32 x 4 ms = 128 ms
      atlasCols: 8,
    }, opts);
    o.atlasRows = Math.ceil(o.numSlices / o.atlasCols);
    o.maxBins = Math.ceil(o.maxDuration / o.binDuration);
    o.omniRows = Math.ceil(o.maxBins / OMNI_W);
    o.numBins = o.maxBins;          // bins of the last run's window, set by run()
    if (o.rays === undefined) o.rays = o.batches * o.rayTexSize ** 2;
    o.bands = BANDS;                // fixed: the shaders accumulate exactly two groups of 4 bands

    this.read = floatTarget(o.rayTexSize, o.rayTexSize, 3);     // state0, state1, hit
    this.write = floatTarget(o.rayTexSize, o.rayTexSize, 3);
    this.omniTarget = floatTarget(OMNI_W, 2 * o.omniRows);                              // 2 band groups
    this.dirTarget = floatTarget(o.sliceWidth * o.atlasCols, 2 * o.sliceHeight * o.atlasRows);
    this.omni = this.omniTarget.texture;
    this.directional = this.dirTarget.texture;

    this._listenerPos = new THREE.Vector3().copy(o.listener);
    this._worldToListener = new THREE.Matrix3();

    // ---- trace pass (full-screen quad)
    this.traceMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: FULLSCREEN_VERT,
      fragmentShader: traceFragSource,
      uniforms: {
        uBVHNodes: { value: o.bvh.nodes }, uBVHTris: { value: o.bvh.tris },
        uState0: { value: null }, uState1: { value: null },
        uInit: { value: 1 }, uSeed: { value: 0 }, uRayTexSize: { value: o.rayTexSize },
        uSource: { value: o.source },
        uListener: { value: this._listenerPos },
        uWorldToListener: { value: this._worldToListener },
        uListenerRadius: { value: o.listenerRadius },
        uMaxPathLength: { value: 0 },             // set by run() from plan()
        uScattering: { value: o.scattering },
      },
      depthTest: false, depthWrite: false, blending: THREE.NoBlending,
    });
    this.quadScene = new THREE.Scene();
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.traceMat);
    quad.frustumCulled = false;
    this.quadScene.add(quad);

    // ---- accumulate passes (one point per ray, additive blending)
    const nRays = o.rayTexSize * o.rayTexSize;
    const groups = (a) => [new THREE.Vector4().fromArray(a, 0), new THREE.Vector4().fromArray(a, 4)];
    const common = {
      uHit: { value: null },
      uSpeedOfSound: { value: o.speedOfSound },
      uAbsorption: { value: groups(o.absorption) },
      uAirAbsorption: { value: groups(o.airAbsorption) },
      uGroup: { value: 0 },
      uRayWeight: { value: 0 },                 // set by setRayCount()
    };
    const additive = {
      glslVersion: THREE.GLSL3, fragmentShader: ACCUM_FRAG,
      blending: THREE.CustomBlending, blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
      blendSrcAlpha: THREE.OneFactor, blendDstAlpha: THREE.OneFactor,
      depthTest: false, depthWrite: false, transparent: true,
    };
    this.omniMat = new THREE.ShaderMaterial({
      ...additive, vertexShader: OMNI_VERT,
      uniforms: {
        ...common, uBinDuration: { value: o.binDuration }, uNumBins: { value: o.numBins },
        uOmniWidth: { value: OMNI_W }, uOmniRows: { value: o.omniRows },
      },
    });
    this.dirMat = new THREE.ShaderMaterial({
      ...additive, vertexShader: DIR_VERT,
      uniforms: {
        ...common,
        uSliceDuration: { value: o.sliceDuration }, uNumSlices: { value: o.numSlices },
        uCols: { value: o.atlasCols },
        uSliceSize: { value: new THREE.Vector2(o.sliceWidth, o.sliceHeight) },
        uAtlasSize: { value: new THREE.Vector2(o.sliceWidth * o.atlasCols, o.sliceHeight * o.atlasRows) },
      },
    });

    const g = this._rayGeometry = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(nRays * 3), 3));
    const mkScene = (mat) => {
      const s = new THREE.Scene();
      const p = new THREE.Points(g, mat);
      p.frustumCulled = false;
      s.add(p);
      return s;
    };
    this.omniScene = mkScene(this.omniMat);
    this.dirScene = mkScene(this.dirMat);

    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.setRayCount(o.rays);
  }

  /** Replaces the room geometry (from makeBVHTextures) and disposes the previous textures. Takes effect on the next
   *  run(). */
  setGeometry(bvh) {
    const u = this.traceMat.uniforms;
    u.uBVHNodes.value.dispose(); u.uBVHTris.value.dispose();
    u.uBVHNodes.value = bvh.nodes; u.uBVHTris.value = bvh.tris;
    this.opts.bvh = bvh;
  }

  /** Total rays, rounded down to whole rows of the ray texture (>= one row). Runs as full rayTexSize^2 batches plus
   *  one partial batch for the remainder. Takes effect on the next run(). */
  setRayCount(rays) {
    const o = this.opts, row = o.rayTexSize;
    o.rays = Math.max(row, Math.floor(rays / row) * row);
    o.batches = Math.ceil(o.rays / (row * row));
    // uniform object is shared by both accumulate materials
    this.omniMat.uniforms.uRayWeight.value = 1 / (o.rays * Math.PI * o.listenerRadius ** 2);
  }

  /** Total rays = batches * rayTexSize^2. Takes effect on the next run(). */
  setBatches(batches) { this.setRayCount(batches * this.opts.rayTexSize ** 2); }

  /**
   * Window and bounce budget for the next run(), from a CPU probe of the room (rays cast from the source through
   * the BVH, reflecting diffusely): the mean free path, and the fraction of segments that leave an open room.
   * Energy then decays per metre by  -ln(1 - a_b) / mfp  (walls)  - ln(1 - escape) / mfp  (open sky)  + air_b;
   * the window ends where the slowest band has lost minEnergyDb (>= the directional window, <= maxDuration).
   * Bounces cover that path at the mean free path with 50 % margin (shorter-than-average paths need more), but stop
   * where the walls alone have taken minEnergyDb from every band.
   * @returns { duration (s), numBins, bounces, mfp (m), escape (0..1) }
   */
  plan() {
    const o = this.opts, c = o.speedOfSound, minDuration = o.numSlices * o.sliceDuration;
    const { mfp, escape } = probeRoom(o.bvh, o.source);
    const plan = (duration, bounces) => ({
      duration, numBins: Math.min(o.maxBins, Math.ceil(duration / o.binDuration)), bounces, mfp, escape,
    });
    if (!(mfp < Infinity)) return plan(minDuration, 1);   // nothing to reflect off: direct segment only
    const lnKeep = (k) => Math.log(Math.max(k, 1e-6));
    let slowest = Infinity;                       // energy decay of the slowest band, nepers per metre
    for (let b = 0; b < o.bands.length; b++) {
      slowest = Math.min(slowest, -(lnKeep(1 - o.absorption[b]) + lnKeep(1 - escape)) / mfp + o.airAbsorption[b]);
    }
    const nepers = o.minEnergyDb / 10 * Math.LN10;
    const duration = Math.min(o.maxDuration, Math.max(minDuration, nepers / slowest / c));
    const keep = Math.max(...o.absorption.map((a) => 1 - a));
    const wallBounces = keep >= 1 ? Infinity : Math.floor(nepers / -lnKeep(keep)) + 1;
    const bounces = Math.min(o.maxBounces, wallBounces, Math.ceil(1.5 * c * duration / mfp) + 1);
    return plan(duration, Math.max(1, bounces));
  }

  /** Upper limit on the bounces plan() may pick: fewer means a faster trace, but rays may then stop before the
   *  window ends (its tail comes out too weak). Takes effect on the next run(). */
  setMaxBounces(n) { this.opts.maxBounces = n; }

  /** Energy absorption per octave band (BANDS), 0..1, as [8]. Takes effect on the next run(). */
  setAbsorption(a) {
    this.opts.absorption = [...a];
    const [lo, hi] = this.omniMat.uniforms.uAbsorption.value;   // shared by both accumulate materials
    lo.fromArray(a, 0); hi.fromArray(a, 4);
  }

  /** Energy attenuation of air per band (BANDS), 1/m, as [8] (see airAbsorption()). Takes effect on the next run(). */
  setAirAbsorption(a) {
    this.opts.airAbsorption = [...a];
    const [lo, hi] = this.omniMat.uniforms.uAirAbsorption.value;   // shared by both accumulate materials
    lo.fromArray(a, 0); hi.fromArray(a, 4);
  }

  /** Fraction of reflections that are diffuse (0 = mirror, 1 = Lambert). Takes effect on the next run(). */
  setScattering(s) { this.opts.scattering = s; this.traceMat.uniforms.uScattering.value = s; }

  /** Copy camera position / orientation into the listener uniforms. */
  updateListener() {
    const o = this.opts;
    if (o.camera) {
      o.camera.updateWorldMatrix(true, false);
      this._listenerPos.setFromMatrixPosition(o.camera.matrixWorld);
      if (o.cameraRelative) {
        const q = o.camera.getWorldQuaternion(new THREE.Quaternion()).invert();
        this._worldToListener.setFromMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(q));
      } else {
        this._worldToListener.identity();
      }
    } else {
      this._worldToListener.identity();
    }
  }

  _tracePass(init, seed) {
    const u = this.traceMat.uniforms, r = this.renderer;
    u.uInit.value = init ? 1 : 0;
    u.uSeed.value = seed;
    u.uState0.value = this.read.textures[0];
    u.uState1.value = this.read.textures[1];
    r.setRenderTarget(this.write);
    r.render(this.quadScene, this.camera);
    [this.read, this.write] = [this.write, this.read];   // newest state is now in this.read
  }

  _accumulate() {
    const hit = this.read.textures[2], r = this.renderer;
    this.omniMat.uniforms.uHit.value = hit;
    this.dirMat.uniforms.uHit.value = hit;
    for (let g = 0; g < 2; g++) {                        // band groups 0-3, 4-7 (uGroup is shared)
      this.omniMat.uniforms.uGroup.value = g;
      r.setRenderTarget(this.omniTarget); r.render(this.omniScene, this.camera);
      r.setRenderTarget(this.dirTarget);  r.render(this.dirScene, this.camera);
    }
  }

  /** Runs the full simulation over the window plan() picks. Returns { directional, omni } textures and the plan
   *  ({ duration, numBins, bounces, mfp, escape }). */
  run() {
    const r = this.renderer;
    this.updateListener();

    const prevTarget = r.getRenderTarget();
    const prevAuto = r.autoClear;
    const prevColor = new THREE.Color(); r.getClearColor(prevColor);
    const prevAlpha = r.getClearAlpha();

    r.autoClear = false;
    r.setClearColor(0x000000, 0);
    for (const t of [this.omniTarget, this.dirTarget]) { r.setRenderTarget(t); r.clear(); }

    let seed = 1;
    const o = this.opts, size = o.rayTexSize, plan = this.plan(), bounces = plan.bounces;
    o.numBins = plan.numBins;
    this.omniMat.uniforms.uNumBins.value = plan.numBins;
    this.traceMat.uniforms.uMaxPathLength.value = o.speedOfSound * plan.duration;
    for (let b = 0; b < o.batches; b++) {
      // a partial last batch traces only its first `rows` rows (scissor) and splats only those rays (draw range)
      const rows = Math.min(size, (o.rays - b * size * size) / size), partial = rows < size;
      for (const t of [this.read, this.write]) { t.scissor.set(0, 0, size, rows); t.scissorTest = partial; }
      this._rayGeometry.setDrawRange(0, partial ? rows * size : Infinity);

      this._tracePass(true, seed++);                       // emit
      for (let k = 0; k < bounces; k++) {
        this._tracePass(false, seed++);                    // one bounce for all rays
        this._accumulate();                                // splat listener hits
      }
    }
    for (const t of [this.read, this.write]) t.scissorTest = false;
    this._rayGeometry.setDrawRange(0, Infinity);

    r.setRenderTarget(prevTarget);
    r.autoClear = prevAuto;
    r.setClearColor(prevColor, prevAlpha);
    return { directional: this.directional, omni: this.omni, ...plan };
  }

  /** Source distance and direction (in the same frame as the directional output), and whether the straight line
   *  between them is free of geometry (CPU test against the BVH). Valid after run(). */
  directPath() {
    const d = this.opts.source.clone().sub(this._listenerPos);
    const distance = d.length();
    const visible = !segmentBlocked(this.opts.bvh, this._listenerPos, d.clone().divideScalar(distance), distance);
    d.normalize().applyMatrix3(this._worldToListener);
    return { distance, direction: d, visible };
  }

  /** texture.repeat / texture.offset that select time-slice i from the atlas. */
  sliceTransform(i) {
    const { atlasCols: c, atlasRows: rws } = this.opts;
    return {
      repeat: new THREE.Vector2(1 / c, 1 / rws),
      offset: new THREE.Vector2((i % c) / c, Math.floor(i / c) / rws),
    };
  }

  /** Read the atlas back to the CPU: Float32Array(w * h * 8), 8 bands per pixel, atlas row 0 = bottom. */
  readDirectional(out) {
    const o = this.opts, w = o.sliceWidth * o.atlasCols, h = o.sliceHeight * o.atlasRows;
    const raw = new Float32Array(w * 2 * h * 4);
    this.renderer.readRenderTargetPixels(this.dirTarget, 0, 0, w, 2 * h, raw);
    out = out || new Float32Array(w * h * 8);
    for (let p = 0; p < w * h; p++) for (let g = 0; g < 2; g++) for (let c = 0; c < 4; c++) out[p * 8 + g * 4 + c] = raw[(g * w * h + p) * 4 + c];
    return out;
  }

  /** Read the fine omni histogram of the last run's window back to the CPU: Float32Array(numBins * 8), 8 bands per
   *  bin. Reads only the rows the window uses. */
  readOmni(out) {
    const { numBins: n, omniRows: R } = this.opts, rows = Math.ceil(n / OMNI_W);
    const raw = [0, 1].map((g) => {
      const px = new Float32Array(OMNI_W * rows * 4);
      this.renderer.readRenderTargetPixels(this.omniTarget, 0, g * R, OMNI_W, rows, px);
      return px;
    });
    out = out || new Float32Array(n * 8);
    for (let i = 0; i < n; i++) for (let g = 0; g < 2; g++) for (let c = 0; c < 4; c++) out[i * 8 + g * 4 + c] = raw[g][i * 4 + c];
    return out;
  }

  /** Direction (listener frame, from listener towards the energy's origin) for tile pixel (i, j). */
  pixelDirection(i, j, target = new THREE.Vector3()) {
    const lon = ((i + 0.5) / this.opts.sliceWidth - 0.5) * 2 * Math.PI;
    const lat = ((j + 0.5) / this.opts.sliceHeight - 0.5) * Math.PI;
    return target.set(Math.cos(lat) * Math.sin(lon), Math.sin(lat), -Math.cos(lat) * Math.cos(lon));
  }

  /** Solid angle (sr) of tile row j — divide pixel energy by this for energy per steradian. */
  pixelSolidAngle(j) {
    const H = this.opts.sliceHeight, W = this.opts.sliceWidth;
    const lat0 = (j / H - 0.5) * Math.PI, lat1 = ((j + 1) / H - 0.5) * Math.PI;
    return (2 * Math.PI / W) * (Math.sin(lat1) - Math.sin(lat0));
  }
}

/* ---------------------------------------------------------------- BVH */
/**
 * CPU twin of traceBVH in rirTrace.frag: nearest triangle of bvh (from makeBVHTextures) along o + t d with
 * tMin < t < tMax, as { t, n: [x, y, z] face normal }, or null. any = true returns the first hit found (shadow test).
 */
function castRay(bvh, o, d, tMin, tMax, any = false) {
  const N = bvh.nodes.image.data, T = bvh.tris.image.data, EPS = 1e-6;
  const [ox, oy, oz] = o, [dx, dy, dz] = d;
  const inv = (c) => (c >= 0 ? 1 : -1) / Math.max(Math.abs(c), 1e-20);   // as in the shader: no 0 * Infinity = NaN
  const ix = inv(dx), iy = inv(dy), iz = inv(dz);
  let best = tMax, hit = -1;
  const boxHit = (n) => {
    const k = 8 * n;
    let t0 = (N[k] - ox) * ix, t1 = (N[k + 4] - ox) * ix, lo = Math.min(t0, t1), hi = Math.max(t0, t1);
    t0 = (N[k + 1] - oy) * iy; t1 = (N[k + 5] - oy) * iy; lo = Math.max(lo, Math.min(t0, t1)); hi = Math.min(hi, Math.max(t0, t1));
    t0 = (N[k + 2] - oz) * iz; t1 = (N[k + 6] - oz) * iz; lo = Math.max(lo, Math.min(t0, t1)); hi = Math.min(hi, Math.max(t0, t1));
    return lo <= hi && hi > tMin && lo < best;
  };
  const stack = [0];
  while (stack.length) {
    const n = stack.pop();
    if (!boxHit(n)) continue;
    const a = N[8 * n + 3], count = N[8 * n + 7];
    if (!count) { stack.push(a, a + 1); continue; }
    for (let k = 3 * a * 4, end = 3 * (a + count) * 4; k < end; k += 12) {   // Moller-Trumbore, as in the shader
      const e1x = T[k + 4], e1y = T[k + 5], e1z = T[k + 6], e2x = T[k + 8], e2y = T[k + 9], e2z = T[k + 10];
      const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
      const det = e1x * px + e1y * py + e1z * pz;
      if (Math.abs(det) < 1e-12) continue;
      const rdet = 1 / det, tx = ox - T[k], ty = oy - T[k + 1], tz = oz - T[k + 2];
      const u = (tx * px + ty * py + tz * pz) * rdet;
      if (u < -EPS || u > 1 + EPS) continue;
      const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
      const v = (dx * qx + dy * qy + dz * qz) * rdet;
      if (v < -EPS || u + v > 1 + EPS) continue;
      const t = (e2x * qx + e2y * qy + e2z * qz) * rdet;
      if (t > tMin && t < best) {
        best = t; hit = k;
        if (any) return { t, n: [T[k + 3], T[k + 7], T[k + 11]] };
      }
    }
  }
  return hit < 0 ? null : { t: best, n: [T[hit + 3], T[hit + 7], T[hit + 11]] };
}

/** True if any triangle crosses o + t d for 0 < t < length. Ends are excluded by 1e-4 m: source / listener on a
 *  wall pass. o, d: THREE.Vector3. */
function segmentBlocked(bvh, o, d, length) {
  return castRay(bvh, [o.x, o.y, o.z], [d.x, d.y, d.z], 1e-4, length - 1e-4, true) !== null;
}

/**
 * Mean free path of the room around `source` and the fraction of segments that leave it (open rooms), from
 * PROBE_RAYS rays reflecting diffusely (Lambert) for up to PROBE_BOUNCES bounces. Seeded, so the same room and
 * source give the same answer (no window jitter between traces). mfp = Infinity if no ray hits anything.
 */
const PROBE_RAYS = 256, PROBE_BOUNCES = 24;
function probeRoom(bvh, source) {
  let seed = 12345;
  const rnd = () => {                            // mulberry32
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const sphere = () => {
    const z = 1 - 2 * rnd(), phi = 2 * Math.PI * rnd(), r = Math.sqrt(Math.max(0, 1 - z * z));
    return [r * Math.cos(phi), r * Math.sin(phi), z];
  };
  let length = 0, hits = 0, escapes = 0;
  for (let r = 0; r < PROBE_RAYS; r++) {
    let o = [source.x, source.y, source.z], d = sphere();
    for (let b = 0; b < PROBE_BOUNCES; b++) {
      const h = castRay(bvh, o, d, 1e-5, Infinity);
      if (!h) { escapes++; break; }
      length += h.t; hits++;
      const n = d[0] * h.n[0] + d[1] * h.n[1] + d[2] * h.n[2] > 0 ? h.n.map((v) => -v) : h.n;   // two-sided
      o = o.map((v, c) => v + d[c] * h.t + n[c] * 1e-4);
      const s = sphere(), l = n.map((v, c) => v + s[c]), len = Math.hypot(...l) || 1;            // Lambert
      d = l.map((v) => v / len);
    }
  }
  return { mfp: hits ? length / hits : Infinity, escape: escapes / Math.max(1, hits + escapes) };
}

// Layout read by rirTrace.frag (see its header). Width = the minimum MAX_TEXTURE_SIZE of WebGL2.
const BVH_TEX_W = 2048;
const OMNI_W = 2048;        // omni histogram row width (bins), same reason
const LEAF_SIZE = 4;        // stop splitting at this many triangles...
const MAX_LEAF = 16;        // ...and keep splitting above this many even if the SAH prefers a leaf
const MAX_DEPTH = 40;       // < STACK in rirTrace.frag
const SAH_BINS = 16;

/**
 * Builds a BVH (binned surface area heuristic) over triangles and packs it into the two textures the trace shader
 * reads. tris: Float32Array, 9 floats (v0, v1, v2) per triangle; degenerate triangles are dropped.
 * Returns { nodes, tris, triangles, nodeCount, depth } (textures + stats).
 */
export function makeBVHTextures(tris) {
  // per-triangle bounds and centroids, skipping degenerate ones
  let n = tris.length / 9;
  const keep = [];
  for (let t = 0; t < n; t++) {
    const p = (k) => tris[9 * t + k];
    const ux = p(3) - p(0), uy = p(4) - p(1), uz = p(5) - p(2), vx = p(6) - p(0), vy = p(7) - p(1), vz = p(8) - p(2);
    if (Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) > 2e-9) keep.push(t);
  }
  n = keep.length;
  if (!n) throw new Error('no triangles');
  const bmin = new Float32Array(3 * n), bmax = new Float32Array(3 * n), cen = new Float32Array(3 * n);
  keep.forEach((t, i) => {
    for (let c = 0; c < 3; c++) {
      const a = tris[9 * t + c], b = tris[9 * t + 3 + c], d = tris[9 * t + 6 + c];
      bmin[3 * i + c] = Math.min(a, b, d); bmax[3 * i + c] = Math.max(a, b, d); cen[3 * i + c] = (a + b + d) / 3;
    }
  });

  const idx = new Uint32Array(n).map((_, i) => i);
  const maxNodes = 2 * n - 1;
  const nlo = new Float32Array(3 * maxNodes), nhi = new Float32Array(3 * maxNodes);
  const nA = new Uint32Array(maxNodes), nCount = new Uint32Array(maxNodes);
  let nodeCount = 1, depth = 0;
  const area = (dx, dy, dz) => (dx < 0 ? 0 : dx * dy + dy * dz + dz * dx);
  const binLo = new Float64Array(3 * SAH_BINS), binHi = new Float64Array(3 * SAH_BINS), binN = new Uint32Array(SAH_BINS);
  const rightArea = new Float64Array(SAH_BINS), rightN = new Uint32Array(SAH_BINS);

  const stack = [[0, 0, n, 0]];                  // [node, first, count, depth]
  while (stack.length) {
    const [node, first, count, d] = stack.pop();
    depth = Math.max(depth, d);
    // node bounds and centroid bounds
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    const clo = [Infinity, Infinity, Infinity], chi = [-Infinity, -Infinity, -Infinity];
    for (let k = first; k < first + count; k++) {
      const i = idx[k];
      for (let c = 0; c < 3; c++) {
        lo[c] = Math.min(lo[c], bmin[3 * i + c]); hi[c] = Math.max(hi[c], bmax[3 * i + c]);
        clo[c] = Math.min(clo[c], cen[3 * i + c]); chi[c] = Math.max(chi[c], cen[3 * i + c]);
      }
    }
    nlo.set(lo, 3 * node); nhi.set(hi, 3 * node);
    const makeLeaf = () => { nA[node] = first; nCount[node] = count; };
    if (count <= LEAF_SIZE || d >= MAX_DEPTH) { makeLeaf(); continue; }

    // best split over SAH_BINS centroid bins on each axis; cost in units of triangle tests (traversal step = 1)
    let best = { cost: Infinity, axis: -1, bin: 0 };
    for (let axis = 0; axis < 3; axis++) {
      const extent = chi[axis] - clo[axis];
      if (extent <= 1e-12) continue;
      const scale = SAH_BINS / extent;
      binLo.fill(Infinity); binHi.fill(-Infinity); binN.fill(0);
      for (let k = first; k < first + count; k++) {
        const i = idx[k], b = Math.min(SAH_BINS - 1, Math.floor((cen[3 * i + axis] - clo[axis]) * scale));
        binN[b]++;
        for (let c = 0; c < 3; c++) {
          binLo[3 * b + c] = Math.min(binLo[3 * b + c], bmin[3 * i + c]);
          binHi[3 * b + c] = Math.max(binHi[3 * b + c], bmax[3 * i + c]);
        }
      }
      // sweep from the right: area / count of bins b..end
      let rx0 = Infinity, ry0 = Infinity, rz0 = Infinity, rx1 = -Infinity, ry1 = -Infinity, rz1 = -Infinity, rn = 0;
      for (let b = SAH_BINS - 1; b > 0; b--) {
        rx0 = Math.min(rx0, binLo[3 * b]); ry0 = Math.min(ry0, binLo[3 * b + 1]); rz0 = Math.min(rz0, binLo[3 * b + 2]);
        rx1 = Math.max(rx1, binHi[3 * b]); ry1 = Math.max(ry1, binHi[3 * b + 1]); rz1 = Math.max(rz1, binHi[3 * b + 2]);
        rn += binN[b];
        rightArea[b] = area(rx1 - rx0, ry1 - ry0, rz1 - rz0); rightN[b] = rn;
      }
      // sweep from the left: split between bins b - 1 and b
      let lx0 = Infinity, ly0 = Infinity, lz0 = Infinity, lx1 = -Infinity, ly1 = -Infinity, lz1 = -Infinity, ln = 0;
      for (let b = 1; b < SAH_BINS; b++) {
        lx0 = Math.min(lx0, binLo[3 * b - 3]); ly0 = Math.min(ly0, binLo[3 * b - 2]); lz0 = Math.min(lz0, binLo[3 * b - 1]);
        lx1 = Math.max(lx1, binHi[3 * b - 3]); ly1 = Math.max(ly1, binHi[3 * b - 2]); lz1 = Math.max(lz1, binHi[3 * b - 1]);
        ln += binN[b - 1];
        if (!ln || !rightN[b]) continue;
        const cost = ln * area(lx1 - lx0, ly1 - ly0, lz1 - lz0) + rightN[b] * rightArea[b];
        if (cost < best.cost) best = { cost, axis, bin: b };
      }
    }
    const nodeArea = area(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    const splitCost = 1 + best.cost / Math.max(nodeArea, 1e-30);
    if (best.axis < 0 || (splitCost >= count && count <= MAX_LEAF)) { makeLeaf(); continue; }

    // partition idx[first .. first + count) by bin; falls back to a median split if every centroid lands on one side
    const { axis, bin } = best, scale = SAH_BINS / (chi[axis] - clo[axis]);
    let mid = first;
    for (let k = first; k < first + count; k++) {
      const i = idx[k];
      if (Math.min(SAH_BINS - 1, Math.floor((cen[3 * i + axis] - clo[axis]) * scale)) < bin) { idx[k] = idx[mid]; idx[mid++] = i; }
    }
    if (mid === first || mid === first + count) mid = first + (count >> 1);
    const left = nodeCount;                      // children are adjacent: the shader finds the right one at left + 1
    nodeCount += 2;
    nA[node] = left; nCount[node] = 0;
    stack.push([left + 1, mid, first + count - mid, d + 1], [left, first, mid - first, d + 1]);
  }

  // pack: nodes 2 texels each, triangles 3 texels each (in BVH order), rows of BVH_TEX_W texels
  const texture = (texels, fill) => {
    const h = Math.max(1, Math.ceil(texels / BVH_TEX_W)), data = new Float32Array(BVH_TEX_W * h * 4);
    fill(data);
    const t = new THREE.DataTexture(data, BVH_TEX_W, h, THREE.RGBAFormat, THREE.FloatType);
    t.minFilter = t.magFilter = THREE.NearestFilter;
    t.generateMipmaps = false;
    t.needsUpdate = true;
    return t;
  };
  // boxes grow by BOX_PAD: a flat, axis-aligned wall has a zero-thickness box, which rounding could make rays miss
  const BOX_PAD = 1e-4;
  const nodes = texture(2 * nodeCount, (data) => {
    for (let i = 0; i < nodeCount; i++) {
      const lo = (c) => nlo[3 * i + c] - BOX_PAD, hi = (c) => nhi[3 * i + c] + BOX_PAD;
      data.set([lo(0), lo(1), lo(2), nA[i], hi(0), hi(1), hi(2), nCount[i]], 8 * i);
    }
  });
  const triTex = texture(3 * n, (data) => {
    for (let k = 0; k < n; k++) {
      const t = keep[idx[k]], p = (j) => tris[9 * t + j];
      const e1 = [p(3) - p(0), p(4) - p(1), p(5) - p(2)], e2 = [p(6) - p(0), p(7) - p(1), p(8) - p(2)];
      const nx = e1[1] * e2[2] - e1[2] * e2[1], ny = e1[2] * e2[0] - e1[0] * e2[2], nz = e1[0] * e2[1] - e1[1] * e2[0];
      const l = Math.hypot(nx, ny, nz);
      data.set([p(0), p(1), p(2), nx / l, ...e1, ny / l, ...e2, nz / l], 12 * k);
    }
  });
  return { nodes, tris: triTex, triangles: n, nodeCount, depth };
}
