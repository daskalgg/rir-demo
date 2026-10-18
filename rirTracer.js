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
//   tracer.omni        : (2 * numBins) x 1 fine-resolution energy histogram (sum over all directions),
//                        bands 0-3 in x < numBins, bands 4-7 after
//
// Usage:
//   const frag = await (await fetch('./rirTrace.frag')).text();
//   const [tex1, tex2] = makeSurfelTextures(positions, normals, 64, 64);
//   const tracer = new RIRTracer(renderer, frag, {
//       tex1, tex2, camera,
//       source: new THREE.Vector3(2, 1.5, 3),
//       surfelRadius: 0.7 * Math.sqrt(totalArea / (64 * 64)),
//   });
//   tracer.run();                                   // call again whenever camera/source moves
//   material.map = tracer.directional;              // or sample a tile via tracer.sliceTransform(i)

import * as THREE from 'three';

/** Octave band centre frequencies (Hz). */
export const BANDS = [125, 250, 500, 1000, 2000, 4000, 8000, 16000];

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
uniform int   uNumBins;
void main() {
    vec4 h = fetchHit();
    float bin = floor(h.z / uSpeedOfSound / uBinDuration);
    if (h.w < 0.0 || bin >= float(uNumBins)) { clipPoint(); return; }
    vEnergy = bandEnergy(h);
    gl_PointSize = 1.0;
    gl_Position = vec4((float(uGroup * uNumBins) + bin + 0.5) / float(2 * uNumBins) * 2.0 - 1.0, 0.0, 0.0, 1.0);
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
      batches: 8,                 // total rays = batches * rayTexSize^2
      maxBounces: 64,
      speedOfSound: 343,
      listenerRadius: 0.15,
      scattering: 0.2,
      absorption:    [0.10, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25],   // per band (BANDS), energy
      // energy attenuation of air, 1/m (ISO 9613-1, 20 °C, 50 % RH)
      airAbsorption: [1.0e-4, 3.0e-4, 6.3e-4, 1.1e-3, 2.3e-3, 6.8e-3, 2.4e-2, 8.6e-2],
      surfelRadius: 0.1,
      // fine omni histogram
      numBins: 2048, binDuration: 0.0005,                         // ~1.02 s
      // directional atlas
      sliceWidth: 128, sliceHeight: 64,                           // equirect tile (2:1)
      numSlices: 32, sliceDuration: 0.004,                        // 32 x 4 ms = 128 ms
      atlasCols: 8,
    }, opts);
    o.atlasRows = Math.ceil(o.numSlices / o.atlasCols);
    o.bands = BANDS;                // fixed: the shaders accumulate exactly two groups of 4 bands

    this.read = floatTarget(o.rayTexSize, o.rayTexSize, 3);     // state0, state1, hit
    this.write = floatTarget(o.rayTexSize, o.rayTexSize, 3);
    this.omniTarget = floatTarget(2 * o.numBins, 1);                                    // 2 band groups
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
        uTex1: { value: o.tex1 }, uTex2: { value: o.tex2 },
        uState0: { value: null }, uState1: { value: null },
        uInit: { value: 1 }, uSeed: { value: 0 }, uRayTexSize: { value: o.rayTexSize },
        uSource: { value: o.source },
        uListener: { value: this._listenerPos },
        uWorldToListener: { value: this._worldToListener },
        uListenerRadius: { value: o.listenerRadius },
        uSurfelRadius: { value: o.surfelRadius },
        uMaxPathLength: { value: o.speedOfSound * Math.max(o.numBins * o.binDuration, o.numSlices * o.sliceDuration) },
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
    const totalRays = o.batches * nRays;
    const groups = (a) => [new THREE.Vector4().fromArray(a, 0), new THREE.Vector4().fromArray(a, 4)];
    const common = {
      uHit: { value: null },
      uSpeedOfSound: { value: o.speedOfSound },
      uAbsorption: { value: groups(o.absorption) },
      uAirAbsorption: { value: groups(o.airAbsorption) },
      uGroup: { value: 0 },
      uRayWeight: { value: 1 / (totalRays * Math.PI * o.listenerRadius ** 2) },
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
      uniforms: { ...common, uBinDuration: { value: o.binDuration }, uNumBins: { value: o.numBins } },
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

    const g = new THREE.BufferGeometry();
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
  }

  /** Total rays = batches * rayTexSize^2. Takes effect on the next run(). */
  setBatches(batches) {
    const o = this.opts;
    o.batches = batches;
    // uniform object is shared by both accumulate materials
    this.omniMat.uniforms.uRayWeight.value = 1 / (batches * o.rayTexSize ** 2 * Math.PI * o.listenerRadius ** 2);
  }

  /** Bounces traced per ray. Takes effect on the next run(). */
  setMaxBounces(n) { this.opts.maxBounces = n; }

  /** Energy absorption per octave band (BANDS), 0..1, as [8]. Takes effect on the next run(). */
  setAbsorption(a) {
    this.opts.absorption = [...a];
    const [lo, hi] = this.omniMat.uniforms.uAbsorption.value;   // shared by both accumulate materials
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

  /** Runs the full simulation. Returns { directional, omni } textures. */
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
    for (let b = 0; b < this.opts.batches; b++) {
      this._tracePass(true, seed++);                       // emit
      for (let k = 0; k < this.opts.maxBounces; k++) {
        this._tracePass(false, seed++);                    // one bounce for all rays
        this._accumulate();                                // splat listener hits
      }
    }

    r.setRenderTarget(prevTarget);
    r.autoClear = prevAuto;
    r.setClearColor(prevColor, prevAlpha);
    return { directional: this.directional, omni: this.omni };
  }

  /** Source distance and direction (in the same frame as the directional output). Valid after run(). */
  directPath() {
    const d = this.opts.source.clone().sub(this._listenerPos);
    const distance = d.length();
    d.normalize().applyMatrix3(this._worldToListener);
    return { distance, direction: d };
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

  /** Read the fine omni histogram back to the CPU: Float32Array(numBins * 8), 8 bands per bin. */
  readOmni(out) {
    const n = this.opts.numBins, raw = new Float32Array(2 * n * 4);
    this.renderer.readRenderTargetPixels(this.omniTarget, 0, 0, 2 * n, 1, raw);
    out = out || new Float32Array(n * 8);
    for (let i = 0; i < n; i++) for (let g = 0; g < 2; g++) for (let c = 0; c < 4; c++) out[i * 8 + g * 4 + c] = raw[(g * n + i) * 4 + c];
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

/** Helper: pack surfels into the two input textures in the requested layout. */
export function makeSurfelTextures(positions, normals, width, height) {
  const n = width * height;
  const d1 = new Float32Array(n * 4), d2 = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    d1.set([positions[3*i], positions[3*i+1], positions[3*i+2], normals[3*i]], i * 4);   // [p.x p.y p.z n.x]
    d2.set([normals[3*i+1], normals[3*i+2], 0, 0], i * 4);                               // [n.y n.z 0 0]
  }
  const mk = (data) => {
    const t = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.FloatType);
    t.minFilter = t.magFilter = THREE.NearestFilter;
    t.generateMipmaps = false;
    t.needsUpdate = true;
    return t;
  };
  return [mk(d1), mk(d2)];
}
