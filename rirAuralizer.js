// rirAuralizer.js — turns the ray-traced energy RIR (rirTracer.js) into audible, binaural output.
//
// PIPELINE
//   1. The tracer gives ENERGY data, not a waveform:
//        omni        numBins x 4   fine-resolution energy per time bin, 4 octave bands
//        directional atlas of equirect tiles (coarse time slices) -> where the energy came from
//   2. synthesizeSpeakerIR()  (pure JS, runs anywhere)
//        Energy -> pressure by multiplying band-limited random noise with sqrt(energy envelope)
//        (the standard "stochastic RIR" construction). The energy is distributed over K virtual
//        loudspeakers (default: icosahedron, 12) according to the directional atlas, with
//        independent noise per speaker so the reverb tail is decorrelated.
//        The direct sound is added analytically as a clean, correctly panned impulse.
//   3. RIRAuralizer (Web Audio)
//        anechoic mono source -> K x ConvolverNode (one IR per speaker)
//                             -> K x PannerNode (HRTF, placed at the speaker direction)
//                             -> master gain -> destination
//
// HEAD TRACKING (no re-trace needed): trace with  cameraRelative:false  in RIRTracer. The speakers
// are then in WORLD axes and setListenerFromCamera(camera) rotates the Web Audio listener with the
// camera. Moving the listener or source still requires tracer.run() + setRIRFromTracer().
//
// Absolute level: the IR is scaled so the direct sound has amplitude 1/distance (1 at 1 m).

const FOUR_PI = 4 * Math.PI;

/* ------------------------------------------------------------------ speaker layouts */
const unit = (arr) => arr.map((v) => { const l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l]; });

export const SPEAKER_LAYOUTS = {
  octahedron: () => unit([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]),
  cube: () => unit([-1, 1].flatMap((x) => [-1, 1].flatMap((y) => [-1, 1].map((z) => [x, y, z])))),
  icosahedron: () => {
    const t = (1 + Math.sqrt(5)) / 2;
    return unit([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                 [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]);
  },
};

/* ------------------------------------------------------------------ small DSP helpers */
function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// RBJ biquad, Butterworth Q. Two identical stages = Linkwitz-Riley 4th order (flat-summing crossover).
function biquadCoefs(type, f0, fs) {
  const w0 = 2 * Math.PI * f0 / fs, c = Math.cos(w0), alpha = Math.sin(w0) / (2 * Math.SQRT1_2);
  let b0, b1, b2;
  if (type === 'lowpass') { b0 = (1 - c) / 2; b1 = 1 - c; b2 = (1 - c) / 2; }
  else { b0 = (1 + c) / 2; b1 = -(1 + c); b2 = (1 + c) / 2; }
  const a0 = 1 + alpha;
  return [b0 / a0, b1 / a0, b2 / a0, (-2 * c) / a0, (1 - alpha) / a0];
}
function applyBiquad(x, [b0, b1, b2, a1, a2]) {
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let n = 0; n < x.length; n++) {
    const xn = x[n];
    const y = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = xn; y2 = y1; y1 = y; x[n] = y;
  }
}

/** Unit-variance random-sign noise, band-limited to octave band `band` (0..3). One channel per speaker. */
function makeBandNoise(K, L, fs, crossovers, band, rng) {
  const WARM = 4096;                       // discard filter start-up transient
  const stages = [];
  if (band > 0) { const c = biquadCoefs('highpass', crossovers[band - 1], fs); stages.push(c, c); }
  if (band < crossovers.length) { const c = biquadCoefs('lowpass', crossovers[band], fs); stages.push(c, c); }
  const ch = [], inv = new Float32Array(K);
  for (let k = 0; k < K; k++) {
    const x = new Float64Array(L + WARM);
    for (let n = 0; n < x.length; n++) x[n] = rng() < 0.5 ? -1 : 1;
    for (const c of stages) applyBiquad(x, c);
    const y = new Float32Array(L);
    let ss = 0;
    for (let n = 0; n < L; n++) { y[n] = x[n + WARM]; ss += y[n] * y[n]; }
    inv[k] = 1 / Math.sqrt(ss / L);        // normalise so band power == requested energy
    ch.push(y);
  }
  return { ch, inv };
}

/* ------------------------------------------------------------------ directional weights */
function equirectDir(u, v) {                // same convention as rirTrace.frag / rirTracer.js
  const lon = (u - 0.5) * 2 * Math.PI, lat = (v - 0.5) * Math.PI, cl = Math.cos(lat);
  return [cl * Math.sin(lon), Math.sin(lat), -cl * Math.cos(lon)];
}

/** Energy-normalised lobe panning: out[k] >= 0, sum(out) == 1. */
function panWeights(dir, speakers, p, out) {
  let sum = 0;
  for (let k = 0; k < speakers.length; k++) {
    const s = speakers[k];
    const c = Math.max(0, dir[0] * s[0] + dir[1] * s[1] + dir[2] * s[2]);
    out[k] = Math.pow(c, p); sum += out[k];
  }
  if (sum < 1e-12) { out.fill(1 / speakers.length); return; }
  for (let k = 0; k < speakers.length; k++) out[k] /= sum;
}

function computeTileGains(W, H, speakers, p) {
  const K = speakers.length, G = new Float32Array(W * H * K), tmp = new Float32Array(K);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    panWeights(equirectDir((i + 0.5) / W, (j + 0.5) / H), speakers, p, tmp);
    G.set(tmp, (j * W + i) * K);
  }
  return G;
}

/** sqrt of the fraction of each slice's band energy that goes to each speaker -> [slice][band][k]. */
function computeSliceSqrtWeights(data, tileG, K) {
  const { sliceWidth: W, sliceHeight: H, atlasCols: C, numSlices: S, directional: D } = data;
  const Wa = W * C, out = new Float32Array(S * 4 * K), acc = new Float64Array(K);
  for (let s = 0; s < S; s++) {
    const col = s % C, row = Math.floor(s / C);
    for (let b = 0; b < 4; b++) {
      acc.fill(0); let tot = 0;
      for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
        const e = D[((row * H + j) * Wa + col * W + i) * 4 + b];
        if (!(e > 0)) continue;
        tot += e;
        const g = (j * W + i) * K;
        for (let k = 0; k < K; k++) acc[k] += e * tileG[g + k];
      }
      const base = (s * 4 + b) * K;
      for (let k = 0; k < K; k++) out[base + k] = tot > 0 ? Math.sqrt(acc[k] / tot) : Math.sqrt(1 / K);
    }
  }
  return out;
}

/* ------------------------------------------------------------------ IR synthesis */
/**
 * @param data {
 *   omni: Float32Array(numBins*4), numBins, binDuration,
 *   directional: Float32Array(atlas RGBA), sliceWidth, sliceHeight, atlasCols, numSlices, sliceDuration,
 *   speedOfSound, airAbsorption:[4],
 *   direct: { distance, direction:[x,y,z] } | null     (direction in the same frame as the atlas)
 * }
 * @returns { channels: Float32Array[K], sampleRate, length }
 */
export function synthesizeSpeakerIR(data, speakers, opts = {}) {
  const { sampleRate: fs, panExponent = 4, crossovers = [250, 1000, 4000], seed = 1 } = opts;
  const K = speakers.length;
  const L = Math.ceil(data.numBins * data.binDuration * fs) + 2;
  const Ns = data.binDuration * fs;                         // samples per energy bin
  const rng = mulberry32(seed);

  const noise = [0, 1, 2, 3].map((b) => makeBandNoise(K, L, fs, crossovers, b, rng));
  const tileG = computeTileGains(data.sliceWidth, data.sliceHeight, speakers, panExponent);
  const sw = computeSliceSqrtWeights(data, tileG, K);
  const uni = Math.sqrt(1 / K);                             // late field beyond the atlas: diffuse

  const out = Array.from({ length: K }, () => new Float32Array(L));
  const directBin = data.direct ? Math.floor(data.direct.distance / data.speedOfSound / data.binDuration) : -10;
  const envScale = FOUR_PI / Ns;                            // energy per bin -> variance per sample (x4pi: 1/d scaling)

  for (let n = 0; n < L; n++) {
    const t = n / fs;
    const bin = Math.floor(t / data.binDuration);
    if (bin >= data.numBins) break;
    if (Math.abs(bin - directBin) <= 1) continue;           // replaced by the analytic direct sound below
    const slice = Math.floor(t / data.sliceDuration);
    const hasDir = slice < data.numSlices;
    for (let b = 0; b < 4; b++) {
      const e = data.omni[bin * 4 + b];
      if (!(e > 0)) continue;
      const amp = Math.sqrt(e * envScale);
      const nb = noise[b];
      for (let k = 0; k < K; k++) {
        const w = hasDir ? sw[(slice * 4 + b) * K + k] : uni;
        out[k][n] += amp * w * nb.ch[k][n] * nb.inv[k];
      }
    }
  }

  if (data.direct) {                                        // clean, panned direct impulse (amplitude 1/d)
    const d = Math.max(data.direct.distance, 0.1);
    let air = 0;
    for (let b = 0; b < 4; b++) air += Math.exp(-data.airAbsorption[b] * d);
    const amp = Math.sqrt(air / 4) / d;
    const w = new Float32Array(K);
    panWeights(data.direct.direction, speakers, panExponent, w);
    const delay = (data.direct.distance / data.speedOfSound) * fs;
    const i0 = Math.floor(delay), fr = delay - i0;
    const norm = 1 / Math.hypot(1 - fr, fr);                 // keep impulse energy for any fractional delay
    for (let k = 0; k < K; k++) {
      const g = amp * Math.sqrt(w[k]) * norm;
      if (i0 < L) out[k][i0] += g * (1 - fr);
      if (i0 + 1 < L) out[k][i0 + 1] += g * fr;
    }
  }
  return { channels: out, sampleRate: fs, length: L };
}

/* ------------------------------------------------------------------ Web Audio playback */
export class RIRAuralizer {
  /**
   * @param opts.audioContext  existing AudioContext (optional)
   * @param opts.speakers      array of unit vectors, or a name from SPEAKER_LAYOUTS (default 'icosahedron')
   * @param opts.panExponent   directional lobe sharpness for distributing energy over speakers (default 4)
   */
  constructor(opts = {}) {
    this.ctx = opts.audioContext || new (window.AudioContext || window.webkitAudioContext)();
    const sp = opts.speakers || 'icosahedron';
    this.speakers = typeof sp === 'string' ? SPEAKER_LAYOUTS[sp]() : sp;
    this.panExponent = opts.panExponent ?? 4;
    this.worldFrame = false;
    this.anechoic = null;
    this.srcNode = null;
    this.bank = null;

    // mono sum of the anechoic signal feeds every speaker convolver
    this.input = this.ctx.createGain();
    this.input.channelCount = 1;
    this.input.channelCountMode = 'explicit';
    this.input.channelInterpretation = 'speakers';

    this.master = this.ctx.createGain();
    this.master.gain.value = 0.5;
    this.master.connect(this.ctx.destination);
  }

  setGain(g) { this.master.gain.value = g; }

  /** Accepts URL string, File/Blob, ArrayBuffer or AudioBuffer. Use an ANECHOIC (dry) recording. */
  async loadAnechoic(src) {
    if (src instanceof AudioBuffer) { this.anechoic = src; return src; }
    let ab;
    if (typeof src === 'string') ab = await (await fetch(src)).arrayBuffer();
    else if (src instanceof ArrayBuffer) ab = src;
    else ab = await src.arrayBuffer();
    this.anechoic = await this.ctx.decodeAudioData(ab);      // resampled to ctx.sampleRate
    return this.anechoic;
  }

  /** Builds the IR from a RIRTracer that has already run(). */
  async setRIRFromTracer(tracer) {
    const o = tracer.opts, dp = tracer.directPath();
    return this.setRIR({
      omni: tracer.readOmni(), numBins: o.numBins, binDuration: o.binDuration,
      directional: tracer.readDirectional(),
      sliceWidth: o.sliceWidth, sliceHeight: o.sliceHeight, atlasCols: o.atlasCols,
      numSlices: o.numSlices, sliceDuration: o.sliceDuration,
      speedOfSound: o.speedOfSound,
      airAbsorption: [o.airAbsorption.x, o.airAbsorption.y, o.airAbsorption.z, o.airAbsorption.w],
      direct: { distance: dp.distance, direction: [dp.direction.x, dp.direction.y, dp.direction.z] },
      worldFrame: !o.cameraRelative,
    });
  }

  /** data: see synthesizeSpeakerIR (+ optional worldFrame flag). Cross-fades if already playing. */
  async setRIR(data) {
    this.worldFrame = !!data.worldFrame;
    if (!this.worldFrame) this._setListenerOrientation(0, 0, -1, 0, 1, 0);

    // let the UI repaint before the (synchronous) synthesis
    await new Promise((r) => setTimeout(r, 0));
    const ir = synthesizeSpeakerIR(data, this.speakers, {
      sampleRate: this.ctx.sampleRate, panExponent: this.panExponent,
    });

    const bank = this._buildBank(ir);
    const old = this.bank, t = this.ctx.currentTime;
    bank.gain.gain.setValueAtTime(0, t);
    bank.gain.gain.linearRampToValueAtTime(1, t + 0.08);
    if (old) {
      old.gain.gain.setValueAtTime(1, t);
      old.gain.gain.linearRampToValueAtTime(0, t + 0.08);
      setTimeout(() => this._disposeBank(old), 250);
    }
    this.bank = bank;
    return ir;
  }

  _buildBank(ir) {
    const ctx = this.ctx;
    const gain = ctx.createGain();
    gain.connect(this.master);
    const convs = [], pans = [];
    this.speakers.forEach((s, k) => {
      const buf = ctx.createBuffer(1, ir.length, ir.sampleRate);
      buf.copyToChannel(ir.channels[k], 0);

      const conv = ctx.createConvolver();
      conv.normalize = false;                 // MUST be set before .buffer, or the IR gets rescaled
      conv.buffer = buf;

      const pan = ctx.createPanner();
      pan.panningModel = 'HRTF';
      pan.distanceModel = 'linear';
      pan.rolloffFactor = 0;                  // no distance gain: level is already inside the IR
      pan.refDistance = 1;
      pan.positionX.value = s[0]; pan.positionY.value = s[1]; pan.positionZ.value = s[2];

      this.input.connect(conv);
      conv.connect(pan);
      pan.connect(gain);
      convs.push(conv); pans.push(pan);
    });
    return { gain, convs, pans };
  }

  _disposeBank(bank) {
    bank.convs.forEach((c) => { try { this.input.disconnect(c); } catch (e) { /* already gone */ } c.disconnect(); });
    bank.pans.forEach((p) => p.disconnect());
    bank.gain.disconnect();
  }

  async play({ loop = true, offset = 0 } = {}) {
    if (!this.anechoic) throw new Error('call loadAnechoic() first');
    await this.ctx.resume();
    this.stop();
    const src = this.ctx.createBufferSource();
    src.buffer = this.anechoic;
    src.loop = loop;
    src.connect(this.input);
    src.start(0, offset);
    this.srcNode = src;
  }

  stop() {
    if (this.srcNode) { try { this.srcNode.stop(); } catch (e) { /* not started */ } this.srcNode.disconnect(); this.srcNode = null; }
  }

  /** World-frame IRs only: rotate the Web Audio listener with a Three.js camera (call every frame). */
  setListenerFromCamera(camera) {
    if (!this.worldFrame) return;
    if (camera.updateWorldMatrix) camera.updateWorldMatrix(true, false);
    const e = camera.matrixWorld.elements;     // columns: X (0-2), Y (4-6), Z (8-10); camera looks down -Z
    this._setListenerOrientation(-e[8], -e[9], -e[10], e[4], e[5], e[6]);
  }

  _setListenerOrientation(fx, fy, fz, ux, uy, uz) {
    const l = this.ctx.listener;
    if (l.forwardX) {
      l.forwardX.value = fx; l.forwardY.value = fy; l.forwardZ.value = fz;
      l.upX.value = ux; l.upY.value = uy; l.upZ.value = uz;
    } else {
      l.setOrientation(fx, fy, fz, ux, uy, uz);
    }
  }
}
