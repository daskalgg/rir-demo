// rirAuralizer.js — turns the ray-traced energy RIR (rirTracer.js) into audible, binaural output.
//
// PIPELINE
//   1. The tracer gives ENERGY data, not a waveform:
//        omni        numBins x B   fine-resolution energy per time bin, B octave bands (data.bands)
//        directional atlas of equirect tiles (coarse time slices) -> where the energy came from
//   2. synthesizeSpeakerIR()  (pure JS, runs anywhere)
//        Energy -> pressure by multiplying band-limited random noise with sqrt(energy envelope)
//        (the standard "stochastic RIR" construction). The energy is distributed over K virtual
//        loudspeakers (default: icosahedron, 12) according to the directional atlas, with
//        independent noise per speaker so the reverb tail is decorrelated.
//        The direct sound is added analytically as a clean, correctly panned impulse.
//   3. RIRAuralizer (Web Audio), two render modes:
//        'speakers'  anechoic mono source -> bank gain -> K x ConvolverNode (one IR per speaker)
//                    -> K x PannerNode (HRTF, at the speaker direction; shared by all banks) -> master
//        'ambisonic' synthesizeAmbisonicIR() instead: 1st- or 3rd-order AmbiX RIR (4 / 16 channels);
//                    anechoic mono source -> bank gain -> 4 / 16 x ConvolverNode -> merger -> Omnitone FOA / HOA
//                    renderer (sound-field rotation for head tracking + binaural decoding) -> trim -> master
//        master -> limiter -> destination
//        A new IR gets a new bank; the switch cross-fades the bank INPUTS, so the old convolvers ring out
//        their tail naturally and the new ones build up from silence (no output discontinuity -> no click).
//
// HEAD TRACKING (no re-trace needed): trace with  cameraRelative:false  in RIRTracer. The speakers / the
// ambisonic field are then in WORLD axes and setListenerFromCamera(camera) rotates the Web Audio listener and
// the HOA field with the camera. Moving the listener or source still requires tracer.run() + setRIRFromTracer().
//
// Absolute level: the IR is scaled so the direct sound has amplitude 1/distance (1 at 1 m).
// Headroom: rooms that are physically very loud (small rooms, focusing points) would clip, so an IR whose
// energy norm exceeds opts.maxLevel is turned down to it (the level change rides on the cross-fade), and a
// limiter on the output catches what is left. Below maxLevel, relative loudness is kept as is.

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

/* ------------------------------------------------------------------ ambisonics (AmbiX: ACN order, SN3D) */
export const AMBI_ORDERS = [1, 3];               // Omnitone: FOA renderer for 1, HOA renderer for 3
export const AMBI_ORDER = 3, AMBI_CHANNELS = (AMBI_ORDER + 1) ** 2;   // highest order / channel count supported
const ambiChannels = (order) => (order + 1) ** 2;
// per order: output trim after Omnitone's decoder so that all render modes play at the same loudness, and the
// signs f of the rotation matrix Omnitone needs (inverse(camera) * diag(f), measured)
const AMBI_RENDER = {
  1: { trim: 10 ** (0.6 / 20), rotSigns: [1, 1, 1] },               // FOA renderer: plain inverse; +0.6 dB
  3: { trim: 10 ** (1.35 / 20), rotSigns: [-1, 1, -1] },             // HOA: identity turns the field by 180°; +1.35 dB
};

/** Ambisonic axes (X front, Y left, Z up) from the tracer's world axes (three.js: -Z = forward of an unrotated
 *  camera, +X right, +Y up). The sound field is encoded in this fixed frame; head tracking rotates it. */
export const worldToAmbi = ([x, y, z]) => [-z, -x, y];

const S3 = Math.sqrt(3), S15 = Math.sqrt(15), S58 = Math.sqrt(5 / 8), S38 = Math.sqrt(3 / 8);
/** Real SN3D spherical harmonics up to 3rd order, ACN order, for a unit vector in ambisonic axes (lower orders
 *  are the first (order + 1)^2 entries). */
export function shEncode(x, y, z, out = new Float32Array(AMBI_CHANNELS)) {
  out[0] = 1;
  out[1] = y; out[2] = z; out[3] = x;
  out[4] = S3 * x * y; out[5] = S3 * y * z; out[6] = 0.5 * (3 * z * z - 1); out[7] = S3 * x * z; out[8] = 0.5 * S3 * (x * x - y * y);
  out[9] = S58 * y * (3 * x * x - y * y); out[10] = S15 * x * y * z; out[11] = S38 * y * (5 * z * z - 1);
  out[12] = 0.5 * z * (5 * z * z - 3); out[13] = S38 * x * (5 * z * z - 1); out[14] = 0.5 * S15 * z * (x * x - y * y);
  out[15] = S58 * x * (x * x - 3 * y * y);
  return out;
}

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

/** Unit-variance random-sign noise, band-limited to octave band `band` (0..B-1). One channel per speaker. */
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
  const { sliceWidth: W, sliceHeight: H, atlasCols: C, numSlices: S, directional: D } = data, B = data.bands.length;
  const Wa = W * C, out = new Float32Array(S * B * K), acc = new Float64Array(K);
  for (let s = 0; s < S; s++) {
    const col = s % C, row = Math.floor(s / C);
    for (let b = 0; b < B; b++) {
      acc.fill(0); let tot = 0;
      for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
        const e = D[((row * H + j) * Wa + col * W + i) * B + b];
        if (!(e > 0)) continue;
        tot += e;
        const g = (j * W + i) * K;
        for (let k = 0; k < K; k++) acc[k] += e * tileG[g + k];
      }
      const base = (s * B + b) * K;
      for (let k = 0; k < K; k++) out[base + k] = tot > 0 ? Math.sqrt(acc[k] / tot) : Math.sqrt(1 / K);
    }
  }
  return out;
}

/* ------------------------------------------------------------------ IR synthesis */
/**
 * @param data {
 *   bands: [B centre frequencies, Hz],
 *   omni: Float32Array(numBins*B), numBins, binDuration,
 *   directional: Float32Array(atlas pixels * B), sliceWidth, sliceHeight, atlasCols, numSlices, sliceDuration,
 *   speedOfSound, airAbsorption:[B],
 *   direct: { distance, direction:[x,y,z] } | null     (direction in the same frame as the atlas)
 * }
 * @returns { channels: Float32Array[K], sampleRate, length }
 */
// crossovers halfway (geometrically) between band centres; crossovers too close to Nyquist are dropped, so the
// top bands then share the highest available band filter
const bandCrossovers = (bands, fs) => bands.slice(1).map((f, i) => Math.sqrt(f * bands[i])).filter((f) => f < 0.45 * fs);

export function synthesizeSpeakerIR(data, speakers, opts = {}) {
  const B = data.bands.length;
  const { sampleRate: fs, panExponent = 4, seed = 1, crossovers = bandCrossovers(data.bands, opts.sampleRate) } = opts;
  const K = speakers.length;
  const L = Math.ceil(data.numBins * data.binDuration * fs) + 2;
  const Ns = data.binDuration * fs;                         // samples per energy bin
  const rng = mulberry32(seed);

  const noise = data.bands.map((_, b) => makeBandNoise(K, L, fs, crossovers, Math.min(b, crossovers.length), rng));
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
    for (let b = 0; b < B; b++) {
      const e = data.omni[bin * B + b];
      if (!(e > 0)) continue;
      const amp = Math.sqrt(e * envScale);
      const nb = noise[b];
      for (let k = 0; k < K; k++) {
        const w = hasDir ? sw[(slice * B + b) * K + k] : uni;
        out[k][n] += amp * w * nb.ch[k][n] * nb.inv[k];
      }
    }
  }

  if (data.direct) {                                        // clean, panned direct impulse (amplitude 1/d)
    const d = Math.max(data.direct.distance, 0.1);
    let air = 0;
    for (let b = 0; b < B; b++) air += Math.exp(-data.airAbsorption[b] * d);
    const amp = Math.sqrt(air / B) / d;
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

/**
 * Ambisonic RIR (AmbiX: ACN / SN3D, opts.order 1..AMBI_ORDER, default 3) from the same tracer data as
 * synthesizeSpeakerIR.
 * Per band, one noise signal carries the energy envelope (as in the speaker version); it is cut into grains
 * (50 % overlap, ~2 periods of the band centre) and each grain is encoded from a direction drawn at random from
 * that time slice's directional energy in that band (uniform directions after the atlas window -> diffuse tail).
 * A grain's SH vector is split into the slice's mean m (shared by all its grains) and a random deviation Y - m:
 * m is windowed with Hann (sums to 1 in amplitude), the deviation with sqrt(Hann) (sums to 1 in energy), so the
 * expected energy of every channel is right for both concentrated and diffuse slices, and W (channel 0, where
 * Y = m = 1) carries exactly the band energies. The direct sound is encoded exactly.
 * Directions in data are in the tracer's world (or listener) frame and are converted with worldToAmbi().
 * @returns { channels: Float32Array[(order + 1)^2], sampleRate, length, order, normChannels: [0] }
 */
export function synthesizeAmbisonicIR(data, opts = {}) {
  const B = data.bands.length;
  const { sampleRate: fs, seed = 1, order = AMBI_ORDER, crossovers = bandCrossovers(data.bands, opts.sampleRate) } = opts;
  const NC = ambiChannels(order);               // channels written; the tables below always hold all AMBI_CHANNELS
  const L = Math.ceil(data.numBins * data.binDuration * fs) + 2;
  const Ns = data.binDuration * fs;
  const rng = mulberry32(seed);
  const { sliceWidth: W, sliceHeight: H, atlasCols: C, numSlices: S, directional: D } = data, Wa = W * C, P = W * H;

  // SH vector of every atlas pixel centre
  const pixY = new Float32Array(P * AMBI_CHANNELS), tmp = new Float32Array(AMBI_CHANNELS);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    pixY.set(shEncode(...worldToAmbi(equirectDir((i + 0.5) / W, (j + 0.5) / H)), tmp), (j * W + i) * AMBI_CHANNELS);
  }
  // per slice and band: cumulative energy over the tile's pixels (for drawing directions) and the mean SH vector
  const cdf = new Float32Array(S * B * P), tot = new Float64Array(S * B), mean = new Float32Array((S + 1) * B * AMBI_CHANNELS);
  const acc16 = new Float64Array(AMBI_CHANNELS);
  for (let s = 0; s < S; s++) {
    const col = s % C, row = Math.floor(s / C);
    for (let b = 0; b < B; b++) {
      const base = (s * B + b) * P;
      let acc = 0;
      acc16.fill(0);
      for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
        const e = D[((row * H + j) * Wa + col * W + i) * B + b], p = j * W + i;
        if (e > 0) {
          acc += e;
          for (let c = 0; c < AMBI_CHANNELS; c++) acc16[c] += e * pixY[p * AMBI_CHANNELS + c];
        }
        cdf[base + p] = acc;
      }
      tot[s * B + b] = acc;
      for (let c = 0; c < AMBI_CHANNELS; c++) mean[(s * B + b) * AMBI_CHANNELS + c] = acc > 0 ? acc16[c] / acc : +(c === 0);
    }
  }
  for (let b = 0; b < B; b++) mean[(S * B + b) * AMBI_CHANNELS] = 1;   // "slice S" = after the atlas: diffuse, mean = W only
  const Y = new Float32Array(AMBI_CHANNELS);
  const drawDirection = (slice, b) => {          // -> SH coefficients in Y
    let dir;
    const t = slice < S ? tot[slice * B + b] : 0;
    if (t > 0) {
      const base = (slice * B + b) * P, u = rng() * t;
      let lo = 0, hi = P - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[base + mid] < u) lo = mid + 1; else hi = mid; }
      dir = equirectDir(((lo % W) + rng()) / W, (Math.floor(lo / W) + rng()) / H);   // jitter within the pixel
    } else {                                      // diffuse: uniform on the sphere
      const z = 1 - 2 * rng(), phi = 2 * Math.PI * rng(), r = Math.sqrt(Math.max(0, 1 - z * z));
      dir = [r * Math.cos(phi), z, r * Math.sin(phi)];
    }
    return shEncode(...worldToAmbi(dir), Y);
  };

  const out = Array.from({ length: NC }, () => new Float32Array(L));
  const directBin = data.direct ? Math.floor(data.direct.distance / data.speedOfSound / data.binDuration) : -10;
  const envScale = FOUR_PI / Ns;
  const sig = new Float32Array(L);
  for (let b = 0; b < B; b++) {
    // band signal: noise x sqrt(energy envelope), direct-sound bins left out (added analytically below)
    const nb = makeBandNoise(1, L, fs, crossovers, Math.min(b, crossovers.length), rng), noise = nb.ch[0], inv = nb.inv[0];
    sig.fill(0);
    for (let n = 0; n < L; n++) {
      const bin = Math.floor(n / Ns);
      if (bin >= data.numBins) break;
      if (Math.abs(bin - directBin) <= 1) continue;
      const e = data.omni[bin * B + b];
      if (e > 0) sig[n] = Math.sqrt(e * envScale) * noise[n] * inv;
    }
    // grains: periodic Hann (amplitude-complementary) and its sqrt (power-complementary), hop G/2
    const G = 2 * Math.max(8, Math.min(1024, Math.round(fs / data.bands[b])));
    const hann = new Float32Array(G).map((_, n) => 0.5 - 0.5 * Math.cos(2 * Math.PI * n / G)), root = hann.map(Math.sqrt);
    for (let start = -G / 2; start < L; start += G / 2) {
      const slice = Math.min(S, Math.floor(Math.max(0, start + G / 2) / fs / data.sliceDuration));
      drawDirection(slice, b);
      const m = mean.subarray((slice * B + b) * AMBI_CHANNELS, (slice * B + b + 1) * AMBI_CHANNELS);
      const n0 = Math.max(0, start), n1 = Math.min(L, start + G);
      for (let n = n0; n < n1; n++) {
        const v = sig[n];
        if (v === 0) continue;
        const wm = v * hann[n - start], wd = v * root[n - start];
        for (let c = 0; c < NC; c++) out[c][n] += wm * m[c] + wd * (Y[c] - m[c]);
      }
    }
  }

  if (data.direct) {                              // clean direct impulse (amplitude 1/d), encoded at its direction
    const d = Math.max(data.direct.distance, 0.1);
    let air = 0;
    for (let b = 0; b < B; b++) air += Math.exp(-data.airAbsorption[b] * d);
    const amp = Math.sqrt(air / B) / d;
    shEncode(...worldToAmbi(data.direct.direction), Y);
    const delay = (data.direct.distance / data.speedOfSound) * fs;
    const i0 = Math.floor(delay), fr = delay - i0, norm = 1 / Math.hypot(1 - fr, fr);
    for (let c = 0; c < NC; c++) {
      if (i0 < L) out[c][i0] += amp * norm * (1 - fr) * Y[c];
      if (i0 + 1 < L) out[c][i0 + 1] += amp * norm * fr * Y[c];
    }
  }
  return { channels: out, sampleRate: fs, length: L, order, normChannels: [0] };
}

/** 32-bit float WAV (WAVE_FORMAT_EXTENSIBLE, no speaker mask) of an IR's channels, e.g. an AmbiX ARIR. */
export function encodeWav(ir) {
  const nc = ir.channels.length, n = ir.length, dataBytes = n * nc * 4;
  const buf = new ArrayBuffer(68 + dataBytes), v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 60 + dataBytes, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 40, true);
  v.setUint16(20, 0xFFFE, true); v.setUint16(22, nc, true); v.setUint32(24, ir.sampleRate, true);
  v.setUint32(28, ir.sampleRate * nc * 4, true); v.setUint16(32, nc * 4, true); v.setUint16(34, 32, true);
  v.setUint16(36, 22, true); v.setUint16(38, 32, true); v.setUint32(40, 0, true);   // cbSize, valid bits, channel mask
  // KSDATAFORMAT_SUBTYPE_IEEE_FLOAT {00000003-0000-0010-8000-00aa00389b71}
  [0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71].forEach((x, i) => v.setUint8(44 + i, x));
  str(60, 'data'); v.setUint32(64, dataBytes, true);
  let o = 68;
  for (let i = 0; i < n; i++) for (let c = 0; c < nc; c++, o += 4) v.setFloat32(o, ir.channels[c][i], true);
  return buf;
}

/* ------------------------------------------------------------------ Web Audio playback */
export class RIRAuralizer {
  /**
   * @param opts.audioContext  existing AudioContext (optional)
   * @param opts.speakers      array of unit vectors, or a name from SPEAKER_LAYOUTS (default 'icosahedron')
   * @param opts.panExponent   directional lobe sharpness for distributing energy over speakers (default 4)
   * @param opts.crossfade     seconds to cross-fade from the old IR to a new one (default 0.4)
   * @param opts.maxLevel      largest IR energy norm played unscaled; 1 = a direct sound at 1 m (default 2)
   * Render mode: 'speakers' (default) or 'ambisonic' — call setMode() to switch (loads Omnitone)
   */
  constructor(opts = {}) {
    this.ctx = opts.audioContext || new (window.AudioContext || window.webkitAudioContext)();
    const sp = opts.speakers || 'icosahedron';
    this.speakers = typeof sp === 'string' ? SPEAKER_LAYOUTS[sp]() : sp;
    this.panExponent = opts.panExponent ?? 4;
    this.crossfade = opts.crossfade ?? 0.4;
    this.maxLevel = opts.maxLevel ?? 2;
    this._fadeDone = Promise.resolve();
    this._queue = Promise.resolve();          // setRIR calls run one after the other
    this.mode = 'speakers';
    this.ambiOrder = AMBI_ORDER;
    this.ambi = {};                           // order -> Omnitone FOA / HOA renderer, created by setMode()
    this._lastData = null;                    // tracer data of the current IR (re-synthesised on mode change)
    this._lastIR = null;
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

    // safety limiter: near-brickwall above -3 dBFS (its built-in makeup gain keeps peaks just under 0 dBFS)
    this.limiter = this.ctx.createDynamicsCompressor();
    this.limiter.threshold.value = -3;
    this.limiter.knee.value = 0;
    this.limiter.ratio.value = 20;
    this.limiter.attack.value = 0.002;
    this.limiter.release.value = 0.15;
    this.master.connect(this.limiter);
    this.limiter.connect(this.ctx.destination);

    // A/B switch, see setBypass(): the reverberant path (panners / ambisonic renderers) ends in wetOut; the dry
    // anechoic signal goes straight to master, loudness-matched to the current IR
    this.bypass = false;
    this.wetOut = this.ctx.createGain();
    this.wetOut.connect(this.master);
    this.dry = this.ctx.createGain();
    this.dry.gain.value = 0;
    this.input.connect(this.dry);
    this.dry.connect(this.master);

    // one HRTF panner per virtual speaker; built once so a new IR never restarts the HRTF state
    this.panners = this.speakers.map((s) => {
      const pan = this.ctx.createPanner();
      pan.panningModel = 'HRTF';
      pan.distanceModel = 'linear';
      pan.rolloffFactor = 0;                  // no distance gain: level is already inside the IR
      pan.refDistance = 1;
      pan.positionX.value = s[0]; pan.positionY.value = s[1]; pan.positionZ.value = s[2];
      pan.connect(this.wetOut);
      return pan;
    });
  }

  setGain(g) { this.master.gain.value = g; }

  /** true: play the anechoic signal without any RIR (A/B comparison), at the loudness of the current IR (its energy
   *  norm after headroom scaling, i.e. the RMS gain of the reverberant path for white noise). Short cross-fade. */
  setBypass(on) {
    this.bypass = !!on;
    this._updateBypass();
  }

  _updateBypass() {
    // setTargetAtTime starts from wherever the gain is, so a switch mid-fade doesn't jump (no cancel needed)
    const t = this.ctx.currentTime, tc = 0.015;
    // the binaural renderers come out ~2.5 dB below the IR's energy norm (measured, white noise, all modes)
    const dryLevel = 10 ** (-2.5 / 20) * (this.bank ? this.bank.level * this.bank.norm : 1);
    this.wetOut.gain.setTargetAtTime(this.bypass ? 0 : 1, t, tc);
    this.dry.gain.setTargetAtTime(this.bypass ? dryLevel : 0, t, tc);
  }

  /** 'speakers' or 'ambisonic' (order 1 or 3). Re-synthesises the current IR in the new format (no re-trace)
   *  and cross-fades. */
  async setMode(mode, order = this.ambiOrder) {
    if (mode !== 'speakers' && mode !== 'ambisonic') throw new Error('unknown render mode ' + mode);
    if (mode === 'ambisonic') {
      if (!AMBI_ORDERS.includes(order)) throw new Error('unsupported ambisonic order ' + order);
      if (!this.ambi[order]) {
        // loaded lazily: keeps the synthesis functions above usable without a browser / Omnitone
        const { default: Omnitone } = await import('omnitone');
        const r = order === 1 ? Omnitone.createFOARenderer(this.ctx)          // ACN channel map by default
                              : Omnitone.createHOARenderer(this.ctx, { ambisonicOrder: order });
        await r.initialize();                 // decodes Omnitone's built-in HRIRs for this order
        const trim = this.ctx.createGain();
        trim.gain.value = AMBI_RENDER[order].trim;
        r.output.connect(trim);
        trim.connect(this.wetOut);
        this.ambi[order] = r;
        this._rotateField(this._camElements);
      }
      this.ambiOrder = order;
    }
    this.mode = mode;
    if (this._lastData) return this.setRIR(this._lastData);
    return null;
  }

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

  /** Builds the IR from a RIRTracer that has already run(). Pass { omni, directional } if they were already read
   *  back (tracer.readOmni() / readDirectional() stall on the GPU); they are kept, not modified. */
  async setRIRFromTracer(tracer, { omni = tracer.readOmni(), directional = tracer.readDirectional() } = {}) {
    const o = tracer.opts, dp = tracer.directPath();
    return this.setRIR({
      bands: o.bands,
      omni, numBins: o.numBins, binDuration: o.binDuration,
      directional,
      sliceWidth: o.sliceWidth, sliceHeight: o.sliceHeight, atlasCols: o.atlasCols,
      numSlices: o.numSlices, sliceDuration: o.sliceDuration,
      speedOfSound: o.speedOfSound,
      airAbsorption: [...o.airAbsorption],
      direct: { distance: dp.distance, direction: [dp.direction.x, dp.direction.y, dp.direction.z] },
      worldFrame: !o.cameraRelative,
    });
  }

  /** data: see synthesizeSpeakerIR (+ optional worldFrame flag). Cross-fades from the previous IR. */
  setRIR(data) {
    this._lastData = data;
    const p = this._queue.then(() => this._setRIRNow(data));
    this._queue = p.catch(() => {});
    return p;
  }

  async _setRIRNow(data) {
    this.worldFrame = !!data.worldFrame;
    if (!this.worldFrame) this._setListenerOrientation(0, 0, -1, 0, 1, 0);

    // let the UI repaint before the (synchronous) synthesis
    await new Promise((r) => setTimeout(r, 0));
    const opts = { sampleRate: this.ctx.sampleRate, panExponent: this.panExponent, order: this.ambiOrder };
    const ir = this.mode === 'ambisonic' ? synthesizeAmbisonicIR(data, opts) : synthesizeSpeakerIR(data, this.speakers, opts);
    ir.kind = this.mode;
    this._lastIR = ir;

    // one fade at a time: scheduled ramps can't be interrupted portably (no cancelAndHoldAtTime in Firefox)
    await this._fadeDone;
    // headroom: energy norm (all speaker channels, or W of the ambisonic IR) ~ output RMS for white noise of RMS 1
    let energy = 0;
    for (const c of ir.normChannels || ir.channels.keys()) for (const v of ir.channels[c]) energy += v * v;
    ir.norm = Math.sqrt(energy);
    ir.level = Math.min(1, this.maxLevel / ir.norm);

    const bank = this._buildBank(ir);         // starts silent (input gain 0)
    bank.level = ir.level;
    bank.norm = ir.norm;
    // start a little ahead: the main thread may still be busy (GPU readback) when the events reach the audio thread
    const old = this.bank, t = this.ctx.currentTime + 0.05, T = Math.max(this.crossfade, 0.01);
    if (!old) {
      bank.gain.gain.value = bank.level;
    } else {
      // equal-power (the two reverbs are uncorrelated, so a linear fade would dip ~3 dB mid-way), built from
      // linear ramps: Firefox steps through setValueCurveAtTime values, which clicks
      const N = 32;
      const fade = (param, shape) => {
        param.setValueAtTime(shape(0), t);
        for (let i = 1; i <= N; i++) param.linearRampToValueAtTime(shape(i / N * Math.PI / 2), t + T * i / N);
      };
      fade(bank.gain.gain, (x) => bank.level * Math.sin(x));
      fade(old.gain.gain, (x) => old.level * Math.cos(x));
      this._fadeDone = new Promise((r) => setTimeout(r, (T + 0.05) * 1000));
      // the old convolvers are still ringing out what they received before the fade; free them once that's gone
      setTimeout(() => this._disposeBank(old), (T + ir.length / ir.sampleRate + 0.1) * 1000);
    }
    this.bank = bank;
    if (this.bypass) this._updateBypass();     // follow the new IR's loudness
    return ir;
  }

  _buildBank(ir) {
    const ctx = this.ctx;
    const gain = ctx.createGain();
    gain.gain.value = 0;                      // silent BEFORE it is connected
    const merger = ir.kind === 'ambisonic' ? ctx.createChannelMerger(ir.channels.length) : null;
    const convs = ir.channels.map((ch, k) => {
      const buf = ctx.createBuffer(1, ir.length, ir.sampleRate);
      buf.copyToChannel(ch, 0);

      const conv = ctx.createConvolver();
      conv.normalize = false;                 // MUST be set before .buffer, or the IR gets rescaled
      conv.buffer = buf;

      gain.connect(conv);
      if (merger) conv.connect(merger, 0, k);  // ACN channel k
      else conv.connect(this.panners[k]);
      return conv;
    });
    if (merger) merger.connect(this.ambi[ir.order].input);
    this.input.connect(gain);
    return { gain, convs, merger };
  }

  _disposeBank(bank) {
    try { this.input.disconnect(bank.gain); } catch (e) { /* already gone */ }
    bank.gain.disconnect();
    bank.convs.forEach((c) => c.disconnect());
    if (bank.merger) bank.merger.disconnect();
  }

  /** The current IR as an AmbiX (ACN / SN3D) ARIR in a float WAV, at the current ambisonic order (3rd order when
   *  rendering 'speakers', synthesised for export). Absolute level (direct sound = 1/distance), no headroom scaling.
   *  @returns { wav: ArrayBuffer, order } */
  exportAmbisonicWav() {
    if (!this._lastData) throw new Error('no RIR yet');
    const ir = this._lastIR?.kind === 'ambisonic' ? this._lastIR
      : synthesizeAmbisonicIR(this._lastData, { sampleRate: this.ctx.sampleRate, order: AMBI_ORDER });
    return { wav: encodeWav(ir), order: ir.order };
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

  /** World-frame IRs only: rotate the Web Audio listener and the ambisonic field with a Three.js camera (every
   *  frame). */
  setListenerFromCamera(camera) {
    if (!this.worldFrame) { this._camElements = null; this._rotateField(null); return; }
    if (camera.updateWorldMatrix) camera.updateWorldMatrix(true, false);
    const e = camera.matrixWorld.elements;     // columns: X (0-2), Y (4-6), Z (8-10); camera looks down -Z
    this._setListenerOrientation(-e[8], -e[9], -e[10], e[4], e[5], e[6]);
    this._camElements = e;
    this._rotateField(e);
  }

  /** For a field encoded with worldToAmbi(), Omnitone wants inverse(camera.matrixWorld) * diag(f), f per order
   *  (measured — the HOA renderer's identity turns the field by 180°, the FOA renderer's doesn't). Rotation part
   *  only: out(i, j) = R(j, i) * f(j). All created renderers are kept in step: a fading bank may use another order. */
  _rotateField(e) {
    const m = this._rot || (this._rot = new Float32Array(16));
    for (const [order, r] of Object.entries(this.ambi)) {
      const f = AMBI_RENDER[order].rotSigns;
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) m[j * 4 + i] = (e ? e[i * 4 + j] : +(i === j)) * f[j];
      m[15] = 1;
      r.setRotationMatrix4(m);
    }
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
