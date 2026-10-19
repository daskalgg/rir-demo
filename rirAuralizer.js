const FOUR_PI = 4 * Math.PI;

export const AMBI_ORDERS = [1, 3];
export const AMBI_ORDER = 3, AMBI_CHANNELS = (AMBI_ORDER + 1) ** 2;
const ambiChannels = (order) => (order + 1) ** 2;
const AMBI_RENDER = {
  1: { trim: 10 ** (0.6 / 20), rotSigns: [1, 1, 1] },
  3: { trim: 10 ** (1.35 / 20), rotSigns: [-1, 1, -1] },
};

export const worldToAmbi = ([x, y, z]) => [-z, -x, y];

const S3 = Math.sqrt(3), S15 = Math.sqrt(15), S58 = Math.sqrt(5 / 8), S38 = Math.sqrt(3 / 8);
export function shEncode(x, y, z, out = new Float32Array(AMBI_CHANNELS)) {
  out[0] = 1;
  out[1] = y; out[2] = z; out[3] = x;
  out[4] = S3 * x * y; out[5] = S3 * y * z; out[6] = 0.5 * (3 * z * z - 1); out[7] = S3 * x * z; out[8] = 0.5 * S3 * (x * x - y * y);
  out[9] = S58 * y * (3 * x * x - y * y); out[10] = S15 * x * y * z; out[11] = S38 * y * (5 * z * z - 1);
  out[12] = 0.5 * z * (5 * z * z - 3); out[13] = S38 * x * (5 * z * z - 1); out[14] = 0.5 * S15 * z * (x * x - y * y);
  out[15] = S58 * x * (x * x - 3 * y * y);
  return out;
}

function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

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

function makeBandNoise(L, fs, crossovers, band, rng) {
  const WARM = 4096;
  const stages = [];
  if (band > 0) { const c = biquadCoefs('highpass', crossovers[band - 1], fs); stages.push(c, c); }
  if (band < crossovers.length) { const c = biquadCoefs('lowpass', crossovers[band], fs); stages.push(c, c); }
  const x = new Float64Array(L + WARM);
  for (let n = 0; n < x.length; n++) x[n] = rng() < 0.5 ? -1 : 1;
  for (const c of stages) applyBiquad(x, c);
  const y = new Float32Array(L);
  let ss = 0;
  for (let n = 0; n < L; n++) { y[n] = x[n + WARM]; ss += y[n] * y[n]; }
  const inv = 1 / Math.sqrt(ss / L);
  for (let n = 0; n < L; n++) y[n] *= inv;
  return y;
}

function equirectDir(u, v) {
  const lon = (u - 0.5) * 2 * Math.PI, lat = (v - 0.5) * Math.PI, cl = Math.cos(lat);
  return [cl * Math.sin(lon), Math.sin(lat), -cl * Math.cos(lon)];
}

const bandCrossovers = (bands, fs) => bands.slice(1).map((f, i) => Math.sqrt(f * bands[i])).filter((f) => f < 0.45 * fs);

export function synthesizeAmbisonicIR(data, opts = {}) {
  const B = data.bands.length;
  const { sampleRate: fs, seed = 1, order = AMBI_ORDER, crossovers = bandCrossovers(data.bands, opts.sampleRate) } = opts;
  const NC = ambiChannels(order);
  const L = Math.ceil(data.numBins * data.binDuration * fs) + 2;
  const Ns = data.binDuration * fs;
  const rng = mulberry32(seed);
  const { sliceWidth: W, sliceHeight: H, atlasCols: C, numSlices: S, directional: D } = data, Wa = W * C, P = W * H;

  const pixY = new Float32Array(P * AMBI_CHANNELS), tmp = new Float32Array(AMBI_CHANNELS);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    pixY.set(shEncode(...worldToAmbi(equirectDir((i + 0.5) / W, (j + 0.5) / H)), tmp), (j * W + i) * AMBI_CHANNELS);
  }
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
  for (let b = 0; b < B; b++) mean[(S * B + b) * AMBI_CHANNELS] = 1;
  const Y = new Float32Array(AMBI_CHANNELS);
  const drawDirection = (slice, b) => {
    let dir;
    const t = slice < S ? tot[slice * B + b] : 0;
    if (t > 0) {
      const base = (slice * B + b) * P, u = rng() * t;
      let lo = 0, hi = P - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[base + mid] < u) lo = mid + 1; else hi = mid; }
      dir = equirectDir(((lo % W) + rng()) / W, (Math.floor(lo / W) + rng()) / H);
    } else {
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
    const noise = makeBandNoise(L, fs, crossovers, Math.min(b, crossovers.length), rng);
    sig.fill(0);
    for (let n = 0; n < L; n++) {
      const bin = Math.floor(n / Ns);
      if (bin >= data.numBins) break;
      if (Math.abs(bin - directBin) <= 1) continue;
      const e = data.omni[bin * B + b];
      if (e > 0) sig[n] = Math.sqrt(e * envScale) * noise[n];
    }
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

  if (data.direct) {
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
  return { channels: out, sampleRate: fs, length: L, order };
}

export function encodeWav(ir) {
  const nc = ir.channels.length, n = ir.length, dataBytes = n * nc * 4;
  const buf = new ArrayBuffer(68 + dataBytes), v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 60 + dataBytes, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 40, true);
  v.setUint16(20, 0xFFFE, true); v.setUint16(22, nc, true); v.setUint32(24, ir.sampleRate, true);
  v.setUint32(28, ir.sampleRate * nc * 4, true); v.setUint16(32, nc * 4, true); v.setUint16(34, 32, true);
  v.setUint16(36, 22, true); v.setUint16(38, 32, true); v.setUint32(40, 0, true);
  [0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71].forEach((x, i) => v.setUint8(44 + i, x));
  str(60, 'data'); v.setUint32(64, dataBytes, true);
  let o = 68;
  for (let i = 0; i < n; i++) for (let c = 0; c < nc; c++, o += 4) v.setFloat32(o, ir.channels[c][i], true);
  return buf;
}

export class RIRAuralizer {
  constructor(opts = {}) {
    this.ctx = opts.audioContext || new (window.AudioContext || window.webkitAudioContext)();
    this.crossfade = opts.crossfade ?? 0.4;
    this.maxLevel = opts.maxLevel ?? 2;
    this._fadeDone = Promise.resolve();
    this._queue = Promise.resolve();
    this.ambiOrder = opts.order ?? 1;
    if (!AMBI_ORDERS.includes(this.ambiOrder)) throw new Error('unsupported ambisonic order ' + this.ambiOrder);
    this.ambi = {};
    this._lastData = null;
    this._lastIR = null;
    this.worldFrame = false;
    this.anechoic = null;
    this.srcNode = null;
    this.bank = null;

    this.input = this.ctx.createGain();
    this.input.channelCount = 1;
    this.input.channelCountMode = 'explicit';
    this.input.channelInterpretation = 'speakers';

    this.master = this.ctx.createGain();
    this.master.gain.value = 0.5;

    this.limiter = this.ctx.createDynamicsCompressor();
    this.limiter.threshold.value = -3;
    this.limiter.knee.value = 0;
    this.limiter.ratio.value = 20;
    this.limiter.attack.value = 0.002;
    this.limiter.release.value = 0.15;
    this.master.connect(this.limiter);
    this.limiter.connect(this.ctx.destination);

    this.bypass = false;
    this.wetOut = this.ctx.createGain();
    this.wetOut.connect(this.master);
    this.dry = this.ctx.createGain();
    this.dry.gain.value = 0;
    this.input.connect(this.dry);
    this.dry.connect(this.master);
  }

  setGain(g) { this.master.gain.value = g; }

  setBypass(on) {
    this.bypass = !!on;
    this._updateBypass();
  }

  _updateBypass() {
    const t = this.ctx.currentTime, tc = 0.015;
    const dryLevel = 10 ** (-2.5 / 20) * (this.bank ? this.bank.level * this.bank.norm : 1);
    this.wetOut.gain.setTargetAtTime(this.bypass ? 0 : 1, t, tc);
    this.dry.gain.setTargetAtTime(this.bypass ? dryLevel : 0, t, tc);
  }

  async setOrder(order) {
    if (!AMBI_ORDERS.includes(order)) throw new Error('unsupported ambisonic order ' + order);
    await this._renderer(order);
    this.ambiOrder = order;
    if (this._lastData) return this.setRIR(this._lastData);
    return null;
  }

  async _renderer(order) {
    if (!this.ambi[order]) {
      const { default: Omnitone } = await import('omnitone');
      const r = order === 1 ? Omnitone.createFOARenderer(this.ctx)
                            : Omnitone.createHOARenderer(this.ctx, { ambisonicOrder: order });
      await r.initialize();
      const trim = this.ctx.createGain();
      trim.gain.value = AMBI_RENDER[order].trim;
      r.output.connect(trim);
      trim.connect(this.wetOut);
      this.ambi[order] = r;
      this._rotateField(this._camElements);
    }
    return this.ambi[order];
  }

  async loadAnechoic(src) {
    if (src instanceof AudioBuffer) { this.anechoic = src; return src; }
    let ab;
    if (typeof src === 'string') ab = await (await fetch(src)).arrayBuffer();
    else if (src instanceof ArrayBuffer) ab = src;
    else ab = await src.arrayBuffer();
    this.anechoic = await this.ctx.decodeAudioData(ab);
    return this.anechoic;
  }

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
      direct: dp.visible ? { distance: dp.distance, direction: [dp.direction.x, dp.direction.y, dp.direction.z] } : null,
      worldFrame: !o.cameraRelative,
    });
  }

  setRIR(data) {
    this._lastData = data;
    const p = this._queue.then(() => this._setRIRNow(data));
    this._queue = p.catch(() => {});
    return p;
  }

  async _setRIRNow(data) {
    this.worldFrame = !!data.worldFrame;
    await this._renderer(this.ambiOrder);

    await new Promise((r) => setTimeout(r, 0));
    const ir = synthesizeAmbisonicIR(data, { sampleRate: this.ctx.sampleRate, order: this.ambiOrder });
    this._lastIR = ir;

    await this._fadeDone;
    let energy = 0;
    for (const v of ir.channels[0]) energy += v * v;
    ir.norm = Math.sqrt(energy);
    ir.level = Math.min(1, this.maxLevel / ir.norm);

    const bank = this._buildBank(ir);
    bank.level = ir.level;
    bank.norm = ir.norm;
    const old = this.bank, t = this.ctx.currentTime + 0.05, T = Math.max(this.crossfade, 0.01);
    if (!old) {
      bank.gain.gain.value = bank.level;
    } else {
      const N = 32;
      const fade = (param, shape) => {
        param.setValueAtTime(shape(0), t);
        for (let i = 1; i <= N; i++) param.linearRampToValueAtTime(shape(i / N * Math.PI / 2), t + T * i / N);
      };
      fade(bank.gain.gain, (x) => bank.level * Math.sin(x));
      fade(old.gain.gain, (x) => old.level * Math.cos(x));
      this._fadeDone = new Promise((r) => setTimeout(r, (T + 0.05) * 1000));
      setTimeout(() => this._disposeBank(old), (T + ir.length / ir.sampleRate + 0.1) * 1000);
    }
    this.bank = bank;
    if (this.bypass) this._updateBypass();
    return ir;
  }

  _buildBank(ir) {
    const ctx = this.ctx;
    const gain = ctx.createGain();
    gain.gain.value = 0;
    const merger = ctx.createChannelMerger(ir.channels.length);
    const convs = ir.channels.map((ch, k) => {
      const buf = ctx.createBuffer(1, ir.length, ir.sampleRate);
      buf.copyToChannel(ch, 0);

      const conv = ctx.createConvolver();
      conv.normalize = false;
      conv.buffer = buf;

      gain.connect(conv);
      conv.connect(merger, 0, k);
      return conv;
    });
    merger.connect(this.ambi[ir.order].input);
    this.input.connect(gain);
    return { gain, convs, merger };
  }

  _disposeBank(bank) {
    try { this.input.disconnect(bank.gain); } catch (e) {  }
    bank.gain.disconnect();
    bank.convs.forEach((c) => c.disconnect());
    bank.merger.disconnect();
  }

  exportAmbisonicWav() {
    if (!this._lastIR) throw new Error('no RIR yet');
    return { wav: encodeWav(this._lastIR), order: this._lastIR.order };
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
    if (this.srcNode) { try { this.srcNode.stop(); } catch (e) {  } this.srcNode.disconnect(); this.srcNode = null; }
  }

  setListenerFromCamera(camera) {
    if (!this.worldFrame) { this._camElements = null; this._rotateField(null); return; }
    if (camera.updateWorldMatrix) camera.updateWorldMatrix(true, false);
    const e = camera.matrixWorld.elements;
    this._camElements = e;
    this._rotateField(e);
  }

  _rotateField(e) {
    const m = this._rot || (this._rot = new Float32Array(16));
    for (const [order, r] of Object.entries(this.ambi)) {
      const f = AMBI_RENDER[order].rotSigns;
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) m[j * 4 + i] = (e ? e[i * 4 + j] : +(i === j)) * f[j];
      m[15] = 1;
      r.setRotationMatrix4(m);
    }
  }
}
