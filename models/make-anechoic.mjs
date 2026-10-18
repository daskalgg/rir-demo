// Generates anechoic.glb: an anechoic-chamber-shaped room — a box whose six faces (floor included, i.e. no
// wire-mesh walkway) are tiled with pyramid wedges pointing into the room.
// Wedges are 0.6 m square x 0.75 m deep: real chambers use slimmer, deeper wedges, but with 64x64 surfels
// (~0.29 m spacing here) anything finer is not resolved by the tracer.
// Y-up, metres, room spans [0..X] x [0..Y] x [0..Z]. Empty nodes "Source" / "Listener" 1.5 m apart in the middle.
// The wedge shape alone only shortens the reverb (~0.12 s instead of ~0.4 s at the demo's default absorption),
// so the scene's extras also set foam absorption for the whole room: { "absorption": [low, ..., high] }.
//
//   node models/make-anechoic.mjs
import { Part, sub, cross, dot, norm, writeGlb } from './glb.mjs';

const CELL = 0.6, DEPTH = 0.75;
const NX = 8, NY = 7, NZ = 8;                   // wedges per edge
const X = NX * CELL, Y = NY * CELL, Z = NZ * CELL; // outer box 4.8 x 4.2 x 4.8; free space between tips 3.3 x 2.7 x 3.3

const wedges = new Part('Wedges', [0.42, 0.47, 0.55, 1]);
const add = (a, s, b) => a.map((v, i) => v + s * b[i]);

// o = corner, u/v = in-plane axes (nu x nv cells), n = into the room
function wall(o, u, nu, v, nv, n) {
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    const c = (a, b) => add(add(o, (i + a) * CELL, u), (j + b) * CELL, v);
    const apex = add(c(0.5, 0.5), DEPTH, n);
    const base = [c(0, 0), c(1, 0), c(1, 1), c(0, 1)];
    for (let k = 0; k < 4; k++) {
      const tri = [base[k], base[(k + 1) % 4], apex];
      let fn = norm(cross(sub(tri[1], tri[0]), sub(tri[2], tri[0])));
      if (dot(fn, n) < 0) fn = fn.map((x) => -x);   // faces look into the room
      wedges.poly(tri, fn);
    }
  }
}
wall([0, 0, 0], [1, 0, 0], NX, [0, 0, 1], NZ, [0, 1, 0]);    // floor
wall([0, Y, 0], [1, 0, 0], NX, [0, 0, 1], NZ, [0, -1, 0]);   // ceiling
wall([0, 0, 0], [1, 0, 0], NX, [0, 1, 0], NY, [0, 0, 1]);    // z = 0
wall([0, 0, Z], [1, 0, 0], NX, [0, 1, 0], NY, [0, 0, -1]);   // z = Z
wall([0, 0, 0], [0, 0, 1], NZ, [0, 1, 0], NY, [1, 0, 0]);    // x = 0
wall([X, 0, 0], [0, 0, 1], NZ, [0, 1, 0], NY, [-1, 0, 0]);   // x = X

writeGlb(new URL('./anechoic.glb', import.meta.url), 'Anechoic chamber', [wedges], {
  Source: [X / 2, Y / 2, Z / 2 - 0.75],
  Listener: [X / 2, Y / 2, Z / 2 + 0.75],
}, {
  // 125 Hz .. 16 kHz; 0.75 m wedges stop working below ~115 Hz (quarter wavelength)
  absorption: [0.90, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
});
