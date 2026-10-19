import { Part, sub, cross, dot, norm, writeGlb } from './glb.mjs';

const CELL = 0.6, DEPTH = 0.75;
const NX = 8, NY = 7, NZ = 8;
const X = NX * CELL, Y = NY * CELL, Z = NZ * CELL;

const wedges = new Part('Wedges', [0.42, 0.47, 0.55, 1]);
const add = (a, s, b) => a.map((v, i) => v + s * b[i]);

function wall(o, u, nu, v, nv, n) {
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    const c = (a, b) => add(add(o, (i + a) * CELL, u), (j + b) * CELL, v);
    const apex = add(c(0.5, 0.5), DEPTH, n);
    const base = [c(0, 0), c(1, 0), c(1, 1), c(0, 1)];
    for (let k = 0; k < 4; k++) {
      const tri = [base[k], base[(k + 1) % 4], apex];
      let fn = norm(cross(sub(tri[1], tri[0]), sub(tri[2], tri[0])));
      if (dot(fn, n) < 0) fn = fn.map((x) => -x);
      wedges.poly(tri, fn);
    }
  }
}
wall([0, 0, 0], [1, 0, 0], NX, [0, 0, 1], NZ, [0, 1, 0]);
wall([0, Y, 0], [1, 0, 0], NX, [0, 0, 1], NZ, [0, -1, 0]);
wall([0, 0, 0], [1, 0, 0], NX, [0, 1, 0], NY, [0, 0, 1]);
wall([0, 0, Z], [1, 0, 0], NX, [0, 1, 0], NY, [0, 0, -1]);
wall([0, 0, 0], [0, 0, 1], NZ, [0, 1, 0], NY, [1, 0, 0]);
wall([X, 0, 0], [0, 0, 1], NZ, [0, 1, 0], NY, [-1, 0, 0]);

writeGlb(new URL('./anechoic.glb', import.meta.url), 'Anechoic chamber', [wedges], {
  Source: [X / 2, Y / 2, Z / 2 - 0.75],
  Listener: [X / 2, Y / 2, Z / 2 + 0.75],
}, {
  absorption: [0.90, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
});
