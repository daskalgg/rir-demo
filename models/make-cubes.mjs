// Generates cube-10m.glb and cube-20m.glb: empty cubic rooms (default material), identical up to scale.
// Source and Listener at head height against opposite walls (x = 0 and x = L), WALL m out from the wall at its
// horizontal centre: 9 m apart in the 10 m cube, 19 m in the 20 m one.
// The scene's extras set a flat absorption of 0.2 in every band (scattering keeps the demo default).
// Y-up, metres, room spans [0..L]^3.
//
//   node models/make-cubes.mjs
import { Part, writeGlb } from './glb.mjs';

const HEAD = 1.6, WALL = 0.5;

for (const L of [10, 20]) {
  const walls = new Part('Walls', [0.74, 0.72, 0.68, 1]);
  walls.poly([[0, 0, 0], [L, 0, 0], [L, 0, L], [0, 0, L]], [0, 1, 0]);    // floor
  walls.poly([[0, L, 0], [L, L, 0], [L, L, L], [0, L, L]], [0, -1, 0]);   // ceiling
  walls.poly([[0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0]], [0, 0, 1]);    // z = 0
  walls.poly([[0, 0, L], [L, 0, L], [L, L, L], [0, L, L]], [0, 0, -1]);   // z = L
  walls.poly([[0, 0, 0], [0, 0, L], [0, L, L], [0, L, 0]], [1, 0, 0]);    // x = 0
  walls.poly([[L, 0, 0], [L, 0, L], [L, L, L], [L, L, 0]], [-1, 0, 0]);   // x = L

  writeGlb(new URL(`./cube-${L}m.glb`, import.meta.url), `Cube ${L} m`, [walls], {
    Source: [WALL, HEAD, L / 2],
    Listener: [L - WALL, HEAD, L / 2],
  }, { absorption: Array(8).fill(0.2) });
}
