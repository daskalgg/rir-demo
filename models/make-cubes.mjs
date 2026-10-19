import { Part, writeGlb } from './glb.mjs';

const HEAD = 1.6, WALL = 0.5;

for (const L of [10, 20]) {
  const walls = new Part('Walls', [0.74, 0.72, 0.68, 1]);
  walls.poly([[0, 0, 0], [L, 0, 0], [L, 0, L], [0, 0, L]], [0, 1, 0]);
  walls.poly([[0, L, 0], [L, L, 0], [L, L, L], [0, L, L]], [0, -1, 0]);
  walls.poly([[0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0]], [0, 0, 1]);
  walls.poly([[0, 0, L], [L, 0, L], [L, L, L], [0, L, L]], [0, 0, -1]);
  walls.poly([[0, 0, 0], [0, 0, L], [0, L, L], [0, L, 0]], [1, 0, 0]);
  walls.poly([[L, 0, 0], [L, 0, L], [L, L, L], [L, L, 0]], [-1, 0, 0]);

  writeGlb(new URL(`./cube-${L}m.glb`, import.meta.url), `Cube ${L} m`, [walls], {
    Source: [WALL, HEAD, L / 2],
    Listener: [L - WALL, HEAD, L / 2],
  }, { absorption: Array(8).fill(0.2) });
}
