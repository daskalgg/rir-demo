import { Part, writeGlb } from './glb.mjs';

const X = 4.8, Y = 4.2, Z = 4.8;

const walls = new Part('Walls', [0.42, 0.47, 0.55, 1]);
walls.poly([[0, 0, 0], [X, 0, 0], [X, 0, Z], [0, 0, Z]], [0, 1, 0]);
walls.poly([[0, Y, 0], [X, Y, 0], [X, Y, Z], [0, Y, Z]], [0, -1, 0]);
walls.poly([[0, 0, 0], [X, 0, 0], [X, Y, 0], [0, Y, 0]], [0, 0, 1]);
walls.poly([[0, 0, Z], [X, 0, Z], [X, Y, Z], [0, Y, Z]], [0, 0, -1]);
walls.poly([[0, 0, 0], [0, 0, Z], [0, Y, Z], [0, Y, 0]], [1, 0, 0]);
walls.poly([[X, 0, 0], [X, 0, Z], [X, Y, Z], [X, Y, 0]], [-1, 0, 0]);

writeGlb(new URL('./flat-chamber.glb', import.meta.url), 'Flat chamber', [walls], {
  Source: [X / 2, Y / 2, Z / 2 - 0.75],
  Listener: [X / 2, Y / 2, Z / 2 + 0.75],
}, {
  absorption: [0.90, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
});
