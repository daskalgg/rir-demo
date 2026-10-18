// Generates flat-chamber.glb: the anechoic chamber (make-anechoic.mjs) without its wedges — same 4.8 x 4.2 x 4.8 m
// outer box, same Source / Listener positions and same foam absorption, but six smooth walls. Load both to hear
// what the wedge geometry adds on top of the material.
//
//   node models/make-flat-chamber.mjs
import { Part, writeGlb } from './glb.mjs';

const X = 4.8, Y = 4.2, Z = 4.8;                // = make-anechoic.mjs outer box

const walls = new Part('Walls', [0.42, 0.47, 0.55, 1]);
walls.poly([[0, 0, 0], [X, 0, 0], [X, 0, Z], [0, 0, Z]], [0, 1, 0]);    // floor
walls.poly([[0, Y, 0], [X, Y, 0], [X, Y, Z], [0, Y, Z]], [0, -1, 0]);   // ceiling
walls.poly([[0, 0, 0], [X, 0, 0], [X, Y, 0], [0, Y, 0]], [0, 0, 1]);    // z = 0
walls.poly([[0, 0, Z], [X, 0, Z], [X, Y, Z], [0, Y, Z]], [0, 0, -1]);   // z = Z
walls.poly([[0, 0, 0], [0, 0, Z], [0, Y, Z], [0, Y, 0]], [1, 0, 0]);    // x = 0
walls.poly([[X, 0, 0], [X, 0, Z], [X, Y, Z], [X, Y, 0]], [-1, 0, 0]);   // x = X

writeGlb(new URL('./flat-chamber.glb', import.meta.url), 'Flat chamber', [walls], {
  Source: [X / 2, Y / 2, Z / 2 - 0.75],
  Listener: [X / 2, Y / 2, Z / 2 + 0.75],
}, {
  absorption: [0.90, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],   // = make-anechoic.mjs
});
