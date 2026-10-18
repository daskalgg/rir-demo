// Generates two elliptical rooms on the same 20 x 12 m elliptical floor, foci at head height:
//   ellipse.glb       "whispering gallery": vertical walls up to head height, and above that the upper half of a
//                     prolate spheroid whose foci are at head height. Every ray leaving one focus upwards reflects
//                     off the dome straight through the other focus; rays leaving downwards still converge on the
//                     vertical line through it.
//   ellipse-flat.glb  elliptic cylinder: the same walls carried straight up to a flat ceiling at the dome's top
//                     height. Only the horizontal projection focuses (onto the vertical line through the other
//                     focus); floor / ceiling reflections spread the arrivals over height.
// Y-up, metres, major axis along X. Empty nodes "Source" / "Listener" sit exactly on the two foci.
//
//   node models/make-ellipse.mjs
import { Part, norm, writeGlb } from './glb.mjs';

const A = 10, B = 6;                            // semi-axes: 20 m long (X), 12 m wide (Z), dome rises B above the foci
const C = Math.sqrt(A * A - B * B);             // focal distance (8 m)
const H = 1.6;                                  // height of the foci = top of the vertical walls
const NT = 96, NP = 32;                         // dome: segments along the major axis / around it

// spheroid of revolution about the X axis through (0, H, 0); theta along X, phi around it (0 = +Z, pi = -Z)
const S = (theta, phi) => [A * Math.cos(theta), H + B * Math.sin(theta) * Math.sin(phi), B * Math.sin(theta) * Math.cos(phi)];
const inward = ([x, y, z]) => norm([-x / (A * A), -(y - H) / (B * B), -z / (B * B)]);
const E = (t, y) => [A * Math.cos(t), y, B * Math.sin(t)];   // ellipse of the walls / floor rim

const dome = new Part('Dome', [0.86, 0.82, 0.74, 1]);
for (let i = 0; i < NT; i++) for (let j = 0; j < NP; j++) {
  const t0 = Math.PI * i / NT, t1 = Math.PI * (i + 1) / NT, p0 = Math.PI * j / NP, p1 = Math.PI * (j + 1) / NP;
  dome.poly([S(t0, p0), S(t1, p0), S(t1, p1), S(t0, p1)], inward(S((t0 + t1) / 2, (p0 + p1) / 2)));
}

// walls and floor share the dome's rim vertices: t = theta on the +Z side, t = 2pi - theta on the -Z side
const N = 2 * NT, tAt = (k) => 2 * Math.PI * k / N;
function walls(height) {
  const part = new Part('Walls', [0.74, 0.7, 0.62, 1]);
  for (let k = 0; k < N; k++) {
    const t0 = tAt(k), t1 = tAt(k + 1), tm = (t0 + t1) / 2;
    part.poly([E(t0, 0), E(t1, 0), E(t1, height), E(t0, height)], norm([-Math.cos(tm) / A, 0, -Math.sin(tm) / B]));
  }
  return part;
}
function disc(name, color, y, n) {               // elliptical floor / ceiling, fan from the centre
  const part = new Part(name, color);
  for (let k = 0; k < N; k++) part.poly([[0, y, 0], E(tAt(k), y), E(tAt(k + 1), y)], n);
  return part;
}
const floor = disc('Floor', [0.5, 0.44, 0.38, 1], 0, [0, 1, 0]);
const markers = { Source: [-C, H, 0], Listener: [C, H, 0] };

writeGlb(new URL('./ellipse.glb', import.meta.url), 'Ellipse', [dome, walls(H), floor], markers);
writeGlb(new URL('./ellipse-flat.glb', import.meta.url), 'Ellipse (flat ceiling)',
  [walls(H + B), disc('Ceiling', [0.86, 0.82, 0.74, 1], H + B, [0, -1, 0]), floor], markers);
