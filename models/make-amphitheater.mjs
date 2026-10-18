// Generates two large Roman-style theatres — semicircular stepped seating (cavea) around a circular orchestra,
// a raised stage with a tall back wall (scaenae frons) and side walls:
//   amphitheater.glb         open to the sky, with a sloped roof over the stage only
//   amphitheater-closed.glb  the same, roofed over at the top of the walls (13.1 m) like an odeon: a closed hall
// Y-up, metres, seating on the +Z side, stage on -Z. Same material (demo default) and start positions in both.
// Empty nodes "Source" (on stage) and "Listener" (in the seats) give the demo its start positions.
//
//   node models/make-amphitheater.mjs
import { Part, cross, norm, writeGlb } from './glb.mjs';

const R0 = 12;                                  // orchestra radius
const ROWS = 24, TREAD = 0.8, RISER = 0.42;     // seating steps
const SEG = 96;                                 // segments over the 180° arc
const R1 = R0 + ROWS * TREAD;                   // outer seating radius (31.2 m)
const TOP = ROWS * RISER;                       // top row height (10.1 m)
const WALL_H = TOP + 3;                         // back wall of the seating and stage walls
const STAGE_FRONT = -4, STAGE_BACK = -12, STAGE_H = 1.2;

const P = (r, phi, y) => [r * Math.sin(phi), y, r * Math.cos(phi)];   // phi = 0 -> +Z (centre of the seating)

const UP = [0, 1, 0], DOWN = [0, -1, 0], TO_AUDIENCE = [0, 0, 1];
const phiAt = (s) => -Math.PI / 2 + Math.PI * s / SEG;

/* ---------------------------------------------------------------- seating */
const seating = new Part('Seating', [0.78, 0.74, 0.66, 1]);
for (let k = 0; k < ROWS; k++) {
  const rIn = R0 + k * TREAD, rOut = rIn + TREAD, y0 = k * RISER, y1 = y0 + RISER;
  for (let s = 0; s < SEG; s++) {
    const p0 = phiAt(s), p1 = phiAt(s + 1), pm = (p0 + p1) / 2;
    seating.poly([P(rIn, p0, y1), P(rOut, p0, y1), P(rOut, p1, y1), P(rIn, p1, y1)], UP);                       // tread
    seating.poly([P(rIn, p0, y0), P(rIn, p1, y0), P(rIn, p1, y1), P(rIn, p0, y1)], [-Math.sin(pm), 0, -Math.cos(pm)]); // riser
  }
  for (const side of [-1, 1]) {                 // end walls of the seating block (plane z = 0, facing the stage)
    seating.poly([[side * rIn, 0, 0], [side * rOut, 0, 0], [side * rOut, y1, 0], [side * rIn, y1, 0]], [0, 0, -1]);
  }
}
for (let s = 0; s < SEG; s++) {                 // back wall above the top row
  const p0 = phiAt(s), p1 = phiAt(s + 1), pm = (p0 + p1) / 2;
  seating.poly([P(R1, p0, TOP), P(R1, p1, TOP), P(R1, p1, WALL_H), P(R1, p0, WALL_H)], [-Math.sin(pm), 0, -Math.cos(pm)]);
}

/* ---------------------------------------------------------------- orchestra + side passages */
const ground = new Part('Orchestra', [0.62, 0.57, 0.5, 1]);
for (let s = 0; s < SEG; s++) ground.poly([[0, 0, 0], P(R0, phiAt(s), 0), P(R0, phiAt(s + 1), 0)], UP);
ground.poly([[-R1, 0, STAGE_FRONT], [R1, 0, STAGE_FRONT], [R1, 0, 0], [-R1, 0, 0]], UP);

/* ---------------------------------------------------------------- stage building */
const stage = new Part('Stage', [0.55, 0.42, 0.32, 1]);
stage.poly([[-R1, STAGE_H, STAGE_BACK], [R1, STAGE_H, STAGE_BACK], [R1, STAGE_H, STAGE_FRONT], [-R1, STAGE_H, STAGE_FRONT]], UP);
stage.poly([[-R1, 0, STAGE_FRONT], [R1, 0, STAGE_FRONT], [R1, STAGE_H, STAGE_FRONT], [-R1, STAGE_H, STAGE_FRONT]], TO_AUDIENCE);

function stageWalls(withStageRoof) {
  const walls = new Part('ScaenaeFrons', [0.82, 0.78, 0.72, 1]);
  walls.poly([[-R1, STAGE_H, STAGE_BACK], [R1, STAGE_H, STAGE_BACK], [R1, WALL_H, STAGE_BACK], [-R1, WALL_H, STAGE_BACK]], TO_AUDIENCE);
  for (const side of [-1, 1]) {                 // side walls, from the stage back wall to the end of the seating
    walls.poly([[side * R1, 0, STAGE_BACK], [side * R1, 0, 0], [side * R1, WALL_H, 0], [side * R1, WALL_H, STAGE_BACK]], [-side, 0, 0]);
  }
  if (withStageRoof) {
    const roofLow = WALL_H - 1.5, roofN = norm(cross([0, roofLow - WALL_H, STAGE_FRONT - STAGE_BACK], [1, 0, 0]));
    walls.poly([[-R1, WALL_H, STAGE_BACK], [R1, WALL_H, STAGE_BACK], [R1, roofLow, STAGE_FRONT], [-R1, roofLow, STAGE_FRONT]],
      roofN[1] < 0 ? roofN : roofN.map((v) => -v));   // sloped roof over the stage, facing down
  }
  return walls;
}

/* ---------------------------------------------------------------- ceiling (closed version) */
// flat, at the top of the walls: half disc over the seating + rectangle over the stage building; the rim
// vertices match the back wall's, and the two pieces meet along z = 0, so the hall is closed
const ceiling = new Part('Ceiling', [0.7, 0.66, 0.6, 1]);
for (let s = 0; s < SEG; s++) ceiling.poly([[0, WALL_H, 0], P(R1, phiAt(s), WALL_H), P(R1, phiAt(s + 1), WALL_H)], DOWN);
ceiling.poly([[-R1, WALL_H, STAGE_BACK], [R1, WALL_H, STAGE_BACK], [R1, WALL_H, 0], [-R1, WALL_H, 0]], DOWN);

/* ---------------------------------------------------------------- GLB */
const markers = {
  Source: [0, STAGE_H + 1.6, (STAGE_FRONT + STAGE_BACK) / 2],
  Listener: P(R0 + 10.5 * TREAD, 20 * Math.PI / 180, 11 * RISER + 1.2),   // row 11, head height
};
writeGlb(new URL('./amphitheater.glb', import.meta.url), 'Amphitheater',
  [seating, ground, stage, stageWalls(true)], markers);
writeGlb(new URL('./amphitheater-closed.glb', import.meta.url), 'Amphitheater (closed)',
  [seating, ground, stage, stageWalls(false), ceiling], markers);
