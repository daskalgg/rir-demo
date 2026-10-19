import { Part, writeGlb } from './glb.mjs';

const N = 8, CELL = 3, H = 3, HEAD = 1.6, SEED = 7;

let seed = SEED;
const rand = () => {
  seed = (seed + 0x6D2B79F5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const wallE = Array.from({ length: N }, () => Array(N).fill(true));
const wallS = Array.from({ length: N }, () => Array(N).fill(true));
const seen = Array.from({ length: N }, () => Array(N).fill(false));
const stack = [[0, 0]];
seen[0][0] = true;
while (stack.length) {
  const [x, z] = stack[stack.length - 1];
  const next = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    .map(([dx, dz]) => [x + dx, z + dz, dx, dz])
    .filter(([nx, nz]) => nx >= 0 && nx < N && nz >= 0 && nz < N && !seen[nx][nz]);
  if (!next.length) { stack.pop(); continue; }
  const [nx, nz, dx, dz] = next[Math.floor(rand() * next.length)];
  if (dx === 1) wallE[x][z] = false;
  if (dx === -1) wallE[nx][nz] = false;
  if (dz === 1) wallS[x][z] = false;
  if (dz === -1) wallS[nx][nz] = false;
  seen[nx][nz] = true;
  stack.push([nx, nz]);
}

const dist = Array.from({ length: N }, () => Array(N).fill(-1));
dist[0][0] = 0;
const queue = [[0, 0]];
let far = [0, 0];
while (queue.length) {
  const [x, z] = queue.shift();
  if (dist[x][z] > dist[far[0]][far[1]]) far = [x, z];
  const open = [
    [x + 1, z, x < N - 1 && !wallE[x][z]], [x - 1, z, x > 0 && !wallE[x - 1]?.[z]],
    [x, z + 1, z < N - 1 && !wallS[x][z]], [x, z - 1, z > 0 && !wallS[x][z - 1]],
  ];
  for (const [nx, nz, ok] of open) {
    if (ok && dist[nx][nz] < 0) { dist[nx][nz] = dist[x][z] + 1; queue.push([nx, nz]); }
  }
}

const L = N * CELL;
const floor = new Part('Floor', [0.55, 0.52, 0.48, 1]);
const ceiling = new Part('Ceiling', [0.82, 0.82, 0.8, 1]);
const walls = new Part('Walls', [0.74, 0.72, 0.68, 1]);
floor.poly([[0, 0, 0], [L, 0, 0], [L, 0, L], [0, 0, L]], [0, 1, 0]);
ceiling.poly([[0, H, 0], [L, H, 0], [L, H, L], [0, H, L]], [0, -1, 0]);
const wallX = (x, z0, z1, n) => walls.poly([[x, 0, z0], [x, 0, z1], [x, H, z1], [x, H, z0]], [n, 0, 0]);
const wallZ = (z, x0, x1, n) => walls.poly([[x0, 0, z], [x1, 0, z], [x1, H, z], [x0, H, z]], [0, 0, n]);
wallX(0, 0, L, 1); wallX(L, 0, L, -1); wallZ(0, 0, L, 1); wallZ(L, 0, L, -1);
for (let x = 0; x < N; x++) {
  for (let z = 0; z < N; z++) {
    if (x < N - 1 && wallE[x][z]) wallX((x + 1) * CELL, z * CELL, (z + 1) * CELL, 1);
    if (z < N - 1 && wallS[x][z]) wallZ((z + 1) * CELL, x * CELL, (x + 1) * CELL, 1);
  }
}

const centre = ([x, z]) => [(x + 0.5) * CELL, HEAD, (z + 0.5) * CELL];
writeGlb(new URL('./maze.glb', import.meta.url), 'Maze', [floor, ceiling, walls], {
  Source: centre([0, 0]),
  Listener: centre(far),
});
const [s, l] = [centre([0, 0]), centre(far)];
console.log(`route: ${dist[far[0]][far[1]]} cells ≈ ${dist[far[0]][far[1]] * CELL} m, ` +
  `straight line ${Math.hypot(l[0] - s[0], l[2] - s[2]).toFixed(1)} m`);
