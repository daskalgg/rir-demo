// Tiny helpers shared by the model generators: flat-shaded mesh parts and a GLB writer.
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a) => { const l = Math.hypot(...a); return a.map((v) => v / l); };

export class Part {
  constructor(name, color) { Object.assign(this, { name, color, pos: [], nrm: [], idx: [] }); }
  /** Flat convex polygon facing n (winding is fixed up to match n; repeated points, e.g. at poles, are dropped). */
  poly(pts, n) {
    pts = pts.filter((p, i) => Math.hypot(...sub(p, pts[(i + pts.length - 1) % pts.length])) > 1e-9);
    if (pts.length < 3) return;
    if (dot(cross(sub(pts[1], pts[0]), sub(pts[2], pts[0])), n) < 0) pts = pts.slice().reverse();
    const base = this.pos.length / 3;
    for (const p of pts) { this.pos.push(...p); this.nrm.push(...n); }
    for (let i = 1; i + 1 < pts.length; i++) this.idx.push(base, base + i, base + i + 1);
  }
}

/** Writes parts (+ empty marker nodes { name: [x, y, z] }, + optional scene extras) as a GLB and prints a summary. */
export function writeGlb(url, sceneName, parts, markers = {}, extras = undefined) {
  const chunks = [], bufferViews = [], accessors = [], meshes = [], materials = [], nodes = [];
  let offset = 0;
  const addView = (typed, target) => {
    const bytes = Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength);
    bufferViews.push({ buffer: 0, byteOffset: offset, byteLength: bytes.length, target });
    chunks.push(bytes);
    offset += bytes.length;                     // float32 / uint32 data keeps 4-byte alignment
    return bufferViews.length - 1;
  };
  for (const part of parts) {
    const pos = new Float32Array(part.pos), min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < pos.length; i++) { min[i % 3] = Math.min(min[i % 3], pos[i]); max[i % 3] = Math.max(max[i % 3], pos[i]); }
    accessors.push({ bufferView: addView(pos, 34962), componentType: 5126, count: pos.length / 3, type: 'VEC3', min, max });
    accessors.push({ bufferView: addView(new Float32Array(part.nrm), 34962), componentType: 5126, count: pos.length / 3, type: 'VEC3' });
    accessors.push({ bufferView: addView(new Uint32Array(part.idx), 34963), componentType: 5125, count: part.idx.length, type: 'SCALAR' });
    const a = accessors.length;
    materials.push({ name: part.name, doubleSided: true,
      pbrMetallicRoughness: { baseColorFactor: part.color, metallicFactor: 0, roughnessFactor: 0.9 } });
    meshes.push({ name: part.name, primitives: [{ attributes: { POSITION: a - 3, NORMAL: a - 2 }, indices: a - 1, material: materials.length - 1 }] });
    nodes.push({ name: part.name, mesh: meshes.length - 1 });
  }
  for (const [name, translation] of Object.entries(markers)) nodes.push({ name, translation });

  const bin = Buffer.concat(chunks);
  const json = {
    asset: { version: '2.0', generator: 'rir-demo models/' },
    scene: 0, scenes: [{ name: sceneName, nodes: nodes.map((_, i) => i), extras }],
    nodes, meshes, materials, accessors, bufferViews, buffers: [{ byteLength: bin.length }],
  };
  const pad = (buf, fill) => Buffer.concat([buf, Buffer.alloc((4 - buf.length % 4) % 4, fill)]);
  const jsonChunk = pad(Buffer.from(JSON.stringify(json)), 0x20), binChunk = pad(bin, 0);
  const header = Buffer.alloc(12), jh = Buffer.alloc(8), bh = Buffer.alloc(8);
  header.writeUInt32LE(0x46546C67, 0); header.writeUInt32LE(2, 4);
  header.writeUInt32LE(12 + 8 + jsonChunk.length + 8 + binChunk.length, 8);
  jh.writeUInt32LE(jsonChunk.length, 0); jh.writeUInt32LE(0x4E4F534A, 4);
  bh.writeUInt32LE(binChunk.length, 0); bh.writeUInt32LE(0x004E4942, 4);
  const path = fileURLToPath(url);
  writeFileSync(path, Buffer.concat([header, jh, jsonChunk, bh, binChunk]));

  let area = 0, tris = 0;
  for (const p of parts) for (let i = 0; i < p.idx.length; i += 3, tris++) {
    const v = (k) => p.pos.slice(3 * p.idx[i + k], 3 * p.idx[i + k] + 3);
    area += 0.5 * Math.hypot(...cross(sub(v(1), v(0)), sub(v(2), v(0))));
  }
  console.log(`wrote ${path}: ${tris} triangles, ${area.toFixed(0)} m²`);
  writeIndex(dirname(path));
}

/** Writes index.json in dir: every .glb / .gltf there as { file, name } (name = glTF scene name). The demo's room
 *  dropdown reads it — a static page can't list a folder itself. */
export function writeIndex(dir) {
  const rooms = readdirSync(dir).filter((f) => /\.(glb|gltf)$/i.test(f)).map((file) => {
    let name;
    try {
      const b = readFileSync(join(dir, file));
      const json = file.toLowerCase().endsWith('.glb') ? JSON.parse(b.subarray(20, 20 + b.readUInt32LE(12))) : JSON.parse(b);
      name = json.scenes?.[json.scene ?? 0]?.name;
    } catch { /* unreadable: fall back to the file name */ }
    return { file, name: name || file.replace(/\.(glb|gltf)$/i, '') };
  }).sort((a, b) => a.name.localeCompare(b.name));
  writeFileSync(join(dir, 'index.json'), JSON.stringify(rooms, null, 2) + '\n');
  console.log(`wrote ${join(dir, 'index.json')}: ${rooms.length} rooms`);
}
