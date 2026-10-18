// Rebuilds models/index.json (the demo's room list) after adding or removing a .glb / .gltf by hand.
// The generators call this automatically.
//
//   node models/build-index.mjs
import { fileURLToPath } from 'node:url';
import { writeIndex } from './glb.mjs';

writeIndex(fileURLToPath(new URL('.', import.meta.url)));
