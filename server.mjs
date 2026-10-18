// Minimal static server for the demo (no dependencies). Serves this folder, including node_modules/three.
//
//   npm start            -> http://localhost:8080/        (PORT=3000 npm start for another port)
//   npm run dev          -> same, via nodemon (see nodemon.json)
import http from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('.', import.meta.url));
const PORT = +(process.env.PORT || 8080);
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.frag': 'text/plain; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.glb': 'model/gltf-binary', '.gltf': 'model/gltf+json', '.bin': 'application/octet-stream',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml',
  '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.ogg': 'audio/ogg', '.oga': 'audio/ogg', '.opus': 'audio/ogg',
  '.flac': 'audio/flac', '.m4a': 'audio/mp4', '.aac': 'audio/aac', '.webm': 'audio/webm',
};

http.createServer((req, res) => {
  let path;
  try { path = decodeURIComponent(new URL(req.url, 'http://x').pathname); } catch { res.writeHead(400).end(); return; }
  if (path === '/') path = '/demo.html';
  const file = normalize(join(ROOT, path));
  if (!file.startsWith(ROOT) || file.split(sep).includes('.git')) { res.writeHead(403).end(); return; }   // no ../ escapes
  let stat;
  try { stat = statSync(file); } catch { stat = null; }
  if (!stat?.isFile()) { res.writeHead(404, { 'Content-Type': 'text/plain' }).end('not found: ' + path); return; }
  res.writeHead(200, {
    'Content-Type': TYPES[extname(file).toLowerCase()] || 'application/octet-stream',
    'Content-Length': stat.size,
    'Cache-Control': 'no-cache',                // edits show up on reload
  });
  if (req.method === 'HEAD') { res.end(); return; }
  createReadStream(file).pipe(res);
}).listen(PORT, () => console.log(`RIR demo: http://localhost:${PORT}/`));
