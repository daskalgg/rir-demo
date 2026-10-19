// Minimal static server for the demo (no dependencies). Serves this folder, including node_modules/three.
//
//   npm start            -> http://0.0.0.0:8080/, reachable from other devices on the network
//                           (PORT=3000 / HOST=127.0.0.1 npm start for another port / this machine only)
//   npm run dev          -> same, via nodemon (see nodemon.json)
import http from 'node:http';
import { networkInterfaces } from 'node:os';
import { createReadStream, statSync } from 'node:fs';
import { extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('.', import.meta.url));
const PORT = +(process.env.PORT || 8080);
const HOST = process.env.HOST || '0.0.0.0';
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
}).listen(PORT, HOST, () => {
  console.log(`RIR demo: http://${HOST}:${PORT}/`);
  if (HOST === '0.0.0.0') {                     // the addresses other devices can use
    for (const a of Object.values(networkInterfaces()).flat()) {
      if (a.family === 'IPv4' && !a.internal) console.log(`          http://${a.address}:${PORT}/`);
    }
  }
});
