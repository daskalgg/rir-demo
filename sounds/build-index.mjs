// Rebuilds sounds/index.json — the demo's list of anechoic sounds — from the audio files in this folder.
// A static page can't list a folder itself, so run this after adding or removing sounds.
//
//   node sounds/build-index.mjs
import { readdirSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const dir = fileURLToPath(new URL('.', import.meta.url));
const AUDIO = /\.(wav|mp3|ogg|oga|opus|flac|m4a|aac|webm)$/i;
const sounds = readdirSync(dir).filter((f) => AUDIO.test(f))
  .map((file) => ({ file, name: file.replace(AUDIO, '') }))
  .sort((a, b) => a.name.localeCompare(b.name));
writeFileSync(new URL('./index.json', import.meta.url), JSON.stringify(sounds, null, 2) + '\n');
console.log(`wrote ${dir}index.json: ${sounds.length} sounds`);
