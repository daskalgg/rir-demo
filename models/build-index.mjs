import { fileURLToPath } from 'node:url';
import { writeIndex } from './glb.mjs';

writeIndex(fileURLToPath(new URL('.', import.meta.url)));
