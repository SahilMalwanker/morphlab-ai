// Starts the AI proxy (server/main.ts) and the Vite dev server together.
// Vite keeps running when the proxy cannot start; the app then shows AI as offline.
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const vite = fileURLToPath(new URL('../node_modules/vite/bin/vite.js', import.meta.url));
let stopping = false;

const proxy = spawn(process.execPath, ['server/main.ts'], { cwd: root, stdio: 'inherit' });
proxy.on('exit', (code) => {
  if (!stopping && code !== 0) {
    console.warn('MorphLab: the AI proxy is not running. Browser-only features keep working; AI features show as offline.');
  }
});

const web = spawn(process.execPath, [vite, ...process.argv.slice(2)], { cwd: root, stdio: 'inherit' });
web.on('exit', (code) => {
  stopping = true;
  proxy.kill();
  process.exit(code ?? 0);
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    stopping = true;
    proxy.kill(signal);
    web.kill(signal);
  });
}
