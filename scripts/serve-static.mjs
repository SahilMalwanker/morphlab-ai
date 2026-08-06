// Minimal static file server for the production build (`dist`), used to check
// that MorphLab works from any static host and under a sub-path, with a strict
// Content Security Policy. Usage: node scripts/serve-static.mjs [root] [port] [prefix]
import { createReadStream, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, normalize, resolve, sep } from 'node:path';

const root = resolve(process.argv[2] ?? 'dist');
const port = Number(process.argv[3] ?? 4174);
const prefix = `/${(process.argv[4] ?? '').replace(/^\/+|\/+$/g, '')}/`.replace('//', '/');

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.map': 'application/json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
};

// WebAssembly needs 'wasm-unsafe-eval'. The @mujoco/mujoco 3.14.0 embind glue
// builds its function invokers with `new Function`, so 'unsafe-eval' is also
// required (measured: without it the engine fails to load). The page connects
// only to its own origin (the AI proxy, when present, is same-origin).
const CSP = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-eval' 'wasm-unsafe-eval'",
  "worker-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "connect-src 'self'",
  "object-src 'none'",
  "base-uri 'self'",
  "frame-ancestors 'none'",
].join('; ');

function send(res, status, text) {
  res.writeHead(status, { 'Content-Type': 'text/plain; charset=utf-8', 'X-Content-Type-Options': 'nosniff' });
  res.end(text);
}

createServer((req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') return send(res, 405, 'Method not allowed');
  const url = new URL(req.url ?? '/', 'http://localhost');
  if (`${url.pathname}/` === prefix) {
    res.writeHead(301, { Location: prefix });
    return res.end();
  }
  if (!url.pathname.startsWith(prefix)) return send(res, 404, 'Not found');
  let rel;
  try {
    rel = decodeURIComponent(url.pathname.slice(prefix.length));
  } catch {
    return send(res, 400, 'Bad request');
  }
  if (rel === '' || rel.endsWith('/')) rel += 'index.html';
  const file = resolve(root, normalize(rel));
  if (!file.startsWith(root + sep)) return send(res, 403, 'Forbidden');
  let stat;
  try {
    stat = statSync(file);
  } catch {
    return send(res, 404, 'Not found');
  }
  if (!stat.isFile()) return send(res, 404, 'Not found');
  const isHtml = extname(file) === '.html';
  // The policy is sent with every file: a worker takes its policy from its own script response.
  res.writeHead(200, {
    'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream',
    'Content-Length': stat.size,
    'X-Content-Type-Options': 'nosniff',
    'Referrer-Policy': 'no-referrer',
    'Cache-Control': isHtml ? 'no-cache' : 'public, max-age=31536000, immutable',
    'Content-Security-Policy': CSP,
  });
  if (req.method === 'HEAD') return res.end();
  createReadStream(file).pipe(res);
}).listen(port, () => console.log(`Serving ${root} at http://localhost:${port}${prefix}`));
