# MorphLab user interface: technology

Source reference reviewed on 2026-10-16. This is the developer guide to
the frontend stack, ownership boundaries and technology restrictions, not
a new claim that every workflow has been browser-tested. Implementation
and verification evidence belongs in [status.md](status.md).

| Question | Read |
|---|---|
| What powers the UI, and what can we change? | This document |
| What does each action do, and what happens on failure? | [interface.md](interface.md) |
| Where do controls, results and the viewport belong? | [layout.md](layout.md) |
| Why was an alternative chosen or rejected? | [decisions.md](decisions.md); later amendments take precedence |

Quick navigation: [stack](#1-stack-in-use), [ownership](#2-how-the-pieces-fit),
[styling](#3-styling), [browser requirements](#4-browser-requirements-and-deployment),
[choices](#5-technology-choices-not-blanket-bans),
[restrictions](#6-hard-restrictions-and-current-limitations),
[source map](#8-source-map), [lifecycle](#9-state-and-resource-lifetimes),
[change checklist](#10-changing-the-ui), [verification](#11-verification-contract).

## 1. Stack in use

Versions below are pinned declarations from [package.json](../package.json),
not recommendations to install a new version. Runtime dependencies, build
tools and optional server dependencies serve different execution contexts.

| Concern | Technology | Version | Where |
|---|---|---|---|
| Language | TypeScript, `strict` and `noUncheckedIndexedAccess` | 7.0.2 | all of `src/` |
| Build and dev server | Vite with `@vitejs/plugin-react` | 8.3.2, 6.1.1 | [vite.config.ts](../vite.config.ts); Node tooling is CI-only here |
| Component framework | React and React DOM (`createRoot`, `StrictMode`) | 19.3.0 | [src/main.tsx](../src/main.tsx), `src/ui` |
| UI state | Own `Store<T>` read through `useSyncExternalStore` | - | [src/ui/store.ts](../src/ui/store.ts) |
| 3D rendering | Three.js, imperative; add-ons `OrbitControls`, `TransformControls` | 0.186.1 | `src/render`, viewport components in `src/ui` |
| Plots | Canvas 2D API (`TimePlot`, `StaticPlot`) | - | [src/ui/plot.tsx](../src/ui/plot.tsx) |
| Styling | One plain CSS file with custom properties, plus inline `style` props | - | [src/ui/theme.css](../src/ui/theme.css) |
| Simulation in the page | MuJoCo WebAssembly, imported on demand | `@mujoco/mujoco` 3.14.0 | [src/physics/loader.ts](../src/physics/loader.ts) |
| Background jobs | ES module Web Workers behind `JobWorkerClient` | - | [src/skills/workerClient.ts](../src/skills/workerClient.ts), [src/nav/workerClient.ts](../src/nav/workerClient.ts) |
| Local storage | IndexedDB through `idb` | 8.0.3 | `src/persistence`, [src/ui/storage.ts](../src/ui/storage.ts) |
| Schema validation | Zod (AI tool schemas the UI calls through) | 4.6.5 | [src/ai/tools.ts](../src/ai/tools.ts) |
| Speech | Web Speech API: `speechSynthesis`; `SpeechRecognition` or `webkitSpeechRecognition` | browser | [src/ui/voice.ts](../src/ui/voice.ts) |
| AI access | Same-origin `fetch` to `/api` | - | [src/ai/client.ts](../src/ai/client.ts) |
| Session and unit tests | Vitest (Node environment, no DOM), `fake-indexeddb` | 5.0.3, 6.2.5 | `tests/ui` and the other `tests/` folders |
| Browser tests | Playwright, projects `chromium`, `static` (CSP, sub-path), `firefox` | 1.63.0 | `tests/e2e`, [playwright.config.ts](../playwright.config.ts) |

Fastify 5.12.5 is also a declared dependency, but belongs to the optional
Node server, not the frontend. `server/serve.ps1` serves the same HTTP
contract through PowerShell 7 and .NET on machines without Node.js. Neither server performs the live
robot simulation or replaces the browser's local controllers.

Build settings ([vite.config.ts](../vite.config.ts)): `base: './'` (static hosting and
sub-path), `worker.format: 'es'`, `build.target: 'es2022'`,
`@mujoco/mujoco` excluded from dependency pre-bundling so its loader finds
the `.wasm` file, and `/api` forwarded to the proxy on `127.0.0.1:8787` in
`vite` and `vite preview`.

## 2. How the pieces fit

```text
 index.html -> src/main.tsx -> App          opens IndexedDB, restores the workspace
                               Workspace    open view, preferences, sessions,
                                            task manager, AI controller
   mode components (React)   useStore(session.store)   re-render on snapshot change
   viewport components       Viewport (Three.js)       rAF loop reads the session
   CommandBar                local rules -> AI controller (if available); Stop
```

- Each mode has a session controller with a `Store` snapshot and the
  runtime/worker resources that mode needs. Build and Kinematics do not
  own a MuJoCo runtime. Kinematics is created with the workspace; the
  other sessions are initialized on demand, including demand from a
  command or a cross-view hand-off, not only from opening a view.
- React renders the shell, controls, status summaries and canvas host
  elements. High-rate transforms and telemetry bypass reconciliation:
  the viewport's `requestAnimationFrame`
  loop advances the live simulation (frame time capped at 0.1 s) and
  reads transforms from the session; `TimePlot` redraws from telemetry on
  its own 66 ms timer. Store snapshots are replaced, never mutated.
- `src/render` takes plain data (`Float64Array` transforms, link
  descriptors) and computes no robotics quantities. `src/ui` wires modules
  together and contains no numerical algorithms (architecture section 1).
- `Viewport` ([src/render/viewport.ts](../src/render/viewport.ts)) uses the world frame directly:
  right-handed, Z up, camera up +Z, grid in XY, no axis swap. Device pixel
  ratio is capped at 2, size follows a `ResizeObserver`, one directional
  light casts shadows, and an `afterRender` hook draws the Navigate inset
  camera.
- `RobotView` draws the base, links, joint hubs, tips and wheels from
  transforms; `KinematicsOverlay` draws the IK target, residual line,
  manipulability ellipsoid and workspace samples.
- Each view renders inside an `ErrorBoundary` keyed by the mode. It handles
  React subtree errors, not arbitrary asynchronous failures; command,
  worker and loading paths need their own error handling.
- Disposal: viewports dispose geometries, materials, controls and the
  renderer on unmount; sessions free WASM objects and terminate workers
  (architecture section 9).

## 3. Styling

[src/ui/theme.css](../src/ui/theme.css) is the only style sheet; there are no CSS modules,
preprocessors or CSS-in-JS. Tokens:

| Group | Tokens |
|---|---|
| Surfaces | `--bg-0` #111415 page and navigation, `--bg-1` #1b2021 bars and panels, `--bg-2` #242b2d inputs and buttons, `--bg-3` #2f383b hover |
| Borders | `--border` #3a4548, `--border-strong` #637275 |
| Text | `--text-0` #f2f5f3 primary, `--text-1` #c0cbc8 secondary, `--text-2` #9caaa6 muted |
| Accent | `--accent` #9bd6c3 soft mint (focus, selection, primary), `--accent-dim` #243e37, `--accent-ink` #142921 for filled actions; `--highlight` #e5c58b for secondary headings |
| Status | `--ok` #92ceaa, `--warn` #f1ca89, `--error` #ff9b9b; `--error-dim` #382728 for Stop |
| Fonts | `--font-sans` Bahnschrift, Segoe UI Variable Text, Segoe UI; `--font-mono` Cascadia Mono, Consolas, Liberation Mono (installed fonts only, none downloaded) |
| Type scale | `--fs-xs` 12, `--fs-sm` 13, `--fs-md` 14 (body), `--fs-lg` 17, `--fs-xl` 24 px; line height 1.5, zero letter spacing |
| Spacing and shape | `--space-1` to `--space-4`: 4, 8, 12, 16 px; `--radius` 6 px |
| Sizes | `--control-h` 32 px (buttons, inputs, selects); `--nav-w` 188 px (164 px at 1200 px and below); `--pg-side-w` Playground side panel `clamp(320px, 28vw, 400px)` |

- `color-scheme: dark`; focus is a 2 px accent outline with a 2 px offset.
- Section headings use sentence case; the brand and Playground title use
  the 24 px size. Numbers in inputs and readouts use tabular monospace.
- The active Playground entry and primary actions use a solid mint fill;
  secondary tools use an inset current-view marker. Side-view buttons
  remain pressed-state buttons with an underline, not ARIA tabs.
- Shared classes: `.btn` (`.primary`, `.stop`), `.status` (`.ok`,
  `.warn`, `.error`, `.idle`), `.note` (`.warn`, `.error`), `.muted`,
  `.mono`, `.section`, `.field`, `.row`, `.grow`, `.toggle`, `.readout`,
  `.joint`, `table.chain`, `.explain`, `.viewport-label`.
- Shared components ([src/ui/components.tsx](../src/ui/components.tsx)): `Section`, `Toggle`,
  `Readout`, `NumberField`, `StatusBadge`, `FileButton`, plus `fmt` and
  `downloadText`.
- Scene and plot colors are hex literals in `src/render`, [src/ui/plot.tsx](../src/ui/plot.tsx)
  and the viewport components, chosen to match the tokens; a theme change
  must update them too.

## 4. Browser requirements and deployment

- The supported baseline in [README.md](../README.md#requirements) is a
  current browser with WebAssembly and WebGL2, plus a keyboard for driving.
  Feature detection is not a browser certification: the startup probe in
  [src/ui/capabilities.ts](../src/ui/capabilities.ts) accepts either a
  WebGL2 or WebGL1 context, which does not establish WebGL1 renderer support.
  Actual rendering must be checked in the target browser.
- Missing WebAssembly prevents MuJoCo simulation, including Navigate.
  Missing Web Workers prevents background evaluation and training;
  missing IndexedDB removes persistence; missing graphics support prevents
  3D views. The startup notices are diagnostics, not complete fallback
  implementations for every unsupported browser.
- Content Security Policy: the `@mujoco/mujoco` 3.14.0 embind glue builds
  functions with `new Function`, so `script-src` needs `'unsafe-eval'` in
  addition to `'wasm-unsafe-eval'`. The static deployment check keeps
  `default-src 'self'`, `object-src 'none'`, `frame-ancestors 'none'`
  (decisions, M8).
- The MuJoCo glue loads on demand; Build and Kinematics need no physics.
- A static-only deployment has no AI proxy. Local computation and typed
  rules do not require one, but the HTML, JavaScript, worker and WASM assets
  must still be accessible. This is not an installable offline app: no
  service worker guarantees a cold start without a host or cached assets.
  Browser speech recognition may also require the vendor's network service.
- On a machine without Node.js, serve a production build with
  [server/serve.ps1](../server/serve.ps1) (D15). It serves the app even
  without AI credentials. Source edits are not reflected until a new build
  is installed. See the [run instructions](../README.md#run-without-nodejs).

## 5. Technology choices, not blanket bans

An absent dependency is not automatically forbidden. These are existing
architecture choices or deferred work; changing them needs a justified,
scoped decision, not a claim that the technology is technically impossible.
Reasons below summarize the recorded decisions, not a fresh survey of
external packages or their latest releases.

| Technology | Current choice | Reason or boundary | Record |
|---|---|---|---|
| React Three Fiber | Imperative `Viewport` class | Explicit render loop, transform updates and disposal with fewer dependencies. This is MorphLab's choice, not a claim that R3F cannot support efficient per-frame updates | D2, R |
| Zustand or another state library | `Store<T>` with `useSyncExternalStore` | A small typed store is enough; revisit if panels grow | D2 |
| Pinocchio compiled to WASM | TypeScript kinematics in `src/core` | Would add a second large WASM module | R |
| Rapier JS | MuJoCo | Historical fallback considered before the physics spikes; there is no runtime engine-switching control | D1 and Phase 1 updates |
| MuJoCo multithreaded build (`/mt`) | Single-threaded build | Requires cross-origin isolation and suitable hosting headers; not impossible on all static hosts | D1 |
| MuJoCo `MjSpec` API | Generated MJCF text | The bindings documentation examined for D1 called it untested in web apps | D1 |
| TensorFlow.js | Own MLP and evolution strategies | D3 judged bundle and backend overhead unnecessary for the small networks in scope | D3 |
| ONNX Runtime Web | Own MLP and evolution strategies | D3 rejected its offline-artifact training workflow for this browser-local use case; inference itself is a separate capability | D3 |
| Streaming model replies | Each tool call and result shown when complete | Little gain, an event-stream relay in two servers | D18 (deferred) |
| Express 5 | Fastify proxy and PowerShell counterpart | Fastify supplies request validation hooks, body limits and injected route tests; Express is not prohibited | D11, D15 |
| WebGPU training | CPU Web Workers | Adapter availability was recorded, but no GPU training or GPU MuJoCo rollout path is implemented | D33 |
| react-resizable-panels 4.14.5 | Fixed and clamped panel widths; collapse and restore button | Available on npm (MIT, no runtime dependencies) but not adopted, so drag resizing waits for that integration | D34 |
| Radix UI primitives (react-dialog 1.3.1, react-dropdown-menu 2.2.1) | Native buttons, `aria-expanded` disclosures and `details` | Available on npm (MIT). The redesign has no modal or menu surface that needs them | D34 |
| lucide-react 1.58.0 | Text labels | Available on npm (ISC). Operational controls keep text labels; icons were not needed | D34 |

## 6. Hard restrictions and current limitations

### Policy and safety rules

These are requirements, not optional frontend preferences. A new library
or framework does not override them.

| Restriction | Consequence for the UI | Source |
|---|---|---|
| No provider credentials in the browser | No key/endpoint fields in the UI, no `VITE_` secrets, no client `import.meta.env`, storage, logging or direct credential-bearing provider calls. Only the server reads credentials | D11, D14 |
| No image submissions to the language model | The visible camera is display-only. Provider image capability is not permission to send screenshots; both proxies reject image-bearing messages | D31 |
| No model-generated code or control-rate commands | Models select bounded, validated tools and JSON drafts; local controllers generate actuator commands | D13 |
| No silent reuse of a learned policy on another body | Preserve morphology checks and show incompatibility, rather than hiding or overriding it in a panel | [Architecture](architecture.md#8-persistence) |
| No external training service | D33 uses browser-local compute, not a remote GPU service. Large VLA/RL deployments are outside the browser control loop | D33 and [runtime tiers](architecture.md#7b-ai-layer-srcai-server) |

### Product choices requiring explicit reconsideration

- Keyboard-only driving is a user requirement. Gamepad code was removed;
  adding it back is a product change, not a missing dependency to install.
- React/Three.js belong at the UI/render boundary, never in numerical,
  learning, language or persistence modules. Do not move algorithms into
  components to make a visualization convenient.
- Keep provider code in the server process, not in a Vite plugin that
  shares the frontend build configuration (D11).

### Current technical or infrastructure limits

| Limitation | What it means | What it does not mean |
|---|---|---|
| No Playwright CI browser job | [CI configuration](../.github/workflows/ci.yml) runs the type check, unit tests, spikes and build; the Playwright suites run where matching browsers are installed | Playwright itself is forbidden, or the presence of test files proves a current pass |
| MuJoCo glue needs `'unsafe-eval'` and `'wasm-unsafe-eval'` in the current CSP | A host that forbids those directives cannot run this particular build unchanged | Arbitrary AI-generated JavaScript may be executed |
| Provider deployment uses chat completions | Realtime provider audio is unavailable here; browser speech is a separate, optional path | Every provider lacks realtime audio, or browser recognition always works offline |
| No GPU training path | WebGPU availability alone does not accelerate the current CPU workers | WebGPU is prohibited, or GPU performance has been measured |
| No proxy on a static-only host | AI commands report unavailable while local workflows remain independent of AI | Every possible network feature, notably speech recognition, is offline |

## 7. Not present

These are absent from the current source and dependency manifest, not
strict bans. A proposed addition should solve a demonstrated need and
respect the boundaries above; significant architecture changes belong in
[decisions.md](decisions.md).

- No client-side router: the open view is React state and the URL does not
  change.
- No component library, CSS framework, icon package or charting library.
  Tailwind, Bootstrap, Material UI, shadcn/ui, Lucide, D3 and Chart.js are
  examples of absent options, not approved dependency recommendations.
- No web fonts, internationalization (English only, `lang="en"`), service
  worker or install manifest.
- No DOM-level component tests: Vitest runs in the Node environment and
  tests session classes; rendered UI is covered by Playwright only.

## 8. Source map

Start with the component for presentation and its session for stateful
behavior. Follow into framework-free modules when the change concerns
calculation, validation, training or action execution.

| Responsibility | Presentation or entry point | State and behavior |
|---|---|---|
| Workspace and navigation | [src/ui/App.tsx](../src/ui/App.tsx), [src/ui/WorkspaceNav.tsx](../src/ui/WorkspaceNav.tsx), [src/ui/modes.ts](../src/ui/modes.ts) | Mode selection (Playground default), session lifetime, robot hand-offs, task invalidation, focus after a view switch, the slot after the command bar that the Playground side panel is portaled into |
| Build | [src/ui/builder/BuilderMode.tsx](../src/ui/builder/BuilderMode.tsx) | [src/ui/builder/session.ts](../src/ui/builder/session.ts) |
| Kinematics | [src/ui/kinematics/KinematicsMode.tsx](../src/ui/kinematics/KinematicsMode.tsx) | [src/ui/kinematics/session.ts](../src/ui/kinematics/session.ts) |
| Physics | [src/ui/physics/PhysicsMode.tsx](../src/ui/physics/PhysicsMode.tsx) | [src/ui/physics/session.ts](../src/ui/physics/session.ts) |
| Control lab | [src/ui/control/ControlMode.tsx](../src/ui/control/ControlMode.tsx) | [src/ui/control/session.ts](../src/ui/control/session.ts) |
| AI Playground (Skills) | [src/ui/skills/SkillsMode.tsx](../src/ui/skills/SkillsMode.tsx), [src/ui/skills/TeachPanel.tsx](../src/ui/skills/TeachPanel.tsx), [src/ui/skills/WalkPanel.tsx](../src/ui/skills/WalkPanel.tsx), [src/ui/skills/LastPlay.tsx](../src/ui/skills/LastPlay.tsx) | [src/ui/skills/session.ts](../src/ui/skills/session.ts) |
| Navigate | [src/ui/nav/NavMode.tsx](../src/ui/nav/NavMode.tsx) | [src/ui/nav/session.ts](../src/ui/nav/session.ts), [src/ui/nav/keys.ts](../src/ui/nav/keys.ts), [src/nav/drive.ts](../src/nav/drive.ts) (which bodies can be driven, and their drive geometry) |
| Commands and AI feedback | [src/ui/CommandBar.tsx](../src/ui/CommandBar.tsx) | [src/ui/ai/bridge.ts](../src/ui/ai/bridge.ts), [src/ui/ai/outcome.ts](../src/ui/ai/outcome.ts) |
| Restore and warnings | [src/ui/Notices.tsx](../src/ui/Notices.tsx) | [src/ui/storage.ts](../src/ui/storage.ts), [src/persistence/db.ts](../src/persistence/db.ts) |
| Geometry and display overlays | [src/render/robotView.ts](../src/render/robotView.ts), [src/render/kinematicsOverlay.ts](../src/render/kinematicsOverlay.ts) | Consumes transforms and results, not a second kinematics solver |

## 9. State and resource lifetimes

| State or resource | Owner and lifetime | Maintenance rule |
|---|---|---|
| Field text, open view, open disclosures, Playground side view | React component state | Keep temporary editing state distinct from committed robot state |
| Robot descriptions, selected joints, results, job summaries | Mode sessions and their stores, retained while the workspace lives | Use session methods; do not mutate a snapshot from JSX |
| Scene, canvas, controls, listeners and resize observer | Mounted viewport | Unsubscribe and dispose on unmount; retained session state does not keep a hidden viewport mounted |
| MuJoCo runtime | Physics-capable session | Delete WASM resources during replacement/disposal; never save live WASM views as durable data |
| Training and evaluation jobs | Worker clients and their owning session | Expose progress and cancel; reject stale results after the relevant setup changes |
| Saved records and preferences | Origin-scoped IndexedDB | Validate on load, handle unavailable storage, and preserve unreadable records until explicitly removed |

Switching modes retains session objects, not unrestricted background
activity. The shell invalidates AI tasks; mode cleanup stops or cancels
its owned activity. Workspace unmount disposes resource-owning sessions,
with deferred cleanup for the development `StrictMode` remount.

The render loop, simulation timestep, telemetry sampling and React
notification rate are different clocks. A 66 ms plot timer is not a
physics timestep or a frame-rate guarantee. Training can continue in
workers independently of drawing; live viewport-driven simulation may
slow or stop when the browser throttles a hidden page.

## 10. Changing the UI

1. Identify the owner in the source map. Layout changes start in the
  stylesheet/component; behavior changes start in the session or core
  module that decides the outcome.
2. Reuse the existing field, section, readout and status components. Keep
  display units separate from SI values at the model boundary.
3. Define idle, loading, busy, success, refusal, failure and cancellation
  states before adding a command. Keep local stopping independent of AI.
4. Keep frame-rate updates out of React state. Add listeners, timers,
  render resources or workers only with an explicit cleanup owner.
5. Preserve labels for measured versus commanded data, replay versus
  simulation, and model text versus evidence. Do not equate a policy
  selector or training completion with demonstrated skill success.
6. Check the [interaction contract](interface.md) and
  [layout review checklist](layout.md). Retain numeric alternatives to
  pointer-only actions and keyboard access to operational controls.
7. Justify new dependencies by a concrete capability the current helpers
  cannot provide. Verify their API/license and internal availability;
  update the manifest and obtain the lockfile through CI, never a local
  package install.

These are maintenance requirements, not a proposal to replace the UI
framework, theme, render architecture or existing controls.

## 11. Verification contract

| Change | Relevant evidence |
|---|---|
| Session behavior or validation | Existing session tests under `tests/ui`, plus the owning core tests, run in CI |
| Stored preferences/import/restore | [tests/ui/storage.test.ts](../tests/ui/storage.test.ts) and [tests/e2e/persistence.spec.ts](../tests/e2e/persistence.spec.ts) |
| Keyboard or focus | [tests/ui/navKeys.test.ts](../tests/ui/navKeys.test.ts) and [tests/e2e/a11y.spec.ts](../tests/e2e/a11y.spec.ts) |
| Layout, guide or comparison | [tests/e2e/polish.spec.ts](../tests/e2e/polish.spec.ts), narrow-window checks, real screenshots |
| Rendering or resource cost | Relevant mode smoke tests and [tests/e2e/perf.spec.ts](../tests/e2e/perf.spec.ts); measure on the actual browser/build |
| Documentation only | Check source claims, relative links, heading anchors, table structure and code fences; do not imply runtime tests ran |

CI runs `npm run typecheck`, `npm test`, `npm run spikes` and
`npm run build`; `npm run test:e2e` runs the Playwright suites where
matching browsers are installed.

For browser evidence, identify the commit and build, serve it
(`npm run serve:dist`, or `server/serve.ps1` without Node.js), record
viewport dimensions and visible outcomes, and report
console errors. A source edit, clean editor diagnostics or historical
test count is not proof that the currently served artifact includes it.
Record new measured results in [status.md](status.md); leave unchecked
browser and assistive-technology behavior explicitly unverified.
