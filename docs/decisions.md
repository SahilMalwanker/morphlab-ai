# MorphLab decisions

Each decision records the alternatives, the reasoning, the evidence and how
it will be verified. "Verified" means checked against official documentation
or installed artifacts on 2026-10-03; "pending" means a spike must confirm
it. Changes to decisions are appended, never silently rewritten.

## Evidence sources and documentation access
Web access was available. Sources consulted:
- MuJoCo JavaScript bindings README (google-deepmind/mujoco, `wasm/`).
- MuJoCo XML reference (mujoco.readthedocs.io, stable).
- npm registry metadata for every package named below.
- Package tarballs of `@mujoco/mujoco@3.14.0` and
  `@dimforge/rapier3d-compat@0.21.0`, inspected locally (type declarations,
  loader source, artifact sizes).
- Rapier JavaScript user guide (joints, motors), version 0.21.
- ONNX Runtime Web tutorial and on-device training documentation.
- Reference repositories listed in section R.

## R. Reference material and licenses

| Repository | License | What we learned | How we use it |
|---|---|---|---|
| NxRLab/ModernRobotics | MIT | Product-of-exponentials FK (`FKinSpace`, `FKinBody`), space Jacobian, Newton-Raphson IK. Library is explicitly educational and "optimized neither for efficiency nor robustness". Python, MATLAB and Mathematica only. | Numerical oracle. A Python script in `tools/oracle` calls the published `modern_robotics` package to generate JSON fixtures; our TypeScript FK uses a different formulation (local transform chain), so agreement is an independent check. No code is copied. |
| google-deepmind/mujoco `wasm/` | Apache-2.0 | Canonical bindings, npm `@mujoco/mujoco`, ESM with bundled `.wasm` and `.d.ts`; single-threaded default build and `/mt` build that needs COOP/COEP headers; embind objects must be freed with `.delete()`; large arrays such as `data.qpos` are live views into WASM memory; `data.contact` is a copy; matrices are row-major. README states the bindings are work in progress and Windows build support is experimental (Nov 2025). | Physics engine (D1). Patterns: `MjModel.from_xml_string`, `new MjData(model)`, `mj_step`, explicit `.delete()` in `finally`. |
| ttktjmt/mjswan | Apache-2.0 | Browser MuJoCo + ONNX policy inference + Three.js, static hosting, policies trained offline (mjlab) and run at a control rate with physics decimation. | Pattern only: separate control rate from physics rate, observation assembly from sensor data, policy inference loop. We train in the browser, so we do not adopt ONNX. No code copied. |
| zalo/mujoco_wasm | MIT | Now a demo suite on the official bindings (MuJoCo 3.12). Shows Emscripten virtual file system use (`FS.mkdir`, `FS.mount(MEMFS)`, `FS.writeFile`) and `MjModel.mj_loadXML`, `mj_applyFT`, plus Three.js rendering of `xpos`/`xquat`. Includes a fix for physics freezing when frames exceed a time budget. | Pattern only: fixed-timestep accumulator with a cap on substeps per frame; loading from strings avoids the virtual file system. No code copied. |
| Mostafasaad1/robot-analyzer-js | MIT | React + Vite + React Three Fiber, Pinocchio compiled to WASM, DLS IK, Jacobian and center-of-mass panels, Zustand with transient subscriptions to avoid React re-renders per frame, Jest tests. README states most code was AI-generated. | UI pattern: keep per-frame data out of React state. We do not adopt Pinocchio (adds a second large WASM module) or R3F (see D2). No code copied. |

## D1. Physics engine: MuJoCo WebAssembly (`@mujoco/mujoco` 3.14.0)

Status: decided, verification pending spikes S1 and S2.

| Criterion | MuJoCo WASM 3.14.0 | Rapier JS 0.21.0 (`rapier3d-compat`) |
|---|---|---|
| Articulated bodies | Generalized coordinates; joints are part of the kinematic tree; designed for robots | Impulse joints (maximal coordinates) or multibody joints (reduced coordinates) |
| Contact for legged robots | Soft-constraint convex optimization (Newton solver by default), widely used for legged locomotion research | Game-oriented PGS solver; stable, less established for legged robot control |
| Actuators | Motor, position, velocity, PID, DC motor models with `ctrlrange` and `forcerange` clamping | Joint motors (spring-damper PD) with max impulse |
| Sensors | Built in: jointpos, jointvel, framequat, gyro, accelerometer, touch, rangefinder, contact, camera rangefinder | None built in; ray casts and contact events must be assembled manually |
| Programmatic model generation | Generate MJCF text and call `MjModel.from_xml_string` (verified in `.d.ts`); `MjSpec` API exists but README calls it untested in web apps | Native JS API to create bodies and joints |
| Download (gzip) | `mujoco.wasm` 10.1 MB raw, 2.5 MB gzip; loader 67 KB gzip | WASM 3.0 MB raw, 1.2 MB gzip (compat build inlines it as base64: 1.7 MB gzip) |
| Node support for tests | Loader detects Node (`ENVIRONMENT_IS_NODE`), so headless tests are possible (pending S1) | Supported |
| Risks | Bindings are WIP; Windows build support experimental (consumer use of the npm artifact is OS independent); larger download | Sensors, robot conventions and actuator models must be built by us; contact quality for legged robots unproven here |

Decision: MuJoCo. Built-in robot-grade actuators and sensors directly serve
the control, sensor-only and skill milestones, and its contact model is the
standard for legged-robot simulation. The extra ~1.3 MB gzip is acceptable
for an application whose core is physics. Rapier remains the documented
fallback if S1 or S2 fails.

Consequences:
- Use the single-threaded build (no COOP/COEP requirement for static hosting).
- Generate MJCF from our robot description; never hand-edit MJCF for user robots.
- Free every embind object exactly once.
- Pass `locateFile` to the loader with a bundler-resolved `.wasm` URL so the
  asset path survives Vite builds (loader uses `import.meta.url` otherwise).

## D2. Frontend stack: TypeScript strict, Vite, React, Three.js (imperative)

Status: decided, verified versions.

- TypeScript 7.0.2 (`strict`, `noUncheckedIndexedAccess`), Vite 8.3.2,
  `@vitejs/plugin-react` 6.1.1 (peer `vite ^8`), React 19.3.0,
  Three.js 0.186.1 with `@types/three` 0.186.0.
- Three.js is used imperatively from `src/render` (a `Viewport` class), not
  through React Three Fiber. Reasons: per-frame data must not flow through
  React reconciliation, explicit control of the render loop and disposal,
  one fewer abstraction between computed transforms and rendered matrices,
  fewer dependencies. React renders panels only.
- No state library initially: a small typed store using React's
  `useSyncExternalStore`. Revisit if panels grow.

## D3. Browser-side learning: custom lightweight MLP and evolution strategies

Status: decided, verification pending spike S4.

| Option | Assessment |
|---|---|
| TensorFlow.js 4.22.0 | Full training API, but last npm release October 2024; large bundle; WebGL backend overhead is unnecessary for networks with a few thousand weights |
| ONNX Runtime Web 1.30.0 | Inference-focused; on-device training requires training artifacts generated offline with Python, which conflicts with "no backend" and with user-defined morphologies |
| Custom MLP + evolution strategies (chosen) | A few hundred lines of TypeScript, no dependency, deterministic with a seeded RNG, weights serialize to JSON, testable with finite-difference gradient checks |

Plan: dense MLP (tanh hidden layers) with manual backpropagation and Adam
for imitation (behavior cloning); OpenAI-style evolution strategies with
antithetic sampling and rank-based fitness shaping for reward-based
improvement of controller parameters or policy weights. Training runs in a
Web Worker. Revisit only if a milestone needs networks larger than about
50k parameters.

## D4. Persistence layout and skill artifact schema

Status: decided; verified in M6 (see "M6 updates").

- IndexedDB via `idb` 8.0.3 (ISC, small promise wrapper). Database
  `morphlab`, version 1, stores: `robots` (key path `['id',
  'morphologyHash']`), `skills` (key path `id`, index `morphologyHash`),
  `settings` (key path `key`). Tests use `fake-indexeddb` 6.2.5.
- Export and import bundles: `{ kind: 'morphlab.bundle', schemaVersion: 1,
  exportedAt, robots: RobotDescription[], skills: SkillArtifact[] }`.
  Import validates every record and reports rejected records by reason.
- Skill artifact (v1):

```ts
interface SkillArtifact {
  kind: 'morphlab.skill'; schemaVersion: 1;
  id: string; name: string; skillType: 'foot_lift' | string;
  method: 'replay' | 'state-machine' | 'tuned-parameters' | 'imitation-mlp' | 'es-improved';
  createdAt: string; updatedAt: string;
  robot: { robotId: string; morphologyHash: string };
  limbs: string[];                       // limbs it was evaluated on
  parameters: Record<string, number>;    // controller parameters
  policy: null | { layers: number[]; activation: 'tanh'; weights: number[];
                   inputMean: number[]; inputStd: number[] };
  observation: { name: string; source: 'joint_encoder' | 'imu' | 'contact' | 'time' | 'command'; size: number; units: string }[];
  action: { name: string; type: 'joint_position_target'; units: 'rad'; size: number }[];
  training: null | { algorithm: string; config: Record<string, number | string>;
                     seed: number; iterations: number; wallTimeS: number };
  evaluation: null | { trials: number; successes: number; successRate: number;
                       conditions: Record<string, [number, number]>;
                       metrics: Record<string, number>; simulator: string; evaluatedAt: string };
}
```

A skill executes only when `robot.morphologyHash` equals the active robot's
hash and the requested limb is in `limbs`.

## D5. Testing stack

Status: decided, verified versions.
- Vitest 5.0.3 (peer `vite ^6.4 || ^7 || ^8`) for unit and integration tests
  in Node, including headless MuJoCo tests.
- Playwright 1.63.0 (`@playwright/test`, Chromium) for browser smoke tests,
  plus the IDE browser tools for interactive inspection.
- Python `modern_robotics` (MIT) in a local virtual environment under
  `tools/oracle/.venv` (git-ignored) to generate committed JSON fixtures.

## D6. Control computed in TypeScript, applied through motor actuators

Status: decided, verification pending S1.
PD control runs in `src/core/control` at the physics rate and writes torque
to `data.ctrl`. Motor actuators have `ctrlrange = [-effort, effort]` as a
safety clamp. This keeps gains live-tunable without rebuilding the model,
makes saturation explicit in our code, and keeps commanded, applied and
measured quantities distinct. Joint armature is set (default 0.01 kg*m^2)
because MuJoCo documents that armature improves stability, and the
`implicitfast` integrator is used.

## D7. Threading

Status: decided, verification pending S1 and S4.
Interactive simulation runs on the main thread if a frame of physics costs
under 4 ms (measured). Training and multi-trial evaluation run in dedicated
Web Workers that load their own MuJoCo instance.

## D8. Language layer

Status: decided, verification pending S6.
A deterministic rule-based parser (tokenizer, intent patterns, number and
duration parsing, limb-name resolution with aliases) produces a typed
`SkillCall` or a `Refusal` with a suggestion. The optional cloud adapter
implements the same interface, must return JSON that passes the same
validator, and receives its key at runtime only.

## D9. Robot description conventions
URDF-like origins (`xyz`, `rpy` with `Rz*Ry*Rx`), unit axes in the joint
frame, capsule links from joint to the next joint or tip, SHA-256
morphology hash over canonical JSON of physical fields. See
`docs/architecture.md` sections 2 and 3.

## Phase 1 updates (2026-10-03)

### D1 verified
Spikes S1 and S2 passed with MuJoCo (see `docs/status.md`). Rapier is not
needed. New evidence recorded in the `mujoco-wasm-web` skill: `data.warning`
is a vector owned by `MjData` and must not be deleted (deleting it aborts
the module on the next free); `data.contact` and `warning.get(i)` results
are copies and must be deleted.

### D6 verified
TypeScript PD at 500 Hz with motor actuators and armature 0.01 holds the
quadruped stance (height drift 0.1 mm over 5 s) without instability.

### D7 verified
Controller plus `mj_step` costs about 0.13-0.26 ms per 2 ms step in the
browser, so a 60 Hz frame needs about 1-2 ms of physics: interactive
simulation stays on the main thread. Training ran in a Web Worker (S4).

### D3 amended: reward-based improvement starts from a baseline
Evidence (S4): behavior cloning reached 94 % success (14.3 mm mean error);
evolution strategies from random weights reached 19 % (188 mm) in the same
order of compute; ES fine-tuning of the imitation policy reduced the mean
error to 9.4 mm. Change: the "reward-based improvement" stage in M5 always
starts from an existing baseline (hand-designed parameters or an imitation
policy) and optimizes either low-dimensional controller parameters or a
pre-trained network with small perturbations. ES from scratch is not offered
as a training mode for precise skills.

## M2 updates (2026-10-03)

### D6 amended: closed-loop control uses the split step
Evidence: the first version of the M2 step-response test (1-DOF vertical
hinge, explicit PD at 500 Hz) read sensors after `mj_step`. Those readings
are computed before integration, so the loop carried one timestep of
sensing delay. Measured overshoot was 75.8 % against an analytical 72.7 %
(kp 8, kd 0.12, zeta 0.10) and settling time was 15 % long, consistent
with the predicted damping loss of about `wn * h / 2`. Change: every
closed-loop path calls `MujocoRuntime.controlStep`, which runs `mj_step1`,
evaluates the controller on current encoder readings, then runs `mj_step2`.
After the change the three test cases match the analytical response within
0.16 overshoot points and 0.5 % settling time. Touch and accelerometer
readings still lag one step because they need contact forces. Spikes S1,
S2 and S5 were re-run and still pass.

### Test harness: end-to-end tests run one browser at a time
Evidence: with parallel Playwright workers, headless Chromium (software
WebGL) took more than 10 s to instantiate the 10 MB MuJoCo module and
Emscripten logged "still waiting on run dependencies" as console errors.
Alone it was ready 3.8 s after opening physics mode. `workers: 1` keeps the
strict no-console-error assertion meaningful.

## M3 updates (2026-10-03)

### D9 extended: optional stance and sanity bounds
`RobotDescription.stance` (joint id to home position) replaces the special
case that recognized the quadruped by its id. It is not part of the
morphology hash, like names and the initial pose. Validation now also
enforces size, mass, radius, effort and velocity bounds so imported or
edited robots stay within what the simulator handles well (see
`docs/architecture.md` section 3). In the builder, a limit edit clamps the
stance into the new limits and says so in the version history, instead of
refusing an otherwise valid edit the user has no control to fix.

### Mirror convention
Mirrored limbs reflect across the base XZ plane with revolute axes treated
as pseudovectors, so equal joint angles produce mirrored poses (verified:
every frame of a mirrored limb equals the reflected original to 1e-12 at
random configurations, including yawed mounts and a prismatic joint).
Mirror partners are found geometrically from the mount point. A mirrored
edit regenerates the partner as the exact mirror image of the edited limb
while keeping the partner's ids and name. The hexapod preset's right-side
coxa axes follow the opposite convention; a mirrored edit converts them.

### Builder robots start on the ground
After geometric edits the builder sets the initial base height so the
lowest point at the home configuration is 2 mm above the ground (verified
against forward kinematics). Imports and presets keep their stored pose so
export and import stay byte-identical.

### Bundle format
Robot export uses the D4 bundle unchanged (`skills` empty). Skills in an
imported bundle are reported as not importable yet instead of being
dropped silently; skill import arrives with M5 and M6.

## M4 updates (2026-10-03)

### Gravity compensation as the second controller
The control lab compares PD with PD plus model-based gravity feedforward,
computed in `src/core/dynamics/gravity.ts` from the description's mass
model (capsule links, foot spheres, wheels). Evidence: it matches MuJoCo's
bias forces at rest to 1e-9 N*m on 8 robots, and it turns the reach and
hold challenges from fail to pass on the arms where PD alone sags. The
mass-model convention is now documented next to `FOOT_MASS` in
`src/core/model/types.ts`, shared by the MJCF generator and the dynamics.

### Test stand for joint experiments
Floating robots run step tests and the hold challenge with the base fixed
and raised by the largest limb reach, so ground contact cannot mask joint
dynamics. The UI states when a stand is used.

### Single experiment runs stay on the main thread
A step test pair or a challenge is a single short simulation (a two-run
step test took 0.31 s in the browser). It runs as a generator in 12 ms
slices so the page stays responsive, consistent with D7. Multi-trial
evaluation (M5) still goes to a Web Worker.

## M5 updates (2026-10-03)

### D4 amended: replay skills carry a trajectory
`SkillArtifact` gains `trajectory: null | { dt, jointIds, frames }` for
replay skills; other methods store `null`. Evaluation metrics also record
failure counts (`failed:<reason>`). `parseSkillArtifact` validates the
shape for import and storage.

### Controller-agnostic referee
`src/skills/evaluation.ts` replaces the Phase 1 evaluator: any skill
controller (state machine, replay, neural policy) runs through the same
trial, and the support-contact criterion is measured while the lifted foot
is actually clear (ground truth), not during a controller-reported phase.
The S5 spike was re-run on the new evaluator and still passes 20/20.

### Reward for improvement is dense
The first reward (clear-hold fraction, success bonus, tilt and fall
penalties) was flat at 0 for a baseline whose foot never cleared 3 cm, so
ES made no progress (measured: 0.000 to 0.000). Adding a capped clearance
term (0.5 x clearance / 3 cm) gave ES a usable ordering: held-out reward
0.23 to 2.75 and success 0/20 to 20/20 in 60 trials.

### Imitation features and output scaling
The policy sees joint offsets from the stance, the gravity direction in
the body frame from the IMU (two components), binary foot contact and the
elapsed fraction of the skill; it outputs joint offsets. Output
normalization is folded into the last layer so the stored weights plus
input normalization fully define the policy (no D4 schema change for
output scaling).

### Worker job handler shared with tests
The worker is a thin wrapper around `handleSkillJob` (`src/skills/jobs.ts`);
Node tests drive the same handler in-process, and Playwright verifies the
real worker in the browser.

## M6 updates (2026-10-03)

### D4 verified: storage, restore and damaged records
The IndexedDB layout is as designed. Every load validates each record
(`parseRobot`, `parseSkillArtifact`, and a check that the stored key
matches the robot's id and hash). Records that fail are listed with reasons
in a notice and kept until the user removes them, so nothing is dropped
silently. The workspace stores the last builder and skills robots in
`settings` and restores them after a reload. `setSetting` refuses
credential-like keys (api key, token, secret, password, credential).

### D4 amended: skills in bundles, imported skills are labeled
The Skills tab exports the library (current robot plus every skill) and
imports skills; the Build tab keeps importing robots. `SkillArtifact` gains
optional `importedAt`: imported skills keep their morphology binding and
their cards state that the evaluation came from the file and was not re-run
here. Validation became stricter because imported policies run in the
simulator: the weight count must match the layer sizes, input scales must
be positive, replay frames need one finite value per joint, evaluation
counts must be consistent, and only known fields are kept.
`summarizeEvaluation` no longer stores NaN means for an evaluation without
trial results (found by the stricter validation).

### D8 verified: resolver and cloud adapter
The local parser stays the default. The resolver chooses the compatible
skill with the highest evaluated success rate; a different hold is applied
only to parameterized methods and only within 0.5-5 s, because replay and
imitation skills have fixed timing. The cloud adapter targets an
OpenAI-compatible chat completions API: https only (http for localhost),
`credentials: 'omit'`, key only in the `Authorization` header, 8 s timeout,
JSON output validated like the local parser, fallback to the local rules
with a note. It was tested against mocked responses only; no real provider
was called during development.

### Skill worker start retry
Evidence: in one full Playwright run the skill worker's script request
ended with `net::ERR_ABORTED` while the page was compiling MuJoCo under
heavy CPU load, and the evaluation failed with "unknown error"; the same
test passed in the run before and after. Change: if a worker fails before
it has sent any message and without an error message (the script did not
load), the client sends the pending jobs once more to a fresh worker and
otherwise reports "the worker could not be started". A worker that crashes
after starting is not retried and its error text is kept (unit tested with
fake workers).

## M7 updates (2026-10-03)

### D10. Rover navigation stack
Status: decided and verified (see `docs/status.md`).
- Sensor-only by construction: the navigation controller receives
  `{ t, sensors }` (encoders, IMU, touch, ranges); ground-truth mode adds
  `groundTruth.pose` and the true map and is labeled as a reference. A test
  audits the keys handed to the controller.
- Pose: wheel-encoder distance with IMU heading. Map: log-odds occupancy
  grid, 0.05 m cells, updated along each range ray. Planning: 8-connected
  A* with an octile heuristic on the grid.
- Range rays tilt up by 0.035 rad: horizontal rays hit the infinite floor
  plane at very long range as soon as the body pitches slightly, which
  would turn the floor into phantom obstacles at pitches above about 0.04
  rad.
- Clearance: the first version inflated obstacles by 0.30 m and used a
  0.5 m pursuit lookahead; it collided on 2 of 5 maps in sensor-only mode
  and 3 of 5 with ground truth, because paths hugged the inflated boundary
  and pure pursuit cut corners. Change: hard inflation at 0.32 m (rover
  half-diagonal plus margin) from a distance transform, a quadratic
  proximity cost up to 0.75 m (weight 3), lookahead 0.3 m and 0.4 m/s.
  Result: 5 of 5 maps in both modes. A* stays optimal for non-negative cell
  costs; this is tested against an independent Dijkstra.
- Remote control maps keys (W/S/A/D, arrows) or a gamepad's left stick to a
  forward speed and turn rate, converted to wheel speeds; keys are ignored
  while a text field has focus. (Keyboard only since 2026-10-04; see
  "Keyboard is the only remote input".)

## M8 updates (2026-10-03)

### Deployment: relative base, static host check, CSP
`vite.config.ts` uses `base: './'` so the same build runs from any path.
A second Playwright project serves `dist/` with a minimal Node static
server (`scripts/serve-static.mjs`) under `/morphlab/` with a Content
Security Policy and runs the `@smoke` tests there. Evidence: with
`script-src 'self' 'wasm-unsafe-eval'` MuJoCo failed to load ("Evaluating a
string as JavaScript violates the following Content Security Policy
directive"); the @mujoco/mujoco 3.14.0 embind glue builds invokers with
`new Function`. The policy therefore includes `'unsafe-eval'`; everything
else stays restricted (`default-src 'self'`, `object-src 'none'`,
`frame-ancestors 'none'`).

### Engine glue loads on demand
`loadMujocoBrowser` imports `@mujoco/mujoco` dynamically, so the first
bundle no longer carries the 106 KB engine glue (kinematics needs no
physics). A failed download is not cached, so a later attempt can succeed.
Measured in headless Chromium, five runs each: first script 1,134 KB to
1,029 KB; time to the first rendered view 907-1320 ms (median 1233) before,
838-1045 ms (median 927) after; physics start 2484-7177 ms (median 3294)
before, 2349-3049 ms (median 2756) after.

### Tabs follow the WAI-ARIA tabs pattern with manual activation
Arrow keys, Home and End move focus between mode tabs and Enter or Space
opens one, because opening a mode can load the physics engine. The mode
content sits in a `role="tabpanel"` wrapper with `display: contents` so the
grid layout is unchanged; a test confirms through the Chromium
accessibility tree that the panel is still exposed.

### Unedited fields do not commit
Evidence: keyboard traversal reached only 27 of 59 Build controls. Leaving
a number field re-submitted its display value; the builder treated that as
an edit, created a new version and re-rendered the section, which dropped
focus. Now a field commits only when its text changed, and the builder
treats an edit that leaves the description identical as a no-op.

### Learn mode and guide are local preferences
Both are stored in the `settings` store (`ui.preferences`) and default to
guide shown, learn mode off. Explanations are static text next to the
numbers they explain, written from the implemented algorithms (for
example the ES reward terms are copied from `trialReward`).

### Attempt comparison replays recorded states
Each playback in the Skills tab records base and link transforms from the
simulator every 40 ms. The comparison draws two recordings side by side on
one timeline with the foot-clearance plot; it is labeled as a replay of
recorded simulator states and never simulates. At most six attempts are
kept, in memory only.

## Hardening updates (2026-10-04)

### Joint targets are rate-limited; damping uses measured velocity
Live joint control (Physics and Skills), every evaluation and training
trial, the Control lab step test and the reach challenge pass their
requested joint targets through `TargetRateLimiter`
(`src/core/control/limits.ts`): the commanded target moves toward the
request no faster than the joint's declared velocity limit, and
continuous joints move the shorter way. (The hold and push challenges
command the constant home pose. Navigate's wheel speed targets were
already clamped to the wheels' velocity limits.) The PD law wraps the
position error of continuous joints to (-pi, pi]. The D term damps the
measured velocity only. Evidence: feeding the limiter's velocity as
`qdCmd` dropped the imitation policy from 19/20 to 1/20, because a 100 Hz
target held for five 2 ms physics steps produces a velocity spike of
delta / 2 ms in one step; without the feed-forward term the policy is back
above 16/20. Speeds above the limit are still possible under external
forces; the Physics tab shows the time over the limit and the fastest
joint's speed ratio.

### Control lab: rate-limited step by default, ideal step as a reference
The step test commands the step at the joint's velocity limit, as live
control does, and the linear model is integrated (RK4, same 2 ms
zero-order hold) with that same command, so measured and predicted
responses are compared under the same input. The instant jump stays
available as "Ideal step (analytical reference)"; the M2 comparison
against the closed-form second-order response uses it explicitly, with the
same tolerances. The rate-limited comparison on the gravity-free hinge
meets the same tolerances. Results keep the setup they ran with; changing
the joint, step, record time, command or gains cancels a running
experiment, and shown results that no longer match the settings are
labeled as such.

### Units follow the joint type
`src/core/model/units.ts` names rotation and sliding quantities (rad, deg
for display, rad/s, N*m, N*m/rad, kg*m^2 versus m, m/s, N, N/m, kg). The
hold challenge judges prismatic joints in metres: 5 mm and 12.5 mm/s,
which are the revolute limits (0.02 rad, 0.05 rad/s) times a 0.25 m lever
arm.

### IK: "unreachable" requires proof
`unreachable` is reported only when the target is beyond the limb's reach
bound (a proof that no configuration reaches it). A solver that stops
improving without that proof reports `stalled` with a message saying the
target may still be reachable. Previously a stall was reported as
unreachable.

### Live playback uses the evaluation referee
`FootLiftReferee` (`src/skills/evaluation.ts`) judges both evaluation
trials and live playback: clearance, the continuous hold the command
requested (or the evaluation's 2 s), tilt, base height and support
contact. A playback ends as succeeded, failed (criteria not met),
aborted (the skill's own tilt check), interrupted (reset, robot change,
another playback) or faulted (controller error, numerical instability),
and every path settles the run exactly once. A command reports "Lifted"
only when the referee passed; anything else is a failure with the
measured numbers. The live run's differences from the evaluation setup
(gains, mass and friction outside the evaluated ranges) are listed with
the result.

### Evaluation provenance and command eligibility
Skills record `evaluationOrigin` (`local` or `imported`; older records are
read as imported when they carry `importedAt`). Evaluations store their
context: seed, criteria, gains, stance, randomization ranges, nominal
mass, limb, time step and control period. Base-mass randomization is
relative to the robot (plus or minus 12.5 %; 3.5-4.5 kg for the 4 kg
quadruped as before). Parsing checks that success counts, rate and the
Wilson interval agree, parameters lie in the editable bounds, policy
layer sizes match the robot, and replays have unique joints and at most
600 s (demonstration recording stops there, so every recording made here
stays valid). Text commands use only skills with a consistent, locally
measured evaluation with at least one success; imported skills need
"Re-evaluate here", which keeps the file's claim next to the local result.

### Commands act on the tab they are typed in
Stand and reset act on the live simulation of the active tab (Physics,
Skills or Navigate) and are refused in tabs without one. A foot lift
typed outside the Skills tab runs only if that tab shows the Skills
robot's morphology, otherwise it is refused naming both robots. The
command is interpreted against the robot of the active tab.

### Storage restore never blocks the app
Restoring the workspace runs under the same timeout as opening the
database and inside a try/catch; on failure the app starts with the
default robots and says so, leaving stored records untouched. Stored
robot pointers must be `[id, 64-hex hash]`; malformed or dangling ones are
ignored with a notice. Exports include every robot the library's skills
were evaluated on (from the session and from storage), and imports store
the robots their skills refer to.

### Navigation batch runs in a worker
The five-map evaluation moved from main-thread time slices to a Web
Worker (`src/nav/nav.worker.ts`, same job protocol as skills through the
shared `JobWorkerClient`). The controller mode is fixed when the batch
starts; the batch can be cancelled, and the worker is terminated on
cancel and when the session is disposed.

### Collision policy (characterized, unchanged)
Generated models exclude contacts between the base and every limb link;
MuJoCo filters parent-child pairs; all other pairs collide (links of
different limbs, non-adjacent links of one limb, links with the floor and
obstacles). Consequence: a leg swung into the body passes through it.
Evidence (`tests/physics/collisions.test.ts`): a link inside the base box
gives no contact, and the same model without the exclusions reports the
base-link contact; a joint-range scan of the quadruped's front-left leg
finds configurations that overlap the body, none of which is reported;
overlapping links of two limbs and the folded third link of a chain do
collide, parent-child pairs do not. No change was made: no test or run
showed a failure caused by the policy, and enabling base contacts would
make the hip mounts collide with the body.

### Remote driving stops on focus loss
Remote control stops when the window loses focus, the page is hidden or
the driving gamepad disconnects. Driving keys are ignored in text fields,
selects, sliders, spin buttons, tab lists and with Ctrl, Alt or Meta held.
(Changed on 2026-10-04: no gamepad, and W A S D drive on tabs and sliders;
see "Keyboard is the only remote input".)

### Sessions are disposed with the workspace
The workspace disposes its physics, control, skills and navigation
sessions (WASM objects and workers) when it unmounts. Disposal is
deferred by one task and cancelled if the effect runs again, so React
StrictMode's development remount keeps the sessions alive.

## Keyboard-only driving and CI (2026-10-04)

### Keyboard is the only remote input
The user will not use a gamepad or joystick; anyone opening the app must
be able to drive with the keyboard. The gamepad code (polling, disconnect
handling, state field, readout) was removed rather than kept as an unused
path. Driving keys are the arrow keys and W A S D, matched by
`KeyboardEvent.code`. Nothing drives while text entry has focus (text
inputs, selects, text areas, editable content, spin buttons, list boxes,
menus, combo boxes). On tabs, tab lists, sliders and radio groups the
arrow keys keep their own meaning and W A S D drive. Ctrl, Alt or Meta
never drive. Driving stops on window blur and page hide.
Alternative rejected: ignoring every key on a focused tab, as before. A
keyboard-only user opens Navigate with Enter on its tab, so focus is still
on the tab, and the keys would do nothing with no visible reason.

### Builds and tests run in CI
Type checking, unit tests, the feasibility spikes and the production
build run in CI on Linux with Node.js 24, so every result comes from a
clean install of the lock file. Node floors from the installed package
manifests: vite 8.3.2 and rolldown 1.2.12 need `^20.19.0 || >=22.12.0`;
vitest 5.0.3 needs `^22.12.0 || ^24.0.0 || >=26.0.0`, so Node.js 20
cannot run the tests. The lock file resolves every package from the
public npm registry, and `.github/workflows/ci.yml` runs `npm ci`, the
type check, the unit tests, the spikes and the build. The Playwright
suites are not in CI, because they need matching browser builds that the
CI image does not provide; they run where those browsers are installed.

## Embodied-AI phase 1 (2026-10-04)

### D11. AI proxy: Fastify on Node.js, run with type stripping
The language model is reached only through a local proxy in `server/`;
the browser never sees the key or the endpoint. Options considered:
- Vite dev-server plugin: exists only while `vite` or `vite preview`
  runs, and puts the code that reads the secrets into the process and
  configuration that also build the client bundle, where one wrong
  `define` would copy them into it. Rejected.
- Express 5: workable, but body limits, JSON parsing with prototype
  poisoning protection and in-process request tests (supertest) would be
  added by hand or as more dependencies.
- Fastify 5.12.5 (MIT; 15 direct dependencies according to the npm
  registry metadata): body limit, safe JSON parsing, async hooks for the
  guards, logging off by default, and `inject()` to test routes without a
  socket. Chosen.

`node server/main.ts` runs the TypeScript sources with Node's built-in
type stripping (no build step; unflagged from Node.js 22.18, so `engines`
is now `>=22.18`). The server code uses erasable syntax only
(`tsconfig.server.json` sets `erasableSyntaxOnly`) and `.ts` import
extensions (`allowImportingTsExtensions` in `tsconfig.json`). `npm run dev`
(`scripts/dev.mjs`) starts the proxy and Vite together; Vite keeps running
when the proxy cannot start. The proxy:
- listens on 127.0.0.1:8787 only; Vite's dev and preview servers forward
  `/api` to it, so the browser calls it same-origin;
- refuses origins other than localhost and 127.0.0.1 on ports 5173 and
  4173, POST requests without an Origin, `Sec-Fetch-Site: cross-site`, and
  Host headers other than its own (DNS rebinding); sends no CORS headers;
- accepts only JSON (the text/plain parser is removed), 256 KB at most,
  validated by a strict Zod schema (at most 40 messages of 8000
  characters, 48,000 characters in total, 24 tools, 2048 output tokens)
  with tool names restricted to MorphLab's tools, so it is not a general
  relay;
- limits each session (header `x-morphlab-session`) to 20 requests per
  minute and all sessions together to 100;
- answers with fixed messages: upstream bodies are never forwarded, only an
  error `code` and `param` that match `[A-Za-z0-9_.-]{1,64}`; credentials
  live in private fields whose JSON, string and inspect forms name only the
  provider; there is no request logging.
A static deployment (for example GitHub Pages) has no proxy; the AI
features are then unavailable and everything else works. A public deployment with AI
needs a hosted proxy with user authentication and quotas; none is
provided.

### D12. Provider: Azure OpenAI chat completions
The `ENDPOINT` in `.env` was classified by its shape only (values never
printed): https, a `*.openai.azure.com` host, the path
`/openai/deployments/{deployment}/chat/completions` and an `api-version`
query parameter. The adapter follows Microsoft's Azure OpenAI REST
reference: `api-key` header; roles system, user, assistant and tool;
`tools` as `{type: 'function', function: {name, description,
parameters}}`; `tool_choice`; `tool_calls` with arguments as a JSON
string; `finish_reason`; errors as `{error: {code, message, param}}`. It
sends `max_completion_tokens` for api-versions from 2024-09-01 and
`max_tokens` before (the reference marks `max_tokens` as deprecated and not
supported by reasoning models), and never sends `temperature`. Parallel
tool calls are not requested; the client runs one tool call at a time.
Tool calling and image input depend on the deployed model; the proxy
reports them as unknown because it does not probe. `scripts/ai-probe.ps1`
(PowerShell, no Node.js) checked the configured deployment on 2026-10-04:
gpt-4o-2024-08-06 with an api-version before 2024-09-01, tool calling,
image input and streaming all answered (`docs/status.md`). A chat
completions deployment offers no realtime audio, so voice will use the
browser's speech API (Tier A). Other adapters (generic OpenAI-compatible,
Puter.js) are not implemented.

### D13. Tools and task manager
Zod 4.6.5 (MIT) defines the tool schemas in `src/ai/tools.ts`: strict
objects, bounded numbers, integer counts and identifier patterns. The
model receives JSON Schema generated from the same definitions
(`z.toJSONSchema`, input side) and only for tools the current robot
supports (`RobotCapabilities`); every call is validated again before
anything runs. Results are always `{taskId, status, reason, measurements,
fallback}`; refusals are `rejected` with a reason. `TaskManager`
(`src/ai/tasks.ts`) runs one task at a time: a new task cancels the
running one, commands planned against an older state version are
refused, cancel and modify need the current task ID, the Stop control
applies the body's safe stop first (arm: hold pose; rover: brake; legged:
stance) with or without a task and without any AI service, and an
executor error fails the task and applies the safe stop. Results that
arrive after a task ended are ignored.

### Dependency changes are locked and reviewed
A dependency change regenerates `package-lock.json` with `npm install
--package-lock-only --ignore-scripts`; the lock is reviewed (every
`resolved` URL and integrity hash) and committed, and CI installs exactly
the locked versions with `npm ci`.

### D14. AI command path in the app (D8 amended)
The browser-side cloud interpreter of M6 is removed: it accepted an
endpoint and key in the page and called the provider directly, which the
secrets rule forbids. Commands now go to the local rules first. The parser
marks the refusals that a model with tools may handle (`escalate`: not
recognized, navigation); only those go to the AI path, and only when the
proxy reports a configured provider. Other refusals (walking, torques,
several legs) stay local, so safety refusals never depend on a model.
One AI turn allows up to five model calls and runs at most one action per
model answer; every call is validated before it runs and its measured
result goes back to the model. The command bar's status line is taken from
the last action's measured result; the model's reply is shown separately,
labeled as not a measurement. Changing tabs cancels a running AI task and
makes earlier plans stale; the task manager stops the body the task
started on. Conversation memory keeps the last eight messages for the
current robot and scene and is cleared when they change. Executors exist
for the Navigate tab (navigateTo on the existing navigation stack) and the
Skills tab (executeSkill with the resolver's compatibility, eligibility and
hold rules); `navigateTo` takes the arena bounds and has no speed
parameter, because the navigation stack has no per-goal speed limit.

### D15. Node-free local server
Some machines cannot run Node.js (a managed workstation, for example), so
`npm run dev` is not available there, yet the app with AI should still
run on them. Options considered: another JavaScript runtime (Deno, Bun)
would run npm packages and is against the spirit of such a policy;
serving the app from a shared static host and calling a local proxy
across origins would expose the proxy to every site on that host;
storing the key in CI variables would only test, not run, the app.
Chosen: a production build (`dist/`, from CI or another machine) served
by `server/serve.ps1`,
PowerShell 7 with .NET 8 classes only (HttpListener on 127.0.0.1,
System.Text.Json, HttpClient). App and AI routes share one origin, so the
browser code and its Content Security Policy are unchanged. The script
mirrors the Fastify proxy: routes, host, origin, cross-site and session
checks, 256 KB body limit, the request rules of `ChatRequestSchema`, the
tool allowlist, rate limits, the Azure request and answer mapping and the
fixed error messages. It answers static files while a model call is in
flight (a single-threaded loop waiting on the listener and the upstream
tasks). Two implementations of one contract can drift:
`tests/server/servePs1.test.ts` (CI) checks the tool list and limits
against `src/ai`, and `server/serve.tests.ps1` (local) tests the guards,
mappings and path safety, plus one real request with `-Live`. Costs: no
hot reload (a source change needs a new build), and the
script must be kept in step with `server/app.ts` and `server/azure.ts` by
hand beyond what the tests check.

### D16. Three memories: conversation, world and skill
Phase 1.4 asks for conversation, skill and world memories. Principles:
memories hold measurements and user requests, never model text presented
as fact; each record says where it came from; the user can see and delete
every record; nothing secret is stored.
- Conversation memory (D14): the last eight user and model messages for one
  tab, robot and scene, in the page only; cleared when the context changes
  and by "Forget conversation" in the command bar.
- World memory: named places on a Navigate map (at most 20 per map, names
  of 1 to 40 plain characters, matched without regard to case or spacing).
  A place stores x and y in m and its source: the sensor-only estimate
  (where the controller believes the rover is), ground truth (ground-truth
  mode), or coordinates given in a request (checked against the arena and
  obstacles). Tools: `rememberPlace`, `navigateToPlace` (the tool layer
  looks up the coordinates, so the model never copies numbers) and
  `forgetPlace`; the Navigate tab lists the places, drives to them and
  forgets them without AI, and draws them on the floor. A place is a
  record, not a measurement: a run to it is judged by the referee as any
  navigation run.
- Skill memory: per skill, the live runs from every playback (Play button,
  local commands and AI): runs that reached their end, successes, runs
  stopped early, and the last run with its referee failures and measured
  hold. It is labeled as a record, not an evaluation, and is deleted with
  the skill or from the skill card. Evaluation results stay in the skill
  artifact unchanged.
Storage: one IndexedDB store `memories` (database version 2; the upgrade
keeps every earlier record), validated on read like the other stores;
invalid records are reported and kept until removed. Without storage,
memories last until the page closes. Not done: memories of scene objects
(none exist yet), summaries of earlier conversations, and memory for the
arm and the other tabs.

### D17. Live changes to a running task (D14 amended)
An AI command waits for its task's measured outcome, so the AI cannot take
a second request during the run. Changes during a run are therefore
handled locally, without AI: while a command runs, the command bar takes
short commands (at most six words, no destination, no negation): stop,
slower, faster (by 0.1 m/s, or 0.05 m/s for "a little"), and higher,
lower, closer, farther for tasks that support them. They go to the
active task through `TaskManager.modify`, or to a goal run started in the
Navigate tab; otherwise they are refused with the reason. Stop is the same
local Stop control. For the rover the change is a commanded speed limit
(0.1 to 0.4 m/s) that scales both wheel targets, so the planned path and
its curvature stay the same; the Navigate tab shows the limit, accepts it
as a number and shows the speed measured from ground truth. navigateTo and
navigateToPlace take an optional maxSpeed in the same range (D14 said
navigateTo had none, because no limit existed then). A limit set by
maxSpeed or changed during an AI task lasts until that task ends; then the
earlier limit returns, so "drive slowly" does not slow later commands.
Skills playing on the legged robot cannot change while they run and say
so.

### D18. Streaming replies deferred
Phase 1.3 lists streaming; the provider supports it (`scripts/ai-probe.ps1`:
`text/event-stream`). It is deferred because it would gain little now and
cost much. Each model call took 0.56 to 2.32 s in the browser sessions;
replies are limited to one to three sentences; a tool call must be
complete and validated before anything runs, so only the closing text
could stream, and it would appear well under a second earlier. The cost:
an event-stream relay with its own error and cancel paths in both servers,
one of them a single-threaded PowerShell loop, plus their drift checks.
Revisit when long replies appear (the experiment coach); until then the
command bar shows each tool call and result as soon as it completes.

### D19. Vision: the rover's simulated front camera for the AI
`getCameraSnapshot` renders the Navigate inset camera (70 deg vertical,
about 99 deg horizontal field of view, on the rover's front) off screen
into an sRGB target so colors match the inset, flips it and encodes a
JPEG (quality 0.8; 256 x 154 px by default). The chat contract gains an
optional `image` on user messages: a JPEG or PNG data URL of at most
90,000 characters, at most two per request, outside the 48,000-character
text budget, so two images fit the 256 KB body limit. Both servers
validate it the same way and send it to Azure as a low-detail image part
(the drift test compares the limits and the pattern). Chat completions
do not take images in tool messages, so the loop sends the tool's text
result first and the image right after it as a user message; the data URL
never appears in tool text. The command bar trace shows the exact image
that was sent. The image is a rendering of the simulation used as a
simulated camera: only what the camera would see (the path, rays, grid
and markers are overlays it does not draw), and what the model reads from
it is labeled as the model's words, not a measurement. Not done: cameras
in other tabs, and perception that turns images into positions.

### D20. Voice: local speech output, opt-in browser speech input
The provider is a chat completions deployment, so realtime voice through
the server does not exist (the status route says so). Voice therefore
uses the browser:
- Reading replies aloud uses speech synthesis with an installed voice
  that works without a network service (`localService`), English only,
  at most 400 characters. It is off by default; "Read replies" in the
  command bar turns it on and the choice is stored with the other
  preferences. Results are announced by kind (Refused, Failed, Question),
  and the model's own words as "The AI says".
- Voice input uses the browser's speech recognition (English, one
  utterance per press of Speak, 10 s limit). In Chrome and Edge the audio
  goes to the browser maker's speech service, so the first use explains
  that and asks for consent, stored in preferences; MorphLab neither
  receives nor stores audio. A transcript runs exactly as typed text
  would: slower, faster or stop while a command runs, otherwise the normal
  command path with all its checks.
Seen in the integrated browser (Electron 42, Chrome 148): six local voices;
speech recognition exists but gave no events within 6 s under automation
(no microphone), so voice input is not verified there.
Amended after the browser check: nothing is spoken while the microphone
listens and speech stops when listening starts, so the app's own voice
cannot become a command; "Turn off voice input" withdraws the consent.

### D21. Conversation memory keeps the evidence behind each reply (D16 amended)
In the browser check the model answered "is anything on your left?" with
an invented range and once claimed "The rover successfully returned to
the dock" without calling any tool. Probes through the local server with
the real model (gpt-4o-2024-08-06), same system prompt and tools, four
tries each:
- Plain history (earlier request and reply as text) before "is anything on
  your left?": 0 of 4 called a tool; all four stated a distance (0.29 to
  1.80 m) that no tool had measured. Without history: 4 of 4 called
  getRobotState.
- A tag in the stored reply ("[Earlier reply, written from ...]"): tools
  were called, but once the model copied the tag into its answer and
  claimed a result it had not measured. Rejected.
- Earlier tool calls and results kept as real tool messages: 0 of 4 called
  a tool; the model answered from the old readings. Rejected.
- A system note after each stored reply naming the tool results it was
  written from, plus one prompt sentence that state questions need a fresh
  tool result: 4 of 4 called getRobotState for the left question and 4 of
  4 called navigateToPlace for "drive back to the dock" after a stored
  success. Chosen.
Conversation memory therefore keeps the last four exchanges as request,
reply and note ("Note: the reply above was written from tool results
measured then (navigateTo succeeded); it is not a current measurement",
or "used no tool result"). The command bar still labels every reply that
ran no action as the model's words, which caught the false claim.

### D22. Experiment coach in the Control lab
The coach is the AI in the Control lab tab; it explains step tests and
tries gains, through the lab's own experiments:
- Tools: getRobotCapabilities; getRobotState (both gain sets, the
  explicit-integration check, and the shown step-test results described
  with the setup they ran with); runControllerExperiment, which sets gain
  set B and runs the lab's step test for A and B; cancelTask. Gain set A
  is the learner's: the AI cannot change it, the joint or the step, so
  every experiment is a comparison against the learner's baseline that the
  learner also sees as plots.
- Step tests are deterministic, so the `trialCount` of the phase 1 schema
  was removed: repeated trials would report identical numbers.
- The result holds the measured metrics of A and B (rise time, overshoot,
  settling time, steady-state error, saturated fraction, damping ratio).
  A run with B unstable is a failed experiment whose fallback states the
  stability rule. A run that was stopped or whose setup changed reports
  nothing: the executor accepts only new results for the requested gains.
- The context key holds the robot, joint, step, duration and command but
  not the gains, so the coach's own changes to B do not make its plans
  stale or clear the conversation.
- Stop ends the experiment; a task can now say what stopping does
  (`TaskExecutor.stopText`), here "The experiment stopped; its unfinished
  results were discarded." instead of the arm's hold text.
- Questions that mention torque now reach the AI; the local torque rule
  refuses commands only.
Prompt probes (first model call, real model, two or three tries each):
with "explain results from the measured metrics, change one thing at a
time", "How can I reduce the overshoot?" got advice without data (two of
three with LaTeX), "make it settle faster without overshoot" changed kd
and gravity compensation together, and "why is the torque saturating?"
called no tool. With the final wording (read the shown step test with
getRobotState first; change gain set A's settings in one respect; symbols
as plain words) both questions called getRobotState three of three times,
and experiments changed kd only. Streaming stays deferred (D18): coach
replies are still at most three sentences.

Amended 2026-10-05, after the browser check:
- The configuration of runControllerExperiment is optional. A baseline
  had to copy A's settings onto B and overwrote the learner's B; without
  a configuration the tool now runs the lab as set and changes nothing.
  With one, the result names B's earlier gains.
- Measured diagnosis instead of model inference: a run whose final 10 %
  varies by less than the 2 % band while its mean lies outside it "came
  to rest" off target and never settles. The result text says so, the
  summary states what changes a resting error under PD (gravity
  compensation removes it, kp shrinks it, kd does not), and the lab shows
  the same explanation as a note computed from the run. The check uses
  the recorded response rather than the steady-state error alone, so a
  response that is still rising or ringing is not blamed on gravity.
- Replies are rewritten as plain text in the loop (`plainReply`): the
  plain-text instruction reduced but did not stop LaTeX such as \(kd\),
  and the text is also spoken and kept in the conversation memory.
- Final wording, chosen by probes with constructed tool results (status,
  "Coach fixes"): read the shown step test with getRobotState first;
  change exactly one value from gain set A, or from gain set B when its
  results are shown; for each goal the user named, quote the measured
  value from the latest result and say whether it meets the goal.
  "Copy gain set A and change exactly one value" held in fresh
  conversations (11 of 11) but not with conversation memory, where the
  model iterated on B or changed kp and kd together. Iterating on B one
  value at a time is sound experimenting, so the wording allows it;
  quoting values stopped most misread results ("without overshoot" at
  12.4 %).
- A result whose configuration differs in more than one value from both
  A and B's earlier settings says that it does not show the effect of one
  change alone. The tool does not refuse such runs: a learner may ask for
  a specific set of gains.
- With memory the model also sent two experiments in one answer; the
  loop runs one action per answer and refuses the rest. After a refusal
  the model answered instead of calling again (8 of 8, with two refusal
  texts). Turning off parallel tool calls at the provider would stop
  this, but read-only calls the model now makes together would then cost
  one model call each within the budget of five per turn; not done.

### D23. Skills the user teaches (D4 amended)
The vision says users teach skills to their robot; M5 shipped one
built-in skill (the foot lift, obtained five ways), and every other request
was refused, some with suggestions for features that did not exist
("record a demonstration of it first"). Decision:
- A taught skill is a motion the user poses plus goals the user chooses.
  The motion is keyframes captured from the live robot's commanded joint
  targets (joint sliders in the Skills simulation, under the joint
  controller, so the body can fall while it is posed), each with the time
  to reach it, and a final hold. Playback blends from the home stance
  through each keyframe with a cosine profile, clamped to the joint limits
  and velocity limited like every target. It is open loop and labeled so:
  "Keyframe motion you posed (open loop)".
- Goals say what success means and are measured from ground truth by one
  `GoalJudge`, used in evaluation trials and live playback: the body stays
  upright, does not fall, a limb tip is above a height for a time, the
  body is below or above a height for a time, all feet leave the ground,
  the body rises above its resting height, moves forward, back or sideways
  in its start heading, turns (unwrapped yaw), and ends standing (last
  0.5 s). Heights above the start count only after the body first comes
  to rest within 3 mm of its resting height in the home stance, so a
  preset placed slightly above the ground does not "rise" as it settles.
  Fixed bases take limb-tip goals only. Each goal also gives a progress
  score in [0, 1] (`goalReward`) for later reward-based improvement.
- Evaluation is the same 20 randomized trials in the worker; the evaluation
  context stores the goals, and the metrics count trials per goal met.
- Commands: the skill's own name (letters, digits, spaces, hyphens,
  underscores, apostrophes; at most 40) is matched before any other rule,
  with framing words such as "please", "do the", "now" or "again"
  stripped, so a taught name wins over built-in words. The best evaluated
  version for the exact morphology runs (lower Wilson bound first) and the
  reply lists every goal with its measured value. The AI lists taught
  skills with their goals and runs them with executeSkill; holdSeconds
  applies to foot lifts only.
- Artifact (D4): `skillType: 'foot_lift' | 'taught'`; taught skills have
  `method: 'keyframes'`, `motion` (joint ids in global order, keyframes
  with absolute times, end hold) and `taught: { name, goals }`, and their
  `limbs` lists only the limbs their goals name (it may be empty). Import
  validates goals and keyframes like every other field.
Why keyframes: the smallest general mechanism by which a user can teach
any posture or motion the body can perform, fully determined by data the
user sees and edits; physics decides whether it works and the goals measure
it. Not chosen now: recording teleoperation of every joint (sliders move
one joint at a time), learning from goals alone (reinforcement learning at
this scale is too slow in a browser worker and opaque to a learner), and
motions generated by the AI (possible later as validated JSON the user
evaluates; the model never produces joint targets at control rate).
Not yet: improving a taught skill by reward or making it closed loop by
imitation; drafts are not saved until evaluated.

### D24. The AI composes skills (D14 and D23 amended)
After D23 a request for a motion nobody had taught still ended in "teach
it first": the local rules refused walking, jumping, dancing and sitting
without asking the AI (D14), and the AI had no tool that creates a skill.
The robot could do only what someone had posed by hand. Decision: the AI
composes a skill draft as validated JSON, the simulator tries it and judges
it by its goals, and the AI revises, evaluates and runs it, all within one
command and all visible in the Teach panel.
- Poses are described in terms of the body, not joint angles: `body` moves
  the body from its home stance (x, y, z in m; roll, pitch, yaw in rad)
  while the feet stay where they stand; `tips` moves a limb tip from where
  it rests (m, home body frame); `joints` sets joint targets (rad) within
  their limits, last. Every pose is relative to the home stance.
  `src/skills/compose.ts` turns them into joint targets with the kinematics
  core (damped least-squares IK seeded from the previous pose, inside the
  joint limits). Unknown limbs and joints, values out of range, a body
  move on a fixed base and tips on wheels are refused with the reason;
  a tip IK cannot reach is reported with its distance in mm, never hidden.
  Why: a model cannot be expected to know each robot's joint signs, but
  "lower the body 8 cm" and "lift the front left foot 8 cm after shifting
  the body away from it" are physical statements IK can make exact. The
  system prompt describes the body in numbers (tip positions at home,
  joint axes in the body frame, limits, home values, resting height,
  effort limits; `describeBody`).
- Tools (19 in total): `draftSkill` (name, 1 to 8 keyframes with the time
  to reach each, a final hold, 1 to 6 goals with the D23 goal kinds and
  bounds) composes the poses; a refused draft runs nothing. An accepted
  draft replaces the Teach panel's draft only when that is empty or the
  AI's own unedited draft, so the user's unsaved work is never lost; it is
  then tried once in the live simulation and every goal comes back as
  measured, plus any pose the IK could not reach. `evaluateSkillDraft`
  runs the D23 evaluation (20 randomized trials in the worker) and saves
  the skill only when at least one trial met every goal (the Teach panel
  still saves every evaluation, as before). The result gives the success
  rate, its Wilson interval, trials per goal and the saved name. Both are
  actions (one per model answer) and go through the task manager; Stop
  ends a try (stance) or cancels an evaluation (nothing saved).
- Honesty: skills record who composed them (`taught.author`: absent for
  the user, `ai`, or `ai-edited` once the user edits the AI's draft); cards,
  playback labels and the AI's skill list say "Keyframe motion the AI
  composed (open loop)". The prompt asks for goals that measure what the
  user asked for and forbids weakening a goal to make a draft pass; the
  goals are shown in the Teach panel, on the card and in every result, so
  a weak goal is visible. A single try is reported as "one run, not
  evaluated, not saved".
- Routing (D14 amended): walking, jumping, dancing, waving and posture
  requests, and a lift on a robot without legs, now go to the AI when the
  local AI server is available; without it the refusal still explains how
  to teach the motion and adds that the AI can compose it. A foot lift
  that no evaluated skill serves goes to the AI in the Skills tab. Torque
  commands and lifting several legs at once stay local refusals. In a tab
  without AI tools such a request gets the local answer with a note on
  where the AI acts.
- Loop (D14 amended): up to 10 model calls per command (was 5), so the AI
  can compose, try, revise, evaluate and run in one command; still at most
  one action per answer. After an action the loop asks the view for fresh
  capabilities, so a skill saved during the command can run in it. Long
  commands drop the oldest remembered exchanges before the request would
  pass the proxy limits (40 messages, 48,000 characters); if it still does
  not fit, the command ends and says so. Tool results may be 2,500
  characters (was 1,500) and answers 1,600 output tokens (was 800), so a
  draft with several poses fits; arguments over 4,000 characters are
  refused with their length instead of being cut into invalid JSON.
Not done: closed-loop (sensor feedback) versions of composed skills,
reward-based improvement of their poses and timing, composing on robots
outside the Skills tab, and multi-skill sequences planned ahead.

### D25. Physical-AI pipeline: checked goals, gaits, improvement, sequences, tools in every tab (D23 and D24 amended)
The browser check of D24 showed the gap: asked to "sit down like a dog",
the model composed a level crouch and judged it by height and tilt only,
then called it a sit. Goals were the model's choice and nothing measured
what the request meant. The robot also could not walk (keyframes play open
loop and have no contact pattern), could not get better at a skill except
by the model rewriting it, and the AI had no tools outside the Navigate,
Skills and Control lab tabs. Research behind the choices:
`docs/research-paths.md` (L2R, SayTap, PMTG, Eureka, SayCan, Inner
Monologue, Text2Motion, Salimans et al.).
- Goals (D23 amended): four new kinds, all judged from simulator ground
  truth. `bodyAngle` (pitch or roll within a band for a time; positive
  pitch lowers the front, positive roll raises the left side), `footDown`
  (a foot on the ground for a share of the run), `tipSwing` (strokes of a
  limb tip along x, y or z of the start heading, each at least a given
  length; the first stroke counts once the range seen spans it) and
  `tipReach` (a tip within a tolerance of a point given relative to where
  it started, for a time). A fixed base takes the three limb-tip kinds.
  Every kind has a progress score for the reward.
- Request check: `src/skills/intent.ts` maps motion words in a skill's
  name (or, when the name has none, in a request for exactly one motion)
  to a measurable property: sit pitches nose up, bow nose down, lean left
  or right rolls, lie down and crouch lower the body against its resting
  height, stand tall raises it, jump lifts every foot, walk and turn need
  travel and turn goals in the asked direction ("in place" excepted), wave
  needs a swinging tip, point or reach a reached point, lift a raised tip
  (or the body). Negated words are skipped, limb words ("front right",
  "back left") bind the goal to that limb. A draft from the AI whose goals
  miss one is refused with what to add; the Teach panel shows the same
  finding as a warning on the user's drafts. Why literal rules: they are
  predictable and testable; the lexicon is small on purpose and words
  without a rule are not judged.
- Gaits (`src/skills/gait.ts`, after SayTap and PMTG): a skill may be a
  gait instead of keyframes: a contact pattern (trot, walk, pace, bound,
  pronk, wave, stand, or one phase per limb), duty factor, cycle time,
  stride, side step and turn per cycle, step height, body height and
  duration, optional legs held up. Phases come from where the feet stand
  (left and right, front to rear), so a trot is a tripod on six legs.
  Stance feet slide back in the body frame, swing feet arc forward, the
  step fades in and out over a cycle, and leg IK turns foot targets into
  joint targets at 100 Hz. IMU feedback (sensor only) changes foot heights
  in proportion to body roll and pitch and their rates; with non-zero
  gains the skill is labeled closed loop. Not learned: the trajectory
  generator is hand-designed.
- Improvement (D3 amended; L2R, Eureka, Salimans et al.): evolution
  strategies in the worker adjust a draft's keyframe joint values and
  durations, or a gait's parameters and feedback gains (stride, side step
  and turn keep their direction), to raise the goal reward in randomized
  trials with common conditions per iteration. The reward is half the mean
  goal progress and half the worst goal's progress, plus 1 for a run that
  met every goal: in the browser check a reward of the mean alone let the
  search trade staying upright for four swinging feet. The draft changes only when
  the result scores better than the original on held-out conditions, and
  the record (rollouts, held-out rewards and successes, seed) goes into
  the saved skill's training field. Defaults: 10 iterations of 4 pairs,
  one trial each, 8 held-out trials (96 simulated runs).
- Sequences (SayCan, Text2Motion): `runSkillSequence` runs up to 12 saved
  skill runs one after another, each from the home stance and judged on its
  own, stops at the first that does not succeed, and reports the chance of
  success estimated beforehand (the product of evaluated success rates)
  next to what was measured. Names are checked before anything runs.
- Tools in every tab (27 in total): `improveSkillDraft`,
  `runSkillSequence` and `selectRobot` (a preset or the Build tab's robot)
  in the Skills tab; `pushRobot` (an impulse along the body heading, with
  2 s of measured response from the session telemetry: largest tilt,
  lowest height, largest joint error) and `resetSimulation` in the Physics
  tab; `moveTipTo` (IK status and residual) and `setJointAngles` (clamped,
  clamping reported) in the Kinematics tab, labeled geometry only;
  `buildRobot` (a preset, a generated legged robot or arm, validated by the
  builder; a refused design changes nothing) in the Build tab. After
  `selectRobot` succeeds the controller adopts the new robot for the rest
  of the command instead of treating the change as stale (D14 amended),
  and the system prompt is rebuilt for it.
- Limits (D11 and D14 amended): one message may hold 16,000 characters
  (was 8,000). In the browser check the Skills tab's system text (rules,
  starting points, the body description and the skill list) passed 8,000
  as soon as a skill was saved during a command, and the request was
  refused. The skill list now closes the system text and is shortened to
  fit (without goals, then cut, with `listSkills` for the rest; `listSkills`
  takes an optional skill name for one skill's goals and records); the
  system text is cut at the limit as a last resort. A tool result longer
  than 2,500 characters leaves out its measurements, then the end of its
  reason, and stays valid JSON. The 48,000-character total is unchanged.
- The AI may not run a saved skill whose goals miss a motion word of its
  name or of the request (a level crouch saved as "sit"); the skill list
  says so, and a command by name prefers the version whose goals fit.
- Feedback from a try (Inner Monologue): the result names every goal as
  measured and gives checkpoints from ground truth: the body (forward,
  left, height, pitch, roll, heading change) and the tips of the limbs the
  goals name, at each keyframe, each second of a gait and the end. In the
  browser check the model kept a second keyframe that undid the pose until
  it could see where the body was at each keyframe.
- Follow-up (D14 amended): the model often answered "I will revise it"
  after a try and stopped. When it answers right after a try that missed a
  goal, the loop reminds it once to revise (up to three drafts) or improve,
  and after a try or improvement that met every goal, to evaluate and save
  it; at most two reminders per command, within the 10 model calls.
- Travel and turn goals take an optional maximum, so "about 90 degrees" is
  judged between both bounds (a 185 degree turn no longer passes), and a
  gait draft whose stride, side step or turn points the other way than its
  goals is refused before it runs (the model once turned left for "turn
  right").
Not done: learned locomotion policies (Tier C), feasibility between the
skills of a sequence (each restarts from home), feedback on anything but
body attitude, and a larger request lexicon.

### D26. Mirror images of skills; review of nine physical-AI repositories (D25 amended)
Nine repositories were reviewed for the pipeline: RSL-RL, MimicKit,
TD-MPC2, DreamerV3, Diffusion Policy, ACT, robomimic, rlkit and BeyondMimic
whole-body tracking (`docs/research-paths.md` has the table with licenses,
requirements and the decision for each). All are Python training
frameworks for GPUs; none runs in the browser, and their training belongs
to Tier C. Two ideas fit the browser pipeline without replacing anything:
- Mirror images (RSL-RL's symmetry extension, after Mittal et al., ICRA
  2024; mirrored motion data in motion imitation): `src/skills/mirror.ts`
  turns a taught skill into the draft of its other side. Limbs pair by
  mount point; joints pair in order and keep or flip their value depending
  on whether the partner axis is the mirror image of this one or its
  opposite, so both the builder's mirror convention and the presets'
  opposite axes (quadruped hip roll, hexapod coxa yaw) work. The map is
  verified with forward kinematics before use (home stance symmetric, every
  limb tip of a mirrored pose at the mirror image of its partner's); a body
  that fails is refused with the reason. Goals follow (left and right,
  turn direction, roll band, limbs, the side of a reach); a gait's side step
  and turn change sign and each leg takes its partner's phase, so a mirrored
  trot starts with the other diagonal pair. Why not trust the mirror: the
  mirrored draft is tried once and evaluated over 20 randomized trials
  like any draft, because geometric symmetry does not prove the same
  physical result. The AI calls `mirrorSkill` (28 tools in total) instead
  of composing the other side again, which also removes a source of sign
  errors seen in the D25 browser check; the Teach panel has a Mirror left
  and right button for the user's draft.
- Robustness after randomized evaluation (BeyondMimic's emphasis on the
  conditions that fail; domain randomization in all of them): an
  evaluation below 80 % of trials tells the AI that `improveSkillDraft`
  over the same randomized ranges can make the skill more reliable before
  a second evaluation; commands run the version with the better
  evaluation. Failure-weighted sampling of conditions inside ES was
  considered and not adopted without a measurement showing it helps.
Not adopted: PPO, world models (TD-MPC2, DreamerV3), diffusion and
transformer policies (Diffusion Policy, ACT), offline RL (robomimic,
rlkit) and adversarial motion priors (MimicKit). Each needs GPU training
on large batches or datasets, and the policies would not run at control
rate in the browser.

### D27. A learned feedback policy in the joints of a gait (plan Phase 1; D25 amended)
The user asked for physical AI in the strict sense: a learned network that
sets joint targets every control step, trained in simulation, with the
language model starting the training instead of composing every motion.
`docs/physical-ai-plan.md` holds the research, the feasibility numbers and
the phases; this is Phase 1, on the ARS core of Phase 0.
- Structure, after Policies Modulating Trajectory Generators (Iscen et al.
  2018): the gait generator stays, and a linear policy adds
  e(t) * 0.15 rad * clip(W x + b, -1, 1) to every joint target at 100 Hz,
  where e(t) is the gait's fade-in and fade-out envelope. The sum is
  clamped to the joint limits and goes through the rate limiter and PD
  loop like every skill. A zero policy is the gait exactly (tested, also
  in MuJoCo); a non-finite output hands the rest of the run to the gait
  alone and the phase says so.
- Observation v1, sensor-only, 12 values: gravity direction x and y in the
  body frame from the IMU, gyro, the gait phase as sine and cosine of the
  phase and of twice the phase, and the gait's command; normalized by
  running statistics stored with the policy. 13 parameters per joint (156
  on the quadruped, 234 on the hexapod).
- Training, in the skills worker: ARS V2-t (Mania, Guy and Recht 2018),
  30 iterations of 8 antithetic directions, top 4, step 0.02, noise 0.03,
  common randomized conditions per iteration (the evaluation's friction,
  mass and joint-offset ranges); 497 simulated runs with the comparison.
  The return comes from simulator ground truth, which the policy never
  sees: per 10 ms step exp(-|v_xy - target|^2 / 0.01) and
  0.5 exp(-(w_z - target)^2 / 0.05), faded like the gait, minus
  2 (g_x^2 + g_y^2), 0.5 v_z^2 and 0.01 per squared action change; a fall
  ends the run with -2; plus twice the skill's goal reward (D25). The
  target is the gait's own speed, measured in one run of the gait alone,
  along each direction it commands, and zero along the others: the policy
  removes drift and wobble without changing the motion the goals were set
  for. The first version followed the nominal command (stride per cycle
  over the cycle time); in the browser it kept a policy that halved the
  travel of a hexapod side step whose goal had been tuned to the gait's
  real, faster speed. That version is replaced.
- A policy is kept only when, on 8 held-out trials with separate seeds, its
  return is higher than the gait alone with no more falls, no fewer trials
  meeting every goal and a mean goal reward at most 0.01 lower; otherwise
  the draft is unchanged and the result says so.
- Skills: method `gait-learned` (gait, policy and training record in the
  artifact; sizes validated on load; bound to the morphology hash; joint
  count checked). Any edit of the draft drops the policy; trial-and-error
  improvement runs on the gait alone and drops a policy trained for the
  earlier gait; a mirror image does not carry the policy over.
- AI: tool `trainGaitPolicy` (29 tools; legged bodies, Skills tab). When
  the request asks for learning (a learned policy, controller or network,
  training) and the draft is a gait, the results of the try, the
  improvement and the evaluation, and the follow-up, say to train before
  evaluating: in the first browser check the model evaluated without
  training. 12 model calls per command (was 10). The model never sets
  weights or joint targets; it starts training and reads measured results.
- Why ARS and a linear policy now rather than PPO: 14,000 to 17,500 physics
  steps per second per thread in CI gait trials give about 500 runs of
  4.5 s in 70 to 130 s; ARS matched deep RL on MuJoCo locomotion with linear
  policies (Mania et al.), and PMTG used linear policies over a trajectory
  generator. PPO stays Phase 3; a command-conditioned controller per body
  is Phase 2.

### D28. A learned walking controller per body, commanded by speed or target (plan Phase 2)
The user asked to build Phase 2 after the Phase 1 report: one walking
controller per legged body that the language model commands, instead of a
gait or a policy per skill.
- Structure: the gait generator (trot on four legs, tripod on six, 0.5 s
  cycles, IMU feedback) is driven by a walking command in the body frame
  (forward, sideways and turn rate). Two learned parts sit on top:
  - A calibration from the speed asked for to the generator's command
    (system identification). 18 runs at mid conditions (each axis both
    ways at 0.8 and 0.4 of its range, and pairs of axes) measure the
    velocity each generator command gives; least squares fits 3 rows of 6
    coefficients over the positive and the negative part of each axis, so
    the fit holds sign-dependent gains and the drift one axis causes in
    another (on the quadruped a side step drifts backward and turns). A
    command is inverted over the 8 sign patterns; the smallest consistent
    solution wins, and per-cycle values are clamped to the generator's
    bounds (the result says when).
  - A feedback policy, `LearnedGaitController` of D27 (linear,
    Observation v1, at most 0.15 rad per joint at 100 Hz), trained by ARS
    over random commands: 40 iterations of 8 directions, top 4, every
    candidate scored on 2 random commands under the evaluation's
    randomized friction, mass and joint offsets. The return is that of
    D27 with the command as the velocity target, plus a pose term (0.5
    exp(-|p - path|^2 / 0.01) + 0.5 exp(-(yaw - path yaw)^2 / 0.05) per
    second, where the path is the command integrated with the fade).
- Keep guard: on 8 held-out commands the policy must beat the calibrated
  generator in return, with no more falls, and must not end further from
  where the commands lead (mean position error and mean heading error).
  The first version (09bf7d5) kept a quadruped policy by return alone
  (3.33 to 3.79) whose heading error rose from 0.032 to 0.128 rad; the pose
  term and the endpoint guard came from that.
- Evaluation: 20 randomized commands (the six single-axis commands at
  three quarters of each range, then random ones) for the generator
  without calibration, with it, and the saved controller. A walk is within
  tolerance when the body stays up (no fall, tilt at most 0.5 rad) and
  ends within a quarter of the commanded travel and turn (at least 5 cm
  and 0.15 rad) of where the command leads. Wilson interval as in D4.
- Measured (CI, deterministic, identical in the browser on 4 workers):
  the policy was rejected on both bodies. Quadruped: held-out return 6.380
  to 6.829, position error 0.062 to 0.051 m, heading error 0.040 to
  0.094 rad; evaluation 7/20 calibrated against 2/20 uncalibrated. Hexapod:
  return 8.573 to 8.127; 20/20 calibrated against 0/20. The calibration
  carries the accuracy; a linear policy that cannot see heading or
  position cannot meet a pose reward. Next: heading and position error in
  the observation (Observation v2), interaction terms in the calibration.
- Worker pool: `RolloutPool` splits each batch of rollouts into contiguous
  chunks over `min(4, max(1, hardwareConcurrency - 1))` skill workers (job
  `rollouts`) and joins the outcomes in order. `augmentedRandomSearchBatched`
  makes the same draws and updates as the sequential version, and running
  statistics travel as plain parts, so the result does not depend on the
  number of workers (tested). 1354 runs take 89 s in the browser on 4
  workers, 141 to 310 s in CI on one thread.
- Storage: one controller per morphology in the settings store under
  `locomotion.<morphology hash>` (kind `morphlab.locomotion`, schema 1,
  strict parsing; a controller for another morphology is refused). Not in
  bundles yet.
- AI: tools `trainLocomotion` (legged bodies, Skills tab) and `locomote`
  (31 tools). `locomote` takes either a target endpoint (forwardM, leftM,
  turnRad from where the body stands) or speeds with durationS. For a
  target, `walkForTarget` solves for constant speeds: the end position is
  linear in the two speeds for a fixed turn rate, and the 2 x 2 system has
  the squared length of the faded heading average as determinant (positive
  below a full turn); it picks the shortest duration, in 0.1 s steps, whose
  speeds stay within 80 % of the trained ranges. In the browser the model
  computed durations from speeds itself and walked 0.135 m for a 0.15 m
  request, so distances and angles go through the target form. Every walk
  reports where the command leads and where the body went (ground truth,
  start frame); the result is named by where the walk ended and whether
  the body stayed up, never "succeeded" next to an error outside the
  tolerance. The controller reads sensors only; the model never sets joint
  targets or weights.

### D29. A walking policy that sees where the body is (D28 amended); jumps must land; walks chain
After D28 the user asked to train the walking controller properly, so the
learned part is what walks, and to try many motions. D28's policy was
rejected on both presets: it could not observe how far the body was from
where the command leads.
- Sensor-only state estimate (`src/skills/odometry.ts`): heading from the
  IMU orientation relative to the start, body velocity from leg odometry
  (a foot whose touch sensor reads more than 1 N now and at the previous
  update is planted, so the body moves at -(dp/dt + w x p) in its own
  frame, p from forward kinematics of the joint encoders, w the gyro),
  integrated into the start frame. No simulator state. Measured over the
  40 evaluation runs: 0.018 m from ground truth on the quadruped, 0.014 m
  on the hexapod; heading below 0.001 rad (the simulated IMU has no drift).
- Observation v2 (18 values, `src/skills/walkPolicy.ts`): Observation v1's
  gravity direction, gyro and phase, the walking command, the heading
  error, the position error and the velocity error in the body frame
  (path minus estimate; the path is the command integrated with the
  stepping envelope), and the envelope.
- Action: the policy changes the gait generator's stride, side step and
  turn per cycle (at most 0.05 m, 0.03 m and 0.15 rad, low-pass filtered
  with 0.1 s), the way PMTG modulates the generator's parameters
  (`GaitController.setMotion`). Optional per-joint corrections (Action v1,
  0.15 rad) were measured and are off by default: with them the quadruped
  policy was rejected (held-out return 7.824 to 7.748, heading error 0.036
  to 0.045 rad); with the command outputs only (57 parameters) it was kept.
- Reward: D28's, with the pose term at the scale of the walk tolerance
  (exp(-|e|^2 / 0.0025) and exp(-e_yaw^2 / 0.01), weight 1 each). At the
  earlier widths (0.01 m^2, 0.05 rad^2) a heading error of 0.04 rad cost
  almost nothing, while the keep guard compares heading errors strictly.
- Training: 80 iterations of 8 directions on 2 random commands each, 12
  held-out commands, 40 evaluation commands; 2722 runs. 254 s for the
  quadruped and 468 s for the hexapod in the browser on 4 workers; 352 to
  419 s and 509 to 675 s in CI on one thread. The keep guard is D28's.
- Measured (CI, identical in the browser): quadruped policy kept,
  held-out return 7.824 to 10.064, position error 0.067 to 0.014 m,
  heading error 0.036 to 0.015 rad; evaluation 40/40 (95 % interval 91 to
  100 %), 0.013 m, 0.014 rad, against 19/40 for the calibrated generator
  alone and 2/40 without calibration. The learned gains are proportional
  feedback from the position and heading errors to the stride, side step
  and turn. Hexapod policy rejected: the calibrated generator already ends
  within 0.004 m on 40/40, and the policy added error (0.004 to 0.014 m).
- Locomotion records are schema 2; a schema 1 record still walks with its
  calibration but a policy trained for Observation v1 is refused.
- Jumps: a saved "jump" that only asked for all feet off the ground passed
  20/20 while the robot flipped onto its back (feet in the air). The
  airborne goal now counts only while the tilt is at most 1 rad; the
  request check requires a jump to measure its landing (standsAtEnd or
  upright); and a run whose goals pass while the body tips past 90 degrees
  says so in its result with a suggestion to add the missing goal.
- Paths: every live playback used to reset the simulation, so a square of
  walks and turns restarted at the origin after each step. Walks by
  command and saved gait skills now start where the body stands when it is
  up (tilt under 0.35 rad, body above 70 % of its resting height; ground
  truth used only for this reset decision), and the result says where each
  started. Keyframe skills and foot lifts still start from the home stance.
  In one model answer, locomote and executeSkill calls run in order while
  each succeeds (other actions stay one per answer), so a path needs one
  answer instead of one model call per side.
- The AI summary routes walking, turning and side steps by a distance or
  an angle to locomote when a controller exists; the model had composed or
  mirrored gait skills for a right turn and a side step instead.

### D30. Learned feedback policies for keyframe skills; judging runs as a viewer would
The user asked for real physical AI rather than recorded motion replayed:
jumping and standing on two legs, with the learned part doing the work, no
hand-written numbers, and verification that matches what is seen on screen.
- Policy (`src/skills/posePolicy.ts`): the keyframes, composed by the AI or
  the user, only sketch the motion; a linear network moves every joint
  target by up to 0.5 rad every 10 ms. Observation v4 (23 values on the
  quadruped): gravity direction x and y in the body frame, gyro, one
  contact flag per foot, motion progress and its sine, and each joint
  encoder minus the sketch's target. Sensors only; ground truth only
  judges. NaN output falls back to the keyframes for the rest of the run.
- Training (`src/skills/trainPose.ts`): augmented random search (D27's
  batched ARS), 100 iterations of 12 directions, two randomized runs per
  candidate (friction, body mass, joint offsets) and one push of 0.2 to
  0.8 N*s per run at a random moment and direction; 4828 runs, 84 to 112 s
  in the browser on 4 workers. Contact flags are normalized to -1 and 1.
  Reward per control step: the share of the goals holding now (a hold goal
  already met counts as holding, so the policy is not paid to stay in a
  pose past its hold), standing as an end goal asks once the motion is
  over (weight 4), minus spin and correction rate; a fall costs 2; plus
  twice the goal reward of the verdict. Kept only if, on 12 held-out pushed
  runs, it has no more falls, no fewer runs meeting every goal, and more
  such runs or a goal reward higher by more than 0.05.
- Goals: airborne may name the feet that leave the ground together
  (limbIds); a stand on the hind legs or on two legs must measure its
  lifted feet that way (two tip heights can be met one after the other),
  and a jump must name none. A jump counts as airborne only within 34 deg
  of upright; named feet allow a lean up to 90 deg.
- Body checks (`GoalJudge`, `MujocoRuntime.groundTruthTouchingParts`): for
  every run of a free-standing body, whatever its goals: a part other than
  a foot (the base box, any joint of a limb) within 5 mm of the floor for
  0.02 s or more fails the run unless a goal asks the body to lie low, and
  so does a tilt past 90 deg. They are in the verdict of evaluations, live
  runs and tries, counted per trial, shown to the user and the AI, and in
  every learned policy's reward.
- Looking at the run (`reviewLastRun`, `src/render/filmstrip.ts`): six
  frames of the last recorded run from the default camera view are sent
  to the model as an image with the measured result. After a skill is
  evaluated and saved, the AI runs it once and looks before answering; its
  verdict on the image is labeled as its judgment, not a measurement.
- No hand-written pose recipes in the AI's instructions any more: they
  state the physics, and the model chooses the numbers from the body
  description and the measured tries.
- AI loop: an evaluation with no trial meeting every goal is revised
  within three drafts instead of ending the answer; a threshold the model
  chose stricter than the request may be corrected. The tool list is
  counted as the proxy counts it (System.Text.Json escapes more
  characters), and a long command shortens tool results already read.
- Language: "stand", "balance" and similar run the home stance only as a
  short command; requests to stand on the hind legs or on two legs go to
  the AI (seen in the browser: they ran the home stance).

### D31. No images go to the language model (D30 amended); stability margin in checkpoints
The user asked that the app send no pictures to the AI: an image costs
many tokens on every request that carries it.
- Removed: `reviewLastRun` and the filmstrip renderer (D30),
  `getCameraSnapshot` and the off-screen capture of the Navigate camera
  for the AI, and the follow-up that made the model run a saved skill
  again to look at it. The camera inset on screen stays. The model judges
  a run from measurements only: goals, body checks and checkpoints.
- The contract refuses images: user messages in `ChatRequestSchema` have
  no image field, and both proxies (`server/app.ts` through the schema,
  `server/serve.ps1` by its own check) refuse a message that carries one;
  the image mapping for Azure is gone. The provider status still reports
  `imageInput`, a property of the deployment, not of what the app sends.
- In place of looking, checkpoints of free-standing bodies give the static
  stability margin from ground truth: how far the center of mass (MuJoCo
  `subtree_com` of the base) is inside or outside the convex hull of the
  feet on the ground, and where the nearest edge is
  (`src/core/math/support.ts`). Seen in browser trials: the model shifted
  the body toward the foot it lifted, and dropped the shift in the next
  pose. The draftSkill schema now says each pose is whole from the home
  stance and gives the signs of pitch and roll.
- Known gap: goals the model chooses can be weaker than the request (seen:
  a "lasso" judged only by a sideways swing of 1.26 m passed every goal
  while the arm swept around the base). Without images, only the request
  checks of `src/skills/intent.ts` catch this.

### D32. Correct training feedback without changing saved policy playback
The October 14 investigation is recorded in `physical-ai-analysis.md`.
Its proposed fixes are hypotheses, not evidence that two-leg balance is
solved. Start with independently testable corrections:
- ES fitness must include body checks in partial progress, as evaluation
  does. Failed checks already suppress the success bonus; they were not
  completely invisible to ES. Keep the existing search amplitude.
- Tested fixed baseline normalization with physical zero centers and
  minimum input scales, then rejected it: the standard jump measurement
  fell from 20/20 to 18/20, and neither two-leg task succeeded. Restore the
  original trainer and its tests; do not trade an existing working skill
  for a speculative training improvement. Saved playback never changed.
- Keep home-relative keyframe semantics, including empty frames returning
  home. Warn about empty interior frames and omitted body shifts while
  lifting feet; do not infer different targets from neighboring frames.
- Let measured balance failures suggest training, not only request words.
  The tool still reports failure and recommends correcting bad poses first.
- A real-model bow trial repeated a negative pitch while judging a
  positive-pitch goal. Warn when all explicit nonzero body targets oppose
  their body-angle goal, and put that warning in the revision fallback.
  This is advisory: it does not change targets or evaluation. Do not
  interpret a sign error in a sketch as proof of physical infeasibility.
- Defer a forced residual fade, increased ES amplitude and dense reward
  retuning to measured experiments. A fade would change saved controllers
  and could remove needed landing feedback. Static support margin is a
  diagnostic, not a general dynamic-balance success criterion.
Keep evaluation thresholds and randomized held-out checks unchanged.
Verification and measured results belong in `status.md`.

### D33. Local laptop compute and opt-in nonlinear feedback
The user approved proceeding, clarifying that the available GPU is the
laptop's built-in graphics, not a separate training service. WebGPU is
available on Intel gen-12lp in the current browser. Do not equate that with
GPU-accelerated MuJoCo rollouts or an approved remote training environment.
- First bounded browser-local slice: reuse the existing MLP and batched
  ARS implementation to offer a seeded 16-unit hidden layer for keyframe
  feedback. No new dependency, optimizer, action recipe or remote service.
- Keep linear training the default and its normalization/reward unchanged.
  The nonlinear option is experimental and must be judged by measurements.
- Protect a current draft policy by comparing it on the same held-out
  pushes before replacement, not just comparing against bare keyframes.
- Report a sensor ablation on the neural candidate with motion progress
  preserved. Disabling feedback here fixes sensor channels at their
  normalization means; it is not the same as removing the policy.
- Nonlinear policies use the existing versioned observation and bounded
  action contract, artifact validation and morphology checks.
- GPU training, stronger task specifications, larger observation models,
  PPO, and two-leg mastery remain separate work. Do not claim this slice
  implements the full research proposal or solves arbitrary motion tasks.

### D34. Skills-first workspace: the AI Playground (amends M8 tabs, D2 shell)
Approved product direction (2026-10-16): Skills / AI Playground is the
primary experience; Build, Kinematics, Physics, Control lab and Navigate
are secondary engineering tools that stay complete and reachable.
- Navigation: the six equal tabs became a left `nav` of buttons with
  `aria-current="page"`: AI Playground first, then a labeled Lab tools
  group, then Guide and Learn mode. The tabs pattern and its
  `display: contents` tab panel were removed rather than kept partly; the
  work area is a `main` landmark named after the view. Below 1000 px the
  list folds behind a Menu disclosure (not a modal drawer). Rejected: a
  vertical tablist (a group heading inside a tablist is invalid) and an
  icon-only rail (meaning must stay in text labels).
- Startup: the app opens on the AI Playground instead of Kinematics. The
  Skills session, and with it the MuJoCo engine, now loads at startup; the
  engine glue is still a separate dynamic import, Build and Kinematics
  still run without it, and an engine failure is shown on the Playground
  while the navigation keeps working. Nothing moves, trains, evaluates,
  listens or calls a model on its own.
- The internal mode id stays `skills` and its label stays "Skills" in
  messages; only the presentation title is "AI Playground". Renaming the
  "Skills tab" wording in parser, resolver and AI tool texts was deferred:
  those strings are asserted by unit tests and steer model tool use (see
  "Model tool use depends on wording" in the 2026-10-05 probes), so they
  need their own change with tests and probes.
- Playground layout: context header (robot, body facts, morphology hash,
  engine state, side panel controls), a status strip for job progress with
  its Cancel and the latest message, the robot view, and one side panel
  with Teach, Library and Attempts. Two large side columns made the robot
  view small; one panel with a collapse button gives the view most of the
  width. Panel collapse is stored in `ui.preferences`
  (`playgroundPanelCollapsed`), validated like the other preferences and
  written only on toggle.
- Side panel placement (validation pass): the panel is rendered through a
  React portal into a slot after the command bar and placed beside the
  work area and the command bar by the workspace grid. In one column
  (900 px and below) the order is robot view, command field and Stop task,
  then the panel, in the document as on screen, so a long teaching form is
  never scrolled past to reach Stop. Rejected: CSS `order` or a
  `display: contents` main (visual order would differ from Tab order) and
  a fixed or sticky command footer (it would cover panel inputs). The
  panel is a top-level complementary landmark, no longer inside `main`.
- Focus: when a view switch removes the focused control (Back to AI
  Playground, the links in Build), focus moves to the new view's `main`
  (`tabindex="-1"`, not a Tab stop) instead of the page body.
- Teaching reads as Motion, Success criteria, Try, Evaluate and save;
  improvement and training sit in a disclosure, their results stay
  visible. Library records are compact and expandable (one open at a
  time, the newest by default). No session, evaluation or persistence
  logic changed.
- Command bar: field, Run or Send, Stop task, status and a Voice and AI
  disclosure (Speak, Check AI, Read replies, Forget conversation); the
  output scrolls in a bounded height. Stop task now says when nothing was
  playing and that jobs are not affected.
- Dependencies: react-resizable-panels 4.14.5 (MIT), @radix-ui/react-dialog
  1.3.1 and react-dropdown-menu 2.2.1 (MIT) and lucide-react 1.58.0 (ISC)
  were checked in the registry metadata (2026-10-16, nothing
  downloaded). None was adopted: the redesign has no modal or menu
  surface that needs Radix or icons. Drag resizing of the side panel waits for an approved
  react-resizable-panels integration; a home-grown splitter was not built.
- Reduced motion: the stylesheet neutralizes transitions and animations.
  Camera damping is unchanged: a draft that turned OrbitControls damping
  off for `prefers-reduced-motion` was reverted in the validation pass
  because it changed renderer behavior in every view and is not part of
  the shell redesign. It can be proposed separately.
Verification: unit test for the new preference; e2e specs updated and
added for navigation, startup, engine failure, panel collapse, Stop versus
Cancel and leaving the Playground. CI and a browser smoke check passed on
`8e879cc`; the full Playwright suite has no CI job. See `docs/status.md`.

### D35. Graphite and mint visual theme (2026-10-16)
The user accepted the D34 layout but requested a more distinctive theme,
typography and color treatment. Keep that layout and its workflows.
- Graphite surfaces, soft mint primary actions, warm detail accents and
  distinct red Stop styling; no glow, blur or decorative animations.
- Installed Bahnschrift for interface text, tabular monospace for numbers,
  24 px brand and view title, 17 px section headings, 32 px controls.
  No web-font download, package or external asset dependency.
- Current-view markers, underlined side-view choices and filled primary
  actions separate navigation from commands without changing semantics.
- Match the floor, robot materials and plot surfaces to the theme; keep
  camera behavior, simulation, geometry and plotted signals unchanged.
- Validate text contrast, overflow and the real robot rendering on desktop
  and narrow windows against the identified CI artifact. Source checks
  alone do not establish visual quality.

### D36. Navigate drives the chosen wheeled robot (2026-10-16)
The user asked to choose the Navigate robot (presets and their own Build
design) instead of always loading the rover.
- The navigation stack steers by left and right wheel speeds (skid
  steering) with wheel odometry. Its rover constants (four wheel indices,
  0.06 m radius, 0.34 m track, 0.22 x 0.21 m half footprint, 0.32 m planner
  clearance, 0.2 m ray mount) are now derived from the description by
  `src/nav/drive.ts`; for the rover they reproduce those values exactly,
  checked by a unit test, so its tuning and evidence stand.
- Drivable: free-standing, wheel-only bodies, every wheel a single joint
  about the body's left-right axis (either sign), wheels on both sides, one
  wheel radius, wheels below the body and range rays below the 0.3 m
  obstacles. Each refusal names its reason.
- Not drivable, and said so: fixed-base arms; legged robots, because they
  walk only through a trained, morphology-bound walking controller in the
  AI Playground that executes one walk command for a duration and is not
  connected to the planner. Steering a legged robot along a planned path
  would be a separate slice (path following through walk commands, legged
  odometry, its own evaluation), not a relabeled rover controller.
- The picker lists every preset and the Build robot, with undrivable
  entries disabled and the reasons in text. Build gains Drive in Navigate,
  an explicit hand-off like the other views; a Build robot loaded earlier
  stays listed after Build changes it. The choice is not persisted.
- The batch evaluation, AI context (name, wheels, track, ray mount) and
  task body mapping use the chosen robot; the rover keeps its wording.

### D37. Legged robots walk the Navigate planner's path (2026-10-16)
The user asked for the quadruped, the hexapod and legged Build designs to
move in Navigate, not to be refused. This is the separate slice D36 named.
- The stack now outputs a body command (forward speed, turn rate). Wheeled
  bodies turn it into wheel speeds exactly as before; the rover's numbers
  do not change. Legged bodies walk with the gait generator of D25 (trot;
  alternating tripods on six legs): the command sets the stride and turn
  of each 0.5 s cycle. This is a trajectory generator with IMU feedback
  steered by the planner, not a trained walking controller, and it is
  labeled so in the view and the AI context.
- The generator plays a gait of at most 20 s that fades in and out. For an
  open-ended walk its clock is kept in the steady window by whole cycles,
  which leaves every foot's phase unchanged; a stop first slows stride
  and turn to zero, then fades the stepping out. Stride and turn change
  at a bounded rate so planted feet are not dragged. The generator and
  the skills code are unchanged.
- Limits come from the gait sweep (D25 spike, all runs upright): commanded
  speed up to 0.15 m/s (0.075 m per cycle, between the measured 0.06 and
  0.10 m strides) and turn rate up to 0.5 rad/s (0.25 rad per cycle, below
  the measured 0.3). The goal speed limit range is 0.05 to 0.15 m/s; runs
  get 120 s instead of 60 s. Joint control is the evaluation loop: PD
  (kp 40, kd 1) on rate-limited targets, the gait updated at 100 Hz.
- Sensor-only odometry is `LegOdometry` (D29): planted feet by touch
  sensor, forward kinematics of the encoders, IMU heading. In Navigate it
  tracks the point of each foot sphere that touched the ground rather
  than the sphere's center: the quadruped's feet roll as its legs pitch
  through a stance, and the first CI run (95ca128) measured its estimate
  0.23 to 0.27 m short after about 3 m (the hexapod 0.005 m). The walking
  controllers keep the center-based estimate they were trained with.
  Slipping feet still make it drift; nothing corrects it.
- Range rays sit 0.15 m above the floor and 5 cm ahead of the front feet,
  so no ray sees the body's own legs and the rays stay below the 0.3 m
  obstacles (the quadruped's body center stands 0.27 m high). A walking
  body pitches; a hit that the IMU attitude places on the floor clears the
  cells up to it instead of marking an obstacle. Wheeled bodies do not use
  this filter.
- Accepted: free-standing bodies without wheels, with at least four legs
  of at least two joints, feet level within 2 cm in the stance and below
  every other part, and a gait the generator accepts. Two-legged bodies
  are refused (a trot leaves them on one foot). Footprint: the stance
  extents plus 4 cm for the steps and the 2 cm margin. The planner
  clearance (half the footprint diagonal) is 0.43 m for the quadruped,
  whose rear knees reach 0.29 m behind its center, and 0.45 m for the
  hexapod's splayed legs, so the 0.8 m gap is closed to both; that is
  reported as no path, not hidden.
- Stack fix found on that map: after a failed replan the stack kept
  following the stale path until the next replan, so a blocked walker in
  sensor-only mode kept moving and collided (spike on c45f47f, both legged
  bodies). It now stands still until a replan succeeds, and the no-path
  rule ends the run. This applies to wheeled bodies too.
- The referee adds a fall outcome: tilt above 0.8 rad or the base below
  half its standing height, from simulator state. The live view stops
  and asks for a reset, as after a collision.
- Not done: a trained walking controller (D29) steering the path, side
  steps, speed calibration per body, or correcting odometry drift.

### D38. Learned walking and the AI in Navigate (2026-10-16)
The user asked to "make it all AI": legged robots in Navigate should walk
with what was learned, and the AI should run it.
- A walking controller trained in the AI Playground (D28, D29) is stored
  per morphology; Navigate looks it up for its robot and, when it fits,
  walks with it: its calibration turns the planner's speed and turn rate
  into the generator's stride and turn per cycle, and its feedback policy
  corrects them (and every joint, when it has joint outputs) from the
  IMU and its own leg odometry against where the commands lead. The
  observation is the one the policy was trained on (the function is
  shared with `WalkPolicyController`). Without one, the gait generator
  walks as in D37, and the view and the AI context say which walks.
- The policy was trained on constant commands over walks of at most 10 s;
  here the command changes with the path and the walk is open-ended. The
  reference it reads is the integral of the commands, held within 0.1 m
  and 0.3 rad of its estimate so its inputs stay in the range it was
  trained on. It restarts from zero at every start from standing, like
  the walks it was trained on. A controller that arrives while the body
  walks takes over the next time it stands. This transfer is new and is
  measured, not assumed: the locomotion spikes walk two Navigate maps
  with the controller they train.
- Training from Navigate: a button, and trainLocomotion for the AI, train
  the controller of the Navigate robot in the AI Playground's worker pool
  without switching the Playground's robot (the Playground refuses its
  own jobs meanwhile). It uses the gains Navigate walks with (kp 40,
  kd 1) and is saved for every view.
- The AI in Navigate: selectRobot switches the Navigate robot (rover,
  quadruped, hexapod or the Build robot; arms are refused with the
  reason) and trainLocomotion trains a legged robot's walking controller.
  navigateTo results say whether the learned controller or the generator
  walked. The tool names are unchanged, so the proxies need no change.
- Seen in the browser: the model answered "learn to walk" with "the
  robot is learning to walk" and no tool call. The Navigate context now
  says that nothing is trained without trainLocomotion, and a legged body
  without a controller gets one reminder to call it (like the Skills
  tab's reminder for motion requests answered without an action).
- Measured: with its calibration the hexapod walks at the commanded
  speed instead of about 1.7 times faster, so it takes longer to reach a
  goal than with the plain generator; that is the calibration working.
- Not done: a controller trained in the AI Playground with that view's
  live joint gains, when they differ from kp 40, kd 1, walks in Navigate
  with kp 40, kd 1 without being calibrated again.

### D39. Audit fixes and on-demand code loading (2026-10-20)
- Viewport cleanup releases the WebGL context explicitly. Navigation
  unmount stops both autonomous and remote runs; switching views clears
  old command output, and a late answer cannot overwrite the next view.
- Fixed arms stop at their measured joint pose. Idle Stop reports that
  no activity was running and distinguishes background jobs from motion.
- Kinematics capabilities depend on actual targetable limbs. A wheeled
  body with an arm still has IK; a wheel-only body has joint controls but
  no target or displayed end-effector position.
- Recorded Control lab outputs retain their original settings. Edits to
  gains or Compare B make them stale; hidden B outputs are not displayed.
- Secondary UI components and AI-only code load on demand with a loading
  state inside the existing view error boundary. Session ownership,
  explicit robot hand-offs, the primary Playground and local Stop remain.
- The original startup script budget remains 1,200,000 decoded bytes.
  Code splitting improves it but does not meet it; status.md records this
  as open rather than relaxing the test to accept the larger bundle.