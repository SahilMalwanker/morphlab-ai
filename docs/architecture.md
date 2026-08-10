# MorphLab architecture

Status labels in this document describe design intent. Only
`docs/status.md` states what is implemented.

## 1. Layering

```
 src/ui        React panels, inspector, timeline, command bar   (React only)
 src/render    Three.js scene, meshes, overlays                 (Three.js only)
 ---------------------------------------------------------------- boundary
 src/core      math, robot model, kinematics, control           (pure TS)
 src/physics   MJCF generation (pure) + MuJoCo runtime adapter  (TS + WASM)
 src/skills    skill definitions, state machines, evaluation    (pure TS)
 src/learning  MLP, evolution strategies, imitation, workers    (pure TS)
 src/language  rule-based parser, resolver, LLM adapter iface   (pure TS)
 src/ai        tool schemas, task manager, proxy protocol      (pure TS)
 src/nav       occupancy grid, A*, odometry, path follower      (pure TS)
 src/persistence IndexedDB store, bundle import/export          (TS + IDB)
 ---------------------------------------------------------------- process boundary
 server        local AI proxy, the only reader of .env          (Node.js)
```

Rules:
- Nothing below the boundary imports React or Three.js.
- `server/` imports only `src/ai` from the app; nothing in `src` imports
  `server/`, and no browser code reads the key or the endpoint.
- `src/render` consumes plain data (`Float64Array` transforms, link
  descriptors) and never computes robotics quantities itself.
- `src/ui` owns user interaction and wires modules together through small
  controller objects; it does not contain numerical algorithms.
- Every module boundary validates inputs and rejects non-finite numbers.

## 2. Units, frames and transforms

| Quantity | Unit |
|---|---|
| length | m |
| mass | kg |
| time | s |
| angle | rad (UI may display degrees, converted at the UI boundary) |
| force / torque | N / N*m |
| linear / angular velocity | m/s / rad/s |

Frames:
- World `W`: right-handed, Z up, gravity `(0, 0, -9.81)`. Identical to the
  MuJoCo world frame. The renderer uses the same frame (camera `up = +Z`,
  grid in the XY plane); there is no axis conversion anywhere.
- Base `B`: attached to the robot body. For fixed-base robots `T_W_B` is
  constant; for floating-base robots it comes from the simulator or, in
  kinematics mode, is held fixed by the user.
- Joint frame `J_i`: located by a fixed origin transform relative to the
  parent link frame. The joint axis is a unit vector expressed in `J_i`.
- Link frame `L_i`: the joint frame after joint motion. Link geometry is
  expressed in `L_i`.
- Tip frame `E`: a fixed offset from the last link frame of a limb.

Transform convention:
- 4x4 homogeneous matrices, column-vector convention: `p_a = T_a_b * p_b`.
- Stored column-major in `Float64Array(16)` (same memory layout as
  `THREE.Matrix4.elements`, so render code copies without transposition).
- Chains compose left to right:
  `T_W_Li = T_W_B * T_B_J1 * R1(q1) * T_L1_J2 * R2(q2) * ... `
  where `T_Lk_Jk+1` is the fixed origin transform and `Rk(qk)` is the joint
  motion (rotation about the axis for revolute, translation along the axis
  for prismatic).
- Origin rotations are given as roll-pitch-yaw `(r, p, y)` meaning
  `Rz(y) * Ry(p) * Rx(r)` (extrinsic X, then Y, then Z), as in URDF.
- Quaternions inside `src/core` are `[w, x, y, z]` (MuJoCo order). Three.js
  uses `(x, y, z, w)`; conversion happens only in `src/render`.

## 3. Robot description schema (v1)

```ts
interface RobotDescription {
  schemaVersion: 1;
  id: string;                 // stable identity across versions
  name: string;
  base: {
    kind: 'fixed' | 'floating';
    size: [number, number, number];   // box full extents, m
    mass: number;                      // kg
    pose: { xyz: Vec3; rpy: Vec3 };    // initial T_W_B
  };
  limbs: LimbDescription[];
  stance?: Record<string, number>;     // home joint positions by joint id (not morphology)
}
interface LimbDescription {
  id: string;                 // e.g. 'front_left'
  name: string;               // display name, e.g. 'Front left leg'
  role: 'arm' | 'leg' | 'wheel';
  joints: JointDescription[]; // serial chain, base side first
  tip: { xyz: Vec3; radius: number }; // tip offset in last link frame
  wheel?: { radius: number; width: number }; // wheel limbs only
}
interface JointDescription {
  id: string;
  name: string;
  type: 'revolute' | 'prismatic' | 'continuous';
  origin: { xyz: Vec3; rpy: Vec3 };   // parent link (or base) -> joint frame
  axis: Vec3;                          // unit vector in joint frame
  limit: { lower: number; upper: number; effort: number; velocity: number };
  link: { radius: number; mass: number }; // capsule from this joint to the next joint (or tip)
  damping: number; armature: number;
}
```

Derived data:
- `morphologyHash`: SHA-256 of the canonical JSON of every field that
  affects kinematics or dynamics (structure, ids, origins, axes, limits,
  masses, radii, damping, armature, base size, mass and kind). Names, the
  initial base pose and the stance are excluded.
- A morphology version is identified by `(id, morphologyHash)`. Any edit
  that changes the hash creates a new version; skills reference the hash.
- Validation (`validateRobot`) also enforces sanity bounds: offsets within
  2 m, base sides up to 3 m, masses up to 200 kg, radii up to 0.25 m,
  effort up to 1000, velocity up to 100, at most 12 limbs and 8 joints per
  limb, and stance values inside joint limits.

Builder operations (`src/core/model/builder.ts`) are pure functions that
return a new description: legged and arm generators, joint patches, joint
count changes, limb removal and mirroring. Mirroring reflects across the
base XZ plane: points map to `(x, -y, z)`, origin roll-pitch-yaw to
`(-roll, pitch, -yaw)`, revolute axes (pseudovectors) to `(-ax, ay, -az)`
and prismatic axes to `(ax, -ay, az)`, so equal joint angles give mirrored
poses. The builder session validates every candidate before applying it.

## 4. Kinematics core (`src/core/kinematics`)
- `forwardKinematics(model, q, T_W_B)` returns per-link world transforms
  and tip transforms. Pure, allocation-light, deterministic.
- `positionJacobian(model, limbIndex, q)` returns the 3xN geometric
  Jacobian of the tip position in world coordinates.
- `solveIK(...)` implements damped least squares with adaptive damping,
  step clamping, joint-limit clamping and active-set removal of joints held
  at limits. It returns `status` (`converged`, `unreachable`, `stalled`,
  `max-iterations`, `invalid-input`), residual, iterations and the final
  joint vector, which is always inside limits. `unreachable` is reported
  only when the target lies beyond the limb's reach bound (a proof);
  `stalled` means the solver stopped improving without such a proof.
- `manipulability(J)` returns singular values and principal directions of
  `J J^T` for the ellipsoid overlay.
- Reachability: a sampled workspace cloud and a radial reach bound.

## 5. Physics (`src/physics`)
- `generateMjcf(description, options)` is a pure function that converts the
  robot description into an MJCF string (bodies, joints, capsule geoms, tip
  spheres, motor actuators with effort limits, touch and range sensors,
  IMU site). It is unit tested without WASM.
- `MujocoRuntime` owns the WASM module, `MjModel` and `MjData`, steps the
  simulation and exposes typed state snapshots. It frees every embind object
  with `.delete()` on rebuild and disposal.
- Control is computed in TypeScript (`src/core/control`) at the physics rate
  and written to `data.ctrl` as joint torque. Requested joint targets pass
  through `TargetRateLimiter`, so the commanded target moves no faster than
  each joint's velocity limit (continuous joints the shorter way); the PD
  law wraps continuous-joint errors and damps measured velocity.
  Saturation happens in our code and is reported, then MuJoCo clamps again
  with `ctrlrange` as a safety net. Joint quantities are named per joint
  type (`src/core/model/units.ts`).
- Collision policy of generated models: contacts between the base and
  every limb link are excluded, MuJoCo filters parent-child pairs, and all
  other pairs collide (links of different limbs, non-adjacent links of one
  limb, links with the floor and obstacles). A leg swung into the body
  passes through it (`tests/physics/collisions.test.ts`).
- Interactive simulation runs on the main thread when a full frame of
  physics costs under 4 ms (measured in the physics spike); batch trials,
  training and the navigation map batch run in Web Workers with their own
  MuJoCo instance, through one job protocol (`JobWorkerClient`).

## 6. Data flow per frame (simulation mode)

```
user input -> UI controller -> requested targets
  for each physics substep (MujocoRuntime.controlStep):
    mj_step1(model, data)                                   [state, position/velocity sensors]
    measured state  <- sensors (encoders, IMU, contacts)   [sensor-only]
    q_cmd           = rate-limit(requested, velocity limits)
    tau_cmd         = PD(q_cmd, q_meas, qd_meas)            [src/core/control]
    tau_applied     = saturate(tau_cmd, effort limits)
    mj_step2(model, data)                                   [actuation, integration]
    run observer    <- referee samples (ground truth)       [playback only]
  snapshot          -> render (link transforms from data.xpos/xquat)
                    -> plots (commanded vs measured vs applied)
```

Encoder and gyro readings are current at the control step; touch and
accelerometer readings need contact forces and lag one step.

Ground-truth quantities (`data.xpos`, contact list) are available only in
the explicitly labeled ground-truth debug view and in evaluation metrics,
never as controller inputs in sensor-only mode.

## 7. Skills, learning and language
- A skill is a typed controller (`state-machine`, `replay`,
  `tuned-parameters`, `imitation-mlp`, `es-improved`, `keyframes`) with parameters or
  weights, observation and action definitions, and evaluation results.
- Two kinds of skill (D23): the built-in foot lift (`skillType:
  'foot_lift'`, five methods, judged by `FootLiftReferee`), and skills the
  user teaches (`skillType: 'taught'`, method `keyframes`): poses captured
  from the live robot's joint targets, played open loop as a cosine blend
  from the home stance through each keyframe, and judged by the goals the
  user set (`GoalJudge` in `src/skills/taught.ts`, the same judge in
  evaluation trials and live playback). Goals are measured from ground
  truth: tilt, lowest body height, a limb tip above a height for a time,
  body below or above a height for a time, all feet off the ground, rise
  above the resting height, travel along the start heading, turn (unwrapped
  yaw), and standing at the end. A fixed base takes limb-tip goals only.
- A taught skill runs by its name: the command bar matches the name before
  any other rule (optionally with words such as "please" or "now") and the
  AI calls it through executeSkill; both run the best evaluated version for
  the exact morphology and report every goal as measured.
- The AI composes taught skills too (D24): `src/skills/compose.ts` turns
  poses described in terms of the body (move the body with the feet kept
  where they stand, move limb tips from where they rest, set joint
  targets) into keyframes with damped least-squares IK inside the joint
  limits, and `describeBody` gives the model the body in numbers. The
  draft goes to the Teach panel (never over the user's unsaved draft), is
  tried once live, and is saved after a 20-trial evaluation in which at
  least one trial met every goal; the skill records `taught.author`.
- Training and multi-trial evaluation run in a Web Worker that builds its
  own MuJoCo model from the same robot description.
- The language layer maps text to `SkillCall` JSON
  (`{ skillType, limbId?, durationS?, ... }`) or to a refusal with a
  suggestion. A resolver checks the call against the robot (limb names) and
  the skill library (type, morphology hash, evaluated limbs) before any
  execution. Language never emits torques or joint targets directly.
- The resolver considers only skills whose evaluation is consistent and
  was measured here (`evaluationOrigin: 'local'`) with at least one
  success, and ranks them by the lower end of the Wilson 95 % interval,
  then success rate, newest first. A requested hold different from the
  evaluated one is applied only to parameterized methods (state machine,
  tuned parameters, ES-improved) within 0.5-5 s; replay and imitation
  skills have fixed timing and refuse. Playback is judged by the
  evaluation's referee (`FootLiftReferee`) with the requested hold; the
  reply says "Lifted" only when it passed and otherwise reports the
  failure, interruption or fault with the measured numbers, plus any
  difference between the live and the evaluated setup.
- Commands act on the active tab: stand and reset on that tab's live
  simulation (Physics, Skills, Navigate) or refused; a foot lift outside
  the Skills tab only when that tab shows the Skills robot's morphology.
- Requests the local rules do not cover (not recognized, navigation,
  motions nobody taught such as walking, jumping, waving or sitting, and a
  foot lift no evaluated skill serves in the Skills tab) go to the AI path
  through the local proxy when it is available (section 7b); torque
  commands and lifting several legs stay local refusals.
  The page never handles a provider endpoint or key; the browser-side cloud
  adapter of M6 was removed.

## 7a. Navigation (`src/nav`, `src/physics/navigation.ts`)
- Rover: four-wheel skid steer. Wheel speed control (0.6 N*m per rad/s,
  saturated at the effort limit) at 500 Hz; the navigation controller runs
  at 50 Hz.
- Sensors: wheel encoders, IMU, touch readings and nine MuJoCo
  rangefinders from -80 to 80 deg (2 m range, tilted up 0.035 rad so a
  pitching body does not see the floor).
- Inputs: in `sensor-only` mode the controller receives `{ t, sensors }`
  only (tested); `ground-truth` mode adds `groundTruth.pose` and the true
  obstacle map, labeled as a reference.
- Stack: wheel-encoder plus IMU-heading odometry; log-odds occupancy grid
  (0.05 m cells over a 4.2 m square arena) updated along each ray
  (Bresenham); a distance transform gives hard inflation at 0.32 m and a
  soft proximity cost up to 0.75 m; 8-connected A* with an octile heuristic
  (optimal for non-negative cell costs); replanning every 0.5 s; pure
  pursuit with a 0.3 m lookahead and 0.4 m/s maximum speed.
- Referee (ground truth, never a controller input): collision when the
  rover footprint rectangle overlaps an obstacle box (separating axis
  test); success when the stack reports arrival and the true distance to
  the goal is below 0.25 m; failure after 2 s without a path or at the
  60 s limit.
- The Navigate view draws rays, map and path from the controller's data and
  the rover from simulator state; the inset is a virtual camera rendered
  from the scene for display only (not a controller input).
- The live view and the batch evaluation share the referee
  (`src/nav/referee.ts`). Both navigation stacks are created only at an
  explicit reset and observe every control update, so remote driving, goal
  changes and mode switches never re-initialize the sensor-only estimate.
  The five-map batch runs in a Web Worker with the mode fixed at its start.
  Remote driving uses the keyboard only (`src/ui/nav/keys.ts` decides which
  key events drive) and stops on window blur and page hide.

## 7b. AI layer (`src/ai`, `server/`)
Runtime tiers: Tier A runs in the browser only and works offline (builder,
kinematics, physics, control, sensors, navigation, local command parser,
skills, persistence). Tier B adds language model reasoning through the
local proxy. Tier C (training large models) stays outside the app.

```
 browser                                              local proxy (server/)
 command -> local parser --------------------------> tools
        \-> AI loop -- POST /api/ai/chat ----------> guards: host, origin, session, rate
                                                      strict ChatRequest schema, tool names
                                                      Azure OpenAI adapter --> provider
 tool call -> validateToolCall -> TaskManager -> executor -> {taskId, status, reason, measurements, fallback}
```

- `src/ai/result.ts`: the result every tool returns; `rejected` means
  nothing ran.
- `src/ai/capabilities.ts`: `RobotCapabilities`, derived from the current
  robot and scene; the tools offered to a model follow from it.
- `src/ai/tools.ts`: the twenty-eight tool schemas (Zod), JSON Schema for the
  model, and `validateToolCall` (known tool, available on this robot,
  bounded arguments, saved skill names, remembered place names). Limits
  shared with `src/skills` (goal bounds, pose limits, gait bounds and
  patterns, skill names) are copied, because the proxy imports `src/ai`
  alone; a test compares them.
- `src/ai/tasks.ts`: `TaskManager` (one active task, state versions, task
  IDs, the per-body safe stop).
- `src/ai/protocol.ts`: the browser-proxy contract, shared by both sides.
- `server/config.ts` reads `.env` (API_KEY, ENDPOINT) and detects the
  provider; `server/azure.ts` maps requests, responses and errors;
  `server/app.ts` holds the routes and guards; `server/main.ts` starts the
  proxy on 127.0.0.1:8787 and exits with a message naming the variable
  when the configuration is unusable.
- `server/serve.ps1` is the same contract for machines without Node.js
  (PowerShell 7): it serves the CI-built `dist/` and the AI routes on
  127.0.0.1:4173 (D15).
- Routes: `GET /api/ai/status` (provider and capabilities, never secrets)
  and `POST /api/ai/chat`. Realtime routes do not exist yet. Chat
  requests carry text only: both proxies refuse images (D31).
- `src/ai/client.ts` talks to the proxy (status, chat with retries and
  cancel); `src/ai/loop.ts` runs one turn of model calls (at most 10, one
  action per answer), validation and measured results, refreshes the
  capabilities after each action, keeps each request within the proxy
  limits by dropping the oldest remembered exchanges, and rewrites the
  reply as plain text (`plainReply`);
  `src/ai/controller.ts` holds the status, the system prompt,
  conversation memory and the turn, and passes actions to the task
  manager.
- `src/ui/ai/bridge.ts` connects the tools to the active tab's sessions:
  Navigate (navigateTo, navigateToPlace on the navigation stack;
  rememberPlace and forgetPlace on world memory; range readings and the
  front camera), Skills (executeSkill on the live playback; draftSkill,
  improveSkillDraft and evaluateSkillDraft compose, try, improve, evaluate
  and save skills, D24 and D25; runSkillSequence; selectRobot), the
  Control lab (runControllerExperiment: the step test as set, or with
  gains tried on gain set B; the experiment coach, D22), Physics
  (pushRobot, measured from the session telemetry; resetSimulation),
  Kinematics (moveTipTo, setJointAngles; geometry only) and Build
  (buildRobot through the builder's validated generators).
  `src/ui/ai/outcome.ts` turns a turn report into the command bar's
  text, with the status taken from measured results.
- Skills pipeline (D25): `src/skills/intent.ts` checks a draft's goals
  against the words of its name or request; `src/skills/gait.ts` turns a
  contact pattern into foot targets and joint targets with IMU feedback;
  `src/skills/improve.ts` runs evolution strategies on a draft against
  its goal reward in the worker (job `improve-taught`) and keeps the
  result only when held-out trials score better. `src/skills/mirror.ts`
  (D26) makes the left-right mirror image of a taught skill after checking
  the body's symmetry with forward kinematics; the `mirrorSkill` tool and
  the Teach panel load it as a draft to try and evaluate.
- Learned feedback policy in a gait's joints (D27):
  `src/skills/policyGait.ts` (`LearnedGaitController`: the gait's joint
  targets plus a linear policy's bounded correction at 100 Hz from IMU and
  phase inputs), `src/skills/train.ts` (randomized rollouts, the return,
  training with `src/learning/ars.ts`, the held-out comparison and keep
  rule; worker job `train-gait-policy`). Skills of method `gait-learned`
  carry the policy; the `trainGaitPolicy` tool and the Teach panel start
  the training. The policy reads sensors only; the return uses ground
  truth.
- Walking controller per body (D28): `src/skills/locomotion.ts` (the
  calibration from speed asked for to generator command, fitted by least
  squares on measured runs and inverted per command; `trainLocomotion`
  with batched ARS over random commands, the held-out keep guard and the
  evaluation against the generator with and without the calibration;
  `walkForTarget`, a target endpoint to constant speeds and a duration;
  the stored `LocomotionController`, one per morphology) and
  `src/skills/rollouts.ts` (`RolloutPool`: batches of rollouts split over
  up to four skill workers, worker job `rollouts`; results do not depend
  on the number of workers). The Walk panel in the Skills tab and the
  `trainLocomotion` and `locomote` tools use it. D29 adds
  `src/skills/odometry.ts` (`LegOdometry`, a sensor-only estimate of
  position and heading from the IMU and the feet on the ground) and
  `src/skills/walkPolicy.ts` (`WalkPolicyController`: Observation v2 and
  the policy's changes to the generator's stride, side step and turn).
  Walks and saved gait skills start where the body stands when it is up;
  in one model answer, locomote and executeSkill calls run in order while
  each succeeds (`CHAINABLE_ACTIONS` in `src/ai/loop.ts`).
- Memories (D16, D21): conversation memory in `AiController` (last four
  exchanges per tab, robot and scene, each reply followed by a note on the
  tool results behind it; page only); `src/ai/memory.ts`
  `LongTermMemory` holds world memory (named places per Navigate map, with
  the source of each position) and skill memory (live run counts and the
  last run per skill, from every playback). Both are stored in IndexedDB,
  shown in the Navigate tab and on skill cards, and can be deleted there.
- Live changes (D17): while a command runs, `src/language/live.ts` parses
  stop, slower and faster locally and `src/ui/ai/live.ts` applies them to
  the active task (`TaskManager.modify`) or to a goal run in the Navigate
  tab; the rover's change is its commanded speed limit.
- A model selects tools; it never produces joint targets or torques.

## 8. Persistence
- IndexedDB database `morphlab` (version 2) with object stores `robots`
  (key `[id, morphologyHash]`), `skills` (key `id`), `settings` and
  `memories` (key `id`; places and skill run records).
- Every load validates each record; records that fail are reported with
  reasons and kept until the user removes them. The settings store refuses
  credential-like keys. The workspace restores the last builder and skills
  robots from settings; the restore runs under a timeout and a try/catch,
  pointers must be `[id, 64-hex hash]`, and problems are reported as
  notices while the app starts with the default robots.
- Walking controllers (D28) are stored in `settings` under
  `locomotion.<morphology hash>`, validated on load and restored with the
  workspace.
- Bundles (`morphlab.bundle` JSON) export and import robots and skills with
  schema validation and version checks. Export is canonical (sorted keys,
  two-space indent), so equal content gives identical bytes. Import reads
  untrusted JSON through `parseRobot` and `parseSkillArtifact`, which copy
  only known fields and report wrong shapes; robots then pass
  `validateRobot`. Files over 5 MB are refused before parsing. The Build
  tab imports robots; the Skills tab imports skills (marked
  `evaluationOrigin: 'imported'`, their evaluation came from the file) and
  stores the robots they were evaluated on. Skills export with every robot
  their skills were evaluated on.

## 9. Resource management
- Every WASM object is deleted exactly once (`try/finally` on rebuild).
- Three.js geometries, materials and render targets are disposed when a
  robot is rebuilt or the viewport unmounts; the renderer itself is disposed
  on unmount.
- Workers are terminated when a run is cancelled and when their owning
  session is disposed. Sessions live as long as the workspace (switching
  tabs keeps them); the workspace disposes them on unmount, deferred one
  task so React StrictMode's development remount does not dispose live
  sessions.
- A run in progress belongs to the setup it started with: skill jobs are
  bound to the robot and leg (a result that arrives after either changed
  is discarded), Control lab experiments are cancelled when the setup
  changes, and live playback settles once on success, failure, reset,
  robot change, controller error or instability.

## 10. Module structure (created only when a milestone needs it)

```
src/core/math        vec3, mat4 transforms, quaternions, small linear algebra, seeded RNG
src/core/model       robot description types, validation, presets, morphology hash
src/core/kinematics  FK, Jacobian, DLS IK, manipulability, workspace sampling
src/core/control     PD controller, saturation, step-response metrics, telemetry
src/core/dynamics    gravity torques from the description's mass model
src/physics          MJCF generator, MuJoCo runtime adapter, experiments (step test, challenges), rover navigation runs
src/skills           foot-lift state machine, replay, taught skills (keyframes and goals), evaluation harness, artifacts, worker client
src/learning         MLP, evolution strategies, imitation training, worker
src/language         tokenizer, rule-based parser, resolver, local interpreter
src/ai               tool schemas and validation, task manager, proxy protocol
src/nav              occupancy grid, A*, odometry, pure pursuit, test maps, navigation stack
src/persistence      IndexedDB store, bundle import/export
src/render           Three.js viewport, robot meshes, overlays
src/ui               React application shell, modes, guide, learn mode, attempt comparison
server               local AI proxy (Fastify): configuration, provider adapter, routes; serve.ps1: the same
                     routes plus the built app, in PowerShell 7 for machines without Node.js
scripts              static file server (serve:dist), dev launcher (proxy + Vite), leak scan
spikes/              feasibility spikes (Phase 1), not part of the product
tools/oracle         ModernRobotics oracle fixture generator (Python)
```
