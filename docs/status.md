# MorphLab status

This is the only document that states what is implemented. Every entry is
labeled **Implemented and verified** (tests or browser checks were run and
observed), **Implemented but unverified**, **Planned** or **Failed**.

Commit hashes quoted in the other documents (for example in
[decisions.md](decisions.md)) refer to the original development history,
which is not part of this repository.

## Verification of this source

| Check | Result |
|---|---|
| Type check (app and proxy) and production build | Passed in CI on this source |
| Unit and integration tests (`npm test`, Vitest, headless MuJoCo) | 556 tests in 59 files passed in CI on this source |
| Feasibility spikes (`npm run spikes`) | 24 tests in 17 files passed in CI on this source |
| Browser walkthrough of the build of this source | Done on 2026-10-28 in a Chromium-based browser at 1440 x 900, served by `server/serve.ps1` with a real language model; the screenshots and recordings in `docs/media` come from that session |
| Playwright end-to-end suites (`npm run test:e2e`) | Implemented but unverified on this source (no CI browser job) |
| `.github/workflows/ci.yml` | Implemented but unverified: added for this repository, not yet run |

## Summary

| Area | State |
|---|---|
| M1 Kinematics playground | Implemented and verified |
| M2 Physical body (MuJoCo WASM, PD control, push tool) | Implemented and verified |
| M3 Robot builder (generators, joint editor, morphology versions) | Implemented and verified |
| M4 Control lab (step response, gravity compensation, challenges) | Implemented and verified |
| M5 Skills (state machine, tuned, replay, imitation, evolution strategies) | Implemented and verified |
| M6 Persistence and language (IndexedDB, bundles, rule-based parser) | Implemented and verified |
| M7 Sensors and navigation (sensor-only mapping, A*, pure pursuit) | Implemented and verified |
| M8 Polish (guide, learn mode, attempt comparison, keyboard paths) | Implemented and verified (no screen reader test) |
| AI tools, task manager and local proxy (Node.js and PowerShell) | Implemented and verified with a real model |
| Taught skills (keyframes and goals) and AI-composed skills | Implemented and verified with a real model |
| Learned walking controller per body and the `locomote` tool | Implemented and verified with a real model |
| Legged robots in Navigate | Implemented and verified |
| Voice (spoken replies, browser speech recognition) | Implemented; recognition of real speech unverified |
| Streaming model replies, realtime voice | Planned, deferred |

## Measured in the browser walkthrough (2026-10-28)

All numbers below were read from the app while capturing the media in
this repository. Evaluations are 20 randomized trials (friction 0.6-1.2,
body mass within 12.5 %, joint offsets up to 0.02 rad) in a Web Worker.

### Learning and skills

| Skill or job | Result |
|---|---|
| Foot lift, state machine (hand-designed) | 20 / 20 trials (Wilson 95 % interval 84-100 %), 7.4 s |
| Foot lift, state machine with tuned parameters | 20 / 20 trials, 3.4 s |
| Foot lift, imitation learning (64x64 MLP, behavior cloning) | Trained on 7332 expert samples in 7.5 s (loss 1.072 to 2.13e-4); 20 / 20 trials |
| Foot lift, evolution strategies on the tuned parameters | 60 trials in 9.3 s; held-out reward 2.750 before and after (no improvement, reported as such) |
| "crouch", taught with one keyframe and three goals | Preview met 3 of 3 goals; evaluation 20 / 20 trials in 4.8 s |
| Walking controller for the quadruped | 2722 simulated runs on 4 workers, 263.7 s; 40 / 40 randomized commands within tolerance (95 % interval 91-100 %), mean position error 1.3 cm; the gait generator without the learned calibration: 2 / 40, 17.7 cm |
| Command "lift the front left leg and hold for three seconds" (local rules, no AI) | Hold measured at 3.37 s; the reply notes that the skill was evaluated with a 2 s hold |
| Command "walk forward 40 centimeters" (AI) | One `locomote` call, two model calls; measured 0.414 m forward, position error 0.014 m (tolerance 0.100 m) |

### Lab tools

| Tool | Result |
|---|---|
| Kinematics, six-joint arm | Target (0.35, 0.20, 0.40) m converged, residual 0.001 mm; a target 1.3 m high reported Unreachable, closest point 307.3 mm away after 50 iterations |
| Physics, quadruped | Recovered from 3 N*s pushes along +Y, -X and -Y; a 6 N*s push tips it over |
| Control lab, three-joint arm shoulder, 20 degree step, kp 60, kd 3 | PD alone: overshoot 27.0 %, not settled, steady-state error -2.450 deg. With gravity compensation: overshoot 12.4 %, settling time 0.308 s, steady-state error 0.000 deg. Hold-a-pose challenge: A failed (0.0314 rad), B passed |
| Navigate, rover on the slalom map, sensor-only | Reached the goal 18 cm from it (ground truth) after 32 replans |

## Known limitations

- Simulation only. Simulated sensors are idealized; nothing here is
  claimed to transfer to hardware.
- Taught skills are open-loop keyframe motions unless a learned feedback
  policy is trained for them; balancing on two legs is not solved.
- Evaluation randomizes friction, body mass and joint offsets only;
  terrain, latency and sensor noise are not randomized.
- IK is position-only. The reach bound is conservative, so some
  out-of-reach targets are reported as stalled rather than unreachable.
- Odometry is not corrected by the map (no localization or loop closure).
  Obstacles are static boxes inside a 4.2 m arena; rays are a fan from the
  body centre, so low obstacles are not seen.
- The supported language model provider is an Azure OpenAI chat
  completions deployment. Images are never sent to the model. Replies are
  not streamed.
- The model's instruction following varies; explicit restrictions such as
  "do not train" are enforced by guards for common English wording only
  (see [ai-command-testing.md](ai-command-testing.md)).
- Attempts are kept in memory (last six) and lost on reload. Gains are
  shared by every joint; there are no per-joint gains or integral action.
- The MuJoCo glue needs `'unsafe-eval'` in a Content Security Policy.
- First-load JavaScript is 1,234,358 bytes, over the 1,200,000-byte
  budget in `tests/e2e/perf.spec.ts`.
- Changing the camera view needs a mouse or touch; every other action has
  a keyboard path. No screen reader was tested.
- Driving keys use physical key positions (`KeyboardEvent.code`).
