# MorphLab milestones

Each milestone must fully work, be tested and be verified in a browser
before the next one starts. Acceptance criteria are measurable; results are
recorded in `docs/status.md`.

## Phase 0: research and decisions
Deliverables: vision, architecture, milestones, decisions with
citations, status and README.

## Phase 1: feasibility spikes (gated)
Spikes live in `spikes/`. Reusable modules they exercise live in `src/` and
are intended to be kept; spike harness code is not part of the product.

| # | Spike | Pass criterion |
|---|---|---|
| S1 | Physics: 12-joint quadruped in MuJoCo WASM, rendered with Three.js, PD stance, lateral push | Stands 5 s with base height drift < 2 cm and tilt < 0.1 rad; survives a 2 N*s lateral impulse (base height > 60 % of stance height, tilt < 0.5 rad, settles within 2 s); median step time < 0.25 ms at dt = 2 ms (headless); renders in a browser |
| S2 | Dynamic morphology: MJCF generated from JSON with N limbs x M joints, rebuilt live | At least 6 configurations including odd ones compile with expected `nq`, `nv`, `nu`, simulate 2 s without NaN; 50 rebuilds without WASM heap growth beyond 5 % |
| S3 | IK: DLS on a generated limb with limits | 200 FK-generated targets: >= 98 % converge with residual < 1e-4 m; joints always inside limits; unreachable and near-singular targets return finite results with a correct status |
| S4 | Learning: tiny policy for 2-link reaching, trained in the browser, saved, reloaded, evaluated | Trained policy success rate (final error < 2 cm) >= 80 % on 100 held-out targets vs < 10 % for the untrained policy; reloaded policy reproduces the same evaluation exactly; training < 60 s in the browser |
| S5 | Skill: hand-designed state-machine single-foot lift for the quadruped | 20 trials with friction in [0.6, 1.2] and base mass in [3.5, 4.5] kg; success rate reported; gate passes at >= 80 % |
| S6 | Language: rule-based mapping of example prompts | At least 10 prompts, including unsupported ones, map to the expected structured skill call or to a refusal with a suggestion (100 % of the listed cases) |

Gate: a failing spike is diagnosed and retried with reasonable fixes. If it
still fails, the plan changes in `docs/decisions.md`. If S1 or S2 cannot pass
with either MuJoCo or Rapier, work stops and is reported.

## Phase 2: milestones

### M1 Kinematics playground
Configurable serial chain (default 3-joint arm), FK, DLS IK with limits,
draggable target, frame and axis overlays, reachability feedback.
Acceptance:
- FK agrees with ModernRobotics oracle fixtures (PoE, Python library) to
  1e-9 for all fixture configurations, and with closed-form 2R/3R solutions
  to 1e-12.
- Position Jacobian agrees with central finite differences to 1e-6.
- IK on 200 targets generated from random valid configurations: >= 98 %
  converge with residual < 1e-4 m; returned joints always within limits.
- Unreachable target returns status `unreachable` with a residual equal to
  the gap to the reach bound within 1e-3 m for a straight-line case.
- Near-singular targets produce finite joints and bounded steps.
- Renderer link transforms equal core FK transforms to 1e-9 (unit test of
  the render adapter) and visually in the browser.
- Non-finite inputs are rejected without corrupting the scene.

### M2 Physical body
Quadruped and arm under MuJoCo simulation with gravity, contact, PD control,
reset, push tool and live gain tuning with plots.
Acceptance: quadruped stands under PD; push tool applies a labeled impulse;
gains change without model rebuild; plots show commanded, measured and
applied torque; reset restores the initial state exactly; PD step response
on a 1-DOF test model matches the analytical second-order response
(overshoot within 2 percentage points, settling time within 10 %).

### M3 Robot builder
Limb count, DOF per limb, link parameters, symmetry, presets (arm, rover,
quadruped, hexapod), validation messages, morphology versioning, export and
import.
Acceptance: every preset builds and simulates; invalid input shows a
specific message and is not applied; mirror symmetry produces mirrored
geometry; any morphology change produces a new hash; export then import
round-trips byte-identical canonical JSON.

### M4 Control lab
Step response, saturation display, controller comparison, guided
challenges (reach target, hold pose, survive a push).
Acceptance: rise time, overshoot, settling time and steady-state error are
computed and tested against analytical cases; saturation time fraction is
displayed; two gain sets can be compared on the same plot; each challenge
has an objective pass criterion evaluated from measured data.

### M5 Skills
Record demonstration, replay, parameter tuning, imitation learning and
reward-based improvement for the foot-lift skill, evaluation over repeated
trials, and a skill library with skill cards.
Acceptance: each method is labeled; evaluation runs >= 20 randomized trials
in a worker; skill cards show method, morphology hash, success rate and
tested conditions; incompatible morphologies are refused.

### M5 extension: skills the user teaches (2026-10-05)
The foot lift was the only skill; the vision asks that users teach skills.
Teach a skill: pose the live robot with joint sliders, keep poses as
keyframes with times, choose goals that define success, try once, evaluate
over 20 randomized trials, and run the skill by its name from the command
bar and the AI. The foot lift stays as the worked example.
Acceptance: any robot can be taught (fixed bases with limb-tip goals);
each goal is judged from ground truth by the same judge in evaluation and
playback and reported with its measured value; the stored skill is bound to
the morphology and refused elsewhere; a command with the skill's name runs
the best evaluated version and reports every goal; requests for untaught
motions point to teaching.

### M5 extension: skills the AI composes (2026-10-05)
A request for a motion nobody taught ended in "teach it first". The AI
composes such a skill itself (D24): poses described in terms of the body,
turned into joint targets by IK, goals that measure the request, a try in
the live simulation, revisions from the measured goals, an evaluation, and
the run, in one command, visible and editable in the Teach panel.
Acceptance: composed poses reach their targets (feet kept in place while
the body moves, tips moved by the requested offset) within the IK
tolerance, and unreachable tips are reported with their distance; invalid
drafts run nothing; the user's unsaved draft is never replaced; a draft is
saved only after an evaluation with at least one trial meeting every goal,
labeled as composed by the AI; a skill saved during a command can run in
the same command; requests for untaught motions reach the AI when it is
available, torque commands and several-leg lifts stay local refusals.

### M5 extension: physical-AI pipeline (2026-10-06)
The AI's goals could miss the request (a level crouch called a sit), the
robot could not walk, a skill could not improve, and the AI had no tools
in the Build, Kinematics and Physics tabs (D25). Goals are checked against
the request; gaits step from a contact pattern with IMU feedback;
evolution strategies improve a draft against its goal reward; saved skills
run in sequences; every tab has AI tools; the AI can switch the robot it
teaches.
Acceptance: a draft whose goals miss a motion word of its name or request
is refused with what to add; the new goal kinds (body angle, foot down,
tip swing, tip reach) are judged from ground truth with tested rules; a
trot moves the quadruped forward, backward and around in randomized
trials and stays upright; feedback signs level the body; an improvement
changes a draft only when held-out trials score better and records how;
a sequence stops at the first run that fails and reports the estimate and
the measurements; pushes, IK, joint settings and designs are bounded,
measured and refused without side effects when invalid.

### M6 Persistence and language
IndexedDB storage, reload and recover, command bar with the local parser,
optional cloud adapter, capability checks and fallbacks.
Acceptance: robots and skills survive a page reload; corrupted records are
reported, not silently dropped; the flagship command runs a stored skill
and reports measured hold time; unsupported commands are refused with a
suggestion; no API key is persisted or bundled.

### M7 Sensors and navigation
Sensor-only versus ground-truth modes, range rays, virtual camera, rover
with occupancy grid, planner, path follower and keyboard remote play
(gamepad support was dropped on 2026-10-04 at the user's request).
Acceptance: sensor-only controllers receive no ground-truth fields (tested);
A* finds shortest grid paths on test maps; the rover reaches goals in at
least 4 of 5 test maps without collision; remote control works with the
keyboard alone.

### M8 Polish
Onboarding, learn mode with explanations, attempt-comparison replay,
performance tuning, accessibility basics and a deployment build.
Acceptance: keyboard navigation for all panels, visible focus, color
contrast >= 4.5:1 for text; production build served statically passes the
Playwright smoke suite.
