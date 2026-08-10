# MorphLab user interface: interaction

Source reference reviewed on 2026-10-16. This guide explains user actions,
feedback, recovery and data lifetime. It is for people using the lab and
developers maintaining its interaction contract. Technology belongs in
[ui.md](ui.md), spatial arrangement in [layout.md](layout.md), and measured
implementation/verification evidence in [status.md](status.md).

Quick navigation: [workflows](#3-views-and-workflows), [commands](#4-command-bar),
[voice](#5-voice), [keyboard](#6-keyboard-and-pointer),
[failure recovery](#7-feedback-and-status), [honest labels](#8-labels-that-keep-the-interface-honest),
[saved data](#10-preferences-and-stored-data),
[accessibility](#11-accessibility), [review checklist](#12-interaction-review-checklist).

## 1. Principles

- One workspace: the AI Playground first, five lab tools, one command bar
  (D34). Every view works without the AI proxy; AI commands are an
  addition, never a requirement.
- The Stop task control in the command bar always works locally, with or
  without an AI service.
- Data sources are labeled where they appear: simulator ground truth,
  sensor-only controller estimates, recorded replays, the model's own
  words. Plots name commanded, applied and measured signals separately.
- Refusals, failures, solver limits and saturation are shown with the
  reason and the measured numbers, never hidden.
- Controls must map to implemented actions, not placeholders. A disabled
  action cannot run in the current state; an enabled action can still be
  refused by validation or fail physically. An experimental training
  option is not a promise of a successful motion.
- Copy is plain and restrained, in English, without emojis.

## 2. Navigation

- A left navigation ([src/ui/WorkspaceNav.tsx](../src/ui/WorkspaceNav.tsx))
  with text labels: AI Playground first (subtitle "Skills: teach,
  evaluate, run"), then the Lab tools group: Build, Kinematics, Physics,
  Control lab, Navigate. The app opens on the AI Playground.
- It is a navigation landmark of ordinary buttons; the open view has
  `aria-current="page"` and the work area is a `main` landmark named after
  it. There is no tablist: Tab moves between entries, Enter or Space opens
  one. Opening a view can load the physics engine, so nothing opens on
  focus alone.
- Lab tools show a header with "Lab tools / <tool>" and Back to AI
  Playground. When a view switch removes the focused control (Back, or a
  link in Build), focus moves to the new view's `main`, not the page body.
- Guide and Learn mode sit at the bottom of the navigation as
  pressed-state buttons and are stored as preferences. A units note
  ("SI units, radians internally, world frame Z up") follows them.
- At 1000 px and below the navigation becomes a top bar with the current
  view and a Menu button (`aria-expanded`). The menu is a disclosure, not
  a modal: it does not trap focus, closes with Escape or a choice, and
  returns focus to the Menu button.
- The internal mode identifier stays `skills`; messages still name it
  "the Skills tab" (see [status.md](status.md), known limitations).

## 3. Views and workflows

Each view shows a one-line hint in the command bar while it is open
([src/ui/modes.ts](../src/ui/modes.ts)).

| View | What the user does | Physics | Data shown |
|---|---|---|---|
| AI Playground | Teach a keyframe or gait skill, run the foot-lift example five ways, train a walking controller, evaluate over 20 trials, play, compare attempts | MuJoCo, live; evaluation and training in a worker | Library records with method label, trials and compatibility; the open record adds the success rate with its 95 % interval, conditions and the live-run record |
| Build | Load a preset, generate a legged robot or an arm, edit body, limbs and joints, export and import JSON, open the robot in another view | None (forward kinematics preview) | Morphology version and SHA-256 hash, edit history |
| Kinematics | Pick a robot and limb, edit a serial chain, move joints, drag or type an IK target, toggle overlays | None | IK status, residual, iterations, tip position, Jacobian singular values, manipulability |
| Physics | Run, pause and reset a MuJoCo simulation, tune PD gains, push the base, command one joint | MuJoCo, live | Simulated time, real-time factor, step cost, ground-truth base height and tilt, joint plots, events, MuJoCo warnings |
| Control lab | Run step tests with gain sets A and B, compare with a linear model, run challenges | MuJoCo, one fresh run per experiment | Plots after each run, step metrics, saturation, challenge criteria |
| Navigate | Choose the robot (the rover, a wheeled or legged preset or the Build robot), a map and controller inputs, drive or walk to a goal, drive by keyboard, remember places, evaluate all five maps | MuJoCo, live; batch in a worker | Ground-truth and estimated position, speed, replans, range rays, occupancy grid, path, inset camera |

Hand-off: Build's "Use this robot" opens the current robot in
Kinematics, Physics, Control lab or the AI Playground (Teach skills).
Physics, Control lab and the Playground also offer presets and show
"Builder robot: <name>" for a robot from Build.

Each mode owns its own session. A hand-off explicitly sets the destination
robot; changing a robot in Build does not make every other view a live
mirror of it. Check the open view and robot before a command. Switching
views also invalidates active AI tasks and ends a running playback; it is
not a way to continue a task in a different scene.

### Build, inspect and simulate

1. In Build, load a preset or generate an arm/legged robot. Edit its body,
  selected limb and joint in the labeled panels.
2. Commit numeric edits with Enter or blur. Read validation messages before
  assuming the geometry changed; a refused edit leaves the accepted robot
  unchanged. Accepted physical edits change the morphology hash.
3. Use Open in Kinematics to inspect reach and joint limits without
  physics. Move the target with the gizmo or its X/Y/Z fields. Read both
  the status and residual; Stalled is not proof of Unreachable.
4. Return to Build and use Simulate in Physics for gravity/contact checks,
  or Open in Control lab for a measured joint experiment.

A pose that looks valid in Kinematics can still fall or saturate its
actuators in Physics. The preview is geometry, not a stability certificate.

### Tune and compare controllers

1. In Physics, choose a joint, change its commanded position, and compare
  the commanded and measured position traces. The effort plot separates
  commanded effort from what the actuator actually applies.
2. Adjust gains or apply a labeled push; inspect warnings and saturation,
  not just how smooth the animation looks. Use the Simulation Pause
  control when the live scene must stop advancing.
3. In Control lab, choose the robot, joint, step and duration. Set gain
  sets A/B and enable Compare with gain set B when needed.
4. Run step test and wait for recorded plots and metrics. Changing the
  setup can make the shown result stale; rerun before comparing it with
  the new settings. A linear reference is a model, not another measurement.

### Teach, evaluate and use a skill

The Playground opens with the live robot, the Teach view and three real
starting points: New skill, Explore the foot lift example, Open the skill
library. Nothing starts moving, training or listening by itself.

1. Press New skill and give it a name. The draft states "Draft, not
   saved"; it belongs to the current robot and is not a library entry.
2. 1. Motion: pose joints in the live simulation and add keyframes with
   timings, or edit the gait parameters of a gait draft. The note below
   says whether playback is open loop or closed loop.
3. 2. Success criteria: add goals that actually measure the request,
   including body checks and ending conditions. Missing-goal warnings
   stay visible; a nice-looking motion does not substitute for a goal.
4. 3. Try: Try once runs one live preview judged by the goals. It adds
   nothing to the library and is not a randomized evaluation; the last
   playback appears below the button and the run is listed under Attempts
   (in this page only).
5. Optionally open Improve or train (optional). Read the measured
   comparison to learn whether the candidate was kept; completion alone
   does not mean an improvement was found. The comparison results stay
   visible under step 4 even with the disclosure closed.
6. 4. Evaluate and save: 20 randomized trials in a worker. Progress and
   Cancel appear in the status strip under the header. Open Library to
   inspect the record's method, body, successes over trials, confidence
   interval and tested conditions.
7. Use Play in the open library record, or the saved name in the command
   bar. Attempts shows the last playback and the recorded attempts; a live
   result is not the earlier evaluation.

| Action or state | Important distinction |
|---|---|
| Manual Evaluate 20 trials and save | A valid completed evaluation is saved even with zero successes, so it can be inspected. That does not make it eligible for text commands |
| AI evaluates a composed draft | It saves only if at least one trial met every goal; otherwise the draft remains for revision |
| Train a learned feedback policy | Changes a draft only when the applicable held-out comparison accepts it; evaluate again to save the result |
| Feedback model: Linear | Default for floating-base keyframe feedback training |
| Feedback model: Neural (experimental) | Opt-in nonlinear keyframe feedback; not a GPU switch, not a supported nonlinear gait option, and not a guarantee of balance |
| Change the motion or goals | Invalidates the draft's learned feedback; the old policy must not silently control a different sketch |
| Imported skill | Shows Unverified here; re-evaluate on its matching morphology before text commands can use the imported claim |
| Another robot's skill | A different morphology is incompatible; merely renaming it or retrying Play does not transfer the policy |

The UI does not silently replace a user's nonempty unsaved draft with an
AI-composed one. Save or deliberately discard that draft before requesting
a replacement. Robot changes and reloads can discard drafts.

Sources: [src/ui/skills/TeachPanel.tsx](../src/ui/skills/TeachPanel.tsx),
[src/ui/skills/session.ts](../src/ui/skills/session.ts) and
[the D33 evidence](status.md#learning-and-skills).

### Navigate and remember a destination

Choose the robot: every preset and the current Build robot are listed.
Navigate drives wheel-only, free-standing bodies with wheels on both sides
by skid steering, and walks free-standing bodies with four or more legs
with the trot gait generator steered by the planner (D37); geometry,
footprint, planner clearance and range-ray mount come from the robot's
description. Fixed-base arms, two-legged bodies and bodies that mix wheels
with legs are listed but disabled, with the reason shown below the list.
Build has a Drive in Navigate hand-off, disabled with the reason when the
design cannot move. Choosing a robot or a map is an explicit reset; a
batch keeps the robot it started with. A legged robot's button reads Walk
to goal, its speed limits are lower (0.05 to 0.15 m/s), and a fall ends
the run like a collision.

Leaving Navigate stops an autonomous run started from its panel, as well
as keyboard driving. Returning does not silently resume that run.

A legged robot walks with the walking controller trained for its body in
the AI Playground when there is one, otherwise with the gait generator;
the panel says which (D38). Train a walking controller (or asking the AI
to learn to walk) trains one from Navigate in background workers, with
progress and Cancel; it takes over the next time the robot stands. The
AI can also switch the Navigate robot by name.

Choose the map and controller input mode, then type a goal or double-click
the floor and press Drive to goal. Obstacle or out-of-bounds goals are
refused for the robot's footprint. The goal speed limit applies to
autonomous goal driving, not keyboard driving. A driving key takes over
from the autonomous run.

Compare Position, controller estimate with Position, ground truth rather
than treating them as interchangeable. Remember position saves a named,
map-scoped place with its source; Drive here and Forget operate locally.
Run the 5 test maps starts a separate cancellable worker evaluation, not
five visible trips in the current viewport.

## 4. Command bar

In the `command` area below the work area, wrapping as needed. On the
single-column layout it follows the panels in the scrolling page; it is
not a sticky overlay. Main row, left to right:

| Control | Behavior |
|---|---|
| Command (text, at most 300 characters) | Enter or Run submits. While a command runs the button reads Send and the text is taken as a live change |
| Stop task | Calls the local stop path without waiting for a model; cancels command/AI activity and invokes the applicable body stop behavior. It does not pause physics or cancel jobs |
| Running, AI status | "Running" while busy; "AI: <provider>", "AI: unavailable" or "AI: checking" |
| Voice and AI | Disclosure (`aria-expanded`) for the secondary options below; stays open while the microphone listens |

Voice and AI options:

| Control | Behavior |
|---|---|
| Speak | Shown when the browser has speech recognition, otherwise a sentence says it is missing; one English utterance per press |
| Check AI | Asks the proxy again; the reason is shown when AI is unavailable |
| Read replies | Reads results aloud with an installed local English voice |
| Forget conversation | Shown when conversation memory exists and nothing runs; clears it |

The output scrolls inside a bounded height, so long replies and an opened
trace never squeeze the robot view; the latest result line comes first.

How a command is handled ([src/ui/App.tsx](../src/ui/App.tsx)):

1. Live changes (stop, slower, faster and similar short commands) are
   applied locally to the running task.
2. A skill the user taught runs by its name.
3. The local rule-based parser interprets the text against the robot of
   the open view. Commands act on the view they are typed in.
4. Refusals the parser marks for escalation go to the AI only when the
   proxy reports a configured provider. If the view has no AI tools for the
   request, the local answer is shown with a note on where the AI can act.
5. Otherwise the local answer stands. In the Playground a lift that no
   evaluated skill serves can go to the AI, which may compose a skill.

The output shows, in order of relevance:

- the view hint while idle, with the reason when AI commands are off;
- while busy, the last tool call and its status, and that slower, faster
  or stop are accepted;
- the result, prefixed "Refused:", "Failed:" or "Question:" when it is
  one of those, with a suggestion, option buttons for questions, the
  source in parentheses (for example "local rules") and a note;
- "What the AI did": a collapsible trace with every tool call, its
  arguments, whether it was refused before running, its status, task ID,
  reason, measured values and fallback, plus the number of model calls,
  the model name and token count. The status line comes from the last
  measured tool result; the model's reply is labeled as its own words,
  not a measurement (D14);
- messages for Stop, live changes, voice and forgotten conversation.

### Stop, pause, reset and cancel

These controls have different scopes. A command outcome must state what
was actually stopped, not imply that every simulation and worker froze.

| Control | Scope |
|---|---|
| Stop task | Local action/task stopping; Navigate stops motion, Playground legs return to their stance and fixed arms hold their measured joint pose. When nothing was playing it says so and that jobs are not affected. It is not a general-purpose Physics Pause button |
| Physics Run/Pause | Controls advancement of that live simulation; Run is named Run simulation for assistive technology; pausing is not restoring its initial state |
| Physics or Navigate Reset | Restores that mode's simulation setup/start state; it does not clear the saved library or preferences |
| Job Cancel | Next to the job progress (Playground status strip, Navigate batch). Ends that training/evaluation job. Do not treat an interrupted batch as a complete evaluation |
| Switch view | Invalidates AI tasks and runs view cleanup (a playback is recorded as interrupted); clears old command output and discards late answers; session objects may remain for returning to the view |
| Forget conversation | Clears page-local AI conversation context, not skills, remembered places or evaluation records |

While a request is busy, Send is not a general queue of future tasks.
Only supported live changes are applied; an incompatible change receives
a reason. A request such as slower must not be documented as changing all
robots or every playback method.

## 5. Voice

- Read replies uses speech synthesis with a local voice only (no network
  service), English, at most 400 characters per reply. Off by default.
- Speak uses the browser's speech recognition (en-US, one utterance, 10 s
  limit). The first press shows a consent note: in Chrome and Edge the
  audio goes to the browser maker's speech service; MorphLab neither
  receives nor stores it. "Use voice input" stores consent; "Turn off
  voice input" withdraws it.
- A transcript runs exactly like typed text. Nothing is spoken while the
  microphone listens, so the app's own voice cannot become a command.
- Accepting consent does not automatically start recording: press Speak
  again. Pressing it while listening cancels that listening attempt.
  Browser microphone permission is separate from MorphLab's stored consent.
- If recognition is missing, denied or times out, use the text field.
  If no local English output voice exists, the UI reports that condition.
  The presence of either browser API does not prove audio works on a
  particular device. No realtime provider-audio connection is involved.

## 6. Keyboard and pointer

| Where | Input | Effect |
|---|---|---|
| Navigation | Tab between entries; Enter or Space | Opens that view; Escape closes the narrow-window Menu |
| Shared number fields | Enter or leaving the field | Commit changed, nonempty finite values; enforce `min`/`max` when supplied, otherwise the owning session performs domain validation |
| Navigate | Arrow keys or W A S D (`KeyboardEvent.code`) | Up/W forward, Down/S backward, Left/A and Right/D turn; release stops; a drive key takes over from Drive to goal |
| Navigate | Same keys on a focused slider or radio group | Only W A S D drive; the arrows keep their own meaning |
| Navigate | Any key in a text field, select, spin button or with Ctrl, Alt or Meta | Never drives |
| Navigate | Window blur or page hidden | Driving stops |
| Navigate viewport | Double-click on the floor | Sets the goal |
| Kinematics viewport | Drag the translate gizmo on the target sphere | Moves the IK target; the sphere color follows the IK status |
| All 3D viewports | Mouse drag and wheel | Orbit, pan and zoom (`OrbitControls` with damping) |

Keyboard driving is the only remote input; there is no gamepad support.

For shared numeric controls, an unchanged rounded display value must not
be written back during keyboard traversal. Angular fields may display
degrees while the model stores radians; prismatic fields use length units.
Never assume a unit from the slider's visual shape. The reference is
[src/ui/components.tsx](../src/ui/components.tsx) and
[src/core/model/units.ts](../src/core/model/units.ts).

## 7. Feedback and status

- Status badges with a colored dot: ok (green), warn (amber), error (red),
  idle (grey). Examples: Running, Paused, Unstable, Converged,
  Unreachable, Stalled, Playing, "Unverified here".
- Notes: neutral, `.warn` or `.error`; live regions use `role="status"`,
  errors that need attention `role="alert"`.
- Loading states are stated in words: "Opening local storage.", "Loading
  physics engine." (Playground badge and an overlay on its view), job
  progress in percent with a progress bar and a Cancel button.
- If the engine cannot load, the Playground badge reads "Simulation
  unavailable" with the reason over the view; the navigation, Build and
  Kinematics keep working.
- Notice area above the work area: missing browser features and their
  consequence, storage unavailable, restore notices (Dismiss), and stored
  records that could not be read (Remove each). Renders nothing when all
  is well.
- A React subtree error can show the view's error fallback while the
  navigation remains available. Asynchronous command/runtime errors have
  separate handling; the boundary is not a universal exception catcher.

### Recovery reference

| Situation | What to inspect or do | Do not conclude |
|---|---|---|
| AI unavailable | Read the status reason, retain local workflows, check the configured local proxy and use Check AI | Physics or local teaching requires the model |
| Edit refused | Read the field/section error and correct the input; the accepted model should remain unchanged | The displayed draft text was applied |
| IK stalled or not converged | Inspect residual, limits and singularity information; try another target/pose | The target is mathematically unreachable |
| Physics unstable | Read warnings, reset and reconsider gains/inputs | The final animation frame is a valid physical result |
| Control result stale | Rerun with the current setup | Old A/B metrics describe the edited settings |
| Skill job cancelled or failed | Read the job message and inspect the retained draft/library | A partial run earned a success rate |
| No eligible skill | Check morphology, local evaluation and success count; teach or re-evaluate as appropriate | A matching name alone grants permission to execute |
| Corrupt or missing stored record | Read the notice; retain it for diagnosis or explicitly Remove it | The app silently repaired or deleted the record |
| Storage unavailable | Export supported robots/skills before closing; assume in-memory work will be lost | A visible library is necessarily durable |

Some failures have no measured values because validation rejected the
request before execution. The interface should say so, not invent a
measurement to fill the result area.

## 8. Labels that keep the interface honest

| Label | Where |
|---|---|
| "No physics." | Build and Kinematics viewport labels |
| "Rendered from simulator state (ground truth)" | Physics viewport label |
| Ground-truth labels | Physics base height and tilt; Navigate position and speed explicitly name their source |
| "Position, controller estimate" | Navigate, next to the ground-truth position |
| "Sensor-only (encoders, IMU, range rays)" and "Ground truth (simulator pose and known map)" | Navigate controller inputs |
| "commanded", "measured (encoder)", "applied" | Physics joint plots |
| "commanded (rate-limited)" or "(ideal step)", "A measured (encoder)", "A linear model" | Control lab plots |
| "Replay of recorded simulator states, display only: nothing is simulated." | Skills attempt comparison |
| "Inset: virtual camera (display only)" | Navigate viewport label |
| Method labels such as "Replay of a recorded demonstration (open loop)", "State machine (hand-designed)", "Keyframes with a learned feedback policy (reinforcement learning under pushes, closed loop)" | Skill cards and command replies ([src/skills/controllers.ts](../src/skills/controllers.ts), [src/skills/artifact.ts](../src/skills/artifact.ts)) |
| "Unverified here" | Imported skills until "Re-evaluate here" runs |

"All goals met" means the encoded checks passed, not that every possible
interpretation of the original request was satisfied. For example, an
at-least repetition goal does not prove an exact repetition count, and a
body-height/tilt ending check does not prove every foot touched down.
These verification gaps are recorded in [status.md](status.md); UI copy
must not strengthen the claim. Trial counts, intervals and conditions are
evidence for that evaluation, not guarantees on arbitrary bodies or real
hardware.

## 9. Learning aids

- Guide: "Getting started" with six steps, AI Playground first, each a
  link that opens the view. Shown on first visit; Hide guide or the Guide
  button in the navigation hides it.
- Learn mode: shows `Explain` blocks (accent left border) next to the
  numbers they explain, written from the implemented algorithms. Off by
  default.

## 10. Preferences and stored data

| Item | Stored where |
|---|---|
| `guideHidden`, `learnMode`, `speakReplies`, `voiceInputConsent`, `playgroundPanelCollapsed` | IndexedDB `settings`, key `ui.preferences`, validated field by field |
| Last robot in Build and in Skills | IndexedDB `settings` as `[id, hash]` pointers |
| Robots, skills, walking controllers, places and skill run records | IndexedDB `robots`, `skills`, `settings`, `memories` |
| Conversation memory | Page only; cleared on context change and by Forget conversation |
| Attempts for comparison (at most six) | Memory only |

Defaults from [src/ui/storage.ts](../src/ui/storage.ts): guide shown,
Learn mode off, Read replies off, voice-input consent not granted, and the
Playground side panel open. The panel state is written when it is toggled,
not continuously. The open view is not a persisted preference; a reload
starts at the AI Playground.

Restore runs under a timeout; on failure the app starts with default
robots and says so. Export and import use JSON files (`*.morphlab.json`);
files over 5 MB are refused before reading. See
[architecture section 8](architecture.md#8-persistence).

### Backup boundaries

| Operation | Includes | Does not include |
|---|---|---|
| Build: Export JSON | The exported robot description in a MorphLab bundle | All saved versions or the full workspace |
| Skills: Export library | Saved library skills and available robot descriptions they reference; missing descriptions are reported | Unsaved draft, comparison recordings, UI preferences, places, conversation or separately stored walking controllers |
| Reload the same origin | Valid saved records and selected Build/Skills robot pointers | Unsaved drafts, live simulation state, command text or comparison attempts |

Bundles are validated data, not executable scripts. Invalid file structure
is refused; invalid records are reported. Imported evaluation claims stay
labeled as imported until rerun here. See
[src/persistence/bundle.ts](../src/persistence/bundle.ts).

IndexedDB is browser-profile and origin scoped. Changing the hostname,
scheme or port can show a different workspace; `localhost` and `127.0.0.1`
are different origins. Clearing site data removes local records. There is
no account-backed sync, automatic cloud backup, or whole-workspace export
control to rely on. Do not clear site data as a first response to a restore
warning.

## 11. Accessibility

- The navigation is a `nav` landmark ("Workspace") of buttons with
  `aria-current="page"` on the open view; the Lab tools list is labeled by
  its heading. The work area is a `main` landmark named after the view.
  The earlier tabs pattern and its `display: contents` tab panel were
  removed (D34); no partial tab semantics remain.
- Panels are `aside` landmarks with labels ("Robot builder", "Limb
  inspector", "Simulation inspector", "Teach", "Skill library", ...);
  sections are labeled by their heading.
- Inputs use associated labels or accessible names; not every control has
  a separate visible label (Place name uses `aria-label`). Toggles are
  checkboxes; pressed-state buttons use `aria-pressed`.
- A visible focus outline on every focusable element.
- [tests/e2e/a11y.spec.ts](../tests/e2e/a11y.spec.ts) checks the navigation semantics, focus after Back to
  AI Playground, Tab reach and visible focus for every control, text
  contrast of at least 4.5:1 in every view, the narrow-window Menu, and
  that at 390 x 844 the command field and Stop task come before the
  Playground side panel. Results are recorded in [status.md](status.md);
  for the D34 version they are pending.

Preserve numeric alternatives to target dragging and floor double-clicks.
Canvas plots and 3D scenes are not automatically accessible just because
their surrounding controls have labels; full screen-reader understanding
of those visuals is not established by the current tests. The CSS reorder
of lab tool panels on narrow screens changes visual order, not
DOM/keyboard order. The Playground side panel is not reordered by CSS: it
is rendered after the command bar, so its document order is its screen
order in one column.

## 12. Interaction review checklist

This is a checklist for future UI changes, not a record that the checks
were run during this documentation update.

- [ ] All six views can be opened by keyboard from the navigation, with visible current and focused states.
- [ ] A refused edit leaves the accepted robot unchanged and gives a useful reason.
- [ ] A command targets the intended view/body and reports measured results separately from model text.
- [ ] Stop remains local; Pause, Reset and job Cancel retain their distinct meanings.
- [ ] Training/evaluation expose progress, cancellation and failure without claiming partial success.
- [ ] Skill cards distinguish method, morphology, evaluation, live results and imported provenance.
- [ ] Reload/exports preserve only the documented data; unavailable storage gives a visible warning.
- [ ] Voice stays opt-in, cancels cleanly, and text remains usable without audio support.
- [ ] Missing AI or browser capabilities are not mistaken for a successful action.

Run browser checks against an identified CI build, record the actual
outcomes in [status.md](status.md), and leave unchecked accessibility or
device behavior explicitly unverified.
