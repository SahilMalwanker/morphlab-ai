# AI command coverage (2026-10-27)

This is a finite tool-path and scenario audit, not proof of every possible
natural-language command. `docs/status.md` remains the implementation
record.

## Builds and fixes

- Starting browser build: CI build of `fdd7b66`, entry
  bundle `index-aHWsCcjV.js`; served with the local PowerShell proxy.
- Existing saved `qa crouch` is preserved; new audit records use a `qa`
  name and are never substituted for existing skills.
- Implemented and verified: job Cancel produces an explicit cancellation
  result for skill evaluation, improvement and policy/walking training.
  The AI loop ends on that result without asking the model to reinterpret
  it or running any later queued action. Previously it reported an invalid
  draft and the model invented an explanation for a failure.
- Regression tests exercise four job-cancellation paths with silent
  workers, unchanged drafts/no saved partial results, and cancellation
  without aborting the outer AI signal. A loop test forbids later tools,
  context refresh and extra model calls after cancellation.
- Found on the first fixed artifact: a preview request saying "do not
  train or evaluate" nevertheless started training. `asksForLearning`
  matched a negated "train", and automatic follow-ups pushed more work.
  A second fix now recognizes common direct prohibitions and preview-only
  requests, stops contradictory follow-ups, and refuses prohibited tool
  execution even if the model asks for it. Evaluation that implicitly
  saves is also blocked when saving was explicitly declined. This bounded
  English guard is not a general natural-language authorization parser.
- CI caught a cancellation flag applied to the wrong bridge
  branch; corrected in `d32f6ae`. CI then passed the build,
  545 unit tests and 24 spikes. Cancellation was observed on that build
  with no invalid-draft advice or subsequent model explanation.
- A later 1 m/s walking request explicitly forbidding substitution was
  replaced by a slower `fast_walk` gait and weaker travel goals. The model
  trained and saved it. That is a failed request, regardless of its 20/20
  score against the substituted goals. The test artifact was deleted.
  The final guard blocks draft/mirror/training/improvement substitutions
  for common explicit no-substitution wording, and recognizes no-compose,
  no-teach and no-create prohibitions. It also removes contradictory tool
  fallbacks and blocks locomotion training's implicit evaluation/save when
  those side effects were forbidden.
- Final source `18207c9`: CI passed typecheck and build,
  59 unit files / 556 tests, and 17 spike files / 24 tests.
  Installed bundle `index-BpysE_Aj.js` from that build.
  Final browser retests refused 1 m/s without action/substitution, ran
  preview-only without training/evaluation/save or contradictory advice,
  and cancelled neural training without any further model call.

## Scope and evidence

All six workspaces were exercised in VS Code Chromium at 1440 x 900 with
gpt-4o-2024-08-06 through the local PowerShell proxy. Initial cases used
`fdd7b66`, middle cases used the cancellation/request-guard builds
`d32f6ae` and `bd2d4c5`, and the final critical retests used `18207c9`.
Unchanged positive paths were not all repeated on the final artifact.
The matrix covers 30 registered tools: 22 available paths exercised via
model calls (including expected refusals and cancellation), three via
their local command/control routes, and five gated by unavailable
capabilities. This is not exhaustive natural-language or parameter-space
coverage and not 30 successful live tool executions.

## Tool matrix

P = observed passing; E = expected refusal/failure; L = local route,
not a direct model tool invocation; G = gated by current capabilities.
Earlier smoke evidence is identified separately and is not a new pass.

| Tool | Live coverage in this audit | Contract/edge coverage |
|---|---|---|
| getRobotCapabilities | P: correct active quadruped | Existing schema tests |
| getRobotState | P: correct measured playback history, no motion | Read-only request on final artifact |
| getSceneObjects | G | Scene-object capability disabled in workspace bridge |
| pointAt | G | Pointing/scene-object capabilities disabled |
| navigateTo | P: reached 2.8 m goal, 0.155 m remaining; E: 9 m rejected | Active speed change and Stop |
| navigateToPlace | P: named origin; E: nonexistent name refused | Already-within-tolerance return reported |
| rememberPlace | P: current estimated origin stored | Sensor-only provenance retained |
| forgetPlace | P: temporary origin removed through tool | No unrelated place removed |
| executeSkill | P: crouch, all 3 goals met; E: unknown name | Incompatible-arm request refused |
| draftSkill | P: preview; fixed forbidden side effects | Preview-only leaves library unchanged |
| improveSkillDraft | P: 96-run search, no better candidate, draft preserved | Cancel regression passed |
| trainGaitPolicy | P: start and Cancel; full gait path ran during failed substitution case | Final Cancel status, zero later model calls |
| trainLocomotion | P: full training/save, 40/40 evaluation | Cancel regression and implicit-save prohibition |
| locomote | P: forward 0.3 m then right 30 deg | Out-of-range no-substitution case fixed |
| mirrorSkill | P: symmetric crouch mirrored and tried | Not a live asymmetric-leg mirror test |
| evaluateSkillDraft | P: mirrored draft 20/20, saved separately | Cancel and no-save guards tested |
| runSkillSequence | P: exactly two saved crouch runs, both passed | Unknown skill refused separately |
| selectRobot | P: arm3 and back to quadruped | Quad skill refused on arm |
| pushRobot | P: 0.5 N*s left; E: 50 N*s refused | Measured 2 s response |
| resetSimulation | L: Physics reset handled by local parser | Direct model route not separately invoked |
| moveTipTo | P: reachable; E: unreachable with 0.9847 m residual | No false reached claim |
| setJointAngles | P: base_yaw 0.2 rad; clamped 6 rad honestly | Unknown joint refused without substitution |
| buildRobot | P: 3-joint 0.6 m arm; E: zero joints refused | Original quadruped morphology restored |
| modifyTask | L: slower during active navigation, 0.40 to 0.30 m/s | Planning-phase rejection has confusing wording |
| cancelTask | L: Stop during navigation and in-flight model request | Typed task cancellation; no extra model explanation |
| listSkills | P: accurate saved-skill list | Unknown library name refused |
| saveSkillFromDemonstration | G | Demonstration capability disabled in bridge |
| evaluateSkill | G | Saved-artifact evaluation capability disabled |
| runControllerExperiment | P: baseline and configured B; A preserved | High-gain conditional-refusal limitation below |
| saveSkill | G | Artifact-saving capability disabled |

## Observed scenarios

1. "Run the saved qa crouch exactly twice using runSkillSequence. Do not
   teach, retrain or modify it." P: two planned/completed/successful runs,
   three goals passed in each; one tool call, two model calls.
2. "Execute the saved skill named nonexistent qa skill. Do not create a
   new skill." E: rejected before playback; model eventually explained
   the missing skill. Four model calls for one refused tool call is an
   efficiency concern, not an execution success.
3. "What robot is active and which evaluated skills are available?" P:
   quadruped and the existing evaluated skill correctly reported.

## Additional measured results

- Control lab baseline A (kp 60, kd 3, no gravity compensation): 27.0%
  overshoot, -2.45 deg steady-state error, not settled. B with compensation:
  12.4%, then 7.6% at kp 80/kd 4, settling 0.276 s. A stayed 60/3.
- Physics push 0.5 N*s: maximum tilt about 0.3 deg, minimum height
  0.262 m; 50 N*s refused without substituting a permitted impulse.
- Walking: 2722 rollouts on four CPU workers, 297.9 s; 40/40 randomized
  commands within tolerance (Wilson 95% approximately 91-100%), mean
  position error 0.013 m, heading error 0.014 rad. Uncalibrated generator
  achieved 2/40. Subsequent 0.3 m walk measured 0.305 m; right-30-deg
  turn measured about 32 deg, both within stated tolerances.
- Mirrored crouch: 20/20, Wilson 95% approximately 84-100%. Improvement
  tried 96 rollouts; baseline already met all goals in 8/8 held-out runs
  at reward 2.000, so no change was retained. No evaluation/save followed
  the request that explicitly declined them.
- Final cancellation: one tool call/one model call, `cancelled` tool
  result, unchanged draft, zero model calls after Cancel.
- Final acceptance page: 10 live chat responses, all HTTP 200, no image
  fields and no page errors/unhandled rejections. Earlier monitored pages
  also showed no image-bearing requests or page errors; deliberate aborts
  are not provider failures. No global performance claim is made from
  browser automation timing.
- Offline `server/serve.tests.ps1`: 62/62 passed. Leak scan: working tree,
  Git objects and served build PASS. No Node command ran locally.

## Remaining limitations

- Control lab intentionally allows gains outside its conservative explicit
  stability bound and flags them. A conditional request to refuse such
  gains still ran kp 400/kd 40 and the model omitted the warning. The
  simulation completed, but conditional intent was not honored. This audit
  did not silently change the lab's intentional experiment policy.
- While a model request is pending but no robot task has started, "slower"
  is refused with a confusing suggestion to say "slower". Once navigation
  was actually active the same command worked. UI also still prefixes a
  cancelled command with "Failed", although the tool/loop status is now
  cancelled and the false model explanation is gone.
- Unknown skill/place refusals sometimes consume four model calls because
  the retry guidance is over-eager. This is an efficiency issue, not a
  successful action. No unrequested replacement was accepted in those
  tests.
- Explicit-restriction guards recognize bounded common English wording,
  not arbitrary negation, conditional language, goal semantics, or every
  unit conversion. A successful tool score is still not proof that its
  goals fully capture a request. Exact repetition/contact and two-leg
  balance gaps from D32/D33 remain.
- Microphone recognition, spoken-audio quality, provider-outage UI,
  every browser/device, all morphology/map combinations, exhaustive value
  fuzzing, and the Node-based Playwright suite remain unexecuted here.
  Neural training was cancellation-tested; the full completed training
  benchmark in this audit was locomotion. Five schema-only capability-gated
  tools are not functioning live features and are not counted as such.
- The invalid substituted `fast_walk` test artifact was deleted; original
  `qa crouch`, requested `qa mirrored crouch`, and the valid walking
  controller were retained. Build was restored to the original quadruped
  hash. A generated QA arm remains as a saved test version.