# Physical AI failure analysis on the quadruped (2026-10-14)

Status of every implemented feature is tracked in `docs/status.md`. This
document records the root-cause analysis and live MuJoCo WASM worker
measurements gathered while investigating why quadruped Physical AI skills
beyond simple poses and jumping fail or score poorly.

## Evidence qualification (2026-10-14 follow-up)
This is an investigation log, not a validated root-cause proof or a claim
that its proposed changes work. Implementation and verification are tracked
in `status.md`; D32 records the first conservative corrections.
- The hand-set policy weights below were diagnostic probes, not a learned
   controller to ship. A later five-trial probe failed all trials; the
   single favorable result does not establish robust hind-leg balance.
- Several exploratory calls used incorrect draft fields (`limbId` instead
   of `limb`, `endHoldSeconds` instead of `endHoldS`) or ignored a refused
   composition, so results may have used the previous draft. The final
   checked `hind current` composition succeeded, but its training and
   evaluation still met every goal in 0/12 and 0/20 trials respectively.
- The reported normalization values 0.128, 0.115 and 0.635 came from the
   final trained policy, not the four initial open-loop rollouts.
- Changing normalization changes the physical policy, but centering a
   phase input is not inherently incorrect: the bias can compensate. Fixed
   scales and physical zero centers are a training choice to evaluate.
- `standsAtEnd` is not part of `judge.instant`; it is rewarded separately
   after the motion. A failed body check suppresses the success bonus even
   in the old ES score. It does not by itself trigger `FALL_PENALTY`, whose
   separate fall thresholds are in `trainPose.ts`.
- Claims that a cosine fade, larger search amplitude or support-margin
   shaping will solve the failures remain unverified. Do not implement
   hard-coded probe weights or weaken success criteria to match them.
- The proposed fixed normalization was implemented and tested in CI,
   then removed: standard jump evaluation dropped from 20/20 to 18/20
   and neither two-leg task succeeded. See D32 and `status.md` for the
   retained changes and exact verification state. Do not reapply this
   candidate as an already-proven fix.

## 1. Summary of current quadruped behavior (baseline at `3da5ccb`)

| Task | Method | Current result | What happens in physics |
|---|---|---|---|
| Locomotion (`locomote`) | Command-conditioned linear gait residual (ARS) | Working | Walks forward, backward, turns left and right, chains steps |
| Jump and land (`pose-jump`) | Keyframes + learned feedback policy (ARS, 288 params) | 20/20 evaluation, 38/40 pushed | Policy damps pitch and roll through flight and landing (`largest tilt 19 deg` vs `43 deg` open-loop) |
| Hind-leg stand (`pose-hind`) | Keyframes + learned feedback policy (ARS, 288 params) | 0/20 evaluation, 0/12 held-out | In CI (`947a7b2`), pitch held 1.09 s and front feet airborne 1.49 s (needed 1.5 s); in live browser training on the rough sketch (`up = { x: -0.15, pitch: -0.8 }`), pitch reached only `-20 deg` (`0.00 s` in `[-63, -29] deg`) |
| Two diagonal legs (`pose-diagonal`) | Keyframes + learned feedback policy (ARS, 288 params) | 0/20 evaluation, 0/12 held-out | Both lifted feet (`front_left`, `rear_right`) stay off the ground together for at most `0.57 s` of `1.5 s` required; the body rocks across the diagonal support line |
| Play bow | Keyframes composed by `gpt-4o` | 0/20 evaluation | `gpt-4o` wrote an empty hold frame `{ seconds: 1.5 }` between `{ seconds: 0.7, body: { pitch: 0.53 } }` and `{ seconds: 1.0, body: { pitch: 0.0 } }`, which returned to `stanceQ` immediately instead of holding the bow |
| Three-leg paw wave | Keyframes + `improveSkillDraft` (ES) | 7/20 evaluation (`feetOnly` 10/20) | `gpt-4o` dropped `body: { y: 0.05 }` on the wave frames, and `improveTaughtSkill` ignored `checks` (`feetOnly`, `noTipOver`) in its fitness |
| Three-leg scratch (`rear_left`) | Keyframes + `improveSkillDraft` (ES) | 0/20 evaluation (`upright` 0/20, `feetOnly` 0/20) | Same dropped weight shift across keyframes plus `improveTaughtSkill` ignoring body checks |

---

## 2. Observations and hypotheses from the investigation

### 2.1 Why the rough `pose-hind` sketch (`up = { x: -0.15, pitch: -0.8 }`) cannot rear up open-loop and why `POSE_RESIDUAL_RAD = 0.5` is tight on it
In `spikes/skill/pose-hind.spike.test.ts` (changed in `f7a0029` from a 5-frame sketch to a 3-frame rough sketch):
- Frame 0 and Frame 1 specify `body: { x: -0.15, pitch: -0.8 }` and `tips: [{ limb: 'front_left', x: -0.15, z: 0.2 }, { limb: 'front_right', x: -0.15, z: 0.2 }]`.
- Because `composePose` transforms tip targets by `T_B1_B0 = invertRigid(fromXyzRpy([-0.15, 0, 0], [0, -0.8, 0]))`, the rear foot targets in the pitched body frame lie above and behind the rear hips, producing:
  `kf0 = [0, 1.494, -1.988, 0, 1.494, -1.988, 0, 1.181, -2.113, 0, 1.181, -2.113]`.
- In `kf0`, the rear knees are bent to `-2.113 rad` (more bent than home `-1.4 rad`) and the front hips are flexed to `+1.494 rad` (lifting the front feet up in place rather than pushing against the floor).
- Measured open-loop in MuJoCo: the robot simply squats backward onto all four feet; pitch reaches at most `-19 deg` (`0.00 s` in `[-63, -29] deg`) and front feet leave the ground for only `0.052 s`.
- Even if the policy saturates `action = +1` on the rear knees (`+0.5 rad`), the rear knee target only reaches `-2.113 + 0.5 = -1.613 rad`, which is still more bent than home (`-1.4 rad`).
- However, we verified in the live browser worker on this exact `kf0` (`[0, 1.494, -1.988, ..., 1.181, -2.113]`) that a `[23, 12]` linear policy with `POSE_RESIDUAL_RAD = 0.5` **can** dynamically pitch the body onto the hind legs by pushing off with the front legs when the front feet touch the ground (`W[1,5] = -1.5, W[2,5] = 1.5, b1 = -0.6`) and driving the rear hips and knees from pitch `R[6]` and pitch rate `gyro[1]`:
  - Measured on the exact 3-keyframe `pose-hind` motion:
    - `pitchHold = 1.718 s` (goal requires `1.5 s` in `[-63, -29] deg`)
    - `airHold = 2.022 s` (goal requires `1.5 s` with both front feet off the ground)
    - `feetOnly = true`, `noTipOver = true`
  - Why did that policy fail `standsAtEnd` (`largest tilt 36 deg, limit 20 deg` in the last `0.5 s`)? Because `LearnedPoseController` keeps adding `POSE_RESIDUAL_RAD * action` at 100% authority during the final return-to-stance keyframe (`t = 3.5..4.5 s`) and the final hold (`t = 4.5..5.0 s`). When the policy has a positive rear-up bias or feedback gain, it prevents the robot from settling back onto all four feet at the end of the motion.

### 2.2 Why ARS in `trainPosePolicy` (`src/skills/trainPose.ts`) failed to find the balance policy

Four suspected contributors were identified; their causal role is not proven, and the first proposed normalization change was rejected after measurement:

1. **Shifting observation normalization across ARS iterations (`stats.merge(fresh)`)**:
   - `trainPosePolicy` initializes `(mean, std)` from 4 open-loop baseline runs (`policy: null`), and then updates `({ mean, std } = normalization(stats, feet))` at the end of every ARS iteration without rescaling `theta`.
   - Because a linear policy evaluates `W * ((x - mean) / std) + b`, changing `mean` and `std` at iteration `it` silently changes the effective gain and bias of the already-learned weight matrix `theta`.
   - Even worse, in the 4 open-loop baseline runs of `pose-hind` (where the robot barely moves), `std[0]` (`R[6]`) is `0.128`, `std[1]` (`R[7]`) is `0.115`, and `mean[10]` (`sin(pi * p)`) is `0.635` with `std[10] = 0.309`. Subtracting `0.635` from `sin(pi * p)` makes the normalized phase `x[10] = -2.06` at `p = 0` and `p = 1` instead of `0`, so any weight on `sin(pi * p)` injects a large unwanted joint offset at the start and end of the motion.

2. **Step-function `judge.instant` has zero gradient before a threshold is reached and zero safety margin once reached**:
   - In `GoalJudge.add` (`src/skills/taught.ts`), `this.instant` is the fraction of goals whose boolean condition holds at step `t` (or whose hold duration has already been met).
   - On `pose-hind`, any candidate policy that pitches the body to `-28 deg` instead of `-29 deg` gets `0` from the `bodyAngle` term in `judge.instant`—the exact same instant reward as doing nothing at `0 deg`.
   - Conversely, once a candidate barely reaches `-29.01 deg` for `1.500 s` on a nominal run, `judge.instant` gives `1.0` with zero incentive to aim for the interior of the target window (`-46 deg`) or to hold for `1.8 s` so that randomized runs also pass `1.5 s`.

3. **"Do nothing on 4 feet" is a strong local optimum in `posePolicyRollout`**:
   - On `pose-hind`, standing motionless on 4 feet satisfies `noFall`, `standsAtEnd`, `feetOnly`, and `noTipOver` (`4` of `6` conditions in `judge.instant`, plus `W_END = 4` at the end), earning a return of `~6.90` with zero risk of falling or touching a knee.
   - During early exploration, candidates that rear up far enough to reach `-35 deg` pitch often touch a knee on landing (`feetOnly = false`) or tip past `20 deg` at the end, which zeroes `worst = Math.min(...scores)` in `goalReward` and incurs `FALL_PENALTY`. As a result, ARS steps toward small weights (`|W| < 0.42`) that keep all 4 feet near the ground (`pitch -20 deg to 2 deg`, `airborne 0.346 s`).

4. **No continuous CoM-over-support-feet reward for two-leg balance (`pose-diagonal` and `pose-hind`)**:
   - Balancing on two diagonal legs (`front_left` and `rear_right` airborne, `front_right` and `rear_left` supporting) or on two hind legs requires keeping the horizontal center of mass over the support segment between the grounded feet, with sufficient foot clearance (`4 to 6 cm`) on the lifted feet so random pushes do not cause premature ground taps that reset the `airborne` contiguous-time counter (`st.cur = 0`).
   - Currently, `airborne` only checks `!s.tipContact[f]` (boolean contact `< 1 N`). A foot hovering `1 mm` above the floor scores the same `1.0` as a foot lifted `6 cm`, until a push causes a `10 ms` tap and cuts `st.longest` in half.

### 2.3 Why `improveTaughtSkill` (`src/skills/improve.ts`) failed on 3-leg skills (`paw wave`, `scratch`)
- Line 127 of `src/skills/improve.ts` calls:
  `goalReward({ success: r.success, goals: r.goals ?? [] }, r.unstable)`
  without passing `checks: r.checks`.
- Consequently, Evolution Strategies in `improveTaughtSkill` is completely blind to `feetOnly` (knee or base touching the ground) and `noTipOver`, Accepting candidate keyframes that wave a paw while resting a knee or hip on the floor (`feetOnly` failed in 10/20 trials on `paw wave` and 20/20 on `scratch`), which then fail evaluation!
- In addition, `POSE_SCALE = 0.08` rad in `src/skills/improve.ts` is too small when an open-loop keyframe draft needs a `0.15 to 0.25 rad` stance-leg adjustment to keep the center of mass inside the 3-leg support triangle.

### 2.4 Why AI-composed multi-keyframe skills (`play bow`, `paw wave`, `scratch`) lost their body posture mid-skill
- In `src/skills/compose.ts`, every frame is composed relative to `c.home` (`stanceQ`), not relative to the previous frame:
  - When `gpt-4o` wrote `[{ seconds: 0.7, body: { pitch: 0.53 } }, { seconds: 1.5 }, { seconds: 1.0, body: { pitch: 0.0 } }]` for `"play bow"`, it intended Frame 1 (`{ seconds: 1.5 }`) to hold the bow from Frame 0 and Frame 2 (`{ seconds: 1.0, body: { pitch: 0.0 } }`) to return to level. Instead, Frame 1 returned to `stanceQ` immediately (`0/20`).
  - When `gpt-4o` wrote `"paw wave"` and `"scratch"`, it put `body: { y: 0.05 }` in Frame 0 to shift weight onto the supporting legs, and then wrote `tips: [{ limb: 'front_left', z: 0.1, y: 0.06 }]` in Frames 1..4 **without** repeating `body: { y: 0.05 }`. Because `composePose` starts from `c.home`, omitting `body` on Frames 1..4 dropped the weight shift as soon as the foot lifted!

---

## 3. Original candidate fixes (not all accepted; see D32)

1. **`src/skills/posePolicy.ts` (`LearnedPoseController` return-to-stance envelope)**:
   - When a motion has at least 2 keyframes and its final keyframe returns to `stanceQ` (while an earlier keyframe moves away from `stanceQ`), smoothly taper the policy residual scale with a cosine envelope `0.5 * (1 + cos(pi * u))` during the final return-to-stance segment (or during `endHoldS` if the last two keyframes are already `stanceQ`).
   - When the final keyframe does not return to `stanceQ` (such as `liftMotion` in `tests/skills/posePolicy.test.ts`), the envelope remains `1.0` throughout `t <= durationS`, preserving every existing unit test assertion.

2. **`src/skills/trainPose.ts` (Fixed physical normalization + smooth interior-margin training return)**:
   - Fix `normalization(stats, feet)` so that:
     - Zero-centered physical inputs (`R[6]`, `R[7]`, `gyro[0..2]`, `sin(pi * p)`, and joint tracking errors `q - sketch`) use `mean = 0` and physical lower bounds on `std` (`0.2` for gravity and joint errors, `0.5` for gyro), while `sin(pi * p)` uses `mean = 0, std = 1` so `x = 0` at `p = 0` and `p = 1`.
     - Foot contacts (`5..5+feet-1`) keep `mean = 0.5, std = 0.5`.
     - Progress `p` (`5+feet`) keeps running mean and `std >= 0.25`.
     - Keep `(mean, std)` fixed across the ARS loop so `theta` is never silently rescaled mid-search.
   - Add a dense, physics-grounded shaping signal during training (`job.training`) that rewards:
     - Continuous proximity and interior margin on `bodyAngle` (peaking at the midpoint of `[minRad, maxRad]` and continuing to reward holds up to `1.35 * holdS`).
     - Continuous foot clearance (`up to 6 cm`) on `airborne` limbs together with horizontal CoM support margin (`supportMargin` from `src/core/math/support.ts`) over the remaining grounded feet, so two-leg balance (`pose-diagonal` and `pose-hind`) gets a smooth gradient toward keeping the CoM over the support line/feet and lifting the airborne feet cleanly clear of the floor.
     - Active-goal gating so standing motionless on 4 feet when an active goal (`airborne`, `bodyAngle`, `tipHeight`, `tipSwing`, `tipReach`, `baseHeight`, `baseRise`, `travel`, `turn`) is unmet cannot outscore policies that actively achieve the motion.

3. **`src/skills/improve.ts` (Include body checks in ES fitness + authority)**:
   - Pass `...(r.checks ? { checks: r.checks } : {})` to `goalReward` in `improveTaughtSkill` so Evolution Strategies penalizes knee/body ground touches and tip-overs.
   - Increase `POSE_SCALE` from `0.08` to `0.14` rad so ES can adjust supporting legs enough to stabilize 3-leg waves and scratches.

4. **`src/skills/compose.ts` & `src/ui/ai/bridge.ts` (Hold-frame semantics & weight-shift note)**:
   - In `composeKeyframes`, if an interior keyframe `frames[i]` (`0 < i < frames.length - 1`) omits `body`, `tips`, and `joints` while `frames[i - 1]` moved away from home and `frames[i + 1]` explicitly specifies a return (such as `body: { pitch: 0 }`), treat `frames[i]` as holding `keyframes[i - 1].q` and add a note explaining that the empty middle frame held the previous pose.
   - When an earlier frame shifted `body` and a later frame lifts a leg tip (`z > 0.02`) without specifying `body`, add a diagnostic note reminding the caller that `body` is relative to home stance and must be repeated in each frame that needs the weight shift.
