import type { RobotCapabilities } from '../../ai/capabilities';
import type { BridgeSnapshot, RobotBridge } from '../../ai/controller';
import { ACTION_TOOLS, asksForLearning, type ToolStep } from '../../ai/loop';
import { PLACE_SOURCE_LABELS, type LongTermMemory, type PlaceRecord } from '../../ai/memory';
import { rejected, type Measurement, type ToolResult } from '../../ai/result';
import type { FinalResult, TaskExecutor } from '../../ai/tasks';
import { SKILL_NAME_PATTERN } from '../../ai/tools';
import { restsOffTarget } from '../../core/control/stepMetrics';
import { baseTransform, buildKinematicModel, clampToLimits, forwardKinematics } from '../../core/kinematics/fk';
import { morphologyHash } from '../../core/model/hash';
import type { JointDescription, RobotDescription } from '../../core/model/types';
import { isQuestionText } from '../../language/parser';
import { RAY_ANGLES, RAY_MAX_RANGE } from '../../nav/stack';
import type { ControllerSpec, StepTestResult } from '../../physics/experiments';
import { groundTruthPose } from '../../physics/navigation';
import { checkSkillCompatibility, commandEligibility, skillLabel, type SkillArtifact } from '../../skills/artifact';
import { describeBody, type ComposedFrame } from '../../skills/compose';
import { gaitText, type GaitParams } from '../../skills/gait';
import { checkIntent, type IntentFinding } from '../../skills/intent';
import { rangeProblem, walkForTarget, type LocomotionController, type WalkCommand } from '../../skills/locomotion';
import { GOAL_LABELS, goalText, type BodyCheck, type SkillGoal } from '../../skills/taught';
import type { PosePolicyResult } from '../../skills/trainPose';
import { homeConfiguration, PRESETS, type PresetId } from '../../core/model/catalog';
import type { LeggedSpec } from '../../core/model/builder';
import type { BuilderSession } from '../builder/session';
import type { ControlSession, StabilityNote, StepRunSetup } from '../control/session';
import type { KinematicsSession } from '../kinematics/session';
import { MODES, type Mode } from '../modes';
import { rememberPlace } from '../nav/places';
import { walkerInfo, type NavSession, type WalkingTraining } from '../nav/session';
import type { PhysicsSession } from '../physics/session';
import { describePlay, OUTCOME_LABELS, walkOutcomeText, type PlayReport, type RunCheckpoint, type SkillsSession } from '../skills/session';

export interface WorkspaceAccess {
  mode(): Mode;
  nav(): NavSession | null;
  skills(): SkillsSession;
  memory(): LongTermMemory;
  control(): ControlSession | null;
  physics?(): PhysicsSession | null;
  kinematics?(): KinematicsSession | null;
  builder?(): BuilderSession | null;
  /** The Build tab's robot: the open design, or the saved one before Build is opened. */
  builtRobot?(): RobotDescription | null;
}

const NO_FEATURES: RobotCapabilities['features'] = {
  pointing: false,
  navigation: false,
  sceneObjects: false,
  demonstrations: false,
  controllerExperiments: false,
  skillEvaluation: false,
  skillAuthoring: false,
  skillSequences: false,
  robotChoice: false,
  physicsTools: false,
  kinematicsTools: false,
  designTools: false,
};

const round = (v: number, digits = 3): number => Number(v.toFixed(digits));
const m = (value: number, unit: string, digits = 3): Measurement => ({ value: round(value, digits), unit });

// Robot descriptions are replaced, never mutated, so object identity marks a robot change.
const robotIds = new WeakMap<object, number>();
let robotSeq = 0;
function identity(o: object): number {
  let id = robotIds.get(o);
  if (id === undefined) {
    id = ++robotSeq;
    robotIds.set(o, id);
  }
  return id;
}

function ok(reason: string, measurements: Record<string, Measurement> = {}): ToolResult {
  return { taskId: null, status: 'succeeded', reason, measurements, fallback: null };
}

function jointCount(d: RobotDescription): number {
  return d.limbs.reduce((n, l) => n + l.joints.length, 0);
}

function capabilityText(caps: RobotCapabilities, tools: string): string {
  return `${caps.name} (${caps.bodyKind}, ${caps.jointCount} joints${caps.morphologyHash ? `, morphology ${caps.morphologyHash.slice(0, 12)}` : ''}). Tools here: ${tools}.`;
}

function placeText(p: PlaceRecord): string {
  return `"${p.name}" (${p.x.toFixed(2)}, ${p.y.toFixed(2)}) m, from ${PLACE_SOURCE_LABELS[p.source]}`;
}

/** Measurement names say where a remembered position came from. */
const PLACE_PREFIX: Record<PlaceRecord['source'], string> = { 'sensor-estimate': 'estimated', 'ground-truth': 'groundTruth', given: 'given' };

function prefixed(x: TaskExecutor, prefix: string): TaskExecutor {
  return {
    run: async (signal): Promise<FinalResult> => {
      const r = await x.run(signal);
      return { ...r, reason: `${prefix}${r.reason}` };
    },
    ...(x.modify ? { modify: x.modify } : {}),
  };
}

/** The robot's range rays as the sensor reported them; never the obstacle map or simulator geometry. */
function rangeReadings(ranges: readonly number[], aheadM: number, who: string): { text: string; measurements: Record<string, Measurement> } {
  if (ranges.length !== RAY_ANGLES.length) return { text: 'Range sensor: no reading yet.', measurements: {} };
  const measurements: Record<string, Measurement> = {};
  const parts: string[] = [];
  let nearest: { range: number; bearing: number } | null = null;
  for (let i = 0; i < RAY_ANGLES.length; i++) {
    const a = RAY_ANGLES[i]!;
    const r = ranges[i]!;
    const deg = Math.round((a * 180) / Math.PI);
    if (!(r >= 0 && r <= RAY_MAX_RANGE)) {
      parts.push(`${deg} none`);
      continue;
    }
    parts.push(`${deg} ${r.toFixed(2)}`);
    if (a === 0) measurements['rangeAhead'] = m(r, 'm');
    if (!nearest || r < nearest.range) nearest = { range: r, bearing: a };
  }
  if (nearest) Object.assign(measurements, { nearestRange: m(nearest.range, 'm'), nearestRangeBearing: m(nearest.bearing, 'rad') });
  return {
    text: `Range sensor, measured from ${aheadM} m ahead of ${who}'s center (bearing in deg, positive to the left, then range in m; none means no hit within ${RAY_MAX_RANGE} m): ${parts.join(', ')}.`,
    measurements,
  };
}

const capitalize = (s: string): string => s.charAt(0).toUpperCase() + s.slice(1);

/** Navigate: the wheeled or legged robot being driven, its navigation stack and the places remembered on its map. */
async function navSnapshot(n: NavSession, memory: LongTermMemory, built: () => RobotDescription | null): Promise<BridgeSnapshot | { unsupported: string }> {
  const rt = n.runtime;
  if (n.state.status !== 'ready' || !rt) return { unsupported: n.state.status === 'error' ? (n.state.error ?? 'The Navigate simulation failed.') : 'The Navigate simulation is still loading.' };
  const d = rt.description;
  const drive = n.drive;
  const legged = drive.kind === 'legged';
  const walking = n.walkingController;
  const who = d.id === 'rover' ? 'the rover' : d.name;
  const aheadM = drive.rayOrigin[0];
  const map = n.map;
  const places = memory.places(map.id);
  const caps: RobotCapabilities = {
    name: d.name,
    morphologyHash: await morphologyHash(d),
    bodyKind: legged ? 'legged' : 'wheeled',
    jointCount: jointCount(d),
    skills: [],
    places: places.map((p) => p.name),
    ...(legged ? { walking: walking ? walking.ranges : null } : {}),
    features: { ...NO_FEATURES, navigation: true, robotChoice: true },
  };
  const sensorOnly = n.state.mode === 'sensor-only';
  const [lo, hi] = n.limits.speedLimitRange;
  const training = n.state.walkingJob;
  const summary = [
    drive.kind === 'wheeled'
      ? `Robot: ${d.name}, a wheeled robot (${drive.wheels.length} wheels, skid steering, track ${drive.trackWidth.toFixed(2)} m) in the Navigate tab, on the map "${map.name}". The arena floor spans x -0.4 to 3.4 m and y -1.9 to 1.9 m; obstacles are boxes.`
      : `Robot: ${d.name}, a legged robot (${drive.legs.length} legs) in the Navigate tab, on the map "${map.name}". It walks with ${walkerInfo(walking).text} The planner's speed and turn rate are its walking commands. The arena floor spans x -0.4 to 3.4 m and y -1.9 to 1.9 m; obstacles are boxes.`,
    legged
      ? training
        ? `A walking controller for this body is being trained now (${Math.round(training.fraction * 100)} %); it walks with it from the next time it stands.`
        : walking
          ? 'A request to learn to walk again or to retrain its walking needs a trainLocomotion call (four to eight minutes); it replaces the stored controller.'
          : 'No walking controller is trained for this body yet. A request to learn to walk, to train its walking, or for a learned, trained or AI walking controller needs a trainLocomotion call: it trains one in background workers (four to eight minutes), and nothing is trained without that call. Afterwards it walks planned paths with that controller.'
      : '',
    'selectRobot switches the Navigate robot: "rover", "quadruped", "hexapod", or "built" (the robot designed in the Build tab); arms cannot move here. It resets the run to the start of the map.',
    sensorOnly
      ? `Its controller runs in sensor-only mode: it localizes from ${legged ? 'leg odometry (joint encoders and foot touch sensors)' : 'wheel odometry'} and the IMU and maps obstacles with its range sensors; it never receives the simulator pose.`
      : 'Its controller runs in ground-truth mode: it receives the simulator pose, labeled as ground truth.',
    `getRobotState also reports the range sensor: nine rays from -80 to 80 deg (positive to the left), measured from ${aheadM} m ahead of ${who}'s center up to ${RAY_MAX_RANGE} m; use it for distances to obstacles.`,
    `navigateTo plans a path on the occupancy grid and ${legged ? 'walks' : 'drives'} it. A run ends when the goal is reached, on a collision${legged ? ', if the robot falls over' : ''}, when no path exists, or after ${n.limits.timeLimitS} s of simulated time; outcomes are judged from simulator ground truth.`,
    `The speed limit is ${n.state.speedLimit.toFixed(2)} m/s; navigateTo and navigateToPlace take an optional maxSpeed from ${lo} to ${hi} m/s for this robot, and during a run the user can say slower, faster or stop.`,
    places.length
      ? `Remembered places on this map (world memory): ${places.map(placeText).join('; ')}. navigateToPlace ${legged ? 'walks' : 'drives'} to one by name.`
      : 'No places are remembered on this map yet.',
    `rememberPlace stores ${who}'s current position under a name (the tool reads the position itself), or given coordinates.`,
  ]
    .filter(Boolean)
    .join(' ');
  const query = (tool: string): ToolResult => {
    const s = n.state;
    switch (tool) {
      case 'getRobotCapabilities':
        return ok(capabilityText(caps, `getRobotState, navigateTo, rememberPlace${places.length ? ', navigateToPlace, forgetPlace' : ''}, ${legged ? 'trainLocomotion, ' : ''}selectRobot, cancelTask`));
      case 'listSkills':
        return ok(`${capitalize(who)} has no saved skills; it ${legged ? 'walks' : 'drives'} with its navigation stack.`);
      case 'getRobotState': {
        const pose = sensorOnly ? s.estimate : s.truth;
        const label = sensorOnly ? 'estimated' : 'groundTruth';
        const measurements: Record<string, Measurement> = { goalX: m(s.goal[0], 'm'), goalY: m(s.goal[1], 'm'), simTime: m(s.simTime, 's', 2) };
        if (pose) Object.assign(measurements, { [`${label}X`]: m(pose.x, 'm'), [`${label}Y`]: m(pose.y, 'm'), [`${label}Yaw`]: m(pose.yaw, 'rad') });
        const source = sensorOnly ? 'Pose: sensor-only estimate (odometry and IMU).' : 'Pose: simulator ground truth (ground-truth mode).';
        const rays = rangeReadings(s.ranges, aheadM, who);
        Object.assign(measurements, rays.measurements);
        return ok(`Driving: ${s.driving}; navigation: ${s.navStatus}; ${s.collided ? 'collided, reset needed' : s.fallen ? 'fell over, reset needed' : s.unstable ? 'unstable, reset needed' : 'no collision'}. ${source} ${rays.text}`, measurements);
      }
      default:
        return rejected(`${tool} is not available for ${who}.`);
    }
  };
  const navRefusal = (): ToolResult | null => {
    if (n.state.collided || n.state.fallen || n.state.unstable) return rejected(`${capitalize(who)} has to be reset first: it collided, fell over or the simulation became unstable.`, 'Press Reset in the Navigate tab.');
    if (n.state.evaluation.running) return rejected('A batch evaluation is running in the Navigate tab.', 'Wait for it to finish or cancel it.');
    return null;
  };
  const action = (tool: string, args: Record<string, unknown>): TaskExecutor | ToolResult => {
    const maxSpeed = typeof args['maxSpeed'] === 'number' ? args['maxSpeed'] : undefined;
    switch (tool) {
      case 'navigateTo':
        return navRefusal() ?? navigateExecutor(n, Number(args['x']), Number(args['y']), maxSpeed);
      case 'navigateToPlace': {
        const p = memory.findPlace(map.id, String(args['place']));
        if (!p) return rejected(`No place named "${String(args['place'])}" is remembered on this map.`);
        return navRefusal() ?? prefixed(navigateExecutor(n, p.x, p.y, maxSpeed), `${placeText(p)}: `);
      }
      case 'rememberPlace': {
        const x = args['x'];
        const y = args['y'];
        const r = rememberPlace(n, memory, String(args['name']), typeof x === 'number' && typeof y === 'number' ? { x, y } : undefined);
        if (!r.ok) return rejected(r.reason);
        const p = r.place;
        const k = PLACE_PREFIX[p.source];
        return ok(`${r.replaced ? 'Updated' : 'Remembered'} ${placeText(p)} on the map "${map.name}".`, { [`${k}X`]: m(p.x, 'm'), [`${k}Y`]: m(p.y, 'm') });
      }
      case 'forgetPlace': {
        const p = memory.forgetPlace(map.id, String(args['place']));
        return p ? ok(`Forgot "${p.name}" on the map "${map.name}".`) : rejected(`No place named "${String(args['place'])}" is remembered on this map.`);
      }
      case 'selectRobot':
        return navSelectRobot(n, String(args['robot']), built);
      case 'trainLocomotion':
        if (!legged) return rejected(`${capitalize(who)} rolls on wheels; walking controllers are trained for legged bodies.`);
        if (n.state.evaluation.running) return rejected('A batch evaluation is running in the Navigate tab.', 'Wait for it to finish or cancel it.');
        return trainLocomotionExecutor((signal) => n.trainWalking(signal), `${d.name} walks planned paths in Navigate with it from the next time it stands.`);
      default:
        return rejected(`${tool} is not available for ${who}.`);
    }
  };
  // Seen in the browser: "learn to walk" was answered with "the robot is learning to walk" and no tool call.
  const followUp = (steps: readonly ToolStep[], request: string): string | null => {
    if (!legged || walking || training || steps.some((x) => x.accepted && ACTION_TOOLS.has(x.tool)) || isQuestionText(request)) return null;
    return asksForLearning(request) && WALK_WORDS.test(request)
      ? 'The user asked the robot to learn to walk. Call trainLocomotion now: nothing is trained until it is called, so do not say that training started without its result.'
      : null;
  };
  return { caps, summary, query, action, followUp };
}

const WALK_WORDS = /\b(walk|walks|walking|gait|gaits|locomotion|step|steps|stepping)\b/i;

const NAV_ROBOTS: readonly PresetId[] = ['rover', 'quadruped', 'hexapod'];

/** selectRobot in Navigate: the rover, a legged preset or the Build tab's robot; arms and other bodies are refused with the reason. */
function navSelectRobot(n: NavSession, which: string, built: () => RobotDescription | null): TaskExecutor | ToolResult {
  let r: { ok: true } | { ok: false; reason: string };
  if (which === 'built') {
    const robot = built();
    if (!robot) return rejected('The Build tab has no robot yet.', 'Design one in the Build tab first.');
    if (robot === n.robot) return ok(`Navigate already shows ${robot.name}.`);
    r = n.setRobot(robot, 'custom');
  } else if (NAV_ROBOTS.includes(which as PresetId) || PRESETS.some((p) => p.id === which)) {
    if (n.state.robotSource === which) return ok(`Navigate already shows ${n.robot.name}.`);
    r = n.setPreset(which as PresetId);
  } else {
    return rejected(`There is no robot "${which}" for Navigate.`, `Robots: ${NAV_ROBOTS.join(', ')}, built.`);
  }
  if (!r.ok) return rejected(r.reason, `Robots that move here: ${NAV_ROBOTS.join(', ')}, or a wheeled or legged Build robot.`);
  return {
    run: async (): Promise<FinalResult> => {
      // The stored walking controller is looked up by morphology before the robot is described.
      await n.refreshWalking();
      const d = n.drive;
      const body = d.kind === 'wheeled' ? `${d.wheels.length} wheels` : `${d.legs.length} legs; it walks with ${walkerInfo(n.walkingController).text}`;
      return { status: 'succeeded', reason: `Navigate now shows ${n.robot.name} (${body}) at the start of the map "${n.map.name}".`, measurements: { joints: { value: jointCount(n.robot), unit: 'count' } }, fallback: null };
    },
  };
}

function navigateExecutor(n: NavSession, x: number, y: number, maxSpeed?: number): TaskExecutor {
  return {
    modify: (c) => {
      if (c.change !== 'slower' && c.change !== 'faster') return { ok: false, reason: `A navigation run can go slower or faster, not ${c.change}.` };
      const r = n.changeSpeed(c.change, c.amount);
      return r.ok ? { ok: true, description: r.description } : r;
    },
    run: (signal) =>
      new Promise<FinalResult>((resolve) => {
        const rt = n.runtime;
        if (!rt) return resolve({ status: 'failed', reason: 'The Navigate simulation is not available.', measurements: {}, fallback: null });
        const t0 = rt.time;
        // maxSpeed and live changes last for this task only; the earlier limit returns when it ends.
        const limitBefore = n.state.speedLimit;
        const walked = n.drive.kind === 'legged' ? ` It walked with ${n.walkingController ? 'its learned walking controller' : 'the gait generator (no walking controller is trained for it)'}.` : '';
        const done = (r: FinalResult): void => {
          n.setSpeedLimit(limitBefore);
          resolve(r);
        };
        const measure = (): Record<string, Measurement> => {
          const p = groundTruthPose(rt);
          const [gx, gy] = n.state.goal;
          return {
            groundTruthDistanceToGoal: m(Math.hypot(p.x - gx, p.y - gy), 'm'),
            groundTruthX: m(p.x, 'm'),
            groundTruthY: m(p.y, 'm'),
            simTimeElapsed: m(rt.time - t0, 's', 2),
            replans: { value: n.state.replans, unit: 'count' },
            commandedSpeedLimit: m(n.state.speedLimit, 'm/s', 2),
          };
        };
        if (maxSpeed !== undefined && !n.setSpeedLimit(maxSpeed)) {
          const [lo, hi] = n.limits.speedLimitRange;
          return done({ status: 'failed', reason: `A speed limit of ${maxSpeed} m/s is outside ${lo} to ${hi} m/s for this robot.`, measurements: {}, fallback: null });
        }
        if (!n.setGoal(x, y)) {
          return done({ status: 'failed', reason: n.state.message ?? 'The goal was refused.', measurements: {}, fallback: 'Choose a goal on free floor, away from obstacles.' });
        }
        n.startAuto();
        if (n.state.driving !== 'auto') return done({ status: 'failed', reason: 'The navigation stack did not start.', measurements: {}, fallback: 'Reset the robot and try again.' });
        let unsubscribe = (): void => {};
        const finish = (r: FinalResult): void => {
          unsubscribe();
          signal.removeEventListener('abort', onAbort);
          done(r);
        };
        const onAbort = (): void => finish({ status: 'cancelled', reason: 'Cancelled.', measurements: measure(), fallback: null });
        unsubscribe = n.store.subscribe(() => {
          const s = n.state;
          if (s.driving === 'auto') return;
          if (s.reached) return finish({ status: 'succeeded', reason: `${s.message ?? 'Reached the goal.'}${walked}`, measurements: measure(), fallback: null });
          if (s.driving === 'remote') return finish({ status: 'cancelled', reason: 'Keyboard driving took over; the navigation run ended.', measurements: measure(), fallback: null });
          if (s.message) return finish({ status: 'failed', reason: `${s.message}${walked}`, measurements: measure(), fallback: s.collided ? 'Reset the robot, then choose a goal with more clearance.' : s.fallen ? 'Reset the robot; it fell over while walking.' : 'Choose another goal, or reset the map.' });
          finish({ status: 'cancelled', reason: 'The run was stopped before the goal was reached.', measurements: measure(), fallback: null });
        });
        signal.addEventListener('abort', onAbort, { once: true });
      }),
  };
}

/** Skill memory as text: live runs on this body, counted from every playback. */
function liveRunsText(memory: LongTermMemory, a: SkillArtifact): string {
  const r = memory.skillRuns(a.id);
  if (!r) return 'no live runs recorded';
  const last = r.last.outcome === 'succeeded' ? 'the last one succeeded' : `the last one ${OUTCOME_LABELS[r.last.outcome]}${r.last.failures.length ? ` (${r.last.failures.join(', ')})` : ''}`;
  return `live runs: ${r.completed} completed, ${r.succeeded} succeeded${r.interrupted ? `, ${r.interrupted} stopped early` : ''}; ${last}`;
}

const pct = (v: number): string => `${Math.round(v * 100)} %`;

/**
 * Where an arm's tip is with every joint at 0 (the links of the chain presets
 * point straight up then): its height, which a tipHeight goal measures, and
 * its offset from the home pose, which a tipReach goal measures.
 */
function zeroTipText(robot: RobotDescription): string {
  const km = buildKinematicModel(robot);
  const i = robot.limbs.findIndex((l) => l.role !== 'wheel');
  const zero = forwardKinematics(km, clampToLimits(km, km.joints.map(() => 0)), baseTransform(km)).tips[i];
  const home = forwardKinematics(km, homeConfiguration(robot), baseTransform(km)).tips[i];
  if (!zero || !home) return 'it has no limb tip';
  const f = (v: number): string => v.toFixed(3);
  return `with every joint at 0 the ${robot.limbs[i]!.id} tip is ${f(zero[14]!)} m above the ground (the joints sag a little under gravity, so a tipHeight goal asks for at most ${f(0.95 * zero[14]!)} m), (${f(zero[12]! - home[12]!)}, ${f(zero[13]! - home[13]!)}, ${f(zero[14]! - home[14]!)}) m from where it starts in the home pose`;
}
/** Characters of the skill list in the Skills tab's system text (the proxy allows 16,000 per message). */
const SKILL_LIST_CHARS = 5000;

/** The first motion word of a taught skill's name (or of the request) that none of its goals measures, if any (D25). */
function unmeasured(a: SkillArtifact | undefined, robot: RobotDescription, request = ''): IntentFinding | undefined {
  return a?.taught ? checkIntent(a.taught.name, request, a.taught.goals, robot).find((f) => f.goalIndex < 0) : undefined;
}

function notThatMotion(name: string, f: IntentFinding): ToolResult {
  return rejected(`"${name}" does not do what was asked: "${f.word}" means ${f.meaning}, and none of its goals measures that.`, `Compose a new skill with draftSkill whose goals measure it (${f.hint}).`);
}

/** The checkpoints of a run as one line for the model (at most nine, evenly spread, and the end). */
function checkpointText(rep: PlayReport, fixedBase: boolean): string {
  const all = rep.checkpoints ?? [];
  if (all.length === 0) return '';
  const step = Math.ceil(Math.max(1, all.length - 1) / 8);
  const shown = all.filter((_, i) => i === all.length - 1 || i % step === 0);
  const parts = shown.map((c) => {
    const body = fixedBase ? '' : `body ${c.forwardM} m forward, ${c.leftM} m left, ${c.heightM} m high, pitch ${c.pitchDeg}, roll ${c.rollDeg}, turned ${c.turnDeg} deg`;
    const tips = c.tips.map((t) => `${t.limbId} tip moved (${t.x}, ${t.y}, ${t.z}) m`).join(', ');
    return `${c.label} at ${c.t} s: ${[body, tips, supportText(c)].filter(Boolean).join('; ')}`;
  });
  return ` Checkpoints (simulator ground truth; x forward, y left, z up from the start): ${parts.join(' | ')}.`;
}

/** Where the centre of mass is over the feet on the ground; left out while every foot is down and it is inside. */
function supportText(c: RunCheckpoint): string {
  const s = c.support;
  if (!s || (s.allFeet && s.marginM !== null && s.marginM > 0)) return '';
  if (s.marginM === null) return 'no foot on the ground';
  const on = `the feet on the ground (${s.feet.join(', ')})`;
  if (s.marginM > 0) return `center of mass ${s.marginM} m inside ${on}`;
  const way = (v: number, pos: string, neg: string) => `${Math.abs(v)} m ${v >= 0 ? pos : neg}`;
  return `center of mass ${-s.marginM} m outside ${on}, whose nearest edge is ${way(s.nearestForwardM, 'forward', 'back')} and ${way(s.nearestLeftM, 'left', 'right')}`;
}

/** The measured result of one try of a skill draft, goal by goal; `learn` when the request asked for a learned policy (or a keyframe balance on a free-standing body). */
function draftResult(name: string, rep: PlayReport, notes: readonly string[], checked: string, fixedBase = false, learn = false, keyframes = false): FinalResult {
  const goals = rep.goals ?? [];
  const met = goals.filter((g) => g.met).length;
  const measurements: Record<string, Measurement> = { goalsMet: { value: met, unit: 'count' }, goals: { value: goals.length, unit: 'count' } };
  goals.forEach((g, i) => {
    if (Number.isFinite(g.measured)) measurements[`goal ${i + 1} (${GOAL_LABELS[g.goal.kind]})`] = m(g.measured, g.unit);
  });
  Object.assign(measurements, { groundTruthMaxTilt: m(rep.maxTiltRad, 'rad'), groundTruthMinBaseHeight: m(rep.minBaseHeightM, 'm') });
  const status = rep.outcome === 'succeeded' ? 'succeeded' : rep.outcome === 'interrupted' ? 'cancelled' : 'failed';
  const balanceFailure = !fixedBase && keyframes && status === 'failed' && (
    goals.some((goal) => !goal.met && (goal.goal.kind === 'upright' || goal.goal.kind === 'noFall' || goal.goal.kind === 'standsAtEnd')) ||
    (rep.checks ?? []).some((check) => !check.met)
  );
  const each = goals.map((g, i) => `goal ${i + 1} ${g.met ? 'met' : 'not met'}: ${g.text}`).join('; ');
  const checks = (rep.checks ?? []).length ? ` Body checks (whatever the goals say): ${(rep.checks ?? []).map((c) => `${c.met ? 'passed' : 'FAILED'}: ${c.text}`).join('; ')}.` : '';
  const poses = notes.length ? ` Poses: ${notes.join('; ')}.` : '';
  const signConflict = notes.find((note) => /^body\.(pitch|roll) sign conflict:/.test(note));
  return {
    status,
    reason: `Tried the draft "${name}" once in the live simulation (one run, not evaluated, not saved): ${met} of ${goals.length} goals met${rep.reason ? ` (${rep.reason})` : ''}. ${each}.${checks}${poses}${checked}${checkpointText(rep, fixedBase)}`,
    measurements,
    fallback:
      status === 'succeeded'
        ? learn
          ? keyframes
            ? 'Every goal was met in this one run, open loop; keyframes do not react to a push or a slip. Call trainGaitPolicy now, in this command, so a feedback policy learned under pushes corrects the keyframes from the sensors of the body, then evaluateSkillDraft. One run is not an evaluation.'
            : 'Every goal was met in this one run, and the user asked for learning: call trainGaitPolicy now, in this command, so a learned feedback policy corrects the gait, then evaluateSkillDraft. One run is not an evaluation.'
          : 'Every goal was met in this one run. Call evaluateSkillDraft now, in this command, to evaluate and save it; one run is not an evaluation.'
        : status === 'failed' && signConflict
          ? `Revise the sketch with draftSkill before optimizing or training it. ${signConflict}.`
        : (learn || balanceFailure) && keyframes
          ? 'If the poses are wrong (the center of mass is not over the feet that stay down, or a goal measures the wrong thing), revise them with draftSkill. If the poses are right but the body cannot hold them open loop (it tips or falls), call trainGaitPolicy: reinforcement learning under pushes trains a feedback policy that balances them from the sensors of the body; then evaluateSkillDraft.'
          : 'Revise the poses, timing or goals that failed and call draftSkill again, or tell the user what the body could not do.',
  };
}

/** A keyframe draft on a free-standing body whose request asks for learning or for balancing on fewer feet: a learned feedback policy (D30) is the next step. */
function learnsKeyframes(robot: RobotDescription, request: string): boolean {
  return robot.base.kind === 'floating' && (asksForLearning(request) || BALANCE_REQUEST.test(request));
}

const BALANCE_REQUEST = /\b(hind legs|back legs|rear legs|two legs|2 legs|front legs|rear up|rears up|handstand|balance|balances|balancing)\b/i;

/** The next step when the draft's request asked for learning and the draft has no learned policy yet (D27, D30), else null. */
function learnNext(s: SkillsSession, text?: string): string | null {
  const d = s.state.draft;
  if (!d || d.policy) return null;
  if (d.gait) return asksForLearning(d.request) ? (text ?? 'The user asked for learning: call trainGaitPolicy now, so a learned feedback policy corrects the gait, then evaluateSkillDraft.') : null;
  return d.keyframes.length > 0 && learnsKeyframes(d.robot, d.request)
    ? (text ?? 'The user asked for learning or balance: call trainGaitPolicy now, so a feedback policy learned under pushes corrects the keyframes from the sensors of the body, then evaluateSkillDraft.')
    : null;
}

/** draftSkill: the poses or gait are checked first (a refusal runs nothing), then the draft is loaded and tried once. */
function draftExecutor(s: SkillsSession, args: Record<string, unknown>, request: string): TaskExecutor | ToolResult {
  const c = s.composeDraft({
    name: String(args['name']),
    ...(args['keyframes'] !== undefined ? { keyframes: args['keyframes'] as ComposedFrame[] } : {}),
    ...(args['gait'] !== undefined ? { gait: args['gait'] as GaitParams } : {}),
    endHoldS: typeof args['finalHoldSeconds'] === 'number' ? args['finalHoldSeconds'] : 1,
    goals: args['goals'] as SkillGoal[],
    request,
  });
  if (!c.ok) return rejected(`The draft was refused: ${c.reason}`, 'Correct it and call draftSkill again.');
  const name = s.state.draft?.name ?? String(args['name']);
  const fixedBase = s.state.robot.base.kind === 'fixed';
  const keyframes = args['gait'] === undefined;
  const learn = keyframes ? learnsKeyframes(s.state.robot, request) : asksForLearning(request);
  const checked = c.intent.length ? ` Request check: ${c.intent.map((f) => `"${f.word}" (${f.meaning}) is measured by goal ${f.goalIndex + 1}`).join('; ')}.` : '';
  return {
    run: async (signal): Promise<FinalResult> => {
      const r = await s.tryDraft(signal);
      if (!r.ok) return { status: signal.aborted ? 'cancelled' : 'failed', reason: r.reason, measurements: {}, fallback: null };
      return draftResult(name, r.report, c.notes, checked, fixedBase, learn, keyframes);
    },
  };
}

/** mirrorSkill: the mirror image of a saved skill becomes the draft (refused when the body is not mirror symmetric), then it is tried once. */
function mirrorExecutor(s: SkillsSession, args: Record<string, unknown>, request: string): TaskExecutor | ToolResult {
  const source = s.state.library.find((a) => a.name === args['skill']);
  if (!source) return rejected(`No saved skill named "${String(args['skill'])}".`, 'List the saved skills and use their exact names.');
  const c = s.mirrorSkill(source.id, request, typeof args['name'] === 'string' ? args['name'] : undefined);
  if (!c.ok) return rejected(`The mirror image of "${source.name}" was refused: ${c.reason}`, 'Compose the other side with draftSkill instead.');
  const fixedBase = s.state.robot.base.kind === 'fixed';
  const keyframes = !s.state.draft?.gait;
  const learn = keyframes ? learnsKeyframes(s.state.robot, request) : asksForLearning(request);
  const checked = c.intent.length ? ` Request check: ${c.intent.map((f) => `"${f.word}" (${f.meaning}) is measured by goal ${f.goalIndex + 1}`).join('; ')}.` : '';
  return {
    run: async (signal): Promise<FinalResult> => {
      const r = await s.tryDraft(signal);
      if (!r.ok) return { status: signal.aborted ? 'cancelled' : 'failed', reason: r.reason, measurements: {}, fallback: null };
      return draftResult(c.name, r.report, [`left-right mirror image of "${source.name}"`], checked, fixedBase, learn, keyframes);
    },
  };
}

/** improveSkillDraft: evolution strategies in the worker; the draft changes only when held-out trials score better. */
function improveExecutor(s: SkillsSession): TaskExecutor {
  return {
    stopText: 'The improvement was cancelled; the draft is unchanged.',
    run: async (signal): Promise<FinalResult> => {
      const r = await s.improveDraftForAi(signal);
      if (!r.ok) return { status: signal.aborted || r.cancelled ? 'cancelled' : 'failed', reason: r.reason, measurements: {}, fallback: signal.aborted || r.cancelled ? null : 'Compose a draft with draftSkill first.' };
      const x = r.result;
      const n = x.heldOutTrials;
      const measurements: Record<string, Measurement> = {
        baselineMeanReward: m(x.baseline.reward, 'reward'),
        resultMeanReward: m(x.result.reward, 'reward'),
        baselineTrialsMeetingEveryGoal: { value: x.baseline.successes, unit: 'count' },
        resultTrialsMeetingEveryGoal: { value: x.result.successes, unit: 'count' },
        heldOutTrials: { value: n, unit: 'count' },
        simulatedRuns: { value: x.rollouts, unit: 'count' },
        wallTime: m(r.wallTimeS, 's', 1),
      };
      return x.improved
        ? {
            status: 'succeeded',
            reason: `Trial and error improved "${r.name}": on ${n} held-out randomized trials the mean goal reward rose from ${x.baseline.reward.toFixed(3)} to ${x.result.reward.toFixed(3)} and the trials meeting every goal from ${x.baseline.successes} to ${x.result.successes} (${x.rollouts} simulated runs). The draft now uses the improved version; it is not saved until evaluateSkillDraft.`,
            measurements,
            fallback: learnNext(s),
          }
        : {
            status: 'failed',
            reason: `Trial and error found nothing better than "${r.name}" on ${n} held-out randomized trials (mean goal reward ${x.baseline.reward.toFixed(3)}, ${x.baseline.successes} of ${n} met every goal; ${x.rollouts} simulated runs). The draft is unchanged.`,
            measurements,
            fallback: 'Evaluate the draft as it is, or revise it with draftSkill.',
          };
    },
  };
}

/** trainGaitPolicy: augmented random search in the worker (D27 for a gait, D30 for keyframes under pushes); the draft keeps the policy only when held-out runs score better. */
function trainPolicyExecutor(s: SkillsSession, architecture: 'linear' | 'mlp'): TaskExecutor {
  return {
    stopText: 'The training was cancelled; the draft is unchanged.',
    run: async (signal): Promise<FinalResult> => {
      const r = await s.trainDraftPolicyForAi(signal, architecture);
      if (!r.ok) return { status: signal.aborted || r.cancelled ? 'cancelled' : 'failed', reason: r.reason, measurements: {}, fallback: signal.aborted || r.cancelled ? null : 'Compose a draft with draftSkill first.' };
      if (r.kind === 'pose') return posePolicyResult(r.name, r.result, r.wallTimeS);
      const x = r.result;
      const n = x.heldOutTrials;
      const b = x.baseline;
      const a = x.result;
      const measurements: Record<string, Measurement> = {
        gaitOnlyReturn: m(b.meanReturn, 'reward'),
        learnedReturn: m(a.meanReturn, 'reward'),
        gaitOnlyGoalReward: m(b.meanGoalReward, 'reward'),
        learnedGoalReward: m(a.meanGoalReward, 'reward'),
        gaitOnlyVelocityError: m(b.trackingErrorMps, 'm/s'),
        learnedVelocityError: m(a.trackingErrorMps, 'm/s'),
        gaitOnlyTrialsMeetingEveryGoal: { value: b.successes, unit: 'count' },
        learnedTrialsMeetingEveryGoal: { value: a.successes, unit: 'count' },
        gaitOnlyFalls: { value: b.falls, unit: 'count' },
        learnedFalls: { value: a.falls, unit: 'count' },
        heldOutTrials: { value: n, unit: 'count' },
        simulatedRuns: { value: x.rollouts, unit: 'count' },
        wallTime: m(r.wallTimeS, 's', 1),
      };
      const travel = (v: typeof b) => `${v.meanForwardM.toFixed(2)} m forward, ${v.meanLeftM.toFixed(2)} m left, ${((v.meanTurnRad * 180) / Math.PI).toFixed(0)} deg turned`;
      const [tf, tl, tw] = x.target;
      const compared = `on ${n} held-out randomized trials, gait alone versus with the policy: training return ${b.meanReturn.toFixed(3)} and ${a.meanReturn.toFixed(3)}, goal reward ${b.meanGoalReward.toFixed(3)} and ${a.meanGoalReward.toFixed(3)}, velocity error ${b.trackingErrorMps.toFixed(3)} and ${a.trackingErrorMps.toFixed(3)} m/s, trials meeting every goal ${b.successes} and ${a.successes}, falls ${b.falls} and ${a.falls}; mean travel ${travel(b)} versus ${travel(a)}. The reward asked for ${tf.toFixed(3)} m/s forward, ${tl.toFixed(3)} m/s left and ${tw.toFixed(3)} rad/s turning while stepping (the gait's own speed where it steps, none elsewhere) and the skill's goals (${x.parameters} parameters, ${x.rollouts} simulated runs)`;
      return x.improved
        ? {
            status: 'succeeded',
            reason: `A learned feedback policy now corrects the joints of "${r.name}" every 10 ms from the IMU and the gait phase: ${compared}. It is not saved until evaluateSkillDraft.`,
            measurements,
            fallback: null,
          }
        : {
            status: 'failed',
            reason: `The learned feedback policy did not beat the gait of "${r.name}" alone, so the draft is unchanged: ${compared}.`,
            measurements,
            fallback: 'Evaluate the gait as it is, or change the gait and train again.',
          };
    },
  };
}

/** The result of training a feedback policy for keyframes (D30): keyframes alone versus with the policy on the same held-out pushed runs. */
function posePolicyResult(name: string, x: PosePolicyResult, wallTimeS: number): FinalResult {
  const n = x.heldOutTrials;
  const b = x.baseline;
  const a = x.result;
  const [p0, p1] = x.pushImpulseNs;
  const measurements: Record<string, Measurement> = {
    keyframesOnlyTrialsMeetingEveryGoal: { value: b.successes, unit: 'count' },
    learnedTrialsMeetingEveryGoal: { value: a.successes, unit: 'count' },
    keyframesOnlyFalls: { value: b.falls, unit: 'count' },
    learnedFalls: { value: a.falls, unit: 'count' },
    keyframesOnlyGoalReward: m(b.meanGoalReward, 'reward'),
    learnedGoalReward: m(a.meanGoalReward, 'reward'),
    keyframesOnlyShareOfGoalsHolding: m(b.meanInstant, 'fraction'),
    learnedShareOfGoalsHolding: m(a.meanInstant, 'fraction'),
    keyframesOnlyReturn: m(b.meanReturn, 'reward'),
    learnedReturn: m(a.meanReturn, 'reward'),
    pushImpulseMin: m(p0, 'N*s'),
    pushImpulseMax: m(p1, 'N*s'),
    heldOutTrials: { value: n, unit: 'count' },
    simulatedRuns: { value: x.rollouts, unit: 'count' },
    wallTime: m(wallTimeS, 's', 1),
  };
  if (x.incumbent) {
    measurements['currentPolicySuccesses'] = m(x.incumbent.successes, 'count');
    measurements['currentPolicyFalls'] = m(x.incumbent.falls, 'count');
    measurements['currentPolicyGoalReward'] = m(x.incumbent.meanGoalReward, 'reward');
  }
  measurements['hiddenUnits'] = m(x.policy.layers.length > 2 ? x.policy.layers[1]! : 0, 'count');
  if (x.feedbackAblation) {
    measurements['sensorAblationSuccesses'] = m(x.feedbackAblation.successes, 'count');
    measurements['sensorAblationFalls'] = m(x.feedbackAblation.falls, 'count');
    measurements['sensorAblationGoalReward'] = m(x.feedbackAblation.meanGoalReward, 'reward');
  }
  const ablation = x.feedbackAblation ? ` Sensor ablation (inputs fixed at training means, phase preserved): ${x.feedbackAblation.successes}/${n} successes, ${x.feedbackAblation.falls} falls, goal reward ${x.feedbackAblation.meanGoalReward.toFixed(3)}. This is a paired diagnostic, not proof of general feedback robustness.` : '';
  const compared = `on ${n} held-out randomized runs, each pushed once with ${p0} to ${p1} N*s at a random moment and direction, keyframes alone (open loop) versus with the policy (closed loop): every goal met in ${b.successes} and ${a.successes}, falls ${b.falls} and ${a.falls}, goal reward ${b.meanGoalReward.toFixed(3)} and ${a.meanGoalReward.toFixed(3)}, share of goals holding over time ${b.meanInstant.toFixed(2)} and ${a.meanInstant.toFixed(2)} (${x.parameters} parameters, ${x.rollouts} simulated runs)`;
  return x.improved
    ? {
        status: 'succeeded',
        reason: `A feedback policy learned by reinforcement learning under pushes now corrects the joints of "${name}" every 10 ms from the IMU, the foot contacts and the joint encoders: ${compared}.${ablation}${a.successes === 0 ? ' No held-out run met every goal yet: the policy only came closer to them.' : ''} It is not saved until evaluateSkillDraft.`,
        measurements,
        fallback: null,
      }
    : {
        status: 'failed',
        reason: `${x.incumbent ? `The candidate did not beat both the keyframes and the current feedback policy of "${name}"` : `The learned feedback policy did not beat the keyframes of "${name}" alone`}, so the draft is unchanged: ${compared}.${x.incumbent ? ` Current policy: ${x.incumbent.successes}/${n} successes, ${x.incumbent.falls} falls, goal reward ${x.incumbent.meanGoalReward.toFixed(3)}.` : ''}${ablation}`,
        measurements,
        fallback: 'Revise the poses so the center of mass is over the feet that stay down and train again, or evaluate the keyframes as they are.',
      };
}

const rangeText = ([lo, hi]: readonly [number, number]): string => `${lo} to ${hi}`;

function walkingRanges(c: LocomotionController): string {
  return `forward ${rangeText(c.ranges.forwardMps)} m/s, sideways ${rangeText(c.ranges.leftMps)} m/s (positive left), turning ${rangeText(c.ranges.turnRadps)} rad/s (positive left)`;
}

/** trainLocomotion: calibration and a feedback policy for this body in the worker pool (D28); it replaces an earlier controller of this morphology. */
function trainLocomotionExecutor(train: (signal: AbortSignal) => Promise<WalkingTraining>, afterwards: string): TaskExecutor {
  return {
    stopText: 'The training was cancelled; nothing was saved.',
    run: async (signal): Promise<FinalResult> => {
      const r = await train(signal);
      if (!r.ok) return { status: signal.aborted || r.cancelled ? 'cancelled' : 'failed', reason: r.reason, measurements: {}, fallback: null };
      const c = r.controller;
      const e = c.evaluation;
      const x = r.result;
      const g = x.heldOut.generator;
      const l = x.heldOut.learned;
      const measurements: Record<string, Measurement> = {
        evaluationSuccesses: { value: e.final.successes, unit: 'count' },
        evaluationTrials: { value: e.trials, unit: 'count' },
        meanPositionError: m(e.final.meanPositionErrorM, 'm'),
        meanHeadingError: m(e.final.meanHeadingErrorRad, 'rad'),
        uncalibratedSuccesses: { value: e.uncalibrated.successes, unit: 'count' },
        uncalibratedMeanPositionError: m(e.uncalibrated.meanPositionErrorM, 'm'),
        uncalibratedMeanHeadingError: m(e.uncalibrated.meanHeadingErrorRad, 'rad'),
        simulatedRuns: { value: x.rollouts, unit: 'count' },
        workers: { value: x.workers, unit: 'count' },
        wallTime: m(r.wallTimeS, 's', 1),
        ...(e.odometry ? { odometryPositionError: m(e.odometry.meanPositionErrorM, 'm') } : {}),
      };
      const ends = `position error ${l.meanPositionErrorM.toFixed(3)} against ${g.meanPositionErrorM.toFixed(3)} m, heading error ${l.meanHeadingErrorRad.toFixed(3)} against ${g.meanHeadingErrorRad.toFixed(3)} rad`;
      const policy = x.policy
        ? `and a feedback policy, kept because on ${x.heldOut.trials} held-out commands it beat the calibrated generator (return ${l.meanReturn.toFixed(3)} against ${g.meanReturn.toFixed(3)}, falls ${l.falls} and ${g.falls}) and ended at least as close to where the commands lead (${ends})`
        : `but no feedback policy: on ${x.heldOut.trials} held-out commands it did not beat the calibrated generator both in return and in where the runs ended (return ${l.meanReturn.toFixed(3)} against ${g.meanReturn.toFixed(3)}, ${ends})`;
      return {
        status: 'succeeded',
        reason:
          `Trained and saved this body's walking controller (${x.rollouts} simulated runs on ${x.workers} worker${x.workers === 1 ? '' : 's'}, ${r.wallTimeS.toFixed(1)} s): a calibration from the speed asked for to the generator's command, fitted on ${x.calibrationRuns.length} measured runs, ${policy}. ` +
          `Evaluation over ${e.trials} randomized commands (friction, mass and joint offsets varied): ${e.final.successes} ended within tolerance (95 % interval ${pct(e.final.wilson95[0])} to ${pct(e.final.wilson95[1])}), mean position error ${e.final.meanPositionErrorM.toFixed(3)} m, heading error ${e.final.meanHeadingErrorRad.toFixed(3)} rad; ` +
          `the generator given the same commands without the calibration: ${e.uncalibrated.successes}, ${e.uncalibrated.meanPositionErrorM.toFixed(3)} m, ${e.uncalibrated.meanHeadingErrorRad.toFixed(3)} rad.${e.odometry ? ` The controller's sensor-only estimate of where the body went ended ${e.odometry.meanPositionErrorM.toFixed(3)} m and ${e.odometry.meanHeadingErrorRad.toFixed(3)} rad from ground truth on average.` : ''} Ranges: ${walkingRanges(c)}. ${afterwards}`,
        measurements,
        fallback: null,
      };
    },
  };
}

/** locomote: one walk by command or to a target in the live simulation, measured against where the command leads. */
function locomoteExecutor(s: SkillsSession, args: Record<string, unknown>): TaskExecutor | ToolResult {
  const c = s.state.walking;
  if (!c) return rejected(`No walking controller is trained for ${s.state.robot.name} yet.`, 'Call trainLocomotion first.');
  const num = (k: string) => Number(args[k] ?? 0);
  const target = ['forwardM', 'leftM', 'turnRad'].some((k) => args[k] !== undefined) ? ([num('forwardM'), num('leftM'), num('turnRad')] as const) : null;
  let cmd: WalkCommand = { forwardMps: num('forwardMps'), leftMps: num('leftMps'), turnRadps: num('turnRadps') };
  let durationS = Number(args['durationS']);
  if (target) {
    const plan = walkForTarget(target, c.ranges, c.base.cycleS);
    if ('problem' in plan) return rejected(`The walk was refused: ${plan.problem}`, `Ranges: ${walkingRanges(c)}.`);
    cmd = plan.command;
    durationS = plan.durationS;
  }
  const problem = rangeProblem(cmd, c.ranges, durationS);
  if (problem) return rejected(`The walk was refused: ${problem}`, `Ranges: ${walkingRanges(c)}; durationS 1 to 10.`);
  return {
    run: async (signal): Promise<FinalResult> => {
      const r = await s.walk(cmd, durationS, signal);
      if (!r.ok) return { status: signal.aborted ? 'cancelled' : 'failed', reason: r.reason, measurements: {}, fallback: null };
      const w = r.report;
      const j = w.judgement;
      const f = (v: number) => v.toFixed(3);
      const d = (v: number) => deg(v).toFixed(0);
      const measurements: Record<string, Measurement> = {
        expectedForward: m(j.expected[0], 'm'), expectedLeft: m(j.expected[1], 'm'), expectedTurn: m(j.expected[2], 'rad'),
        measuredForward: m(j.measured[0], 'm'), measuredLeft: m(j.measured[1], 'm'), measuredTurn: m(j.measured[2], 'rad'),
        positionError: m(j.positionErrorM, 'm'), headingError: m(j.headingErrorRad, 'rad'), groundTruthMaxTilt: m(w.maxTiltRad, 'rad'),
      };
      return {
        status: w.outcome === 'interrupted' ? 'cancelled' : j.success ? 'succeeded' : 'failed',
        reason:
          `${target ? `Target ${f(target[0])} m forward, ${f(target[1])} m left, turned ${d(target[2])} deg: ` : ''}Walked ${cmd.forwardMps} m/s forward, ${cmd.leftMps} m/s left and ${cmd.turnRadps} rad/s turning for ${durationS} s with the learned walking controller (${w.policy ? 'calibrated generator and feedback policy' : 'calibrated generator'}): ${walkOutcomeText(w)}. ${w.continued ? 'It started where the body stood.' : 'It started from the home stance (the simulation was reset first).'} ` +
          `Where the command leads: ${f(j.expected[0])} m forward, ${f(j.expected[1])} m left, turned ${d(j.expected[2])} deg; measured (simulator ground truth, start frame): ${f(j.measured[0])} m forward, ${f(j.measured[1])} m left, turned ${d(j.measured[2])} deg. ` +
          `Position error ${f(j.positionErrorM)} m (tolerance ${f(j.toleranceM)}), heading error ${d(j.headingErrorRad)} deg (tolerance ${d(j.toleranceRad)}).${w.clamped ? ' The generator command hit its bounds, so the body could not reach that speed.' : ''}${w.reason ? ` (${w.reason})` : ''}`,
        measurements,
        fallback: j.success ? null : j.upright ? 'Call locomote again for the remaining distance and turn, measured from where the body stands now.' : 'The body did not stay up; walk more slowly.',
      };
    },
  };
}

/** runSkillSequence: saved skills one after another, each judged on its own. */
function sequenceExecutor(s: SkillsSession, args: Record<string, unknown>): TaskExecutor | ToolResult {
  const steps = (args['steps'] as { skill: string; times?: number }[]).map((x) => ({ skill: x.skill, times: x.times ?? 1 }));
  for (const step of steps) {
    const f = unmeasured(s.state.library.find((a) => a.name === step.skill), s.state.robot);
    if (f) return notThatMotion(step.skill, f);
  }
  return {
    run: async (signal): Promise<FinalResult> => {
      const r = await s.runSequence(steps, signal);
      if (!r.ok) return { status: 'failed', reason: r.reason, measurements: {}, fallback: 'List the saved skills and use their exact names.' };
      const lines = r.runs.map((x, i) => {
        const goals = x.report.goals;
        return `${i + 1}. "${x.skill}" ${OUTCOME_LABELS[x.report.outcome]}${goals ? ` (${goals.filter((g) => g.met).length} of ${goals.length} goals met)` : ''}`;
      });
      const succeeded = r.runs.filter((x) => x.report.outcome === 'succeeded').length;
      const done = r.stopped === null;
      return {
        status: done ? 'succeeded' : signal.aborted ? 'cancelled' : 'failed',
        reason: `${done ? 'Ran' : 'Stopped after'} ${r.runs.length} of ${r.planned} planned runs${r.stopped ? ` (${r.stopped})` : ''}: ${lines.join('; ')}. Each skill started from the home stance. Chance of success estimated beforehand from the evaluations: ${pct(r.predicted)}.`,
        measurements: {
          runsPlanned: { value: r.planned, unit: 'count' },
          runsDone: { value: r.runs.length, unit: 'count' },
          runsSucceeded: { value: succeeded, unit: 'count' },
          predictedSuccess: m(r.predicted, 'fraction'),
        },
        fallback: done ? null : 'Run the failed skill on its own to see what happened, or improve it.',
      };
    },
  };
}

const SKILL_ROBOTS: readonly PresetId[] = ['quadruped', 'hexapod', 'arm3', 'arm6'];

/** selectRobot: the Skills tab shows a preset or the Build tab's robot; skills bound to it come with it. */
function selectRobot(s: SkillsSession, which: string, built: () => RobotDescription | null): ToolResult {
  if (which === 'built') {
    const r = built();
    if (!r) return rejected('The Build tab has no robot yet.', 'Design one in the Build tab first (buildRobot there).');
    if (r === s.state.robot) return ok(`The Skills tab already shows ${r.name}.`);
    if (!s.setRobot(r)) return rejected(s.state.message?.text ?? 'The robot was refused.');
  } else if (SKILL_ROBOTS.includes(which as PresetId)) {
    // Replacing the robot clears the draft and the live run, so the same robot is not loaded again.
    if (s.state.source === which) return ok(`The Skills tab already shows ${s.state.robot.name}.`);
    s.setPreset(which as PresetId);
  } else {
    return rejected(`There is no robot "${which}" for the Skills tab.`, `Robots: ${SKILL_ROBOTS.join(', ')}, built.`);
  }
  const robot = s.state.robot;
  const label = PRESETS.find((p) => p.id === which)?.label ?? robot.name;
  return ok(`The Skills tab now shows ${label} (${robot.limbs.length} limbs, ${jointCount(robot)} joints, ${robot.base.kind} base); its saved skills and the body description come with the next tool result.`, {
    joints: { value: jointCount(robot), unit: 'count' },
    limbs: { value: robot.limbs.length, unit: 'count' },
  });
}

/** evaluateSkillDraft: 20 randomized trials in the worker; the skill is saved when at least one met every goal. */
function evaluateDraftExecutor(s: SkillsSession): TaskExecutor {
  return {
    stopText: 'The evaluation was cancelled; nothing was saved.',
    run: async (signal): Promise<FinalResult> => {
      const r = await s.evaluateDraftForAi(signal);
      if (!r.ok) return { status: signal.aborted || r.cancelled ? 'cancelled' : 'failed', reason: r.reason, measurements: {}, fallback: signal.aborted || r.cancelled ? null : 'Compose a draft with draftSkill first.' };
      const e = r.evaluation;
      const perCheck = (kind: BodyCheck['kind']) => e.results.filter((t) => t.checks?.find((c) => c.kind === kind)?.met !== false).length;
      const perGoal = [
        ...r.goals.map((g, i) => `${GOAL_LABELS[g.kind]} ${e.results.filter((t) => t.goals?.[i]?.met).length}/${e.trials}`),
        ...(e.results.some((t) => t.checks?.length) ? [`only the feet touched the ground ${perCheck('feetOnly')}/${e.trials}`, `never tipped over ${perCheck('noTipOver')}/${e.trials}`] : []),
      ].join(', ');
      const measurements: Record<string, Measurement> = {
        successes: { value: e.successes, unit: 'count' },
        trials: { value: e.trials, unit: 'count' },
        successRate: m(e.successRate, 'fraction'),
        wilson95Low: m(e.wilson95[0], 'fraction'),
        wilson95High: m(e.wilson95[1], 'fraction'),
        evaluationWallTime: m(r.wallTimeS, 's', 1),
      };
      if (!r.artifact) {
        return {
          status: 'failed',
          reason: `Evaluated "${r.name}": none of ${e.trials} randomized trials met every goal, so it was not saved. Trials per goal met: ${perGoal}.`,
          measurements,
          fallback: 'Revise the draft with draftSkill, starting with the goal met least often, or tell the user what the body could not do.',
        };
      }
      return {
        status: 'succeeded',
        reason: `Evaluated "${r.name}": ${e.successes} of ${e.trials} randomized trials met every goal (95 % interval ${pct(e.wilson95[0])} to ${pct(e.wilson95[1])}; friction, body mass and joint offsets varied). Saved to the library as "${r.artifact.name}": executeSkill runs it under that name, and the command "${r.name}" runs it. Trials per goal met: ${perGoal}.`,
        measurements,
        // Domain randomization exposes the conditions a skill fails under; trial and error over the same ranges can make it more reliable.
        fallback: learnNext(s, 'Saved without a learned feedback policy, although the request asks for learning or balance: call trainGaitPolicy on the draft, then evaluateSkillDraft again; commands run the version with the better evaluation.') ?? (e.successRate < 0.8 ? 'It is saved, but some randomized trials missed a goal. If it should be more reliable, call improveSkillDraft on the draft and evaluate it again; the command then runs the better version.' : null),
      };
    },
  };
}

/** Skills tab: the robot, its evaluated skills (taught by the user, and the foot-lift example) and their live run records. */
async function skillsSnapshot(s: SkillsSession, memory: LongTermMemory, built: () => RobotDescription | null): Promise<BridgeSnapshot | { unsupported: string }> {
  const st = s.state;
  const robot = st.robot;
  const hash = st.hash ?? (await morphologyHash(robot));
  const usable = st.library.filter((a) => checkSkillCompatibility(a, robot, hash).ok && commandEligibility(a) === null);
  const callable = usable.filter((a) => SKILL_NAME_PATTERN.test(a.name));
  const bodyKind: RobotCapabilities['bodyKind'] = robot.limbs.some((l) => l.wheel) ? 'wheeled' : robot.base.kind === 'fixed' ? 'arm' : 'legged';
  const walking = bodyKind === 'legged' && st.walking?.robot.morphologyHash === hash ? st.walking : null;
  const caps: RobotCapabilities = {
    name: robot.name, morphologyHash: hash, bodyKind, jointCount: jointCount(robot), skills: callable.map((a) => a.name), places: [],
    ...(bodyKind === 'legged' ? { walking: walking ? walking.ranges : null } : {}),
    features: { ...NO_FEATURES, skillAuthoring: true, skillSequences: true, robotChoice: true },
  };
  const limbName = (id: string) => robot.limbs.find((l) => l.id === id)?.name.toLowerCase() ?? id;
  const goalsOf = (a: SkillArtifact) => (a.taught ? `; succeeds when ${a.taught.goals.map((g) => goalText(g, limbName)).join(', and ')}` : '');
  const fitOf = (a: SkillArtifact) => {
    const f = unmeasured(a, robot);
    return f ? `; its goals do not measure "${f.word}" (${f.meaning}), so it is not that motion` : '';
  };
  const listing = callable.map((a) => `"${a.name}" (${skillLabel(a)}, evaluation ${a.evaluation ? `${a.evaluation.successes}/${a.evaluation.trials}` : 'none'}; ${liveRunsText(memory, a)}${goalsOf(a)}${fitOf(a)})`);
  // The skill list closes the system text and is shortened to fit it: without goals first, then cut (listSkills has the rest).
  let skillList = listing.join(', ');
  if (skillList.length > SKILL_LIST_CHARS) skillList = `${callable.map((a) => `"${a.name}" (${skillLabel(a)}, evaluation ${a.evaluation ? `${a.evaluation.successes}/${a.evaluation.trials}` : 'none'}${fitOf(a)})`).join(', ')}; listSkills with a skill name gives its goals`;
  if (skillList.length > SKILL_LIST_CHARS) skillList = `${skillList.slice(0, SKILL_LIST_CHARS)}... (listSkills gives the whole list)`;
  const kindText = bodyKind === 'arm' ? 'a fixed-base arm' : bodyKind === 'wheeled' ? 'a wheeled robot' : 'a legged robot';
  const draft = st.draft && st.draft.robot === robot ? st.draft : null;
  const summary = [
    `Robot: ${robot.name}, ${kindText} in the Skills tab (morphology ${hash.slice(0, 12)}).`,
    'Skills are keyframe motions or gaits judged by their goals (set by the user or composed by you), or the built-in foot lift.',
    'Live runs (skill memory) count every playback on this body, from the Play button, commands and AI; they are a record, not an evaluation.',
    'executeSkill plays one in the live simulation; a referee judges the run from simulator ground truth (a taught skill by each of its goals; a foot lift by foot clearance, hold time, tilt and support contact). holdSeconds (0.5 to 5) applies to adjustable foot lifts only. Skills are never reused on a different body. runSkillSequence runs several saved skills one after another (for "twice" or "then").',
    walking ? `Walking, turning and side steps by a distance or an angle are done with locomote (this body's learned walking controller, below), not with draftSkill, mirrorSkill or a saved gait skill, unless the user asks to teach or save a named skill.` : '',
    'When the user asks for a motion, act: run a saved skill that does it with executeSkill, or compose one; do not only suggest a command. When the user asks for a motion that no saved skill does, compose it instead of refusing. Procedure: 1. draftSkill loads poses or a gait, and goals, into the Teach panel and tries them once. 2. Read every measured goal and the checkpoints; if a goal was missed, change what the checkpoints point to (for example a later keyframe that undid the pose) and call draftSkill again under the same name, at most three drafts in all. 3. If the best draft still misses a goal by a little, call improveSkillDraft once (up to a minute; it searches the poses and timing, or the gait parameters, against the goals). 4. When a try or the improvement met every goal, call evaluateSkillDraft in the same command, so the skill is saved and runs by name next time; when the user asked for learning (a learned policy, controller or network), or for balancing on fewer feet (two legs, the hind legs), call trainGaitPolicy first (for a gait, or for keyframes on a free-standing body). 5. Answer with the measured result. Never end your answer right after a failed try. For the other side of a saved skill (a left wave from a right wave, a right turn from a left turn, a lean the other way), call mirrorSkill with that skill instead of composing it again, then evaluateSkillDraft. To work on another robot, call selectRobot first.',
    'Poses are relative to the home stance, and what a pose leaves out stays at home. body moves the body (m and rad; positive roll raises the left side, positive pitch lowers the front, positive yaw turns left) while the feet stay where they stand. tips moves a limb tip from where it rests (m, in the home body frame). joints sets joint targets (rad) within their limits. A legged body falls when its weight is not over the feet on the ground: before lifting a foot, shift the body away from it; on three feet or more a pose can hold open loop, on fewer it needs a learned feedback policy. A jump needs a crouch and then an extension within about 0.1 s.',
    robot.base.kind === 'floating'
      ? 'Keyframes play open loop: they cannot hold an unstable pose (balancing on two legs, standing on the hind legs) or recover from a push. trainGaitPolicy also trains a feedback policy for a keyframe draft by reinforcement learning (about five minutes): in randomized runs in which the body is pushed at a random moment, a small network learns to correct every joint target every 10 ms from the IMU, the foot contacts and the joint encoders, rewarded by the share of the goals that hold at each moment and by the goals at the end; the draft keeps it only when, on held-out pushed runs, it comes clearly closer to the goals than the keyframes alone without more falls. Your keyframes only sketch the motion; the policy learns the balance. Use it for balance on fewer feet, for landings, and when the user asks for a learned or robust motion; sketch the poses so the center of mass is over the feet that stay down, then train even if the open-loop try fell.'
      : '',
    'Walking, turning, side steps and stepping in place use a gait, not keyframes: a contact pattern with a cycle time, stride per cycle, step height and duration; IMU feedback (feedbackKp, feedbackKd) raises or lowers feet against body tilt, which makes it closed loop. A gait draft can also get a learned feedback policy: trainGaitPolicy (up to two minutes) trains a small network by random search that corrects every joint target every 10 ms from the IMU and the gait phase, rewarded for keeping the measured speed of the gait where it steps, no drift elsewhere, staying upright and meeting the goals; use it when the user asks for a learned or more accurate gait, or when a gait drifts or wobbles, then call evaluateSkillDraft. Start from a trot (diagonal legs together): dutyFactor 0.6, cycleS 0.5, stepHeightM 0.04. A walk or wave pattern lifts one leg at a time, is slower, and needs dutyFactor of at least 1 - 1/legs (0.75 on four legs). Measured on the quadruped preset over 4 s (5 randomized trials, defaults otherwise): strideM 0.06 went 0.39 m forward, strideM 0.1 went 0.75 m, strideM -0.06 went 0.75 m backward, turnRad 0.3 with no stride turned 3.2 rad, lateralM 0.04 went 0.36 m left but also turned 0.4 rad; the hexapod tripod went 0.73 m with strideM 0.06. A side step keeps strideM near 0 (strideM moves the body forward). Scale durationS to the distance or angle asked and check the measured travel; for "about" an amount give the travel or turn goal both a minimum and a maximum (for example 75 to 105 deg for about 90). Measure walking with a travel goal and turning with a turn goal, plus upright.',
    'Goals must measure what the user asked for (a sit pitches the body nose up, a wave swings a limb tip, a bow pitches it nose down); a draft whose goals miss a motion word of the request is refused. Never weaken a goal below what the user asked for to make a draft pass (a threshold you chose stricter than the request may be corrected). When the body cannot do it, say so and quote the measured values.',
    robot.base.kind === 'floating'
      ? 'How motions work on a legged body (choose every number yourself from the body description above, and let the measured try correct it): a sit lowers the rear so the body pitches nose up while the rear feet stay down; a bow lowers the front; a lean rolls the body toward one side; lying down lowers the body toward the ground; standing tall raises it; to lift or wave a leg, first move the body away from that leg so the other feet carry the weight, and keep that shift in every keyframe while the leg is up; a jump crouches, then extends every leg within about 0.1 s, and must land; standing on fewer feet (two legs, the hind legs) puts the center of mass over the feet that stay down and needs a learned feedback policy to hold it.'
      : `How motions work on an arm (choose every number yourself from the body description above, and let the measured try correct it): a wave raises the tip and swings it sideways back and forth (tipSwing); pointing or reaching moves the tip to a point (tipReach); for pointing straight up, ${zeroTipText(robot)}; a nod moves the tip down and up (tipSwing on z). A tip target beyond the arm's reach stops short; the try reports how far.`,
    draft
      ? `The Teach panel holds a draft "${draft.name || 'unnamed'}" (${draft.gait ? `a gait: ${gaitText(draft.gait)}` : `${draft.keyframes.length} keyframes`}), ${draft.author === 'user' ? 'posed by the user' : draft.author === 'ai' ? 'composed by you' : 'composed by you and edited by the user'}${draft.training ? ', improved by trial and error' : ''}${draft.policy ? ', with a learned feedback policy' : ''}.`
      : 'The Teach panel holds no draft.',
    bodyKind !== 'legged'
      ? ''
      : walking
        ? `This body has a learned walking controller (D28; ${walkingRanges(walking)}; ${walking.evaluation.final.successes} of ${walking.evaluation.trials} randomized commands ended within tolerance, mean position error ${walking.evaluation.final.meanPositionErrorM.toFixed(3)} m${walking.policy ? '; with a learned feedback policy' : ''}). To walk, turn or step sideways by a distance or an angle now, call locomote with the target (forwardM, leftM, turnRad, measured from where the body stands; it picks the speeds and the duration); give speeds in these ranges with durationS only when the user asks for a speed or a time. It reports where the body went, so call it again to correct a remaining error. Walks and saved gait skills start where the body stands (when it is up), so they chain into a path; keyframe skills and foot lifts start from the home stance at the origin. For a path (a square, a zigzag), send its locomote calls in one answer: they run in order and stop at the first that does not succeed. Compose a gait with draftSkill only when the user wants a named skill to save.`
        : 'This body has no learned walking controller yet. When the user asks it to learn to walk, or for a walking controller, call trainLocomotion (four to eight minutes, once per body), then locomote walks by speed or to a target.',
    describeBody(robot),
    `Skills evaluated here on this exact body: ${callable.length ? skillList : 'none yet'}.`,
  ].join(' ');
  const query = (tool: string, args: Record<string, unknown> = {}): ToolResult => {
    const now = s.state;
    switch (tool) {
      case 'getRobotCapabilities':
        return ok(capabilityText(caps, `getRobotState, listSkills, draftSkill, improveSkillDraft, ${bodyKind === 'legged' ? `trainGaitPolicy, trainLocomotion, ${walking ? 'locomote, ' : ''}` : ''}evaluateSkillDraft${caps.skills.length ? ', executeSkill, runSkillSequence, mirrorSkill' : ''}, selectRobot, cancelTask`));
      case 'listSkills': {
        const one = typeof args['skill'] === 'string' ? callable.find((a) => a.name === args['skill']) : undefined;
        const measurements: Record<string, Measurement> = {};
        for (const a of one ? [one] : callable) {
          if (a.evaluation) measurements[`${a.name} success rate`] = m(a.evaluation.successRate, 'fraction');
          const runs = memory.skillRuns(a.id);
          if (runs) {
            measurements[`${a.name} live runs completed`] = { value: runs.completed, unit: 'count' };
            measurements[`${a.name} live runs succeeded`] = { value: runs.succeeded, unit: 'count' };
          }
        }
        if (one) return ok(listing[callable.indexOf(one)]!, measurements);
        const omitted = usable.length - callable.length;
        const brief = callable.map((a) => `"${a.name}" ${a.evaluation ? `${a.evaluation.successes}/${a.evaluation.trials}` : 'not evaluated'}${unmeasured(a, robot) ? ' (its goals do not measure its name)' : ''}`);
        return ok(`${brief.length ? `${brief.join('; ')}. Call listSkills with a skill name for its goals and live runs.` : 'No skills have been evaluated here for this robot.'}${omitted ? ` ${omitted} more cannot be called by name (unusual characters in the name).` : ''}`, measurements);
      }
      case 'getRobotState': {
        const last = now.lastPlay;
        const measurements: Record<string, Measurement> = !last ? {} : last.goals ? { lastPlayGoalsMet: { value: last.goals.filter((g) => g.met).length, unit: 'count' }, lastPlayGoals: { value: last.goals.length, unit: 'count' }, lastPlayMaxTilt: m(last.maxTiltRad, 'rad') } : { lastPlayGroundTruthHold: m(last.holdS, 's', 2), lastPlayMaxClearance: m(last.maxClearanceM, 'm') };
        return ok(`${now.playing ? 'A skill is playing.' : now.teleop.active ? 'Demonstration control is active.' : 'No skill is playing; the robot holds its commanded pose.'}${last ? ` Last playback (${last.skillName}): ${last.outcome} (simulator ground truth).` : ''}`, measurements);
      }
      default:
        return rejected(`${tool} is not available for this robot.`);
    }
  };
  const action = (tool: string, args: Record<string, unknown>, ctx?: { request: string }): TaskExecutor | ToolResult => {
    if (tool === 'draftSkill') return draftExecutor(s, args, ctx?.request ?? '');
    if (tool === 'mirrorSkill') return mirrorExecutor(s, args, ctx?.request ?? '');
    if (tool === 'evaluateSkillDraft') return evaluateDraftExecutor(s);
    if (tool === 'improveSkillDraft') return improveExecutor(s);
    if (tool === 'trainGaitPolicy') return trainPolicyExecutor(s, args['architecture'] === 'mlp' ? 'mlp' : 'linear');
    if (tool === 'trainLocomotion') return trainLocomotionExecutor((signal) => s.trainLocomotionForAi(signal), 'locomote walks with it.');
    if (tool === 'locomote') return locomoteExecutor(s, args);
    if (tool === 'runSkillSequence') return sequenceExecutor(s, args);
    if (tool === 'selectRobot') return selectRobot(s, String(args['robot']), built);
    if (tool !== 'executeSkill') return rejected(`${tool} is not available for this robot.`);
    const name = String(args['skill']);
    const misfit = unmeasured(s.state.library.find((a) => a.name === name), robot, ctx?.request ?? '');
    if (misfit) return notThatMotion(name, misfit);
    const hold = typeof args['holdSeconds'] === 'number' ? args['holdSeconds'] : undefined;
    return {
      run: async (signal): Promise<FinalResult> => {
        const r = await s.runSkillByName(name, hold, signal);
        if (!r.ok) return { status: 'failed', reason: r.reason, measurements: {}, fallback: r.fallback };
        const rep = r.report;
        const d = describePlay(rep, r.limb, rep.requiredHoldS);
        const measurements: Record<string, Measurement> = {};
        if (rep.goals) {
          measurements['goalsMet'] = { value: rep.goals.filter((g) => g.met).length, unit: 'count' };
          measurements['goals'] = { value: rep.goals.length, unit: 'count' };
          rep.goals.forEach((g, i) => {
            if (Number.isFinite(g.measured)) measurements[`goal ${i + 1} (${GOAL_LABELS[g.goal.kind]})`] = m(g.measured, g.unit);
          });
          measurements['groundTruthMaxTilt'] = m(rep.maxTiltRad, 'rad');
        } else {
          Object.assign(measurements, {
            groundTruthHold: m(rep.holdS, 's', 2),
            requiredHold: m(rep.requiredHoldS, 's', 2),
            groundTruthMaxClearance: m(rep.maxClearanceM, 'm'),
            groundTruthMaxTilt: m(rep.maxTiltRad, 'rad'),
            supportContactFraction: m(rep.supportContactFraction, 'fraction'),
          });
        }
        return {
          status: rep.outcome === 'succeeded' ? 'succeeded' : rep.outcome === 'interrupted' ? 'cancelled' : 'failed',
          reason: `${d.text} ${r.note}`,
          measurements,
          fallback: d.suggestion ?? null,
        };
      },
    };
  };
  // A motion request answered without any action (a suggestion to run a skill) gets one reminder to act.
  const followUp = (steps: readonly ToolStep[], request: string): string | null => {
    if (steps.some((x) => x.accepted && ACTION_TOOLS.has(x.tool)) || isQuestionText(request)) return null;
    return checkIntent('', request, [], robot).length > 0
      ? 'The user asked the robot for a motion. Act now: run a saved skill that does it with executeSkill, or compose one with draftSkill; do not only suggest a command.'
      : null;
  };
  return { caps, summary, query, action, followUp };
}

const deg = (rad: number): number => (rad * 180) / Math.PI;

/** Joints of a Control lab robot, in the order of the lab's joint selector. */
function controlJoints(robot: RobotDescription): { label: string; angular: boolean; joint: JointDescription }[] {
  return robot.limbs.flatMap((l) => l.joints.map((j) => ({ label: `${l.name}: ${j.name}`, angular: j.type !== 'prismatic', joint: j })));
}

function gainText(s: ControllerSpec): string {
  return `kp ${round(s.gains.kp)}, kd ${round(s.gains.kd)}, ${s.gravityCompensation ? 'with' : 'without'} gravity compensation`;
}

type StepSetupLike = Pick<StepRunSetup, 'joint' | 'step' | 'durationS' | 'command'>;

function setupText(robot: RobotDescription, s: StepSetupLike, limits: boolean): string {
  const j = controlJoints(robot)[s.joint];
  if (!j) return `joint ${s.joint}`;
  const lim = limits ? ` (effort limit ${j.joint.limit.effort} ${j.angular ? 'N*m' : 'N'}, velocity limit ${j.joint.limit.velocity} ${j.angular ? 'rad/s' : 'm/s'})` : '';
  return `joint "${j.label}"${lim}, step ${s.step} ${j.angular ? 'deg' : 'm'}, ${s.durationS} s, ${s.command === 'rate-limited' ? 'target moved at the velocity limit' : 'target jumped at once'}`;
}

function stepResultText(r: StepTestResult, angular: boolean): string {
  if (r.unstable) return 'the simulation became numerically unstable';
  const x = r.metrics;
  const sse = !Number.isFinite(x.steadyStateError) ? 'not measured' : angular ? `${deg(x.steadyStateError).toFixed(2)} deg` : `${(x.steadyStateError * 1000).toFixed(1)} mm`;
  const share = Math.abs(x.steadyStateError / (r.target - r.y0)) * 100;
  const unsettled = restsOffTarget(r.qMeas, r.y0, r.target) ? `not settled (came to rest ${share.toFixed(0)} % of the step off target, outside the 2 % band)` : 'not settled';
  const parts = [
    `rise ${x.riseTimeS === null ? 'not reached' : `${x.riseTimeS.toFixed(3)} s`}`,
    `overshoot ${Number.isFinite(x.overshootPct) ? `${x.overshootPct.toFixed(1)} %` : 'not measured'}`,
    `settling ${x.settlingTimeS === null ? unsettled : `${x.settlingTimeS.toFixed(2)} s`}`,
    `steady-state error ${sse}`,
    `saturated ${(r.saturationFraction * 100).toFixed(0)} %`,
    `damping ratio ${r.zeta.toFixed(2)}`,
  ];
  return parts.join(', ') + (r.clamped ? ' (the step stopped at a joint limit)' : '');
}

function stepMeasurements(prefix: 'a' | 'b', r: StepTestResult, angular: boolean): Record<string, Measurement> {
  const out: Record<string, Measurement> = {};
  if (r.unstable) return out;
  const add = (key: string, v: number | null, unit: string, digits = 3): void => {
    if (v !== null && Number.isFinite(v)) out[`${prefix}${key}`] = m(v, unit, digits);
  };
  add('RiseTime', r.metrics.riseTimeS, 's');
  add('OvershootPct', r.metrics.overshootPct, '%', 1);
  add('SettlingTime', r.metrics.settlingTimeS, 's');
  add('SteadyStateError', r.metrics.steadyStateError, angular ? 'rad' : 'm', 4);
  add('SaturationFraction', r.saturationFraction, 'fraction');
  return out;
}

/** The shown step-test results as text and measurements, described with the setup they were produced with. */
function stepResults(robot: RobotDescription, setup: StepRunSetup, a: StepTestResult, b: StepTestResult | null): { text: string; measurements: Record<string, Measurement> } {
  const angular = controlJoints(robot)[setup.joint]?.angular ?? true;
  const parts = [`A (${gainText(setup.a)}): ${stepResultText(a, angular)}`];
  if (b && setup.b) parts.push(`B (${gainText(setup.b)}): ${stepResultText(b, angular)}`);
  return {
    text: `Step test of ${setupText(robot, setup, false)}. ${parts.join('; ')}.`,
    measurements: { ...stepMeasurements('a', a, angular), ...(b && setup.b ? stepMeasurements('b', b, angular) : {}) },
  };
}

function stabilityText(n: StabilityNote): string {
  return `${n.ok ? 'passes' : 'fails'} (wn*h ${n.wnH.toFixed(3)}, kd*h/I ${n.dampingNumber.toFixed(3)})`;
}

const sameSpec = (x: ControllerSpec, y: ControllerSpec): boolean => x.gains.kp === y.gains.kp && x.gains.kd === y.gains.kd && x.gravityCompensation === y.gravityCompensation;

/** The settings that differ between two controllers: kp, kd, gravity compensation. */
function settingChanges(x: ControllerSpec, y: ControllerSpec): string[] {
  const out: string[] = [];
  if (x.gains.kp !== y.gains.kp) out.push('kp');
  if (x.gains.kd !== y.gains.kd) out.push('kd');
  if (x.gravityCompensation !== y.gravityCompensation) out.push('gravity compensation');
  return out;
}

const listText = (items: string[]): string => (items.length < 2 ? items.join('') : `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`);

/**
 * Runs the lab's step test and reports the measured metrics. With a configuration it first sets gain set B
 * and compares A with B; without one it runs the lab as it is set and changes nothing.
 */
function experimentExecutor(c: ControlSession, spec: ControllerSpec | null): TaskExecutor {
  return {
    stopText: 'The experiment stopped; its unfinished results were discarded.',
    run: async (signal): Promise<FinalResult> => {
      const previousB = c.state.b;
      if (spec) {
        if (!c.setController('b', { kp: spec.gains.kp, kd: spec.gains.kd, gravityCompensation: spec.gravityCompensation })) {
          return { status: 'failed', reason: c.state.message ?? 'The gains were refused.', measurements: {}, fallback: null };
        }
        c.setCompare(true);
      }
      const expectA = c.state.a;
      const expectB = c.state.compare ? c.state.b : null;
      const beforeA = c.state.stepA, beforeB = c.state.stepB;
      const onAbort = (): void => c.cancel();
      signal.addEventListener('abort', onAbort, { once: true });
      try {
        await c.runStepTest();
      } finally {
        signal.removeEventListener('abort', onAbort);
      }
      if (signal.aborted) return { status: 'cancelled', reason: 'Cancelled.', measurements: {}, fallback: null };
      const s = c.state;
      const setup = s.stepSetup;
      // Only results of this run count: a setup change or a Stop in the lab leaves none or older ones.
      const fresh =
        !!setup && !!s.stepA && s.stepA !== beforeA && sameSpec(setup.a, expectA) &&
        (expectB ? !!s.stepB && s.stepB !== beforeB && !!setup.b && sameSpec(setup.b, expectB) : setup.b === null);
      if (!fresh || !s.stepA || !setup) {
        return { status: 'failed', reason: s.message ?? 'The step test did not finish: the lab setup changed or the run was stopped.', measurements: {}, fallback: 'Run the experiment again.' };
      }
      const r = stepResults(s.robot, setup, s.stepA, s.stepB);
      if (spec && s.stepB?.unstable) {
        return { status: 'failed', reason: `Gain set B made the simulation unstable. ${r.text}`, measurements: r.measurements, fallback: 'Lower kp or kd: the explicit-integration check needs wn*h below 0.2 and kd*h/I below 1.' };
      }
      const replaced = spec && !sameSpec(previousB, spec) ? ` Gain set B was ${gainText(previousB)} before this experiment.` : '';
      const fromA = spec ? settingChanges(expectA, spec) : [];
      const fromB = spec ? settingChanges(previousB, spec) : [];
      const mixed =
        fromA.length > 1 && fromB.length > 1
          ? ` This run changed more than one value at once (from A: ${listText(fromA)}; from B's earlier settings: ${listText(fromB)}), so it does not show the effect of one change alone.`
          : '';
      return { status: 'succeeded', reason: r.text + replaced + mixed, measurements: r.measurements, fallback: null };
    },
  };
}

/** Control lab: the experiment coach explains step tests and tries gains on gain set B. */
async function controlSnapshot(c: ControlSession): Promise<BridgeSnapshot | { unsupported: string }> {
  const st = c.state;
  if (st.status === 'error') return { unsupported: st.error ?? 'The Control lab simulation failed to load.' };
  if (st.status !== 'ready') return { unsupported: 'The Control lab simulation is still loading.' };
  const robot = st.robot;
  const caps: RobotCapabilities = {
    name: robot.name,
    morphologyHash: await morphologyHash(robot),
    bodyKind: robot.limbs.some((l) => l.wheel) ? 'wheeled' : robot.base.kind === 'fixed' ? 'arm' : 'legged',
    jointCount: jointCount(robot),
    skills: [],
    places: [],
    features: { ...NO_FEATURES, controllerExperiments: true },
  };
  const summary = [
    `Robot: ${robot.name} in the Control lab tab${st.onStand ? ', mounted on a fixed stand so only its joints move' : ''}. Each joint is held by a PD controller (torque = kp times the position error minus kd times the velocity, optionally plus gravity compensation), limited to the joint's effort limit.`,
    'A step test moves one joint by a set step while the others hold, and measures rise time (10 to 90 %), overshoot (percent of the step), settling time (2 % band), steady-state error and the fraction of time the actuator saturated. The damping ratio kd / (2 sqrt(kp I)) uses the joint inertia I at the home pose; gravity, coupling and saturation make the measured response differ from that linear model.',
    `Setup: ${setupText(robot, st, true)}. Gain set A (the learner's): ${gainText(st.a)}. Gain set B: ${gainText(st.b)}.`,
    "You are the experiment coach. For questions about results or how to improve them, first call getRobotState to read the shown step test; if none is shown, call runControllerExperiment without a configuration, which runs the step test as set and changes nothing. Explain from the measured metrics. To test a change, call runControllerExperiment with a configuration that differs in exactly one value (kp, kd or gravity compensation) from gain set A, or from gain set B when its results are shown, so that change can be judged alone; it sets gain set B and runs the step test for A and B. You cannot change gain set A, the joint or the step. Call a change an improvement only when the measured metrics show it, and for each goal the user named, quote the measured value from the latest result and say whether it meets the goal.",
    'In the linear model a larger kp speeds the rise and adds overshoot, and a larger kd damps overshoot and slows the rise. Under PD alone a joint comes to rest where kp times the error balances the gravity torque: kd does not change that steady-state error, gravity compensation removes it and a larger kp shrinks it. A response that comes to rest outside the 2 % band never settles, whatever kd is; the results say when that happened.',
  ].join(' ');
  const query = (tool: string): ToolResult => {
    const now = c.state;
    switch (tool) {
      case 'getRobotCapabilities':
        return ok(capabilityText(caps, 'getRobotState, runControllerExperiment, cancelTask'));
      case 'listSkills':
        return ok('The Control lab runs controller experiments; skills are kept in the Skills tab.');
      case 'getRobotState': {
        const parts = [`Gain set A: ${gainText(now.a)}; gain set B: ${gainText(now.b)}.`];
        if (now.stability) parts.push(`Explicit-integration check, lightest joint (needs wn*h below 0.2 and kd*h/I below 1): A ${stabilityText(now.stability.a)}, B ${stabilityText(now.stability.b)}.`);
        if (now.running) parts.push(`Running now: ${now.running.label}.`);
        if (!now.stepA || !now.stepSetup) {
          parts.push(`No step test results are shown; the setup is ${setupText(now.robot, now, false)}.`);
          return ok(parts.join(' '));
        }
        const r = stepResults(now.robot, now.stepSetup, now.stepA, now.stepB);
        parts.push(`Shown results: ${r.text}`);
        return ok(parts.join(' '), r.measurements);
      }
      default:
        return rejected(`${tool} is not available in the Control lab.`);
    }
  };
  const action = (tool: string, args: Record<string, unknown>): TaskExecutor | ToolResult => {
    if (tool !== 'runControllerExperiment') return rejected(`${tool} is not available in the Control lab.`);
    const cfg = args['configuration'] as { kp: number; kd: number; gravityCompensation: boolean } | undefined;
    return experimentExecutor(c, cfg ? { gains: { kp: cfg.kp, kd: cfg.kd }, gravityCompensation: cfg.gravityCompensation } : null);
  };
  return { caps, summary, query, action };
}

/** Physics tab: pushes and resets of the live simulation, measured from ground truth. */
async function physicsSnapshot(p: PhysicsSession): Promise<BridgeSnapshot | { unsupported: string }> {
  const st = p.state;
  if (st.status === 'error') return { unsupported: st.error ?? 'The Physics simulation failed to load.' };
  const robot = st.description;
  if (st.status !== 'ready' || !robot || !p.runtime) return { unsupported: 'The Physics simulation is still loading.' };
  const caps: RobotCapabilities = {
    name: robot.name, morphologyHash: await morphologyHash(robot), bodyKind: robot.limbs.some((l) => l.wheel) ? 'wheeled' : robot.base.kind === 'fixed' ? 'arm' : 'legged', jointCount: jointCount(robot), skills: [], places: [],
    features: { ...NO_FEATURES, physicsTools: true },
  };
  const summary = [
    `Robot: ${robot.name} in the Physics tab, simulated live with gravity and contact. Every joint holds its target under a PD controller (kp ${st.gains.kp}, kd ${st.gains.kd}); you cannot set torques or targets here.`,
    'getRobotState reads the body height and tilt from simulator ground truth. pushRobot applies a short impulse and reports the response over 2 s; resetSimulation returns the robot to its start. Skills run in the Skills tab, the step-response experiments in the Control lab.',
    describeBody(robot),
  ].join(' ');
  const query = (tool: string): ToolResult => {
    const rt = p.runtime;
    if (tool === 'getRobotCapabilities') return ok(capabilityText(caps, 'getRobotState, pushRobot, resetSimulation'));
    if (tool !== 'getRobotState' || !rt) return rejected(`${tool} is not available in the Physics tab.`);
    const b = rt.groundTruthBaseState();
    return ok(`${p.state.running ? 'The simulation is running' : 'The simulation is paused'}${p.state.unstable ? '; it became numerically unstable and needs a reset' : ''}. Body height ${b.pos[2].toFixed(3)} m, tilt ${((b.tilt * 180) / Math.PI).toFixed(1)} deg (simulator ground truth).`, {
      groundTruthBaseHeight: m(b.pos[2], 'm'),
      groundTruthTilt: m(b.tilt, 'rad'),
      simTime: m(rt.time, 's', 2),
    });
  };
  const action = (tool: string, args: Record<string, unknown>): TaskExecutor | ToolResult => {
    if (tool === 'resetSimulation') {
      p.reset();
      return ok('Reset the Physics simulation: the robot is back in its start state and home pose.');
    }
    if (tool !== 'pushRobot') return rejected(`${tool} is not available in the Physics tab.`);
    return pushExecutor(p, String(args['direction']), Number(args['impulseNs']));
  };
  return { caps, summary, query, action };
}

/**
 * Pushes along the body's heading and reports 2 s of simulated response
 * from the session's ground-truth telemetry (body tilt and height, and how
 * far the joints were driven from their targets).
 */
function pushExecutor(p: PhysicsSession, direction: string, impulseNs: number): TaskExecutor {
  return {
    stopText: 'The observation of the push stopped; the simulation keeps running.',
    run: (signal) =>
      new Promise<FinalResult>((resolve) => {
        const rt = p.runtime;
        const robot = p.state.description;
        if (!rt || !robot) return resolve({ status: 'failed', reason: 'The Physics simulation is not available.', measurements: {}, fallback: null });
        if (p.state.unstable || !p.state.running) return resolve({ status: 'failed', reason: p.state.unstable ? 'The simulation became numerically unstable; it needs a reset first.' : 'The Physics simulation is paused.', measurements: {}, fallback: p.state.unstable ? 'Call resetSimulation.' : 'Ask the user to resume the simulation.' });
        const floating = robot.base.kind === 'floating';
        const T = rt.groundTruthBaseTransform();
        const yaw = floating ? Math.atan2(T[1]!, T[0]!) : 0;
        const c = Math.cos(yaw), sn = Math.sin(yaw);
        const dir: [number, number, number] = direction === 'forward' ? [c, sn, 0] : direction === 'backward' ? [-c, -sn, 0] : direction === 'left' ? [-sn, c, 0] : [sn, -c, 0];
        const t0 = rt.time;
        const z0 = rt.groundTruthBaseState().pos[2];
        const joints = jointCount(robot);
        p.push(dir, impulseNs);
        const measure = (): { maxTilt: number; minZ: number; maxDev: number; values: Record<string, Measurement> } => {
          let maxTilt = 0, minZ = Infinity, maxDev = 0;
          const tel = p.telemetry;
          if (tel && p.runtime === rt) {
            const ti = tel.channelIndex('baseTilt'), zi = tel.channelIndex('baseHeight');
            for (let k = 0; k < tel.length; k++) {
              if (tel.time(k) < t0) continue;
              maxTilt = Math.max(maxTilt, tel.value(k, ti));
              minZ = Math.min(minZ, tel.value(k, zi));
              for (let j = 0; j < joints; j++) maxDev = Math.max(maxDev, Math.abs(tel.value(k, tel.channelIndex(`qMeas${j}`)) - tel.value(k, tel.channelIndex(`qCmd${j}`))));
            }
          }
          if (!Number.isFinite(minZ)) minZ = z0;
          const values: Record<string, Measurement> = { largestJointErrorFromTarget: m(maxDev, 'rad'), simTimeObserved: m(rt.disposed ? 0 : rt.time - t0, 's', 2) };
          if (floating) Object.assign(values, { groundTruthMaxTilt: m(maxTilt, 'rad'), groundTruthMinBaseHeight: m(minZ, 'm') });
          return { maxTilt, minZ, maxDev, values };
        };
        const finish = (r: FinalResult): void => {
          clearInterval(poll);
          clearTimeout(timer);
          signal.removeEventListener('abort', onAbort);
          resolve(r);
        };
        const onAbort = (): void => finish({ status: 'cancelled', reason: 'Cancelled.', measurements: measure().values, fallback: null });
        const timer = setTimeout(() => finish({ status: 'failed', reason: 'The simulation did not advance 2 s within 15 s; the Physics tab may be paused or hidden.', measurements: measure().values, fallback: null }), 15_000);
        const poll = setInterval(() => {
          if (rt.disposed || p.runtime !== rt || rt.time < t0) return finish({ status: 'failed', reason: 'The simulation was reset or replaced during the observation.', measurements: {}, fallback: null });
          if (rt.time - t0 < 2) return;
          const r = measure();
          const b = rt.groundTruthBaseState();
          const fell = floating && (b.tilt > 0.8 || (z0 > 0.05 && b.pos[2] < 0.5 * z0));
          const deg = (rad: number): string => ((rad * 180) / Math.PI).toFixed(1);
          const body = floating
            ? `the body tilted up to ${deg(r.maxTilt)} deg and its lowest height was ${r.minZ.toFixed(3)} m; it ${fell ? 'fell over' : `ended at ${deg(b.tilt)} deg tilt and ${b.pos[2].toFixed(3)} m`}`
            : 'the arm (pushed at its last link) recovered under its joint controllers';
          finish({
            status: 'succeeded',
            reason: `Pushed ${direction} with ${impulseNs} N*s. Over the next ${(rt.time - t0).toFixed(1)} s ${body}; the joints were driven up to ${deg(r.maxDev)} deg from their targets (simulator ground truth).`,
            measurements: r.values,
            fallback: fell ? 'Reset the simulation; a smaller push or stiffer joint gains may keep it up.' : null,
          });
        }, 100);
        signal.addEventListener('abort', onAbort, { once: true });
      }),
  };
}

const tipText = (t: readonly number[]): string => `(${t.map((v) => v.toFixed(3)).join(', ')}) m`;

/** Kinematics tab: geometry only, so the tools set joint values directly and solve IK without physics. */
async function kinematicsSnapshot(k: KinematicsSession): Promise<BridgeSnapshot> {
  const st = k.state;
  const robot = st.description;
  const targetable = robot.limbs.filter((l) => l.role !== 'wheel').map((l) => l.id);
  const caps: RobotCapabilities = {
    name: robot.name, morphologyHash: await morphologyHash(robot), bodyKind: robot.limbs.some((l) => l.wheel) ? 'wheeled' : robot.base.kind === 'fixed' ? 'arm' : 'legged', jointCount: jointCount(robot), skills: [], places: [],
    targetableLimbs: targetable,
    features: { ...NO_FEATURES, kinematicsTools: true },
  };
  const summary = [
    `Robot: ${robot.name} in the Kinematics tab. This tab is geometry only (forward and damped least-squares inverse kinematics within joint limits): nothing is simulated, so setting joints here is not commanding a physical robot.`,
    targetable.length
      ? 'moveTipTo solves IK for a limb tip to reach a world point and reports the status (converged, unreachable with the closest distance, or stalled); setJointAngles sets joints directly. Positions are in the world frame, m, z up.'
      : 'This robot has wheels only and no limb that can follow a position target; setJointAngles sets wheel joint angles directly.',
    describeBody(robot),
  ].join(' ');
  const tips = (): Record<string, Measurement> => {
    const out: Record<string, Measurement> = {};
    k.state.fk.tips.forEach((T, i) => {
      const id = robot.limbs[i]!.id;
      out[`${id} tip x`] = m(T[12]!, 'm');
      out[`${id} tip y`] = m(T[13]!, 'm');
      out[`${id} tip z`] = m(T[14]!, 'm');
    });
    return out;
  };
  const query = (tool: string): ToolResult => {
    if (tool === 'getRobotCapabilities') return ok(capabilityText(caps, targetable.length ? 'getRobotState, moveTipTo, setJointAngles' : 'getRobotState, setJointAngles'));
    if (tool !== 'getRobotState') return rejected(`${tool} is not available in the Kinematics tab.`);
    const s = k.state;
    const joints = s.model.joints.map((j, i) => `${j.id} ${s.q[i]!.toFixed(3)}`).join(', ');
    const limb = robot.limbs[s.limbIndex];
    const target = limb ? `Selected limb: ${limb.id}; IK target ${tipText(s.target)}${s.ik ? `, last IK ${s.ik.status} with ${(s.ik.residual * 1000).toFixed(1)} mm left` : ''}.` : 'No limb can follow a position target.';
    return ok(`Joint values (rad, m for sliding joints): ${joints}. ${target}`, tips());
  };
  const action = (tool: string, args: Record<string, unknown>): TaskExecutor | ToolResult => {
    if (tool === 'setJointAngles') {
      const list = args['joints'] as { joint: string; value: number }[];
      for (const j of list) {
        const i = k.state.model.joints.findIndex((x) => x.id === j.joint);
        if (i < 0) return rejected(`${robot.name} has no joint "${j.joint}".`, `Joints: ${k.state.model.joints.map((x) => x.id).join(', ')}.`);
      }
      for (const j of list) k.setJoint(k.state.model.joints.findIndex((x) => x.id === j.joint), j.value);
      const s = k.state;
      const clamped = list.filter((j) => Math.abs(s.q[s.model.joints.findIndex((x) => x.id === j.joint)]! - j.value) > 1e-9).map((j) => j.joint);
      return ok(`Set ${list.length} joint${list.length === 1 ? '' : 's'}${clamped.length ? `; ${clamped.join(', ')} stopped at a joint limit` : ''}.`, tips());
    }
    if (tool !== 'moveTipTo') return rejected(`${tool} is not available in the Kinematics tab.`);
    if (targetable.length === 0) return rejected('This robot has no limb whose tip can be placed.');
    const id = args['limb'];
    if (typeof id === 'string') {
      const li = robot.limbs.findIndex((l) => l.id === id);
      if (li >= 0) k.selectLimb(li);
      if (li < 0 || k.state.limbIndex !== li) return rejected(`${robot.name} has no limb "${id}" whose tip can be placed.`, `Limbs: ${targetable.join(', ')}.`);
    }
    k.setTarget([Number(args['x']), Number(args['y']), Number(args['z'])]);
    const ik = k.solve();
    if (!ik) return rejected('This robot has no limb whose tip can be placed.');
    const limb = robot.limbs[k.state.limbIndex]!;
    const T = k.state.fk.tips[k.state.limbIndex]!;
    const status = ik.status === 'converged' ? 'succeeded' : 'failed';
    const km = k.state.model;
    const lk = km.limbs[k.state.limbIndex]!;
    const joints = km.joints.slice(lk.start, lk.start + lk.count).map((j, i) => `${j.id} ${k.state.q[lk.start + i]!.toFixed(3)}`).join(', ');
    return {
      taskId: null,
      status,
      reason: `IK for the ${limb.name.toLowerCase()} tip: ${ik.status}. ${ik.message} The tip is at ${tipText([T[12]!, T[13]!, T[14]!])}; joints ${joints} (rad, within their limits).`,
      measurements: { residual: m(ik.residual, 'm', 4), iterations: { value: ik.iterations, unit: 'count' }, ...tips() },
      fallback: status === 'succeeded' ? null : 'Choose a point closer to the limb, or another limb.',
    };
  };
  return { caps, summary, query, action };
}

function structureText(robot: RobotDescription): string {
  const roles = robot.limbs.reduce<Record<string, number>>((acc, l) => ({ ...acc, [l.role]: (acc[l.role] ?? 0) + 1 }), {});
  return `${robot.name}: ${robot.base.kind} base of ${robot.base.mass} kg, ${Object.entries(roles).map(([r, n]) => `${n} ${r}${n === 1 ? '' : 's'}`).join(', ')}, ${jointCount(robot)} joints`;
}

/** Build tab: the AI designs robots through the builder's validated generators. */
async function buildSnapshot(b: BuilderSession): Promise<BridgeSnapshot> {
  const robot = b.state.robot;
  const caps: RobotCapabilities = {
    name: robot.name, morphologyHash: b.state.hash ?? (await morphologyHash(robot)), bodyKind: robot.limbs.some((l) => l.wheel) ? 'wheeled' : robot.base.kind === 'fixed' ? 'arm' : 'legged', jointCount: jointCount(robot), skills: [], places: [],
    features: { ...NO_FEATURES, designTools: true },
  };
  const summary = [
    `Build tab, version ${b.state.version}: ${structureText(robot)}.`,
    'buildRobot designs a robot from a preset (quadruped, hexapod, arm3, arm6, planar2, rover), as a legged robot (leg pairs 1 to 4, joints per leg 1 to 4, body length, width, height and mass, upper and lower leg length, joint effort) or as an arm (joints 1 to 7, reach). Every design is validated before it replaces the current one and becomes a new version with its own morphology; skills learned on another morphology never carry over.',
    'Act instead of asking: when the user names a kind of robot a preset covers, build the preset; for a custom legged robot, fill in what the user did not say from these defaults and name the ones you used: 2 leg pairs, 3 joints per leg, body 0.4 x 0.2 x 0.08 m and 4 kg, legs 0.16 + 0.16 m, effort 12 N*m. For an arm the defaults are 4 joints and 0.8 m reach.',
    'To teach the new robot, the user (or you, from the Skills tab) switches the Skills tab to it with selectRobot "built".',
  ].join(' ');
  const query = (tool: string): ToolResult => {
    if (tool === 'getRobotCapabilities') return ok(capabilityText(caps, 'getRobotState, buildRobot'));
    if (tool !== 'getRobotState') return rejected(`${tool} is not available in the Build tab.`);
    return ok(`${structureText(b.state.robot)}; version ${b.state.version}. ${describeBody(b.state.robot)}`);
  };
  const action = (tool: string, args: Record<string, unknown>): TaskExecutor | ToolResult => {
    if (tool !== 'buildRobot') return rejected(`${tool} is not available in the Build tab.`);
    const preset = args['preset'];
    const legged = args['legged'] as LeggedSpec | undefined;
    const arm = args['arm'] as { dof: number; reach: number } | undefined;
    const done = typeof preset === 'string' ? b.startFromPreset(preset as PresetId) : legged ? b.generateLegged(legged) : arm ? b.generateArm(arm.dof, arm.reach) : false;
    if (!done) return rejected(`The design was refused: ${b.state.issues.join(' ') || 'it could not be built.'}`, 'Change the values that the message names.');
    if (typeof args['name'] === 'string' && !b.rename(args['name'])) return rejected(`The robot was built but could not be renamed: ${b.state.issues.join(' ')}`);
    const r = b.state.robot;
    return ok(`Built ${structureText(r)} (version ${b.state.version}). ${describeBody(r)}`, { joints: { value: jointCount(r), unit: 'count' }, limbs: { value: r.limbs.length, unit: 'count' }, baseMass: m(r.base.mass, 'kg') });
  };
  return { caps, summary, query, action };
}

/** Connects the AI controller to the sessions of the active tab. */
export function workspaceBridge(w: WorkspaceAccess): RobotBridge {
  const built = (): RobotDescription | null => w.builder?.()?.state.robot ?? w.builtRobot?.() ?? null;
  return {
    contextKey() {
      const mode = w.mode();
      if (mode === 'navigate') {
        const n = w.nav();
        return n ? `navigate:${n.generation}:${n.state.mapId}:${n.state.mode}` : 'navigate:none';
      }
      if (mode === 'skills') return `skills:${identity(w.skills().state.robot)}`;
      if (mode === 'control') {
        // The gains are not part of the key: the coach's own experiments change gain set B.
        const c = w.control();
        return c ? `control:${identity(c.state.robot)}:${c.state.joint}:${c.state.step}:${c.state.durationS}:${c.state.command}` : 'control:none';
      }
      if (mode === 'physics') {
        const d = w.physics?.()?.state.description;
        return d ? `physics:${identity(d)}` : 'physics:none';
      }
      if (mode === 'kinematics') {
        const k = w.kinematics?.();
        return k ? `kinematics:${identity(k.state.description)}` : 'kinematics:none';
      }
      // The robot being designed is not part of the key: designing it is what the Build tab's commands do.
      return mode;
    },
    async snapshot() {
      const mode = w.mode();
      if (mode === 'navigate') {
        const n = w.nav();
        return n ? navSnapshot(n, w.memory(), built) : { unsupported: 'The Navigate simulation has not started.' };
      }
      if (mode === 'skills') return skillsSnapshot(w.skills(), w.memory(), built);
      if (mode === 'control') {
        const c = w.control();
        return c ? controlSnapshot(c) : { unsupported: 'The Control lab has not started.' };
      }
      if (mode === 'physics') {
        const p = w.physics?.();
        if (p) return physicsSnapshot(p);
      }
      if (mode === 'kinematics') {
        const k = w.kinematics?.();
        if (k) return kinematicsSnapshot(k);
      }
      if (mode === 'build') {
        const b = w.builder?.();
        if (b) return buildSnapshot(b);
      }
      const tab = MODES.find((x) => x.id === mode)?.label ?? mode;
      return { unsupported: `The ${tab} tab is not ready for AI commands yet.` };
    },
  };
}
